"""
Train the identification components of the MTG pipeline.

Run train_detector.py first to get mtg_data/yolo_card_best.pt. This script
trains everything else.

Key change from v1: synthetic augmentation now uses a "hard" pipeline that
matches what the benchmark throws at inference time - higher rotation,
glare, occlusion, color jitter, blur, and noise. This trades training
speed for inference robustness on hard photos.

  - Frame classifier: more synthetic exposure (60% vs 30% before) at hard
    intensity. The 66% accuracy on hard benchmark images was caused by the
    model never seeing those conditions during training.
  - Re-rankers: same higher synthetic ratio, with a controllable difficulty
    distribution (40% medium, 30% hard, 30% clean).

Inputs:
  mtg_data/cards_metadata.json
  mtg_data/card_images/{id}.jpg
  mtg_data/yolo_card_best.pt (optional)

Outputs:
  mtg_data/phash_db.npz                 (art + whole_card hashes + frame class)
  mtg_data/frame_classifier.pth
  mtg_data/art_reranker.pth
  mtg_data/art_embeddings.npy
  mtg_data/whole_reranker.pth
  mtg_data/whole_embeddings.npy
  mtg_data/set_classifier.pth
  mtg_data/set_codes.json
  mtg_data/frame_classes_cache.json

Usage:
  python train_identifier.py                    # full run with hard augmentation
  python train_identifier.py --diagnose         # inspect metadata, then exit
  python train_identifier.py --skip frame art   # skip components
  python train_identifier.py --skip-train       # rebuild DB and embeddings only
  python train_identifier.py --easy-aug         # use the old gentler augmentation
                                                # (faster training, worse robustness)
  python train_identifier.py --epochs 8         # shorter training
"""

import argparse
import json
import random
import sys
import time
import warnings
from collections import defaultdict, Counter
from pathlib import Path
from typing import Optional, List

import mlflow

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mtg_layout import (
    CARD_W, CARD_H,
    REGIONS_MODERN, crop_region, extract_art_crop, extract_whole_card,
    classify_frame_from_metadata, FRAME_CLASSES, FRAME_CLASS_TO_IDX,
    phash_64, dhash_64, hamming_distance_vectorized,
    inspect_metadata,
)

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = Path('mtg_data')
IMAGE_DIR = DATA_DIR / 'card_images'
METADATA_PATH = DATA_DIR / 'cards_metadata.json'

HASH_DB_PATH = DATA_DIR / 'phash_db.npz'
FRAME_CLF_PATH = DATA_DIR / 'frame_classifier.pth'
ART_RERANKER_PATH = DATA_DIR / 'art_reranker.pth'
ART_EMB_PATH = DATA_DIR / 'art_embeddings.npy'
WHOLE_RERANKER_PATH = DATA_DIR / 'whole_reranker.pth'
WHOLE_EMB_PATH = DATA_DIR / 'whole_embeddings.npy'
SET_CLF_PATH = DATA_DIR / 'set_classifier.pth'
SET_CODES_PATH = DATA_DIR / 'set_codes.json'
FRAME_CACHE_PATH = DATA_DIR / 'frame_classes_cache.json'

ART_INPUT = 160
WHOLE_INPUT = 224
FRAME_INPUT = 96
SETSYM_INPUT = 64
EMBEDDING_DIM = 256
DEFAULT_EPOCHS = 8
BATCH_SIZE = 128
SEED = 42


# ===========================================================================
# Loading helpers

def load_cards():
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        cards = json.load(f)
    available = [i for i, c in enumerate(cards)
                 if (IMAGE_DIR / f"{c['id']}.jpg").exists()]
    return cards, available


def load_card_image(card_id: str) -> Optional[np.ndarray]:
    path = IMAGE_DIR / f"{card_id}.jpg"
    if not path.exists():
        return None
    img = np.array(Image.open(path).convert('RGB'))
    if img.shape[0] != CARD_H or img.shape[1] != CARD_W:
        img = cv2.resize(img, (CARD_W, CARD_H))
    return img


# ===========================================================================
# Frame class precomputation

def compute_frame_classes(cards, available, use_image_fallback: bool = True,
                          force_recompute: bool = False) -> dict:
    cache: dict = {}
    if FRAME_CACHE_PATH.exists() and not force_recompute:
        with open(FRAME_CACHE_PATH) as f:
            cache = json.load(f)
        avail_ids = {cards[i]['id'] for i in available}
        cache = {k: v for k, v in cache.items() if k in avail_ids}
        missing = [i for i in available if cards[i]['id'] not in cache]
        if not missing:
            print(f"  Loaded {len(cache)} cached frame classes")
            return cache
        print(f"  Cache covers {len(cache)} / {len(available)} cards, "
              f"computing {len(missing)} more...")
        to_compute = missing
    else:
        print(f"  Computing frame class for {len(available)} cards "
              f"(use_image_fallback={use_image_fallback})...")
        to_compute = available

    for idx in tqdm(to_compute, desc="  classifying"):
        card = cards[idx]
        img_path = str(IMAGE_DIR / f"{card['id']}.jpg") if use_image_fallback else None
        cache[card['id']] = classify_frame_from_metadata(card, image_path=img_path)

    with open(FRAME_CACHE_PATH, 'w') as f:
        json.dump(cache, f)
    return cache


def report_frame_distribution(frame_classes: dict, label: str = "Frame distribution"):
    counts = Counter(frame_classes.values())
    total = sum(counts.values())
    print(f"  {label}:")
    for cls in FRAME_CLASSES:
        n = counts.get(cls, 0)
        pct = n / total * 100 if total else 0
        print(f"    {cls:<10} {n:>7,}  ({pct:.1f}%)")
    return counts


# ===========================================================================
# Background generation - shared between difficulty levels

def is_alt_art(card: dict) -> bool:
    """Identify alt-art / Universes-Beyond / weird-frame cards that are
    under-represented in the normal training distribution."""
    if card.get('border_color') == 'borderless':
        return True
    if card.get('full_art') is True:
        return True
    effects = card.get('frame_effects') or []
    if any(e in effects for e in ('extendedart', 'showcase', 'inverted',
                                   'etched', 'fullart', 'colorshifted')):
        return True
    return False


def add_sleeve_overlay(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Simulate a card sleeve with semi-transparent border and rainbow glare
    streaks, replicating what real phone photos look like."""
    h, w = image.shape[:2]
    out = image.copy().astype(np.float32)

    # Slight border expansion (sleeve edge)
    border = random.randint(2, max(3, int(min(h, w) * 0.03)))
    out[:border, :] = out[:border, :] * 0.7 + 40 * 0.3
    out[-border:, :] = out[-border:, :] * 0.7 + 40 * 0.3
    out[:, :border] = out[:, :border] * 0.7 + 40 * 0.3
    out[:, -border:] = out[:, -border:] * 0.7 + 40 * 0.3

    # Rainbow glare streak (horizontal or diagonal)
    n_streaks = random.randint(1, 3)
    for _ in range(n_streaks):
        y0 = random.randint(0, h - 1)
        thickness = random.randint(max(1, h // 30), max(2, h // 10))
        y1 = min(h, y0 + thickness)
        # Random rainbow color per streak
        color = np.array([random.randint(100, 255),
                          random.randint(100, 255),
                          random.randint(100, 255)], dtype=np.float32)
        streak_strength = random.uniform(0.15, strength)
        # Gradient mask across the streak
        mask = np.zeros((y1 - y0, w), dtype=np.float32)
        for row in range(y1 - y0):
            frac = row / max(1, y1 - y0 - 1)
            mask[row, :] = np.sin(frac * np.pi) * streak_strength
        for c in range(3):
            out[y0:y1, :, c] = out[y0:y1, :, c] * (1 - mask) + color[c] * mask

    return np.clip(out, 0, 255).astype(np.uint8)


def add_shadow_gradient(image: np.ndarray, strength: float = 0.4) -> np.ndarray:
    """Add a directional shadow gradient to simulate uneven indoor lighting."""
    h, w = image.shape[:2]
    direction = random.choice(['top', 'bottom', 'left', 'right', 'corner'])
    shadow = np.ones((h, w), dtype=np.float32)

    if direction == 'top':
        grad = np.linspace(1.0 - strength, 1.0, h).reshape(-1, 1)
        shadow = np.broadcast_to(grad, (h, w)).copy()
    elif direction == 'bottom':
        grad = np.linspace(1.0, 1.0 - strength, h).reshape(-1, 1)
        shadow = np.broadcast_to(grad, (h, w)).copy()
    elif direction == 'left':
        grad = np.linspace(1.0 - strength, 1.0, w).reshape(1, -1)
        shadow = np.broadcast_to(grad, (h, w)).copy()
    elif direction == 'right':
        grad = np.linspace(1.0, 1.0 - strength, w).reshape(1, -1)
        shadow = np.broadcast_to(grad, (h, w)).copy()
    else:  # corner
        cx = random.choice([0, w])
        cy = random.choice([0, h])
        ys, xs = np.mgrid[0:h, 0:w]
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).astype(np.float32)
        dist = dist / (dist.max() + 1e-6)
        shadow = 1.0 - strength * (1.0 - dist)

    out = image.astype(np.float32)
    for c in range(3):
        out[:, :, c] *= shadow
    return np.clip(out, 0, 255).astype(np.uint8)


def add_phone_camera_artifacts(image: np.ndarray) -> np.ndarray:
    """Simulate phone-camera artifacts: slight vignette, chromatic aberration,
    JPEG compression artifacts."""
    h, w = image.shape[:2]
    out = image.copy()

    # Vignette
    if random.random() < 0.5:
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2, h / 2
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        vignette = 1.0 - 0.3 * (dist / max_dist) ** 2
        for c in range(3):
            out[:, :, c] = np.clip(out[:, :, c].astype(np.float32) * vignette,
                                   0, 255).astype(np.uint8)

    # JPEG compression artifacts
    if random.random() < 0.4:
        quality = random.randint(50, 85)
        _, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, quality])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    return out


def make_random_background(size: int, difficulty: str = 'medium') -> np.ndarray:
    """
    Background generation parameterized by difficulty. The hard mode adds
    cluttered backgrounds with colored blobs, matching what benchmark_v2
    generates as test images.
    """
    if difficulty == 'easy':
        modes = ['solid', 'gradient']
    elif difficulty == 'medium':
        modes = ['solid', 'gradient', 'noise', 'wood']
    else:  # hard
        modes = ['noise', 'wood', 'cluttered', 'cluttered']

    mode = random.choice(modes)

    if mode == 'solid':
        c = tuple(random.randint(0, 255) for _ in range(3))
        return np.full((size, size, 3), c, dtype=np.uint8)

    if mode == 'gradient':
        c1 = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
        c2 = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
        bg = np.zeros((size, size, 3), dtype=np.uint8)
        if random.random() < 0.5:
            for i in range(size):
                t = i / size
                bg[i, :] = (c1 * (1-t) + c2 * t).astype(np.uint8)
        else:
            for j in range(size):
                t = j / size
                bg[:, j] = (c1 * (1-t) + c2 * t).astype(np.uint8)
        return bg

    if mode == 'noise':
        bg = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        return cv2.GaussianBlur(bg, (21, 21), 0)

    if mode == 'wood':
        base = np.array([random.randint(70, 160), random.randint(50, 130),
                         random.randint(30, 90)], dtype=np.float32)
        var = random.uniform(20, 50)
        bg = np.zeros((size, size, 3), dtype=np.uint8)
        for j in range(size):
            wave = (np.sin(j / random.uniform(8, 25)) * var
                    + np.sin(j / random.uniform(40, 80)) * var * 0.5)
            bg[:, j] = np.clip(base + wave, 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(bg, (5, 5), 0)

    # cluttered
    bg = np.random.randint(40, 200, (size, size, 3), dtype=np.uint8)
    bg = cv2.GaussianBlur(bg, (31, 31), 0)
    for _ in range(random.randint(3, 8)):
        cx, cy = random.randint(0, size), random.randint(0, size)
        radius = random.randint(size // 8, size // 3)
        color = tuple(random.randint(20, 230) for _ in range(3))
        cv2.circle(bg, (cx, cy), radius, color, -1)
    return cv2.GaussianBlur(bg, (51, 51), 0)


def add_glare(image: np.ndarray, strength: float = 0.4) -> np.ndarray:
    h, w = image.shape[:2]
    cx = random.randint(int(w*0.2), int(w*0.8))
    cy = random.randint(int(h*0.2), int(h*0.8))
    radius = random.randint(min(h, w) // 6, min(h, w) // 3)
    glare = np.zeros((h, w), dtype=np.float32)
    cv2.circle(glare, (cx, cy), radius, 1.0, -1)
    glare = cv2.GaussianBlur(glare, (51, 51), 0)
    glare = (glare / glare.max()) * 255 * strength
    out = image.astype(np.float32)
    for c in range(3):
        out[:, :, c] = np.clip(out[:, :, c] + glare, 0, 255)
    return out.astype(np.uint8)


# ===========================================================================
# Synthetic compositing - now with difficulty parameter
#
# The big change vs the previous trainer: each call to synthesize_rectified_card
# can target a specific difficulty level. The defaults match what
# benchmark_v2.py throws at the model, so the training distribution actually
# covers the test distribution.

def synthesize_rectified_card(card_img: np.ndarray,
                               difficulty: str = 'medium') -> np.ndarray:
    """
    Composite a card onto a random background, then re-rectify to 488x680
    with detector-error jitter. Difficulty controls rotation, glare,
    occlusion, blur, color cast, and corner jitter intensity.
    """
    H_card, W_card = card_img.shape[:2]
    canvas = random.randint(500, 900)
    bg = make_random_background(canvas, difficulty=difficulty)

    # Difficulty-dependent parameters - match the benchmark synthesizer
    if difficulty == 'easy':
        scale = random.uniform(0.55, 0.80)
        angle_range = 10
        glare_p = 0.0
        occlusion_p = 0.0
        color_jitter = 15
        bright_jitter = (0.85, 1.15)
        blur_p = 0.2
        noise_p = 0.0
        corner_jitter = 0.02
    elif difficulty == 'medium':
        scale = random.uniform(0.55, 0.85)
        angle_range = 25
        glare_p = 0.3
        occlusion_p = 0.0
        color_jitter = 25
        bright_jitter = (0.7, 1.3)
        blur_p = 0.4
        noise_p = 0.4
        corner_jitter = 0.04
    else:  # hard
        scale = random.uniform(0.45, 0.85)
        angle_range = 60
        glare_p = 0.5
        occlusion_p = 0.4
        color_jitter = 40
        bright_jitter = (0.5, 1.5)
        blur_p = 0.5
        noise_p = 0.5
        corner_jitter = 0.06

    new_h = int(canvas * scale)
    new_w = int(new_h * W_card / H_card)
    if new_w > canvas * 0.9:
        new_w = int(canvas * 0.9)
        new_h = int(new_w * H_card / W_card)
    card_resized = cv2.resize(card_img, (new_w, new_h))

    # Card-level lighting + color jitter (before placement)
    card_f = card_resized.astype(np.float32)
    card_f *= random.uniform(*bright_jitter)
    card_f += random.uniform(-color_jitter, color_jitter)
    card_resized = np.clip(card_f, 0, 255).astype(np.uint8)
    if random.random() < glare_p:
        card_resized = add_glare(card_resized, strength=random.uniform(0.3, 0.6))

    # Rotation + position
    angle = random.uniform(-angle_range, angle_range)
    margin = int(max(new_w, new_h) * 0.7)
    margin = min(margin, canvas // 2 - 1)
    cx = random.randint(margin, max(margin + 1, canvas - margin))
    cy = random.randint(margin, max(margin + 1, canvas - margin))

    M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1.0)
    M[0, 2] += cx - new_w / 2
    M[1, 2] += cy - new_h / 2

    warped = cv2.warpAffine(card_resized, M, (canvas, canvas))
    mask = cv2.warpAffine(np.ones((new_h, new_w), dtype=np.uint8) * 255,
                          M, (canvas, canvas))
    scene = bg.copy()
    scene[mask > 0] = warped[mask > 0]

    # Compute true corners and jitter them (simulates imperfect detection)
    corners = np.array([[0, 0], [new_w, 0], [new_w, new_h], [0, new_h]],
                       dtype=np.float32)
    corners_warped = cv2.transform(corners.reshape(1, -1, 2), M).reshape(-1, 2)

    # Optional occlusion - cover one corner before re-rectifying
    if random.random() < occlusion_p:
        ci = random.randint(0, 3)
        cx_o, cy_o = corners_warped[ci].astype(int)
        rect_w = random.randint(int(canvas*0.08), int(canvas*0.18))
        rect_h = random.randint(int(canvas*0.08), int(canvas*0.18))
        x0 = max(0, cx_o - rect_w // 2)
        y0 = max(0, cy_o - rect_h // 2)
        x1 = min(canvas, x0 + rect_w)
        y1 = min(canvas, y0 + rect_h)
        color = tuple(random.randint(40, 200) for _ in range(3))
        cv2.rectangle(scene, (x0, y0), (x1, y1), color, -1)

    # Sleeve overlay (replicates real card-in-sleeve appearance)
    if random.random() < (0.4 if difficulty == 'hard' else 0.15):
        scene_with_sleeve = add_sleeve_overlay(
            warped, strength=random.uniform(0.2, 0.5))
        scene[mask > 0] = scene_with_sleeve[mask > 0]

    # Shadow gradient (replicates uneven indoor lighting)
    if random.random() < (0.5 if difficulty == 'hard' else 0.2):
        scene = add_shadow_gradient(scene, strength=random.uniform(0.15, 0.45))

    # Whole-scene blur and noise
    if random.random() < blur_p:
        ksize = random.choice([3, 5, 7])
        scene = cv2.GaussianBlur(scene, (ksize, ksize), 0)
    if random.random() < noise_p:
        nz = np.random.randint(-8, 9, scene.shape, dtype=np.int16)
        scene = np.clip(scene.astype(np.int16) + nz, 0, 255).astype(np.uint8)

    # Phone camera artifacts (vignette, JPEG compression)
    if random.random() < (0.3 if difficulty == 'hard' else 0.1):
        scene = add_phone_camera_artifacts(scene)

    # Corner jitter for re-rectification (simulates detector error)
    jitter = random.uniform(0, corner_jitter) * max(new_w, new_h)
    jittered = corners_warped + np.random.uniform(
        -jitter, jitter, corners_warped.shape).astype(np.float32)

    dst = np.array([[0, 0], [CARD_W-1, 0], [CARD_W-1, CARD_H-1], [0, CARD_H-1]],
                   dtype=np.float32)
    try:
        M_rect = cv2.getPerspectiveTransform(jittered.astype(np.float32), dst)
        return cv2.warpPerspective(scene, M_rect, (CARD_W, CARD_H))
    except cv2.error:
        return cv2.resize(card_img, (CARD_W, CARD_H))


def sample_difficulty(easy_aug: bool = False) -> str:
    """
    Pick a synthetic difficulty for one training sample.
    Default mix: 40% medium, 30% hard, 30% clean (returned as 'easy' which
    actually means 'no synthetic compositing' in the calling code).
    --easy-aug mode falls back to the gentler distribution from the v1
    trainer for users who want the old behavior.
    """
    if easy_aug:
        return random.choice(['easy', 'medium'])
    r = random.random()
    if r < 0.30:
        return 'easy'
    if r < 0.70:
        return 'medium'
    return 'hard'


# ===========================================================================
# Frame classifier
#
# The big behavioral change: bumped synthetic exposure to 60% (was 30%),
# and the synthetic samples now span easy/medium/hard difficulty according
# to sample_difficulty(). This is the fix for the 66% hard-benchmark
# accuracy - the v1 classifier never saw 60-degree rotation or occlusion
# during training.

class FrameClassifier(nn.Module):
    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class FrameDataset(Dataset):
    def __init__(self, cards, indices, frame_classes_by_id,
                 training: bool = True, p_synthetic: float = 0.6,
                 easy_aug: bool = False):
        self.cards = cards
        self.indices = indices
        self.training = training
        self.p_synthetic = p_synthetic if training else 0.0
        self.easy_aug = easy_aug
        self.labels = [
            FRAME_CLASS_TO_IDX[frame_classes_by_id[cards[i]['id']]]
            for i in indices
        ]
        self.tf_train = T.Compose([
            T.Resize((FRAME_INPUT, FRAME_INPUT)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.tf_eval = T.Compose([
            T.Resize((FRAME_INPUT, FRAME_INPUT)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = load_card_image(self.cards[idx]['id'])
        if img is None:
            img = np.zeros((CARD_H, CARD_W, 3), dtype=np.uint8)
        if self.training and random.random() < self.p_synthetic:
            difficulty = sample_difficulty(easy_aug=self.easy_aug)
            img = synthesize_rectified_card(img, difficulty=difficulty)
        pil = Image.fromarray(img)
        tf = self.tf_train if self.training else self.tf_eval
        return tf(pil), self.labels[i]


def train_frame_classifier(cards, available, frame_classes_by_id,
                            epochs: int, easy_aug: bool):
    print("\n" + "=" * 70)
    print("Training frame classifier")
    print(f"Augmentation: {'easy (v1 mix)' if easy_aug else 'hard mix (default)'}, "
          f"synthetic ratio = 60%")
    print("=" * 70)

    counts = report_frame_distribution(
        {cards[i]['id']: frame_classes_by_id[cards[i]['id']] for i in available},
        label="Class distribution",
    )

    n_present = sum(1 for c in counts.values() if c > 0)
    if n_present < 2:
        print(f"\n  ABORT: only {n_present} class(es) present.")
        return

    mlflow.start_run(run_name="frame_classifier", nested=True)
    mlflow.log_params({
        'model': 'FrameClassifier',
        'epochs': epochs,
        'batch_size': BATCH_SIZE,
        'easy_aug': easy_aug,
        'p_synthetic': 0.6,
        'lr': 1e-3,
        'weight_decay': 0.01,
        'n_classes': len(FRAME_CLASSES),
        'n_cards': len(available),
    })

    minority = min(c for c in counts.values() if c > 0)
    if minority < 50:
        print(f"\n  WARNING: minority class has only {minority} examples.")

    by_class = defaultdict(list)
    for i in available:
        by_class[frame_classes_by_id[cards[i]['id']]].append(i)
    tr_idx, va_idx = [], []
    for cls, ix in by_class.items():
        random.shuffle(ix)
        s = int(0.85 * len(ix))
        tr_idx.extend(ix[:s])
        va_idx.extend(ix[s:])
    random.shuffle(tr_idx)

    tr_set = FrameDataset(cards, tr_idx, frame_classes_by_id,
                           training=True, p_synthetic=0.6, easy_aug=easy_aug)
    va_set = FrameDataset(cards, va_idx, frame_classes_by_id,
                           training=False)

    label_counts = Counter(tr_set.labels)
    class_weights = {k: 1.0 / max(1, v) for k, v in label_counts.items()}
    sample_weights = [class_weights[lab] for lab in tr_set.labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True)

    tr_loader = DataLoader(tr_set, batch_size=BATCH_SIZE, sampler=sampler,
                           num_workers=6, pin_memory=True, persistent_workers=True)
    va_loader = DataLoader(va_set, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True, persistent_workers=True)

    model = FrameClassifier(n_classes=len(FRAME_CLASSES)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params/1e3:.1f}K")

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        tr_loss = 0
        for x, y in tqdm(tr_loader, desc=f"  ep{epoch+1} train", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optim.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            optim.step()
            tr_loss += loss.item()
        sched.step()

        model.eval()
        correct = total = 0
        per_class = defaultdict(lambda: [0, 0])
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                for p, t in zip(pred.tolist(), y.tolist()):
                    per_class[t][1] += 1
                    if p == t:
                        per_class[t][0] += 1
        acc = correct / total
        per_class_acc = {FRAME_CLASSES[k]: v[0] / max(1, v[1])
                         for k, v in per_class.items()}
        print(f"  ep{epoch+1}: tr_loss={tr_loss/len(tr_loader):.3f} "
              f"val_acc={acc:.3f} per_class={per_class_acc}")

        mlflow.log_metrics({
            'train_loss': tr_loss / len(tr_loader),
            'val_acc': acc,
            **{f'val_acc_{FRAME_CLASSES[k]}': v for k, v in per_class_acc.items()},
        }, step=epoch)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), FRAME_CLF_PATH)
            print(f"    saved best (acc={acc:.3f})")

    mlflow.log_metric('best_val_acc', best_acc)
    mlflow.end_run()
    print(f"  Best val acc: {best_acc:.3f}")


# ===========================================================================
# MobileNet re-ranker (shared between art and whole-card)

class MobileNetReranker(nn.Module):
    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        backbone = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(960, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return F.normalize(self.projection(x), p=2, dim=1)


def triplet_loss(a, p, n, margin: float = 0.3):
    return F.relu(F.pairwise_distance(a, p) - F.pairwise_distance(a, n) + margin).mean()


class TripletRegionDataset(Dataset):
    """
    Triplets where the network sees a specific region of the card. The
    synthetic ratio is bumped to 60% (was 50%) and difficulty is sampled
    via sample_difficulty() so training distribution covers easy/medium/hard.

    Supports hard negative mining: pass a dict mapping anchor_idx -> list of hard negative indices.
    Alt-art / Universes-Beyond / weird-frame cards are oversampled 3x to
    improve re-ranker accuracy on non-standard card frames.
    """
    ALT_ART_OVERSAMPLE = 3

    def __init__(self, cards, indices, region_extractor, input_size: int,
                 p_synthetic: float = 0.6, training: bool = True,
                 easy_aug: bool = False, hard_negatives: dict = None,
                 hard_negative_ratio: float = 0.5):
        self.cards = cards
        self.region_extractor = region_extractor
        self.input_size = input_size
        self.training = training
        self.p_synthetic = p_synthetic if training else 0.0
        self.easy_aug = easy_aug
        self.hard_negatives = hard_negatives or {}
        self.hard_negative_ratio = hard_negative_ratio

        self.name_to_indices = defaultdict(list)
        self.valid = []
        alt_count = 0
        for i in indices:
            self.name_to_indices[cards[i]['name']].append(i)
            self.valid.append(i)
            # Oversample alt-art cards during training
            if training and is_alt_art(cards[i]):
                alt_count += 1
                for _ in range(self.ALT_ART_OVERSAMPLE - 1):
                    self.valid.append(i)
        self.triplet_names = [n for n, ix in self.name_to_indices.items() if len(ix) >= 2]
        hard_msg = f", {len(hard_negatives)} with hard negs" if hard_negatives else ""
        alt_msg = f", {alt_count} alt-art cards oversampled {self.ALT_ART_OVERSAMPLE}x" if training and alt_count else ""
        print(f"    {len(self.valid)} cards (effective), {len(self.triplet_names)} names with 2+ printings{hard_msg}{alt_msg}")

        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.train_tf = T.Compose([
            T.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.3),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
            T.ToTensor(), self.norm,
            T.RandomErasing(p=0.2),
        ])
        self.eval_tf = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(), self.norm,
        ])
        self.synth_tf = T.Compose([T.ToTensor(), self.norm])

    def __len__(self):
        return len(self.valid)

    def _augment(self, idx):
        card_img = load_card_image(self.cards[idx]['id'])
        if card_img is None:
            card_img = np.zeros((CARD_H, CARD_W, 3), dtype=np.uint8)
        if self.training and random.random() < self.p_synthetic:
            difficulty = sample_difficulty(easy_aug=self.easy_aug)
            rectified = synthesize_rectified_card(card_img, difficulty=difficulty)
            region = self.region_extractor(rectified)
            region = cv2.resize(region, (self.input_size, self.input_size))
            return self.synth_tf(region)
        region = self.region_extractor(card_img)
        pil = Image.fromarray(region)
        return self.train_tf(pil) if self.training else self.eval_tf(pil)

    def __getitem__(self, i):
        anchor = self.valid[i]
        name = self.cards[anchor]['name']
        pos_pool = [j for j in self.name_to_indices[name] if j != anchor]
        pos = random.choice(pos_pool) if pos_pool else anchor

        # Hard negative mining: with probability hard_negative_ratio, use a hard negative
        use_hard = (self.training and anchor in self.hard_negatives and
                    random.random() < self.hard_negative_ratio)
        if use_hard:
            neg = random.choice(self.hard_negatives[anchor])
        else:
            neg_name = random.choice([n for n in self.triplet_names if n != name])
            neg = random.choice(self.name_to_indices[neg_name])

        return self._augment(anchor), self._augment(pos), self._augment(neg)


def mine_hard_negatives(model, dataset, top_k: int = 10):
    """
    Find the hardest negatives for each anchor by running the model over
    all cards and finding which non-matching cards have the highest similarity.
    Returns dict mapping anchor_idx -> list of hard negative indices.
    """
    print("  Mining hard negatives...")
    model.eval()

    # Build embeddings for all cards without augmentation
    eval_tf = T.Compose([
        T.Resize((dataset.input_size, dataset.input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embeddings = []
    with torch.no_grad():
        for idx in tqdm(dataset.valid, desc="    embedding", leave=False):
            card_img = load_card_image(dataset.cards[idx]['id'])
            if card_img is None:
                embeddings.append(torch.zeros(EMBEDDING_DIM))
                continue
            region = dataset.region_extractor(card_img)
            t = eval_tf(Image.fromarray(region)).unsqueeze(0).to(DEVICE)
            emb = model(t).cpu()
            embeddings.append(emb[0])

    embeddings = torch.stack(embeddings)

    # For each anchor, find top-k most similar cards that are NOT the same name
    hard_negatives = {}
    for i, anchor_idx in enumerate(tqdm(dataset.valid, desc="    mining", leave=False)):
        anchor_name = dataset.cards[anchor_idx]['name']
        anchor_emb = embeddings[i].unsqueeze(0)

        # Compute similarity to all other cards
        sims = (anchor_emb @ embeddings.T).squeeze()

        # Mask out same-name cards
        mask = torch.ones(len(dataset.valid), dtype=torch.bool)
        for j, other_idx in enumerate(dataset.valid):
            if dataset.cards[other_idx]['name'] == anchor_name:
                mask[j] = False

        # Get top-k highest similarity negatives
        valid_sims = sims.clone()
        valid_sims[~mask] = -1.0
        _, top_indices = torch.topk(valid_sims, min(top_k, mask.sum().item()))

        hard_negatives[anchor_idx] = [dataset.valid[j] for j in top_indices.tolist()]

    print(f"    mined {len(hard_negatives)} anchor -> hard negative mappings")
    return hard_negatives


def train_reranker(cards, indices, output_path: Path,
                   region_extractor, input_size: int,
                   label: str = "", epochs: int = DEFAULT_EPOCHS,
                   easy_aug: bool = False):
    print("\n" + "=" * 70)
    print(f"Training {label} re-ranker on {len(indices)} cards")
    print(f"Augmentation: {'easy (v1 mix)' if easy_aug else 'hard mix (default)'}, "
          f"synthetic ratio = 60%")
    print(f"Hard negative mining: every 3 epochs")
    print("=" * 70)

    if len(indices) < 100:
        print(f"  Too few cards ({len(indices)}). Skipping.")
        return

    mlflow.start_run(run_name=f"reranker_{label.replace('-', '_')}", nested=True)
    mlflow.log_params({
        'model': 'MobileNetReranker',
        'label': label,
        'epochs': epochs,
        'batch_size': BATCH_SIZE,
        'easy_aug': easy_aug,
        'p_synthetic': 0.6,
        'embedding_dim': EMBEDDING_DIM,
        'input_size': input_size,
        'lr': 1e-3,
        'weight_decay': 0.01,
        'hard_negative_mining': True,
        'hard_negative_every_n_epochs': 3,
        'n_cards': len(indices),
    })

    shuffled = indices.copy()
    random.shuffle(shuffled)
    split = int(0.85 * len(shuffled))
    tr_idx, va_idx = shuffled[:split], shuffled[split:]

    print("  Building datasets:")
    tr_set = TripletRegionDataset(cards, tr_idx, region_extractor,
                                   input_size, p_synthetic=0.6, training=True,
                                   easy_aug=easy_aug)
    va_set = TripletRegionDataset(cards, va_idx, region_extractor,
                                   input_size, p_synthetic=0.0, training=False)

    model = MobileNetReranker(EMBEDDING_DIM).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    best = float('inf')
    for ep in range(epochs):
        # Mine hard negatives every 3 epochs (after initial warmup)
        if ep > 0 and ep % 3 == 0:
            hard_negs = mine_hard_negatives(model, tr_set, top_k=10)
            # Rebuild dataset with hard negatives
            tr_set = TripletRegionDataset(cards, tr_idx, region_extractor,
                                           input_size, p_synthetic=0.6, training=True,
                                           easy_aug=easy_aug, hard_negatives=hard_negs,
                                           hard_negative_ratio=0.5)

        tr_loader = DataLoader(tr_set, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=6, pin_memory=True, persistent_workers=True)
        va_loader = DataLoader(va_set, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=4, pin_memory=True, persistent_workers=True)

        model.train()
        tot = 0
        for a, p, n in tqdm(tr_loader, desc=f"  ep{ep+1} tr", leave=False):
            a, p, n = a.to(DEVICE, non_blocking=True), p.to(DEVICE, non_blocking=True), n.to(DEVICE, non_blocking=True)
            optim.zero_grad()
            loss = triplet_loss(model(a), model(p), model(n))
            loss.backward()
            optim.step()
            tot += loss.item()

        model.eval()
        vtot = 0
        with torch.no_grad():
            for a, p, n in va_loader:
                a, p, n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
                vtot += triplet_loss(model(a), model(p), model(n)).item()
        sched.step()

        tr_l = tot / len(tr_loader); va_l = vtot / len(va_loader)
        print(f"  ep{ep+1}: train={tr_l:.4f} val={va_l:.4f}")
        mlflow.log_metrics({'train_loss': tr_l, 'val_loss': va_l}, step=ep)
        if va_l < best:
            best = va_l
            torch.save(model.state_dict(), output_path)
            print(f"    saved best (val={va_l:.4f})")

    mlflow.log_metric('best_val_loss', best)
    mlflow.end_run()
    print(f"  Best val: {best:.4f}")


def build_embeddings(cards, indices, model_path: Path,
                     region_extractor, input_size: int,
                     output_path: Path, label: str = ""):
    print(f"\nBuilding {label} embeddings -> {output_path.name}")
    if not model_path.exists():
        print(f"  Model {model_path} not found. Skipping embeddings.")
        return None

    model = MobileNetReranker(EMBEDDING_DIM).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    eval_tf = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embs = []
    with torch.no_grad():
        for idx in tqdm(indices, desc=f"  embed {label}"):
            img = load_card_image(cards[idx]['id'])
            if img is None:
                continue
            region = region_extractor(img)
            pil = Image.fromarray(region)
            t = eval_tf(pil).unsqueeze(0).to(DEVICE)
            embs.append(model(t).cpu().numpy())
    arr = np.vstack(embs).astype('float32')
    # Explicit L2-normalize before saving to prevent float32 drift
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    arr = arr / norms
    np.save(output_path, arr)
    print(f"  Saved: {arr.shape} (L2-normalized)")
    return arr


# ===========================================================================
# Set symbol classifier

class SetSymbolClassifier(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class SetSymbolDataset(Dataset):
    def __init__(self, cards, indices, set_to_idx, training: bool = True,
                 easy_aug: bool = False):
        self.cards = cards
        self.indices = indices
        self.set_to_idx = set_to_idx
        self.training = training
        self.easy_aug = easy_aug
        self.labels = [set_to_idx[cards[i]['set']] for i in indices]
        self.tf_train = T.Compose([
            T.Resize((SETSYM_INPUT, SETSYM_INPUT)),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.tf_eval = T.Compose([
            T.Resize((SETSYM_INPUT, SETSYM_INPUT)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = load_card_image(self.cards[idx]['id'])
        if img is None:
            img = np.zeros((CARD_H, CARD_W, 3), dtype=np.uint8)
        # Set symbol classifier benefits less from harsh augmentation -
        # the symbol is a small, low-frequency region and heavy synthesis
        # can erase it. Keep at 30% with mostly easy/medium difficulty.
        if self.training and random.random() < 0.3:
            difficulty = 'easy' if random.random() < 0.5 else 'medium'
            img = synthesize_rectified_card(img, difficulty=difficulty)
        sym = crop_region(img, 'set_symbol')
        pil = Image.fromarray(sym)
        return (self.tf_train(pil) if self.training else self.tf_eval(pil),
                self.labels[i])


def train_set_classifier(cards, modern_indices, epochs: int,
                          easy_aug: bool):
    print("\n" + "=" * 70)
    print("Training set symbol classifier")
    print("=" * 70)
    print(f"  Modern-frame cards available: {len(modern_indices)}")

    if len(modern_indices) < 1000:
        print(f"  Too few modern cards ({len(modern_indices)}). Skipping.")
        return

    set_counts = Counter(cards[i].get('set', '<unknown>') for i in modern_indices)
    valid_sets = {s for s, c in set_counts.items() if c >= 20 and s != '<unknown>'}
    keep = [i for i in modern_indices if cards[i].get('set') in valid_sets]
    print(f"  After dropping rare sets (<20 cards): {len(keep)} cards across {len(valid_sets)} sets")

    if len(valid_sets) < 5:
        print(f"  Only {len(valid_sets)} usable sets. Skipping.")
        return

    set_codes = sorted(valid_sets)
    set_to_idx = {s: i for i, s in enumerate(set_codes)}
    with open(SET_CODES_PATH, 'w') as f:
        json.dump(set_codes, f)

    mlflow.start_run(run_name="set_classifier", nested=True)
    mlflow.log_params({
        'model': 'SetSymbolClassifier',
        'epochs': epochs,
        'batch_size': BATCH_SIZE,
        'easy_aug': easy_aug,
        'lr': 1e-3,
        'weight_decay': 0.01,
        'n_classes': len(set_codes),
        'n_modern_cards': len(modern_indices),
        'min_cards_per_set': 20,
    })

    shuffled = keep.copy()
    random.shuffle(shuffled)
    split = int(0.85 * len(shuffled))
    tr_idx, va_idx = shuffled[:split], shuffled[split:]

    tr_set = SetSymbolDataset(cards, tr_idx, set_to_idx,
                                training=True, easy_aug=easy_aug)
    va_set = SetSymbolDataset(cards, va_idx, set_to_idx, training=False)

    label_counts = Counter(tr_set.labels)
    class_weights = {k: 1.0 / max(1, v) for k, v in label_counts.items()}
    sample_weights = [class_weights[lab] for lab in tr_set.labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True)

    tr_loader = DataLoader(tr_set, batch_size=BATCH_SIZE, sampler=sampler,
                           num_workers=6, pin_memory=True, persistent_workers=True)
    va_loader = DataLoader(va_set, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True, persistent_workers=True)

    model = SetSymbolClassifier(n_classes=len(set_codes)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params/1e3:.0f}K, classes: {len(set_codes)}")

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(epochs):
        model.train()
        tot = 0
        for x, y in tqdm(tr_loader, desc=f"  ep{ep+1} tr", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optim.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            optim.step()
            tot += loss.item()
        sched.step()

        model.eval()
        c1 = c5 = total = 0
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                c1 += (logits.argmax(1) == y).sum().item()
                top5 = logits.topk(5, dim=1).indices
                c5 += (top5 == y.unsqueeze(1)).any(1).sum().item()
                total += y.size(0)
        a1 = c1 / total; a5 = c5 / total
        print(f"  ep{ep+1}: tr_loss={tot/len(tr_loader):.3f} top1={a1:.3f} top5={a5:.3f}")
        mlflow.log_metrics({
            'train_loss': tot / len(tr_loader),
            'val_top1': a1,
            'val_top5': a5,
        }, step=ep)
        if a1 > best:
            best = a1
            torch.save(model.state_dict(), SET_CLF_PATH)
            print(f"    saved best (top1={a1:.3f})")
    mlflow.log_metric('best_val_top1', best)
    mlflow.end_run()
    print(f"  Best top-1: {best:.3f}")


# ===========================================================================
# Hash database

def build_hash_db(cards, available, frame_classes_by_id):
    print("\n" + "=" * 70)
    print("Building hash database (art + whole-card)")
    print("=" * 70)

    indices, art_p, art_d, whole_p, whole_d = [], [], [], [], []
    frame_class_arr = []

    for idx in tqdm(available, desc="  hashing"):
        img = load_card_image(cards[idx]['id'])
        if img is None:
            continue
        art = extract_art_crop(img)
        whole = extract_whole_card(img)
        indices.append(idx)
        art_p.append(phash_64(art));     art_d.append(dhash_64(art))
        whole_p.append(phash_64(whole)); whole_d.append(dhash_64(whole))
        cls = frame_classes_by_id.get(cards[idx]['id'], 'modern')
        frame_class_arr.append(FRAME_CLASS_TO_IDX[cls])

    np.savez(
        HASH_DB_PATH,
        indices=np.array(indices, dtype=np.int32),
        art_phash=np.array(art_p, dtype=np.uint64),
        art_dhash=np.array(art_d, dtype=np.uint64),
        whole_phash=np.array(whole_p, dtype=np.uint64),
        whole_dhash=np.array(whole_d, dtype=np.uint64),
        frame_class=np.array(frame_class_arr, dtype=np.int8),
    )
    print(f"  Saved {len(indices)} entries to {HASH_DB_PATH.name}")


# ===========================================================================
# Sanity check

def sanity_check(cards, available, n: int = 30):
    print("\n" + "=" * 70)
    print("Sanity check")
    print("=" * 70)

    if not (HASH_DB_PATH.exists() and ART_EMB_PATH.exists() and ART_RERANKER_PATH.exists()):
        print("  Missing artifacts, skipping.")
        return

    db = np.load(HASH_DB_PATH)
    db_indices = db['indices']
    art_emb = np.load(ART_EMB_PATH)

    if len(art_emb) != len(db_indices):
        print(f"  Embedding count {len(art_emb)} != hash DB count {len(db_indices)}")
        return

    reranker = MobileNetReranker(EMBEDDING_DIM).to(DEVICE)
    reranker.load_state_dict(torch.load(ART_RERANKER_PATH, map_location=DEVICE))
    reranker.eval()

    eval_tf = T.Compose([
        T.Resize((ART_INPUT, ART_INPUT)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_ix = random.sample(available, min(n, len(available)))
    c1 = c5 = 0
    for idx in test_ix:
        card = cards[idx]
        img = load_card_image(card['id'])
        if img is None:
            continue
        art = extract_art_crop(img)
        q_p = phash_64(art); q_d = dhash_64(art)
        d_p = hamming_distance_vectorized(q_p, db['art_phash'])
        d_d = hamming_distance_vectorized(q_d, db['art_dhash'])
        combined = d_p + d_d
        cand = np.argsort(combined)[:50]

        with torch.no_grad():
            t = eval_tf(Image.fromarray(art)).unsqueeze(0).to(DEVICE)
            q_emb = reranker(t).cpu().numpy()
        sims = (q_emb @ art_emb[cand].T).flatten()
        order = cand[np.argsort(-sims)]
        top5 = [cards[int(db_indices[o])]['name'] for o in order[:5]]
        if top5 and top5[0] == card['name']:
            c1 += 1
        if card['name'] in top5:
            c5 += 1

    n_actual = len(test_ix)
    print(f"  On {n_actual} cards: Top-1 {c1/n_actual*100:.1f}%  Top-5 {c5/n_actual*100:.1f}%")


# ===========================================================================
# Augmentation preview (helps verify the synthesizer is doing what we want)

def preview_augmentation(cards, available, n: int = 4, easy_aug: bool = False):
    """Generate one sample at each difficulty and save a visualization."""
    print(f"\nGenerating augmentation preview ({'easy mix' if easy_aug else 'hard mix'})...")
    sample_ix = random.sample(available, n)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.2 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for i, idx in enumerate(sample_ix):
        img = load_card_image(cards[idx]['id'])
        if img is None:
            continue
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original\n{cards[idx]['name'][:25]}", fontsize=9)
        axes[i, 0].axis('off')
        for j, diff in enumerate(['easy', 'medium', 'hard']):
            synth = synthesize_rectified_card(img, difficulty=diff)
            axes[i, j+1].imshow(synth)
            axes[i, j+1].set_title(f"{diff}", fontsize=9)
            axes[i, j+1].axis('off')
    plt.tight_layout()
    out = DATA_DIR / 'augmentation_preview.png'
    plt.savefig(out, dpi=80, bbox_inches='tight')
    plt.show()
    print(f"  Saved preview to {out}")


# ===========================================================================
# Main

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--diagnose', action='store_true',
                   help="Inspect metadata fields and frame distribution, then exit")
    p.add_argument('--preview-aug', action='store_true',
                   help="Show synthetic augmentation samples and exit")
    p.add_argument('--skip', nargs='+', default=[],
                   choices=['frame', 'art', 'whole', 'set'],
                   help="Skip training of specific components")
    p.add_argument('--skip-train', action='store_true',
                   help="Don't train anything; rebuild hash DB and embeddings only")
    p.add_argument('--easy-aug', action='store_true',
                   help="Use the v1 (gentler) augmentation mix instead of the new "
                        "harder one. Faster training, less robust on hard photos.")
    p.add_argument('--no-image-fallback', action='store_true',
                   help="Don't peek at card images for frame classification")
    p.add_argument('--rebuild-frame-cache', action='store_true',
                   help="Recompute frame classes from scratch")
    p.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    p.add_argument('--no-sanity', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    print("=" * 70)
    print("MTG Identifier - Multi-Region Trainer (v2: hard augmentation)")
    print("=" * 70)

    mlflow.set_experiment("mtg_identifier_training")
    mlflow.start_run(run_name="identifier_training")
    mlflow.log_params({
        'epochs': args.epochs,
        'easy_aug': args.easy_aug,
        'seed': SEED,
        'device': str(DEVICE),
        'skip': list(args.skip) if args.skip else [],
        'skip_train': args.skip_train,
    })
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    aug_mode = 'easy (v1 mix)' if args.easy_aug else 'hard mix (default)'
    print(f"Augmentation mode: {aug_mode}")

    # Diagnose mode
    if args.diagnose:
        print("\n" + "=" * 70)
        print("Metadata diagnosis")
        print("=" * 70)
        report = inspect_metadata(str(METADATA_PATH))
        print(f"Total cards in metadata: {report['total_cards']:,}")
        print(f"Sampled: {report['sampled']}")
        print(f"\nFields present (count of cards in sample with each field):")
        for k in ['has_layout', 'has_full_art', 'has_border_color',
                  'has_frame_effects', 'has_frame']:
            field = k.replace('has_', '')
            present = report[k]
            pct = present / report['sampled'] * 100
            status = 'OK' if pct > 80 else 'PARTIAL' if pct > 0 else 'MISSING'
            print(f"  {field:<18} {present:>4}/{report['sampled']} ({pct:5.1f}%)  [{status}]")
        print(f"\nlayout values:        {report['layout_values']}")
        print(f"border_color values:  {report['border_color_values']}")
        print(f"full_art values:      {report['full_art_values']}")
        cards, available = load_cards()
        sample = random.sample(available, min(200, len(available)))
        print(f"\nClassifying {len(sample)} sample cards (metadata only)...")
        meta_only = Counter(classify_frame_from_metadata(cards[i]) for i in sample)
        print(f"  Result: {dict(meta_only)}")
        return

    # Preview mode
    if args.preview_aug:
        cards, available = load_cards()
        preview_augmentation(cards, available, n=4, easy_aug=args.easy_aug)
        return

    # Normal flow
    cards, available = load_cards()
    print(f"\nDataset: {len(cards)} cards, {len(available)} on disk")

    print("\n[0/5] Frame classification (precompute)")
    frame_classes_by_id = compute_frame_classes(
        cards, available,
        use_image_fallback=not args.no_image_fallback,
        force_recompute=args.rebuild_frame_cache,
    )
    report_frame_distribution(frame_classes_by_id, label="Final distribution")

    modern_indices = [i for i in available
                      if frame_classes_by_id[cards[i]['id']] == 'modern']
    fullart_indices = [i for i in available
                       if frame_classes_by_id[cards[i]['id']] == 'fullart']
    special_indices = [i for i in available
                       if frame_classes_by_id[cards[i]['id']] == 'special']

    print(f"\n  -> modern:  {len(modern_indices):,}")
    print(f"  -> fullart: {len(fullart_indices):,}")
    print(f"  -> special: {len(special_indices):,}")

    skip = set(args.skip)
    if args.skip_train:
        skip = {'frame', 'art', 'whole', 'set'}

    if 'frame' not in skip:
        train_frame_classifier(cards, available, frame_classes_by_id,
                                args.epochs, easy_aug=args.easy_aug)
    else:
        print("\nSkipping frame classifier")

    if 'art' not in skip:
        if len(modern_indices) >= 100:
            train_reranker(
                cards, modern_indices,
                output_path=ART_RERANKER_PATH,
                region_extractor=extract_art_crop,
                input_size=ART_INPUT,
                label="art-crop",
                epochs=args.epochs,
                easy_aug=args.easy_aug,
            )
        else:
            print(f"\nSkipping art re-ranker (only {len(modern_indices)} modern cards)")
    else:
        print("\nSkipping art re-ranker")

    if 'whole' not in skip:
        train_reranker(
            cards, available,
            output_path=WHOLE_RERANKER_PATH,
            region_extractor=extract_whole_card,
            input_size=WHOLE_INPUT,
            label="whole-card",
            epochs=args.epochs,
            easy_aug=args.easy_aug,
        )
    else:
        print("\nSkipping whole-card re-ranker")

    if 'set' not in skip:
        train_set_classifier(cards, modern_indices, args.epochs,
                              easy_aug=args.easy_aug)
    else:
        print("\nSkipping set symbol classifier")

    build_hash_db(cards, available, frame_classes_by_id)
    if ART_RERANKER_PATH.exists():
        build_embeddings(cards, available, ART_RERANKER_PATH,
                         extract_art_crop, ART_INPUT,
                         ART_EMB_PATH, label="art")
    if WHOLE_RERANKER_PATH.exists():
        build_embeddings(cards, available, WHOLE_RERANKER_PATH,
                         extract_whole_card, WHOLE_INPUT,
                         WHOLE_EMB_PATH, label="whole-card")

    if not args.no_sanity:
        sanity_check(cards, available, n=30)

    print("\n" + "=" * 70)
    print("Identifier training complete.")
    print("=" * 70)
    mlflow.end_run()


if __name__ == '__main__':
    main()