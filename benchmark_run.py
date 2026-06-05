"""
Benchmark the new multi-region pipeline vs Claude on simulated photos.

Differences from the v1 benchmark:
  - Uses the fine-tuned card detector (yolo_card_best.pt) instead of
    the pretrained YOLO that was finding random sub-regions.
  - Loads the new artifacts: art_reranker, whole_reranker, frame_classifier,
    set_classifier, and the dual-hash database (art + whole).
  - Routes each query by predicted frame class:
      * modern  -> art-crop hash + art re-ranker
      * fullart -> whole-card hash + whole re-ranker
      * special -> whole-card hash + whole re-ranker
  - Stratifies test cards by frame class so we can measure modern,
    fullart, and special accuracy separately.
  - Reports per-frame-class accuracy in addition to the easy/medium/hard
    difficulty breakdown.

Outputs:
  mtg_data/benchmark_samples_v2/        - synthetic test images
  mtg_data/benchmark_results_v2.json    - raw results
  mtg_data/benchmark_v2_comparison.png  - summary plots
"""

import argparse
import base64
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image, ImageOps
from tqdm import tqdm

from mtg_layout import (
    CARD_W, CARD_H,
    extract_art_crop, extract_whole_card, crop_region,
    classify_frame_from_metadata,
    FRAME_CLASSES, FRAME_CLASS_TO_IDX,
    phash_64, dhash_64, hamming_distance_vectorized,
    is_valid_card_detection,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = Path('mtg_data')
IMAGE_DIR = DATA_DIR / 'card_images'
METADATA_PATH = DATA_DIR / 'cards_metadata.json'

YOLO_BEST = DATA_DIR / 'yolo_card_best.pt'
HASH_DB_PATH = DATA_DIR / 'phash_db.npz'
FRAME_CLF_PATH = DATA_DIR / 'frame_classifier.pth'
ART_RERANKER_PATH = DATA_DIR / 'art_reranker.pth'
ART_EMB_PATH = DATA_DIR / 'art_embeddings.npy'
WHOLE_RERANKER_PATH = DATA_DIR / 'whole_reranker.pth'
WHOLE_EMB_PATH = DATA_DIR / 'whole_embeddings.npy'
SET_CLF_PATH = DATA_DIR / 'set_classifier.pth'
SET_CODES_PATH = DATA_DIR / 'set_codes.json'
FRAME_CACHE_PATH = DATA_DIR / 'frame_classes_cache.json'

BENCH_IMG_DIR = DATA_DIR / 'benchmark_samples_v2'
BENCH_ITEMS = DATA_DIR / 'benchmark_items_v2.json'
BENCH_RESULTS = DATA_DIR / 'benchmark_results_v2.json'
BENCH_PLOT = DATA_DIR / 'benchmark_v2_comparison.png'

ART_INPUT = 160
WHOLE_INPUT = 224
FRAME_INPUT = 96
EMBEDDING_DIM = 256


# ---------------------------------------------------------------------------
# Models (compact reproductions matching train_identifier.py)

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


# ---------------------------------------------------------------------------
# Detector: uses the FINE-TUNED weights

class CardDetector:
    CARD_ASPECT = CARD_W / CARD_H
    ASPECT_TOLERANCE = 0.05

    def __init__(self, weights_path: Path = YOLO_BEST, verbose: bool = True):
        from ultralytics import YOLO
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Detector weights not found at {weights_path}. "
                f"Run train_detector.py first.")
        self.model = YOLO(str(weights_path))
        if verbose:
            print(f"  Loaded fine-tuned detector: {weights_path.name}")

    def _looks_like_clean_card(self, image):
        h, w = image.shape[:2]
        if min(h, w) < 200:
            return False
        return abs(w / h - self.CARD_ASPECT) < self.ASPECT_TOLERANCE

    def detect_and_correct(self, image):
        if self._looks_like_clean_card(image):
            return cv2.resize(image, (CARD_W, CARD_H)), 'clean'

        results = self.model(image, verbose=False)
        if len(results[0].obb) == 0:
            return cv2.resize(image, (CARD_W, CARD_H)), 'no_detection'

        # Take the highest-confidence detection
        obb = results[0].obb
        if hasattr(obb, 'conf') and len(obb.conf) > 0:
            best_idx = int(obb.conf.argmax())
        else:
            best_idx = 0
        corners = obb.xyxyxyxy[best_idx].cpu().numpy()
        warped = self._warp(image, corners)

        # Validate the warped output - real cards have varied content
        if not is_valid_card_detection(warped):
            return cv2.resize(image, (CARD_W, CARD_H)), 'invalid_detection'

        return warped, 'detected'

    def _warp(self, image, corners):
        rect = self._order(corners)
        dst = np.array([[0, 0], [CARD_W-1, 0], [CARD_W-1, CARD_H-1], [0, CARD_H-1]],
                       dtype=np.float32)
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (CARD_W, CARD_H))

    def _order(self, pts):
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]; rect[3] = pts[np.argmax(d)]
        return rect


# ---------------------------------------------------------------------------
# Multi-region pipeline

NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

ART_TRANSFORM = T.Compose([
    T.Resize((ART_INPUT, ART_INPUT)),
    T.ToTensor(), NORMALIZE,
])
WHOLE_TRANSFORM = T.Compose([
    T.Resize((WHOLE_INPUT, WHOLE_INPUT)),
    T.ToTensor(), NORMALIZE,
])
FRAME_TRANSFORM = T.Compose([
    T.Resize((FRAME_INPUT, FRAME_INPUT)),
    T.ToTensor(), NORMALIZE,
])


class MultiRegionPipeline:
    """
    Routes queries by predicted frame class:
      modern  -> art crop  -> art_phash + art_reranker
      fullart -> whole card -> whole_phash + whole_reranker
      special -> whole card -> whole_phash + whole_reranker

    Set symbol classifier is queried for modern-frame cards as a tie-breaker.
    """

    def __init__(self, cards, db, frame_clf, art_reranker, art_emb,
                 whole_reranker, whole_emb, set_clf, set_codes, detector):
        self.cards = cards
        self.db_indices = db['indices']
        self.art_phash = db['art_phash']
        self.art_dhash = db['art_dhash']
        self.whole_phash = db['whole_phash']
        self.whole_dhash = db['whole_dhash']
        self.frame_class_db = db['frame_class']

        self.frame_clf = frame_clf
        self.art_reranker = art_reranker
        self.whole_reranker = whole_reranker
        self.set_clf = set_clf
        self.set_codes = set_codes  # list[str] - index -> set code
        self.detector = detector

        self.art_emb = art_emb
        self.whole_emb = whole_emb

    def _classify_frame(self, card_img: np.ndarray) -> str:
        with torch.no_grad():
            t = FRAME_TRANSFORM(Image.fromarray(card_img)).unsqueeze(0).to(DEVICE)
            logits = self.frame_clf(t)
            idx = int(logits.argmax(1).item())
        return FRAME_CLASSES[idx]

    def _classify_set(self, card_img: np.ndarray, top_k: int = 5) -> List[str]:
        if self.set_clf is None or not self.set_codes:
            return []
        sym = crop_region(card_img, 'set_symbol')
        sym_pil = Image.fromarray(sym)
        with torch.no_grad():
            t = T.Compose([
                T.Resize((64, 64)),
                T.ToTensor(), NORMALIZE,
            ])(sym_pil).unsqueeze(0).to(DEVICE)
            logits = self.set_clf(t)
            top = logits.topk(top_k, dim=1).indices[0].cpu().tolist()
        return [self.set_codes[i] for i in top if i < len(self.set_codes)]

    def identify(self, image_path: str, top_k: int = 5,
                 hash_candidates: int = 500):
        timings: Dict[str, float] = {}

        # Stage 1: load + detect
        t0 = time.perf_counter()
        img = np.array(ImageOps.exif_transpose(Image.open(image_path)).convert('RGB'))
        card_img, route = self.detector.detect_and_correct(img)
        timings['detect_ms'] = (time.perf_counter() - t0) * 1000

        # Stage 2: frame class
        t0 = time.perf_counter()
        frame_class = self._classify_frame(card_img)
        timings['frame_ms'] = (time.perf_counter() - t0) * 1000

        # Stage 3: hash retrieval against the right index
        t0 = time.perf_counter()
        if frame_class == 'modern':
            region = extract_art_crop(card_img)
            db_p, db_d = self.art_phash, self.art_dhash
            emb_db = self.art_emb
            reranker_model = self.art_reranker
            transform = ART_TRANSFORM
        else:
            region = extract_whole_card(card_img)
            db_p, db_d = self.whole_phash, self.whole_dhash
            emb_db = self.whole_emb
            reranker_model = self.whole_reranker
            transform = WHOLE_TRANSFORM

        q_p = phash_64(region)
        q_d = dhash_64(region)
        d_p = hamming_distance_vectorized(q_p, db_p)
        d_d = hamming_distance_vectorized(q_d, db_d)
        combined = d_p + d_d

        # Test-time augmentation: if top-1 distance is poor, try rotations
        best_distance = combined.min()
        if best_distance > 10:
            for angle in [90, 180, 270]:
                rotated_img = np.rot90(card_img, k=angle // 90)
                if frame_class == 'modern':
                    rot_region = extract_art_crop(rotated_img)
                else:
                    rot_region = extract_whole_card(rotated_img)

                rot_p = phash_64(rot_region)
                rot_d = dhash_64(rot_region)
                rot_d_p = hamming_distance_vectorized(rot_p, db_p)
                rot_d_d = hamming_distance_vectorized(rot_d, db_d)
                rot_combined = rot_d_p + rot_d_d
                rot_min = rot_combined.min()

                if rot_min < best_distance:
                    best_distance = rot_min
                    combined = rot_combined
                    region = rot_region
                    card_img = rotated_img

        k = min(hash_candidates, len(combined) - 1)
        cand_pos = np.argpartition(combined, k)[:hash_candidates]
        cand_pos = cand_pos[np.argsort(combined[cand_pos])]
        cand_card_idx = self.db_indices[cand_pos]
        timings['hash_ms'] = (time.perf_counter() - t0) * 1000

        # Stage 4: re-rank with the right model
        t0 = time.perf_counter()
        with torch.no_grad():
            t = transform(Image.fromarray(region)).unsqueeze(0).to(DEVICE)
            q_emb = reranker_model(t).cpu().numpy()
        sims = (q_emb @ emb_db[cand_pos].T).flatten()
        order = np.argsort(sims)[::-1]
        ranked_card_idx = cand_card_idx[order]
        ranked_sims = sims[order]
        timings['rerank_ms'] = (time.perf_counter() - t0) * 1000

        # Stage 5: set symbol disambiguation for modern cards only
        timings['set_ms'] = 0.0
        if frame_class == 'modern' and self.set_clf is not None:
            margin = ranked_sims[0] - ranked_sims[1] if len(ranked_sims) > 1 else 1.0
            if margin < 0.05:  # close call - bring in set symbol
                t0 = time.perf_counter()
                set_preds = self._classify_set(card_img, top_k=10)
                timings['set_ms'] = (time.perf_counter() - t0) * 1000
                if set_preds:
                    set_set = set(set_preds)
                    boost = np.array([
                        0.10 if self.cards[int(ci)].get('set') in set_set else 0.0
                        for ci in ranked_card_idx
                    ])
                    ranked_sims = ranked_sims + boost
                    order2 = np.argsort(ranked_sims)[::-1]
                    ranked_card_idx = ranked_card_idx[order2]
                    ranked_sims = ranked_sims[order2]

        timings['total_ms'] = sum(timings.values())
        top = [(int(ranked_card_idx[i]), float(ranked_sims[i]))
               for i in range(min(top_k, len(ranked_card_idx)))]
        return {
            'top_k': top,
            'timings': timings,
            'route': route,
            'predicted_frame': frame_class,
        }


# ---------------------------------------------------------------------------
# Synthetic photo generation (same difficulty buckets as v1 benchmark)

def random_background(size: int, difficulty: str) -> np.ndarray:
    if difficulty == 'easy':
        mode = random.choice(['solid', 'gradient'])
    elif difficulty == 'medium':
        mode = random.choice(['solid', 'gradient', 'noise', 'wood'])
    else:
        mode = random.choice(['noise', 'wood', 'cluttered', 'cluttered'])

    if mode == 'solid':
        c = tuple(random.randint(20, 235) for _ in range(3))
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


def synthesize_photo(card_img: np.ndarray, difficulty: str,
                     output_size: int = 1024) -> np.ndarray:
    H_card, W_card = card_img.shape[:2]
    canvas = output_size
    bg = random_background(canvas, difficulty)

    if difficulty == 'easy':
        scale = random.uniform(0.55, 0.80); angle_range = 10
        blur_p = 0.2; glare_p = 0.0; occlusion_p = 0.0
        color_j = 15; bright_j = (0.85, 1.15)
    elif difficulty == 'medium':
        scale = random.uniform(0.40, 0.70); angle_range = 25
        blur_p = 0.4; glare_p = 0.3; occlusion_p = 0.0
        color_j = 25; bright_j = (0.7, 1.3)
    else:
        scale = random.uniform(0.30, 0.55); angle_range = 60
        blur_p = 0.5; glare_p = 0.5; occlusion_p = 0.4
        color_j = 40; bright_j = (0.5, 1.5)

    new_h = int(canvas * scale)
    new_w = int(new_h * W_card / H_card)
    if new_w > canvas * 0.95:
        new_w = int(canvas * 0.95)
        new_h = int(new_w * H_card / W_card)
    card_resized = cv2.resize(card_img, (new_w, new_h))

    card_f = card_resized.astype(np.float32)
    card_f *= random.uniform(*bright_j)
    card_f += random.uniform(-color_j, color_j)
    card_resized = np.clip(card_f, 0, 255).astype(np.uint8)
    if random.random() < glare_p:
        card_resized = add_glare(card_resized, strength=random.uniform(0.3, 0.6))

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

    if random.random() < occlusion_p:
        corners = np.array([[0, 0], [new_w, 0], [new_w, new_h], [0, new_h]], dtype=np.float32)
        c_warp = cv2.transform(corners.reshape(1, -1, 2), M).reshape(-1, 2)
        cx_o, cy_o = c_warp[random.randint(0, 3)].astype(int)
        rect_w = random.randint(int(canvas*0.08), int(canvas*0.18))
        rect_h = random.randint(int(canvas*0.08), int(canvas*0.18))
        x0 = max(0, cx_o - rect_w // 2); y0 = max(0, cy_o - rect_h // 2)
        x1 = min(canvas, x0 + rect_w);   y1 = min(canvas, y0 + rect_h)
        color = tuple(random.randint(40, 200) for _ in range(3))
        cv2.rectangle(scene, (x0, y0), (x1, y1), color, -1)

    if random.random() < blur_p:
        ksize = random.choice([3, 5, 7])
        scene = cv2.GaussianBlur(scene, (ksize, ksize), 0)
    if difficulty in ('medium', 'hard'):
        noise = np.random.randint(-8, 9, scene.shape, dtype=np.int16)
        scene = np.clip(scene.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return scene


# ---------------------------------------------------------------------------
# Claude

def identify_with_claude(image_path: str, api_key: str) -> Tuple[str, float]:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    with open(image_path, 'rb') as f:
        image_data = base64.standard_b64encode(f.read()).decode('utf-8')
    ext = Path(image_path).suffix.lower()
    media_type = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                  '.png': 'image/png', '.webp': 'image/webp'}.get(ext, 'image/jpeg')
    start = time.time()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                                              "media_type": media_type,
                                              "data": image_data}},
                {"type": "text",
                 "text": "What is the name of this Magic: The Gathering card? "
                         "Reply with ONLY the card name, nothing else."}
            ]
        }]
    )
    return msg.content[0].text.strip(), time.time() - start


# ---------------------------------------------------------------------------
# Test set construction
#
# Stratify by frame class so we get meaningful per-class numbers. Without
# this, ~85% of test cards would be modern and the fullart/special numbers
# would be too noisy to interpret.

def build_test_set(cards, available, frame_classes_by_id,
                   n_per_difficulty: int, n_per_frame_class: dict,
                   seed: int = 42) -> List[Dict]:
    """
    Generate test images stratified across difficulty AND frame class.
    n_per_frame_class is a dict like {'modern': 20, 'fullart': 10, 'special': 5}
    that controls the sampling within each difficulty.
    """
    BENCH_IMG_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # Index cards by frame class
    by_frame = defaultdict(list)
    for i in available:
        cls = frame_classes_by_id.get(cards[i]['id'], 'modern')
        by_frame[cls].append(i)
    print(f"Frame distribution in available cards: "
          f"{ {k: len(v) for k, v in by_frame.items()} }")

    items = []
    difficulties = ['easy', 'medium', 'hard']

    for difficulty in difficulties:
        for frame_cls, n_class in n_per_frame_class.items():
            pool = by_frame.get(frame_cls, [])
            if not pool:
                print(f"  Skipping {difficulty}/{frame_cls} - no cards")
                continue
            sampled = rng.sample(pool, min(n_class, len(pool)))
            for i, card_idx in enumerate(tqdm(sampled,
                                               desc=f"  {difficulty}/{frame_cls}")):
                card = cards[card_idx]
                src = IMAGE_DIR / f"{card['id']}.jpg"
                card_img = np.array(ImageOps.exif_transpose(Image.open(src)).convert('RGB'))
                if card_img.shape[0] != CARD_H or card_img.shape[1] != CARD_W:
                    card_img = cv2.resize(card_img, (CARD_W, CARD_H))
                scene = synthesize_photo(card_img, difficulty=difficulty)
                out_path = (BENCH_IMG_DIR /
                            f"{difficulty}_{frame_cls}_{i:04d}_{card['id']}.jpg")
                Image.fromarray(scene).save(out_path, quality=88)
                items.append({
                    'difficulty': difficulty,
                    'frame_class': frame_cls,
                    'card_idx': card_idx,
                    'card_id': card['id'],
                    'true_name': card['name'],
                    'true_set': card.get('set', '?'),
                    'image_path': str(out_path),
                })
    return items


# ---------------------------------------------------------------------------
# Loading

def load_pipeline():
    print("Loading pipeline components...")
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        cards = json.load(f)
    print(f"  Cards: {len(cards):,}")

    if not HASH_DB_PATH.exists():
        raise FileNotFoundError(f"{HASH_DB_PATH} missing. Run train_identifier.py first.")
    db = np.load(HASH_DB_PATH)
    required_keys = ['indices', 'art_phash', 'art_dhash', 'whole_phash', 'whole_dhash']
    missing = [k for k in required_keys if k not in db.files]
    if missing:
        raise ValueError(
            f"Hash DB is missing keys {missing}. This benchmark requires the v2 "
            f"DB built by the new train_identifier.py. Re-run that script.")
    print(f"  Hash DB: {len(db['indices']):,} entries")

    detector = CardDetector(weights_path=YOLO_BEST)

    frame_clf = FrameClassifier().to(DEVICE)
    frame_clf.load_state_dict(torch.load(FRAME_CLF_PATH, map_location=DEVICE))
    frame_clf.eval()
    print(f"  Frame classifier loaded")

    art_reranker = MobileNetReranker(EMBEDDING_DIM).to(DEVICE)
    art_reranker.load_state_dict(torch.load(ART_RERANKER_PATH, map_location=DEVICE))
    art_reranker.eval()
    art_emb = np.load(ART_EMB_PATH)
    print(f"  Art re-ranker + {art_emb.shape} embeddings")

    whole_reranker = MobileNetReranker(EMBEDDING_DIM).to(DEVICE)
    whole_reranker.load_state_dict(torch.load(WHOLE_RERANKER_PATH, map_location=DEVICE))
    whole_reranker.eval()
    whole_emb = np.load(WHOLE_EMB_PATH)
    print(f"  Whole-card re-ranker + {whole_emb.shape} embeddings")

    set_clf = None
    set_codes = []
    if SET_CLF_PATH.exists() and SET_CODES_PATH.exists():
        with open(SET_CODES_PATH) as f:
            set_codes = json.load(f)
        set_clf = SetSymbolClassifier(n_classes=len(set_codes)).to(DEVICE)
        set_clf.load_state_dict(torch.load(SET_CLF_PATH, map_location=DEVICE))
        set_clf.eval()
        print(f"  Set classifier with {len(set_codes)} classes")
    else:
        print(f"  Set classifier not available - skipping set tie-break")

    pipeline = MultiRegionPipeline(
        cards, db, frame_clf, art_reranker, art_emb,
        whole_reranker, whole_emb, set_clf, set_codes, detector,
    )
    return cards, pipeline


def load_frame_classes_by_id(cards, available):
    """Load the frame class cache. Falls back to metadata-only classification."""
    if FRAME_CACHE_PATH.exists():
        with open(FRAME_CACHE_PATH) as f:
            cache = json.load(f)
        avail_ids = {cards[i]['id'] for i in available}
        return {k: v for k, v in cache.items() if k in avail_ids}
    print("  No frame cache found - computing from metadata...")
    return {cards[i]['id']: classify_frame_from_metadata(cards[i])
            for i in available}


def get_api_key() -> Optional[str]:
    key = os.getenv('ANTHROPIC_API_KEY')
    if key:
        return key
    env_file = DATA_DIR / '.env'
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith('ANTHROPIC_API_KEY='):
                return line.split('=', 1)[1].strip()
    try:
        key = input("Anthropic API key (or Enter to skip): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    return key or None


# ---------------------------------------------------------------------------
# Benchmark loop + reporting

def run_benchmark(cards, pipeline, items, api_key: Optional[str]):
    """Run both pipelines on every item. Returns nested dict of results."""
    # bucket by difficulty -> frame_class
    buckets = defaultdict(lambda: defaultdict(lambda: {
        'hash_top1': 0, 'hash_top5': 0, 'hash_times': [],
        'claude_top1': 0, 'claude_times': [],
        'route_counts': defaultdict(int),
        'frame_pred_counts': defaultdict(int),
        'frame_correct': 0,
        'samples': [], 'n': 0,
    }))

    for item in tqdm(items, desc="Benchmarking"):
        path = item['image_path']
        true_name = item['true_name']
        diff = item['difficulty']
        frame_cls = item['frame_class']
        bucket = buckets[diff][frame_cls]
        bucket['n'] += 1

        # Hash pipeline
        try:
            result = pipeline.identify(path, top_k=5)
            top = result['top_k']
            top_names = [cards[ci]['name'] for ci, _ in top]
            hash_top1 = top_names[0] == true_name
            hash_top5 = true_name in top_names
            bucket['hash_top1'] += int(hash_top1)
            bucket['hash_top5'] += int(hash_top5)
            bucket['hash_times'].append(result['timings']['total_ms'])
            bucket['route_counts'][result['route']] += 1
            bucket['frame_pred_counts'][result['predicted_frame']] += 1
            if result['predicted_frame'] == frame_cls:
                bucket['frame_correct'] += 1
            hash_pred = top_names[0]
            hash_score = top[0][1]
        except Exception as e:
            print(f"  hash error on {path}: {e}")
            hash_pred = f"ERROR: {e}"; hash_score = 0.0
            hash_top1 = hash_top5 = False
            result = {'predicted_frame': '?', 'route': 'error', 'timings': {}}

        # Claude
        if api_key:
            try:
                claude_pred, claude_t = identify_with_claude(path, api_key)
                claude_top1 = claude_pred.lower().strip() == true_name.lower().strip()
                bucket['claude_top1'] += int(claude_top1)
                bucket['claude_times'].append(claude_t * 1000)
            except Exception as e:
                print(f"  claude error: {e}")
                claude_pred = f"ERROR: {e}"; claude_top1 = False
        else:
            claude_pred = "skipped"; claude_top1 = False

        if len(bucket['samples']) < 2:
            bucket['samples'].append({
                'image_path': path,
                'true_name': true_name,
                'true_frame_class': frame_cls,
                'predicted_frame': result.get('predicted_frame', '?'),
                'route': result.get('route', '?'),
                'hash_pred': hash_pred,
                'hash_correct': bool(hash_top1),
                'claude_pred': claude_pred,
                'claude_correct': bool(claude_top1),
            })

    return buckets


def summarize(buckets, has_claude: bool):
    print()
    print("=" * 96)
    print("RESULTS")
    print("=" * 96)

    # Per-difficulty x frame class
    diff_order = ['easy', 'medium', 'hard']
    frame_order = ['modern', 'fullart', 'special']

    print(f"\n{'Difficulty':<10} {'Frame':<10} {'N':>4} "
          f"{'Hash T1':>8} {'Hash T5':>8} {'Claude T1':>10} "
          f"{'Frame OK':>9} {'Hash ms':>9} {'Claude ms':>10}")
    print("-" * 96)

    overall = {'n': 0, 'hash_top1': 0, 'hash_top5': 0,
               'claude_top1': 0, 'frame_correct': 0,
               'hash_times': [], 'claude_times': []}
    by_frame_overall = defaultdict(lambda: {
        'n': 0, 'hash_top1': 0, 'hash_top5': 0, 'claude_top1': 0,
    })

    for diff in diff_order:
        if diff not in buckets:
            continue
        for fc in frame_order:
            if fc not in buckets[diff]:
                continue
            r = buckets[diff][fc]
            n = r['n']
            if n == 0:
                continue
            h1 = r['hash_top1'] / n * 100
            h5 = r['hash_top5'] / n * 100
            c1 = r['claude_top1'] / n * 100 if has_claude else float('nan')
            fok = r['frame_correct'] / n * 100
            ht = np.mean(r['hash_times']) if r['hash_times'] else 0
            ct = np.mean(r['claude_times']) if has_claude and r['claude_times'] else 0

            c1_str = f"{c1:>8.1f}%" if has_claude else f"{'N/A':>9}"
            ct_str = f"{ct:>7.0f}ms" if has_claude else f"{'N/A':>9}"

            print(f"{diff:<10} {fc:<10} {n:>4} "
                  f"{h1:>6.1f}% {h5:>6.1f}% {c1_str} "
                  f"{fok:>7.1f}% {ht:>6.0f}ms {ct_str}")

            overall['n'] += n
            overall['hash_top1'] += r['hash_top1']
            overall['hash_top5'] += r['hash_top5']
            overall['claude_top1'] += r['claude_top1']
            overall['frame_correct'] += r['frame_correct']
            overall['hash_times'].extend(r['hash_times'])
            overall['claude_times'].extend(r['claude_times'])

            by_frame_overall[fc]['n'] += n
            by_frame_overall[fc]['hash_top1'] += r['hash_top1']
            by_frame_overall[fc]['hash_top5'] += r['hash_top5']
            by_frame_overall[fc]['claude_top1'] += r['claude_top1']

    print("-" * 96)
    n = overall['n']
    if n == 0:
        print("No results.")
        return {}
    h1 = overall['hash_top1'] / n * 100
    h5 = overall['hash_top5'] / n * 100
    c1 = overall['claude_top1'] / n * 100 if has_claude else float('nan')
    fok = overall['frame_correct'] / n * 100
    ht = np.mean(overall['hash_times']) if overall['hash_times'] else 0
    ct = np.mean(overall['claude_times']) if has_claude and overall['claude_times'] else 0
    c1_str = f"{c1:>8.1f}%" if has_claude else f"{'N/A':>9}"
    ct_str = f"{ct:>7.0f}ms" if has_claude else f"{'N/A':>9}"
    print(f"{'OVERALL':<10} {'all':<10} {n:>4} "
          f"{h1:>6.1f}% {h5:>6.1f}% {c1_str} "
          f"{fok:>7.1f}% {ht:>6.0f}ms {ct_str}")

    # Per-frame-class summary across difficulties
    print(f"\n{'By frame class:':<20}")
    print(f"  {'frame':<10} {'N':>5} {'Hash T1':>9} {'Hash T5':>9} "
          f"{'Claude T1':>11}")
    for fc in frame_order:
        r = by_frame_overall[fc]
        if r['n'] == 0:
            continue
        h1 = r['hash_top1'] / r['n'] * 100
        h5 = r['hash_top5'] / r['n'] * 100
        c1 = r['claude_top1'] / r['n'] * 100 if has_claude else float('nan')
        c1_str = f"{c1:>9.1f}%" if has_claude else f"{'N/A':>10}"
        print(f"  {fc:<10} {r['n']:>5} {h1:>7.1f}% {h5:>7.1f}% {c1_str}")

    # Speedup / accuracy gap
    if has_claude and ct > 0:
        speedup = ct / ht
        gap = h1 - c1
        print(f"\nSpeed:    hash pipeline is {speedup:.1f}x faster on average")
        if gap > 1:
            print(f"Accuracy: hash pipeline beats Claude by {gap:+.1f} points overall")
        elif gap < -1:
            print(f"Accuracy: Claude beats hash pipeline by {-gap:+.1f} points overall")
        else:
            print(f"Accuracy: within {abs(gap):.1f} points overall (effectively tied)")

    return {
        'overall': {'n': n, 'hash_top1': h1, 'hash_top5': h5,
                    'claude_top1': c1, 'frame_correct': fok,
                    'hash_ms_mean': ht, 'claude_ms_mean': ct},
        'by_frame_class': {
            fc: {
                'n': r['n'],
                'hash_top1': r['hash_top1'] / r['n'] * 100 if r['n'] else 0,
                'hash_top5': r['hash_top5'] / r['n'] * 100 if r['n'] else 0,
                'claude_top1': (r['claude_top1'] / r['n'] * 100
                                if r['n'] and has_claude else None),
            } for fc, r in by_frame_overall.items() if r['n'] > 0
        },
    }


def plot_results(buckets, has_claude: bool, out_path: Path):
    diff_order = [d for d in ['easy', 'medium', 'hard'] if d in buckets]
    frame_order = ['modern', 'fullart', 'special']

    if not diff_order:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: accuracy by difficulty (collapsed across frame class)
    ax = axes[0, 0]
    h1 = []; h5 = []; c1 = []
    for d in diff_order:
        n = sum(buckets[d][fc]['n'] for fc in buckets[d])
        h1.append(sum(buckets[d][fc]['hash_top1'] for fc in buckets[d]) / max(n, 1) * 100)
        h5.append(sum(buckets[d][fc]['hash_top5'] for fc in buckets[d]) / max(n, 1) * 100)
        if has_claude:
            c1.append(sum(buckets[d][fc]['claude_top1'] for fc in buckets[d]) / max(n, 1) * 100)
    x = np.arange(len(diff_order))
    w = 0.27 if has_claude else 0.4
    ax.bar(x - w, h1, w, label='Hash T1', color='#2ecc71', edgecolor='black')
    ax.bar(x,     h5, w, label='Hash T5', color='#27ae60', edgecolor='black')
    if has_claude:
        ax.bar(x + w, c1, w, label='Claude T1', color='#3498db', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels([d.capitalize() for d in diff_order])
    ax.set_ylabel('Accuracy (%)'); ax.set_title('Accuracy by difficulty')
    ax.set_ylim(0, 105); ax.legend(); ax.grid(axis='y', alpha=0.3)

    # Plot 2: accuracy by frame class
    ax = axes[0, 1]
    h1f = []; c1f = []; labels = []
    for fc in frame_order:
        n_total = sum(buckets[d][fc]['n'] for d in diff_order if fc in buckets[d])
        if n_total == 0:
            continue
        labels.append(fc)
        h1_sum = sum(buckets[d][fc]['hash_top1'] for d in diff_order if fc in buckets[d])
        h1f.append(h1_sum / n_total * 100)
        if has_claude:
            c1_sum = sum(buckets[d][fc]['claude_top1'] for d in diff_order if fc in buckets[d])
            c1f.append(c1_sum / n_total * 100)
    x = np.arange(len(labels))
    w = 0.4 if has_claude else 0.6
    ax.bar(x - w/2, h1f, w, label='Hash T1', color='#2ecc71', edgecolor='black')
    if has_claude:
        ax.bar(x + w/2, c1f, w, label='Claude T1', color='#3498db', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Accuracy (%)'); ax.set_title('Accuracy by frame class')
    ax.set_ylim(0, 105); ax.legend(); ax.grid(axis='y', alpha=0.3)

    # Plot 3: latency
    ax = axes[1, 0]
    ht_means = []; ct_means = []
    for d in diff_order:
        all_ht = [t for fc in buckets[d] for t in buckets[d][fc]['hash_times']]
        all_ct = [t for fc in buckets[d] for t in buckets[d][fc]['claude_times']]
        ht_means.append(np.mean(all_ht) if all_ht else 0)
        ct_means.append(np.mean(all_ct) if all_ct else 0)
    x = np.arange(len(diff_order))
    w = 0.4 if has_claude else 0.6
    ax.bar(x - w/2, ht_means, w, label='Hash', color='#2ecc71', edgecolor='black')
    if has_claude:
        ax.bar(x + w/2, ct_means, w, label='Claude', color='#3498db', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels([d.capitalize() for d in diff_order])
    ax.set_ylabel('Mean latency (ms)'); ax.set_title('Latency by difficulty')
    ax.set_yscale('log' if has_claude else 'linear')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    # Plot 4: frame classifier accuracy by difficulty
    ax = axes[1, 1]
    fok = []
    for d in diff_order:
        n_total = sum(buckets[d][fc]['n'] for fc in buckets[d])
        f_total = sum(buckets[d][fc]['frame_correct'] for fc in buckets[d])
        fok.append(f_total / max(n_total, 1) * 100)
    ax.bar(np.arange(len(diff_order)), fok, color='#9b59b6', edgecolor='black')
    ax.set_xticks(np.arange(len(diff_order)))
    ax.set_xticklabels([d.capitalize() for d in diff_order])
    ax.set_ylabel('Frame class accuracy (%)')
    ax.set_title('Frame classifier accuracy at inference')
    ax.set_ylim(0, 105); ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(fok):
        ax.text(i, v + 1, f"{v:.0f}", ha='center')

    plt.tight_layout()
    plt.savefig(out_path, dpi=80, bbox_inches='tight')
    plt.show()
    print(f"\nSaved plot to {out_path}")


def show_samples(buckets):
    print("\nSample predictions (per difficulty / frame class):")
    print("-" * 96)
    for d in ['easy', 'medium', 'hard']:
        if d not in buckets:
            continue
        for fc in ['modern', 'fullart', 'special']:
            if fc not in buckets[d] or not buckets[d][fc]['samples']:
                continue
            print(f"\n{d.upper()} / {fc.upper()}:")
            for s in buckets[d][fc]['samples'][:1]:
                print(f"  true:           {s['true_name']}  (frame={s['true_frame_class']})")
                print(f"  predicted frame: {s['predicted_frame']}, route: {s['route']}")
                mh = '+' if s['hash_correct'] else ' '
                mc = '+' if s['claude_correct'] else ' '
                print(f"  [{mh}] hash:       {s['hash_pred']}")
                print(f"  [{mc}] claude:     {s['claude_pred']}")


# ---------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-modern', type=int, default=550,
                        help="Modern-frame cards per difficulty (default 550)")
    parser.add_argument('--n-fullart', type=int, default=100,
                        help="Full-art cards per difficulty (default 100)")
    parser.add_argument('--n-special', type=int, default=50,
                        help="Special-layout cards per difficulty (default 50)")
    parser.add_argument('--difficulties', nargs='+',
                        default=['easy', 'medium', 'hard'],
                        choices=['easy', 'medium', 'hard'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-claude', action='store_true')
    parser.add_argument('--reuse-images', action='store_true',
                        help="Reuse existing test images")
    args = parser.parse_args()

    print("=" * 96)
    print("BENCHMARK v2: Multi-region Pipeline vs Claude")
    print("=" * 96)

    cards, pipeline = load_pipeline()
    available = [i for i, c in enumerate(cards)
                 if (IMAGE_DIR / f"{c['id']}.jpg").exists()]
    frame_classes_by_id = load_frame_classes_by_id(cards, available)

    api_key = None if args.no_claude else get_api_key()
    has_claude = api_key is not None

    n_per_frame_class = {
        'modern': args.n_modern,
        'fullart': args.n_fullart,
        'special': args.n_special,
    }
    total_per_diff = sum(n_per_frame_class.values())
    total = total_per_diff * len(args.difficulties)

    if args.reuse_images and BENCH_ITEMS.exists():
        with open(BENCH_ITEMS) as f:
            items = json.load(f)
        items = [it for it in items if it['difficulty'] in args.difficulties]
        print(f"\nReusing {len(items)} existing benchmark images")
    else:
        print(f"\nGenerating {total} test images "
              f"({total_per_diff}/diff: "
              f"{args.n_modern} modern + {args.n_fullart} fullart + "
              f"{args.n_special} special)...")
        items = build_test_set(cards, available, frame_classes_by_id,
                                n_per_difficulty=total_per_diff,
                                n_per_frame_class=n_per_frame_class,
                                seed=args.seed)
        with open(BENCH_ITEMS, 'w') as f:
            json.dump(items, f, indent=2)

    if has_claude:
        cost = (1568 * 3e-6 + 10 * 1.5e-5) * len(items)
        print(f"Claude will be called {len(items)} times (~${cost:.3f})")
    else:
        print(f"\nRunning hash pipeline only (no API key).")

    buckets = run_benchmark(cards, pipeline, items, api_key)
    summary = summarize(buckets, has_claude)
    show_samples(buckets)

    # Persist
    serializable = {}
    for d, by_fc in buckets.items():
        serializable[d] = {}
        for fc, r in by_fc.items():
            serializable[d][fc] = {
                'n': r['n'],
                'hash_top1': r['hash_top1'],
                'hash_top5': r['hash_top5'],
                'claude_top1': r['claude_top1'],
                'frame_correct': r['frame_correct'],
                'hash_times': r['hash_times'],
                'claude_times': r['claude_times'],
                'route_counts': dict(r['route_counts']),
                'frame_pred_counts': dict(r['frame_pred_counts']),
                'samples': r['samples'],
            }
    with open(BENCH_RESULTS, 'w') as f:
        json.dump({
            'summary': summary,
            'per_difficulty': serializable,
            'config': vars(args),
            'has_claude': has_claude,
        }, f, indent=2)
    print(f"\nSaved raw results to {BENCH_RESULTS}")

    plot_results(buckets, has_claude, BENCH_PLOT)

    print("\n" + "=" * 96)
    print("Benchmark complete")
    print("=" * 96)


if __name__ == '__main__':
    main()