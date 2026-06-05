"""
Train a YOLO oriented-bounding-box detector specialized for MTG cards.

The pretrained yolo11n-obb.pt was trained on aerial imagery and finds
random sub-regions of card art instead of the card itself. This script
fine-tunes it on synthetic scenes where one MTG card is composited onto
a textured background at a known orientation.

Key change vs v1: synthetic scenes now span easy / medium / hard difficulty
levels matching what benchmark_v2.py produces at inference time. The
previous trainer only generated mild-difficulty scenes, which is why the
detector held up perfectly on the validation set (mAP50-95 = 0.995) but
struggled on hard benchmark images where cards were heavily rotated,
occluded, or sitting on cluttered backgrounds.

Difficulty distribution (controllable via --easy-mix and --hard-mix flags):
  Default: 30% easy, 40% medium, 30% hard
  --easy-mix:  60% easy, 40% medium, 0% hard  (faster, less robust)

Inputs:
  mtg_data/cards_metadata.json
  mtg_data/card_images/{id}.jpg

Outputs:
  mtg_data/yolo_card/                       (synthetic scenes + YOLO labels)
  mtg_data/yolo_card_runs/                  (training run artifacts)
  mtg_data/yolo_card_best.pt                (best fine-tuned weights)

Usage:
  python train_detector.py                  # default: hard-mix scenes, 50 epochs
  python train_detector.py --easy-mix       # use the v1 easy-only mix
  python train_detector.py --gen-only       # generate scenes, don't train
  python train_detector.py --train-only     # data already generated
  python train_detector.py --n-train 10000  # bigger dataset
  python train_detector.py --epochs 30      # shorter training
  python train_detector.py --show           # preview a few generated scenes
"""

import argparse
import json
import random
import shutil
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from mtg_layout import CARD_W, CARD_H

warnings.filterwarnings('ignore')

DATA_DIR = Path('mtg_data')
IMAGE_DIR = DATA_DIR / 'card_images'
METADATA_PATH = DATA_DIR / 'cards_metadata.json'
YOLO_DATA = DATA_DIR / 'yolo_card'
YOLO_RUNS = DATA_DIR / 'yolo_card_runs'
YOLO_BEST_OUT = DATA_DIR / 'yolo_card_best.pt'
DATASET_YAML = YOLO_DATA / 'cards_obb.yaml'
REAL_LABELS_PATH = DATA_DIR / 'real_photo_labels.json'

DEFAULT_N_TRAIN = 5000
DEFAULT_N_VAL = 500
DEFAULT_IMGSZ = 640
DEFAULT_EPOCHS = 50
DEFAULT_BATCH = 16
SEED = 42


# ===========================================================================
# Background generation (matches train_identifier.py and benchmark_v2.py)

def make_background(size: int, difficulty: str = 'medium') -> np.ndarray:
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
# Scene synthesis with corner labels
#
# Returns the scene + the four oriented corners of the card, normalized to
# [0, 1] for YOLO-OBB labels. The corners are post-rotation, so they're the
# actual visible card corners in the scene.

def synthesize_scene_with_corners(card_img: np.ndarray, canvas: int,
                                   difficulty: str = 'medium'):
    """
    Returns (scene, corners_normalized) where:
      scene: HxWx3 uint8 image at canvas resolution
      corners_normalized: 4x2 array of (x, y) in [0, 1], oriented so
                          corners[0] is top-left, [1] top-right,
                          [2] bottom-right, [3] bottom-left.
    """
    H_card, W_card = card_img.shape[:2]
    bg = make_background(canvas, difficulty=difficulty)

    # Difficulty-dependent parameters
    if difficulty == 'easy':
        scale = random.uniform(0.40, 0.70)
        angle_range = 15
        glare_p = 0.0
        occlusion_p = 0.0
        color_jitter = 15
        bright_jitter = (0.85, 1.15)
        blur_p = 0.2
    elif difficulty == 'medium':
        scale = random.uniform(0.35, 0.65)
        angle_range = 30
        glare_p = 0.3
        occlusion_p = 0.0
        color_jitter = 25
        bright_jitter = (0.7, 1.3)
        blur_p = 0.4
    else:  # hard
        scale = random.uniform(0.30, 0.60)
        angle_range = 60
        glare_p = 0.5
        occlusion_p = 0.4
        color_jitter = 40
        bright_jitter = (0.5, 1.5)
        blur_p = 0.5

    new_h = int(canvas * scale)
    new_w = int(new_h * W_card / H_card)
    if new_w > canvas * 0.9:
        new_w = int(canvas * 0.9)
        new_h = int(new_w * H_card / W_card)
    card_resized = cv2.resize(card_img, (new_w, new_h))

    # Card-level color and lighting jitter
    card_f = card_resized.astype(np.float32)
    card_f *= random.uniform(*bright_jitter)
    card_f += random.uniform(-color_jitter, color_jitter)
    card_resized = np.clip(card_f, 0, 255).astype(np.uint8)
    if random.random() < glare_p:
        card_resized = add_glare(card_resized, strength=random.uniform(0.3, 0.6))

    # Rotation + position
    angle = random.uniform(-angle_range, angle_range)
    margin = int(max(new_w, new_h) * 0.6)
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

    # Corner positions in the rotated scene
    corners = np.array([[0, 0], [new_w, 0], [new_w, new_h], [0, new_h]],
                       dtype=np.float32)
    corners_warped = cv2.transform(corners.reshape(1, -1, 2), M).reshape(-1, 2)

    # Optional occlusion - cover one corner. We DON'T modify the corner
    # labels; the detector should learn to predict the true card extent
    # even when one corner is hidden, otherwise it'll always under-predict.
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

    if random.random() < blur_p:
        ksize = random.choice([3, 5, 7])
        scene = cv2.GaussianBlur(scene, (ksize, ksize), 0)

    # Normalize corners to [0, 1] for YOLO labels
    corners_normalized = corners_warped / canvas

    return scene, corners_normalized


def sample_difficulty_for_detector(easy_mix: bool = False) -> str:
    """30% easy, 40% medium, 30% hard by default."""
    if easy_mix:
        r = random.random()
        return 'easy' if r < 0.6 else 'medium'
    r = random.random()
    if r < 0.30:
        return 'easy'
    if r < 0.70:
        return 'medium'
    return 'hard'


# ===========================================================================
# Dataset construction

def write_yolo_obb_label(label_path: Path, corners_normalized: np.ndarray):
    """
    YOLO-OBB label format per line: <class> x1 y1 x2 y2 x3 y3 x4 y4
    All values normalized to [0, 1]. Class 0 = card.
    """
    parts = ['0']
    for x, y in corners_normalized:
        parts.append(f"{x:.6f}")
        parts.append(f"{y:.6f}")
    label_path.write_text(' '.join(parts) + '\n')


def inject_real_photos(real_folder: Path, labels_path: Path, imgsz: int,
                       train_frac: float = 0.8, oversample: int = 10):
    """
    Copy labeled real photos into the YOLO dataset folders.

    Each real photo is resized to imgsz and its label is written in YOLO-OBB
    format.  Because there are very few real photos compared to thousands of
    synthetic ones, each real photo is oversampled (duplicated with a unique
    name) so the model sees them frequently during training.
    """
    if not labels_path.exists():
        print(f"  No real-photo labels at {labels_path} — skipping.")
        print(f"  Run: python label_real_photos.py   to create them.")
        return 0

    labels = json.loads(labels_path.read_text(encoding='utf-8'))
    if not labels:
        print("  Real-photo labels file is empty — skipping.")
        return 0

    train_img = YOLO_DATA / 'train' / 'images'
    train_lab = YOLO_DATA / 'train' / 'labels'
    val_img = YOLO_DATA / 'val' / 'images'
    val_lab = YOLO_DATA / 'val' / 'labels'

    names = list(labels.keys())
    random.shuffle(names)
    split = max(1, int(len(names) * train_frac))
    train_names = names[:split]
    val_names = names[split:] if split < len(names) else names[:1]  # at least 1 val

    count = 0
    for subset, img_dir, lab_dir, reps in [
        (train_names, train_img, train_lab, oversample),
        (val_names, val_img, val_lab, max(1, oversample // 3)),
    ]:
        for name in subset:
            src = real_folder / name
            if not src.exists():
                print(f"  WARNING: {src} not found, skipping")
                continue
            info = labels[name]
            corners = np.array(info['corners_normalized'], dtype=np.float32)

            # Load with EXIF correction, resize to imgsz square
            pil = ImageOps.exif_transpose(Image.open(src)).convert('RGB')
            orig_w, orig_h = pil.size
            pil_resized = pil.resize((imgsz, imgsz))
            img_arr = np.array(pil_resized)

            # corners are already normalized [0,1] — they stay the same
            # after resize because both axes are scaled uniformly to imgsz
            label_line = '0 ' + ' '.join(
                f'{x:.6f} {y:.6f}' for x, y in corners
            ) + '\n'

            for rep in range(reps):
                tag = f"real_{Path(name).stem}_r{rep:02d}"
                Image.fromarray(img_arr).save(img_dir / f"{tag}.jpg", quality=92)
                (lab_dir / f"{tag}.txt").write_text(label_line)
                count += 1

    print(f"  Injected {count} real-photo samples "
          f"({len(train_names)} train × {oversample}, "
          f"{len(val_names)} val × {max(1, oversample // 3)})")
    return count


def generate_dataset(card_paths, n_train: int, n_val: int, imgsz: int,
                     easy_mix: bool, seed: int = SEED,
                     real_folder: Path = None, real_oversample: int = 10):
    rng = random.Random(seed)

    train_img = YOLO_DATA / 'train' / 'images'
    train_lab = YOLO_DATA / 'train' / 'labels'
    val_img = YOLO_DATA / 'val' / 'images'
    val_lab = YOLO_DATA / 'val' / 'labels'
    for d in [train_img, train_lab, val_img, val_lab]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    diff_counts_train = {'easy': 0, 'medium': 0, 'hard': 0}
    diff_counts_val = {'easy': 0, 'medium': 0, 'hard': 0}

    print(f"  train: generating {n_train} scenes...")
    for i in tqdm(range(n_train)):
        card_path = rng.choice(card_paths)
        card_img = np.array(Image.open(card_path).convert('RGB'))
        if card_img.shape[0] != CARD_H or card_img.shape[1] != CARD_W:
            card_img = cv2.resize(card_img, (CARD_W, CARD_H))
        difficulty = sample_difficulty_for_detector(easy_mix=easy_mix)
        diff_counts_train[difficulty] += 1
        scene, corners = synthesize_scene_with_corners(card_img, imgsz, difficulty)
        Image.fromarray(scene).save(train_img / f"scene_{i:06d}.jpg", quality=88)
        write_yolo_obb_label(train_lab / f"scene_{i:06d}.txt", corners)

    print(f"  val: generating {n_val} scenes...")
    for i in tqdm(range(n_val)):
        card_path = rng.choice(card_paths)
        card_img = np.array(Image.open(card_path).convert('RGB'))
        if card_img.shape[0] != CARD_H or card_img.shape[1] != CARD_W:
            card_img = cv2.resize(card_img, (CARD_W, CARD_H))
        difficulty = sample_difficulty_for_detector(easy_mix=easy_mix)
        diff_counts_val[difficulty] += 1
        scene, corners = synthesize_scene_with_corners(card_img, imgsz, difficulty)
        Image.fromarray(scene).save(val_img / f"scene_{i:06d}.jpg", quality=88)
        write_yolo_obb_label(val_lab / f"scene_{i:06d}.txt", corners)

    # YOLO dataset YAML - paths must be relative to the YAML file's directory
    DATASET_YAML.parent.mkdir(parents=True, exist_ok=True)
    DATASET_YAML.write_text(
        f"path: {YOLO_DATA.resolve()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"nc: 1\n"
        f"names: ['card']\n"
    )
    print(f"  Wrote {DATASET_YAML}")
    print(f"  train difficulty mix: {diff_counts_train}")
    print(f"  val difficulty mix:   {diff_counts_val}")

    # Inject real photos if available
    if real_folder is not None:
        inject_real_photos(real_folder, REAL_LABELS_PATH, imgsz,
                           oversample=real_oversample)


def show_samples(n: int = 9):
    """Display a 3x3 grid of generated train scenes for visual inspection."""
    scene_paths = sorted((YOLO_DATA / 'train' / 'images').glob('*.jpg'))
    if not scene_paths:
        print("No generated scenes found. Run with --gen-only first.")
        return
    sample = random.sample(scene_paths, min(n, len(scene_paths)))
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for ax, path in zip(axes.flatten(), sample):
        img = np.array(Image.open(path))
        # Overlay the labeled corners
        label = Path(str(path).replace('images', 'labels').replace('.jpg', '.txt'))
        if label.exists():
            tokens = label.read_text().strip().split()
            coords = np.array([float(x) for x in tokens[1:]]).reshape(-1, 2)
            coords[:, 0] *= img.shape[1]
            coords[:, 1] *= img.shape[0]
            poly = coords.astype(np.int32).reshape(-1, 1, 2)
            img_with_box = img.copy()
            cv2.polylines(img_with_box, [poly], True, (0, 255, 0), 3)
            ax.imshow(img_with_box)
        else:
            ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# ===========================================================================
# Training

def train_yolo(epochs: int, imgsz: int, batch: int):
    from ultralytics import YOLO

    YOLO_RUNS.mkdir(parents=True, exist_ok=True)
    model = YOLO('yolo11n-obb.pt')
    print(f"\nTraining for {epochs} epochs at {imgsz}px (batch={batch})...")

    results = model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(YOLO_RUNS),
        name='cards',
        exist_ok=True,
        # Augmentation tweaks - the synthetic data already has rotation,
        # color jitter, etc. baked in, so we don't want Ultralytics adding
        # heavy mosaic or perspective on top.
        mosaic=0.3,
        mixup=0.0,
        degrees=0.0,
        perspective=0.0001,
        translate=0.05,
        scale=0.2,
        shear=0.0,
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.3,
        cos_lr=True,
        close_mosaic=10,
        patience=10,
    )

    # Find the best.pt file. Ultralytics has a quirk where it sometimes
    # prepends 'runs/obb/' to the project path, so we check both locations.
    if hasattr(results, 'save_dir'):
        run_dir = Path(results.save_dir)
        candidate = run_dir / 'weights' / 'best.pt'
        if candidate.exists():
            shutil.copy(candidate, YOLO_BEST_OUT)
            print(f"\nBest weights copied: {candidate} -> {YOLO_BEST_OUT}")
            return

    # Fallback search
    candidates = [
        YOLO_RUNS / 'cards' / 'weights' / 'best.pt',
        Path('runs/obb') / YOLO_RUNS / 'cards' / 'weights' / 'best.pt',
        Path.cwd() / 'runs' / 'obb' / str(YOLO_RUNS) / 'cards' / 'weights' / 'best.pt',
    ]
    for c in candidates:
        if c.exists():
            shutil.copy(c, YOLO_BEST_OUT)
            print(f"\nBest weights copied: {c} -> {YOLO_BEST_OUT}")
            return

    print(f"\nWARNING: best.pt not found. Searched:")
    for c in candidates:
        print(f"  {c}")
    print(f"You may need to copy it manually to {YOLO_BEST_OUT}")


# ===========================================================================
# Post-training validation on real photos

def validate_on_real_photos(weights_path: Path, real_folder: Path,
                           labels_path: Path = REAL_LABELS_PATH,
                           imgsz: int = DEFAULT_IMGSZ,
                           conf: float = 0.10):
    """
    Run the trained detector on every real photo and report detection results.

    For each labeled photo, checks whether the detector finds the card and
    how close the predicted corners are to the hand-labeled ground truth.
    Also saves a visual summary to mtg_data/real_photo_validation/.
    """
    from ultralytics import YOLO

    if not weights_path.exists():
        print(f"ERROR: weights not found at {weights_path}")
        return

    # Load labels (optional — we can still check detection even without them)
    has_labels = labels_path.exists()
    labels = {}
    if has_labels:
        labels = json.loads(labels_path.read_text(encoding='utf-8'))

    # Find all images in the real folder
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    image_files = sorted(
        p for p in real_folder.iterdir()
        if p.suffix.lower() in extensions
    )
    if not image_files:
        print(f"No images found in {real_folder}")
        return

    model = YOLO(str(weights_path))
    vis_dir = DATA_DIR / 'real_photo_validation'
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Validating detector on {len(image_files)} real photos")
    print(f"Weights: {weights_path}")
    print(f"Confidence threshold: {conf}")
    print(f"{'=' * 70}\n")

    detected = 0
    total = len(image_files)
    results_summary = []

    for img_path in image_files:
        # Load with EXIF correction
        pil = ImageOps.exif_transpose(Image.open(img_path)).convert('RGB')
        img = np.array(pil)

        res = model(img, conf=conf, iou=0.5, verbose=False)
        n_detections = len(res[0].obb) if len(res[0].obb) > 0 else 0
        best_conf = 0.0
        status = 'NO DETECTION'

        if n_detections > 0:
            obb = res[0].obb
            best_idx = int(obb.conf.argmax())
            best_conf = float(obb.conf[best_idx])
            detected += 1
            status = f'DETECTED (conf={best_conf:.3f})'

        print(f"  {img_path.name:<30} {status}")
        results_summary.append({
            'file': img_path.name,
            'detected': n_detections > 0,
            'n_detections': n_detections,
            'best_conf': best_conf,
        })

        # Save visualization: original with detection overlay
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img)
        axes[0].set_title('Original', fontsize=11)
        axes[0].axis('off')

        # Draw detection boxes on a copy
        img_vis = img.copy()
        if n_detections > 0:
            for i in range(n_detections):
                corners = res[0].obb.xyxyxyxy[i].cpu().numpy().astype(np.int32)
                c = float(res[0].obb.conf[i])
                color = (0, 255, 0) if c == best_conf else (255, 165, 0)
                cv2.polylines(img_vis, [corners.reshape(-1, 1, 2)], True, color, 3)
                cv2.putText(img_vis, f'{c:.2f}',
                            tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, color, 2)

        # If we have ground-truth labels, draw them in blue
        if img_path.name in labels:
            gt = np.array(labels[img_path.name]['corners_normalized'],
                          dtype=np.float32)
            gt_px = gt.copy()
            gt_px[:, 0] *= img.shape[1]
            gt_px[:, 1] *= img.shape[0]
            cv2.polylines(img_vis, [gt_px.astype(np.int32).reshape(-1, 1, 2)],
                          True, (0, 100, 255), 2, cv2.LINE_AA)

        axes[1].imshow(img_vis)
        label_str = 'Detected' if n_detections > 0 else 'No Detection'
        axes[1].set_title(f'{label_str}  (green=pred, orange=GT)', fontsize=11)
        axes[1].axis('off')

        plt.suptitle(f"{img_path.name} — {status}", fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig.savefig(vis_dir / f"{img_path.stem}_val.png", dpi=120, bbox_inches='tight')
        plt.close(fig)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"DETECTION RESULTS: {detected}/{total} photos detected "
          f"({100 * detected / total:.0f}%)")
    if detected > 0:
        confs = [r['best_conf'] for r in results_summary if r['detected']]
        print(f"  Avg confidence: {np.mean(confs):.3f}  "
              f"Min: {np.min(confs):.3f}  Max: {np.max(confs):.3f}")
    print(f"Visualizations saved to {vis_dir}/")
    print(f"{'=' * 70}\n")

    # Save JSON summary
    summary_path = vis_dir / 'validation_summary.json'
    summary = {
        'weights': str(weights_path),
        'total': total,
        'detected': detected,
        'detection_rate': detected / total,
        'conf_threshold': conf,
        'results': results_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"Summary saved to {summary_path}")


# ===========================================================================
# Main

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-train', type=int, default=DEFAULT_N_TRAIN)
    parser.add_argument('--n-val', type=int, default=DEFAULT_N_VAL)
    parser.add_argument('--imgsz', type=int, default=DEFAULT_IMGSZ)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH)
    parser.add_argument('--easy-mix', action='store_true',
                        help="Use the gentler easy/medium-only mix (v1 behavior)")
    parser.add_argument('--real-photos', type=str, default=None,
                        help="Folder with labeled real photos (see label_real_photos.py)")
    parser.add_argument('--real-oversample', type=int, default=10,
                        help="How many copies of each real photo to inject (default: 10)")
    parser.add_argument('--gen-only', action='store_true',
                        help="Generate scenes, don't train")
    parser.add_argument('--train-only', action='store_true',
                        help="Skip scene generation; data already exists")
    parser.add_argument('--show', action='store_true',
                        help="Display 9 generated samples and exit")
    parser.add_argument('--validate-only', action='store_true',
                        help="Skip generation and training; just validate on real photos")
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("YOLO Card Detector - Training (v2: hard-mix scenes)")
    print("=" * 70)

    if args.show:
        show_samples()
        return

    # Locate source cards
    if not args.train_only and not args.validate_only:
        card_paths = list(IMAGE_DIR.glob('*.jpg'))
        if not card_paths:
            print(f"ERROR: no cards found in {IMAGE_DIR}")
            return
        print(f"Source cards: {len(card_paths)} on disk")
        mix_label = "easy mix (v1)" if args.easy_mix else "hard mix (default)"
        print(f"Difficulty mix: {mix_label}")
        print(f"Generating {args.n_train} train + {args.n_val} val scenes "
              f"at {args.imgsz}px...")
        real_folder = Path(args.real_photos) if args.real_photos else None
        generate_dataset(card_paths, args.n_train, args.n_val, args.imgsz,
                          easy_mix=args.easy_mix, seed=args.seed,
                          real_folder=real_folder,
                          real_oversample=args.real_oversample)

    if args.gen_only:
        print("\nGeneration complete. Run again with --train-only to train.")
        return

    if not args.validate_only:
        train_yolo(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)

        print("\n" + "=" * 70)
        print("Detector training complete.")
        print(f"Use {YOLO_BEST_OUT} in the identifier instead of yolo11n-obb.pt")
        print("=" * 70)

    # Validate on real photos if a folder was provided
    real_folder = Path(args.real_photos) if args.real_photos else None
    if real_folder is not None and YOLO_BEST_OUT.exists():
        validate_on_real_photos(YOLO_BEST_OUT, real_folder, imgsz=args.imgsz)
    elif args.validate_only and real_folder is None:
        print("ERROR: --validate-only requires --real-photos <folder>")


if __name__ == '__main__':
    REAL_PHOTOS_FOLDER = Path(r"C:\Users\Ben Funk\PycharmProjects\DS-Capstone-2\Mtg-Cards")

    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 70)
    print("YOLO Card Detector - Training (v2: hard-mix scenes)")
    print("=" * 70)

    # Locate source cards
    card_paths = list(IMAGE_DIR.glob('*.jpg'))
    if not card_paths:
        print(f"ERROR: no cards found in {IMAGE_DIR}")
    else:
        print(f"Source cards: {len(card_paths)} on disk")
        print(f"Difficulty mix: hard mix (default)")
        print(f"Generating {DEFAULT_N_TRAIN} train + {DEFAULT_N_VAL} val scenes "
              f"at {DEFAULT_IMGSZ}px...")
        generate_dataset(card_paths, DEFAULT_N_TRAIN, DEFAULT_N_VAL, DEFAULT_IMGSZ,
                         easy_mix=False, seed=SEED,
                         real_folder=REAL_PHOTOS_FOLDER,
                         real_oversample=10)

        train_yolo(epochs=DEFAULT_EPOCHS, imgsz=DEFAULT_IMGSZ, batch=DEFAULT_BATCH)

        print("\n" + "=" * 70)
        print("Detector training complete.")
        print(f"Use {YOLO_BEST_OUT} in the identifier instead of yolo11n-obb.pt")
        print("=" * 70)

        # Validate on real photos
        if YOLO_BEST_OUT.exists():
            validate_on_real_photos(YOLO_BEST_OUT, REAL_PHOTOS_FOLDER, imgsz=DEFAULT_IMGSZ)