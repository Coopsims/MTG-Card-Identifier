"""
Train a YOLO oriented-bounding-box detector specialized for MTG cards.

The pretrained yolo11n-obb.pt was trained on aerial imagery and finds
random sub-regions of card art instead of the card itself. This script
fine-tunes it on synthetic photos of cards.

v3: scenes come from photo_synthesis.py and look like real phone photos -
several cards per photo (scattered, in a grid, fanned in a hand), camera
tilt with true perspective, sleeves, specular glare / streaks / rainbow
sheen, shadows, fingers and dice, phones / ID cards / paper as negatives,
play-mat, wood, cloth and paper backgrounds, white balance, blur, noise and
JPEG. Scenes vary in aspect ratio like real photos.

At inference identify_card.py snaps YOLO's rotated boxes onto the real card
edges (card_detection.refine_quad), so perspective is handled even though
the OBB head can only express rotated rectangles.

Difficulty distribution:
  Default: 25% easy, 40% medium, 35% hard
  --easy-mix:  60% easy, 40% medium, 0% hard  (faster, less robust)

Inputs:
  mtg_data/cards_metadata.json
  mtg_data/card_images/{id}.jpg
  --backgrounds DIR   (optional) photos of your tables / play mats / desks
                      without cards - the most effective realism boost

Outputs:
  mtg_data/yolo_card/                       (synthetic scenes + YOLO labels)
  mtg_data/yolo_card_runs/                  (training run artifacts)
  mtg_data/yolo_card_best.pt                (best fine-tuned weights)

Usage:
  python train_detector.py                  # default: hard-mix scenes, 50 epochs
  python train_detector.py --easy-mix       # gentler mix
  python train_detector.py --gen-only       # generate scenes, don't train
  python train_detector.py --train-only     # data already generated
  python train_detector.py --n-train 15000  # bigger dataset
  python train_detector.py --epochs 30      # shorter training
  python train_detector.py --show           # preview a few generated scenes
  python train_detector.py --backgrounds my_table_photos/
"""

import argparse
import json
import random
import shutil
import warnings
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from mtg_layout import CARD_W, CARD_H
from photo_synthesis import BackgroundBank, synthesize_scene

warnings.filterwarnings('ignore')

DATA_DIR = Path('mtg_data')
IMAGE_DIR = DATA_DIR / 'card_images'
METADATA_PATH = DATA_DIR / 'cards_metadata.json'
YOLO_DATA = DATA_DIR / 'yolo_card'
YOLO_RUNS = DATA_DIR / 'yolo_card_runs'
YOLO_BEST_OUT = DATA_DIR / 'yolo_card_best.pt'
DATASET_YAML = YOLO_DATA / 'cards_obb.yaml'
REAL_LABELS_PATH = DATA_DIR / 'real_photo_labels.json'
# Your folder of real photos; used for validation when --real-photos isn't given
DEFAULT_REAL_PHOTOS_FOLDER = Path(r"C:\Users\Ben Funk\PycharmProjects\DS-Capstone-2\Mtg-Cards")

DEFAULT_N_TRAIN = 8000
DEFAULT_N_VAL = 500
DEFAULT_IMGSZ = 640
DEFAULT_EPOCHS = 50
DEFAULT_BATCH = 16
SEED = 42


# ===========================================================================
# Scene synthesis
#
# Scenes come from photo_synthesis.py: several cards per photo seen through a
# tilted camera (true perspective, not just in-plane rotation), sleeves,
# specular glare, shadows, fingers and dice, phones / ID cards / paper as
# negatives, and play-mat / wood / cloth / paper backgrounds. Labels are
# the corners of every card that is at least 60% visible.

_BANK: Optional[BackgroundBank] = None


def get_background_bank(card_paths, background_dirs=()) -> BackgroundBank:
    global _BANK
    if _BANK is None:
        def art_sampler():
            img = cv2.imread(str(random.choice(card_paths)))
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((CARD_H, CARD_W, 3), np.uint8)
        _BANK = BackgroundBank(art_sampler=art_sampler, photo_dirs=background_dirs)
    return _BANK


def sample_difficulty_for_detector(easy_mix: bool = False) -> str:
    """25% easy, 40% medium, 35% hard by default."""
    r = random.random()
    if easy_mix:
        return 'easy' if r < 0.6 else 'medium'
    if r < 0.25:
        return 'easy'
    if r < 0.65:
        return 'medium'
    return 'hard'


def _load_card(path) -> np.ndarray:
    card_img = np.array(Image.open(path).convert('RGB'))
    if card_img.shape[0] != CARD_H or card_img.shape[1] != CARD_W:
        card_img = cv2.resize(card_img, (CARD_W, CARD_H))
    return card_img


def synthesize_detector_scene(card_paths, imgsz: int, difficulty: str,
                              bank: BackgroundBank, rng: np.random.Generator):
    """
    One training photo. Returns (scene, list of 4x2 corner arrays
    normalised to [0, 1]). Aspect ratio varies like phone photos; YOLO
    letterboxes to imgsz at train time.
    """
    aspect = rng.choice([1.0, 4 / 3, 3 / 4, 16 / 9, 9 / 16])
    w = imgsz if aspect >= 1 else int(imgsz * aspect)
    h = int(imgsz / aspect) if aspect >= 1 else imgsz
    n = int(rng.choice([1, 1, 1, 2, 2, 3, 4, 5, 6, 8], p=None))
    if difficulty == 'easy':
        n = min(n, 2)
    cards = [_load_card(random.choice(card_paths)) for _ in range(n)]
    scene, found = synthesize_scene(cards, (w, h), difficulty, bank, rng)
    corners = [f['corners'] / np.array([w, h], np.float32) for f in found]
    return scene, [np.clip(c, 0, 1) for c in corners]


# ===========================================================================
# Dataset construction

def write_yolo_obb_label(label_path: Path, corners_list):
    """
    YOLO-OBB label format, one line per card: <class> x1 y1 x2 y2 x3 y3 x4 y4
    All values normalized to [0, 1]. Class 0 = card. An empty file is a
    valid negative (photo with no cards).
    """
    if isinstance(corners_list, np.ndarray) and corners_list.ndim == 2:
        corners_list = [corners_list]
    lines = []
    for corners in corners_list:
        parts = ['0'] + [f"{v:.6f}" for xy in corners for v in xy]
        lines.append(' '.join(parts))
    label_path.write_text('\n'.join(lines) + ('\n' if lines else ''))


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
                     real_folder: Path = None, real_oversample: int = 10,
                     background_dirs=()):
    random.seed(seed)
    rng = np.random.default_rng(seed)
    bank = get_background_bank(card_paths, background_dirs)

    train_img = YOLO_DATA / 'train' / 'images'
    train_lab = YOLO_DATA / 'train' / 'labels'
    val_img = YOLO_DATA / 'val' / 'images'
    val_lab = YOLO_DATA / 'val' / 'labels'
    for d in [train_img, train_lab, val_img, val_lab]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    for split, n, img_dir, lab_dir in (('train', n_train, train_img, train_lab),
                                       ('val', n_val, val_img, val_lab)):
        diff_counts = {'easy': 0, 'medium': 0, 'hard': 0}
        n_cards = 0
        print(f"  {split}: generating {n} scenes...")
        for i in tqdm(range(n)):
            difficulty = sample_difficulty_for_detector(easy_mix=easy_mix)
            diff_counts[difficulty] += 1
            scene, corners = synthesize_detector_scene(card_paths, imgsz, difficulty, bank, rng)
            n_cards += len(corners)
            Image.fromarray(scene).save(img_dir / f"scene_{i:06d}.jpg", quality=90)
            write_yolo_obb_label(lab_dir / f"scene_{i:06d}.txt", corners)
        print(f"  {split} difficulty mix: {diff_counts}, {n_cards} labelled cards")

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
            img_with_box = img.copy()
            for line in label.read_text().strip().splitlines():
                tokens = line.split()
                coords = np.array([float(x) for x in tokens[1:]]).reshape(-1, 2)
                coords[:, 0] *= img.shape[1]
                coords[:, 1] *= img.shape[0]
                poly = coords.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(img_with_box, [poly], True, (0, 255, 0), 3)
            ax.imshow(img_with_box)
        else:
            ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# ===========================================================================
# Training

def train_yolo(epochs: int, imgsz: int, batch: int, base_weights: str = 'yolo11n-obb.pt',
               workers: int = 8, device=None):
    from ultralytics import YOLO

    YOLO_RUNS.mkdir(parents=True, exist_ok=True)
    model = YOLO(base_weights)
    print(f"\nTraining {base_weights} for {epochs} epochs at {imgsz}px (batch={batch})...")

    extra = {'device': device} if device is not None else {}
    results = model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        project=str(YOLO_RUNS),
        name='cards',
        exist_ok=True,
        **extra,
        # Augmentation tweaks - the synthetic data already has perspective,
        # colour, glare, etc. baked in, so we don't want Ultralytics adding
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
    parser.add_argument('--backgrounds', type=str, nargs='*', default=[],
                        help="Folders of background photos (tables, mats) without cards")
    parser.add_argument('--base-weights', type=str, default='yolo11n-obb.pt',
                        help="Starting checkpoint (e.g. yolo11s-obb.pt for a larger model, "
                             "or mtg_data/yolo_card_best.pt to continue training)")
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', type=str, default=None, help="e.g. 0, cpu")
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    if args.real_photos is None and DEFAULT_REAL_PHOTOS_FOLDER.is_dir():
        args.real_photos = str(DEFAULT_REAL_PHOTOS_FOLDER)

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("YOLO Card Detector - Training (v3: realistic photo synthesis)")
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
                          real_oversample=args.real_oversample,
                          background_dirs=[Path(d) for d in args.backgrounds])

    if args.gen_only:
        print("\nGeneration complete. Run again with --train-only to train.")
        return

    if not args.validate_only:
        train_yolo(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                   base_weights=args.base_weights, workers=args.workers, device=args.device)

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
    main()
