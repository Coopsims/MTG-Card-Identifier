"""
Evaluate card detection + identification on REAL photos with ground truth.

Synthetic benchmarks (benchmark_run.py) can't tell you how the pipeline does
on actual phone photos - glare on sleeves, busy play mats, perspective,
fingers. This script runs the pipeline over a folder of real photos and
compares what it finds against a ground-truth file.

Ground truth: a JSON file mapping image file name -> list of card names in
that photo (repeat a name for duplicates; order doesn't matter):

    {
      "IMG_2031.jpg": ["Lightning Bolt"],
      "table.jpg":    ["Island", "Island", "Counterspell"]
    }

If no JSON is given, names are taken from the file names instead
("lightning_bolt.jpg", "Lightning Bolt 2.jpg" -> "Lightning Bolt").

Two back-ends:

  --pipeline full        the trained multi-region identifier (identify_card.py,
                         needs the artifacts from train_identifier.py)
  --reference-dir DIR    a training-free identifier built from a folder of
                         reference scans named after the card
                         ("lightning bolt.jpg"). Handy for quick experiments
                         and for checking detection without a trained model.

Usage:
  python evaluate_real_photos.py real_photos/ --pipeline full
  python evaluate_real_photos.py real_photos/ --gt real_photos/gt.json --reference-dir refs/
  python evaluate_real_photos.py real_photos/ ... --save-vis out/   # overlays
  python evaluate_real_photos.py real_photos/ --reference-dir refs/ \
         --yolo mtg_data/yolo_card_best.pt --detector yolo            # detector ablation
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

from card_detection import (CardQuad, find_card_quads, full_image_quad, draw_quads,
                            warp_quad)
from card_matching import (CandidateVerifier, correlation_features, thumbnail_vector,
                           masked_cosine_scan, THUMB_SIZE, quad_variants, remove_glare, fuse)
from mtg_layout import (CARD_W, CARD_H, extract_art_crop, extract_whole_card,
                        phash_64, dhash_64, hamming_distance_vectorized)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def normalize_name(name: str) -> str:
    """Case / punctuation-insensitive card name; drops trailing variant
    numbers ("Island 3", "mountain1")."""
    n = name.lower().strip()
    n = re.sub(r'[\s_]*\(?\d+\)?$', '', n)
    n = n.replace('_', ' ').replace('-', ' ')
    n = re.sub(r"[^a-z0-9' ]", '', n)
    return re.sub(r'\s+', ' ', n).strip()


def load_rgb(path) -> np.ndarray:
    return np.array(ImageOps.exif_transpose(Image.open(path)).convert('RGB'))


# ---------------------------------------------------------------------------
# Training-free identifier over a folder of reference scans

class ReferenceFolderIdentifier:
    """
    Identify cards against a folder of reference images, no training needed.

    Retrieval uses the same art / whole-card pHash+dHash as the trained
    pipeline plus an illumination-normalised correlation over the whole
    (small) database; the top candidates are then verified with glare-aware
    local features. Meant for evaluation and small collections - for 90k
    cards use the trained pipeline, which swaps the correlation scan for
    the learned embeddings.
    """

    def __init__(self, ref_dir: Path, verbose: bool = True):
        paths = sorted(p for p in Path(ref_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not paths:
            raise FileNotFoundError(f"No reference images in {ref_dir}")
        self.paths = paths
        self.names = [re.sub(r'\d+$', '', p.stem).strip().title() for p in paths]
        art_p, art_d, whole_p, whole_d, thumbs = [], [], [], [], []
        t0 = time.perf_counter()
        for p in paths:
            img = self._load(p)
            art, whole = extract_art_crop(img), extract_whole_card(img)
            art_p.append(phash_64(art)); art_d.append(dhash_64(art))
            whole_p.append(phash_64(whole)); whole_d.append(dhash_64(whole))
            thumbs.append(thumbnail_vector(correlation_features(img)))
        self.art_p = np.array(art_p, np.uint64); self.art_d = np.array(art_d, np.uint64)
        self.whole_p = np.array(whole_p, np.uint64); self.whole_d = np.array(whole_d, np.uint64)
        self.thumbs = np.stack(thumbs)
        self.thumbs_sq = self.thumbs ** 2
        self.verifier = CandidateVerifier(lambda i: self._load(self.paths[i]))
        if verbose:
            print(f"Reference index: {len(paths)} cards in {time.perf_counter() - t0:.1f}s")

    @staticmethod
    def _load(p) -> np.ndarray:
        img = load_rgb(p)
        if img.shape[:2] != (CARD_H, CARD_W):
            img = cv2.resize(img, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)
        return img

    def _retrieve(self, card: np.ndarray, query: dict, rotate180: bool):
        if rotate180:
            card = card[::-1, ::-1]
        art, whole = extract_art_crop(card), extract_whole_card(card)
        dist = (hamming_distance_vectorized(phash_64(art), self.art_p)
                + hamming_distance_vectorized(dhash_64(art), self.art_d)
                + hamming_distance_vectorized(phash_64(whole), self.whole_p)
                + hamming_distance_vectorized(dhash_64(whole), self.whole_d))
        hash_prior = 1.0 - dist / 256.0
        qf = query['corr180'] if rotate180 else query['corr']
        valid = query['valid180'] if rotate180 else query['valid']
        q_vec = thumbnail_vector(qf)
        q_valid = np.repeat(cv2.resize(valid.astype(np.uint8), THUMB_SIZE,
                                       interpolation=cv2.INTER_NEAREST).reshape(-1), 3)
        corr = masked_cosine_scan(q_vec, q_valid, self.thumbs, self.thumbs_sq)
        return 0.5 * hash_prior + 0.5 * np.clip(corr, 0, 1)

    def identify_card(self, card: np.ndarray, top_k: int = 5, n_retrieve: int = 20,
                      n_verify: int = 10) -> dict:
        clean, mask = remove_glare(card)
        query = self.verifier.prepare_query(clean, mask)
        # Retrieve for both orientations, verify the best candidates overall
        pool = []
        for rot in (False, True):
            prior = self._retrieve(clean, query, rot)
            for idx in np.argsort(-prior)[:n_retrieve]:
                pool.append((float(prior[idx]), int(idx), rot))
        pool.sort(key=lambda t: -t[0])
        scored = []
        for rank, (prior, idx, rot) in enumerate(pool):
            if rank < n_verify:
                s = self.verifier.score(query, idx, rotate180=rot)
                total = fuse(prior, s['verify'])
            else:
                s = {}
                total = fuse(prior, 0.0)
            scored.append((total, idx, rot, s))
        scored.sort(key=lambda t: -t[0])
        top_total, top_idx, top_rot, top_s = scored[0]
        top_name = self.names[top_idx]
        runner = next((t for t in scored[1:] if self.names[t[1]] != top_name), None)
        seen, top = set(), []
        for t, i, _, _ in scored:
            if (i, self.names[i]) in seen:
                continue
            seen.add((i, self.names[i]))
            top.append((self.names[i], round(t, 4)))
            if len(top) >= top_k:
                break
        return {'name': top_name, 'score': top_total,
                'margin': top_total - (runner[0] if runner else 0.0),
                'rotate180': top_rot, 'top_k': top, 'verify': top_s,
                'glare_frac': query['glare_frac']}

    def identify_quad(self, rgb: np.ndarray, cq: CardQuad, confident: float = 0.45) -> dict:
        best = None
        for label, corners in quad_variants(cq):
            card = warp_quad(rgb, corners)
            res = self.identify_card(card)
            res['variant'] = label
            res['corners'] = corners
            if best is None or res['score'] > best['score']:
                best = res
            if best['score'] >= confident and best['margin'] > 0.08:
                break
        return best


# ---------------------------------------------------------------------------
# Evaluation

class StandaloneDetector:
    """Classical detection, optionally combined with a YOLO-OBB checkpoint."""

    def __init__(self, yolo_weights: Optional[Path] = None, mode: str = 'both'):
        self.model = None
        self.mode = mode
        if yolo_weights is not None and mode != 'classical':
            from ultralytics import YOLO
            self.model = YOLO(str(yolo_weights))

    def __call__(self, rgb: np.ndarray, single: bool = False) -> List[CardQuad]:
        yq, yc = None, None
        if self.model is not None:
            obb = self.model(rgb, conf=0.10, iou=0.5, verbose=False)[0].obb
            if obb is not None and len(obb):
                yq = [c.reshape(4, 2) for c in obb.xyxyxyxy.cpu().numpy()]
                yc = obb.conf.cpu().numpy().tolist()
        quads = find_card_quads(rgb, extra_quads=yq, extra_scores=yc,
                                use_contours=self.mode != 'yolo' or self.model is None)
        if not quads:
            quads = [full_image_quad(rgb)]
        return quads


def detect(rgb: np.ndarray, single: bool) -> List[CardQuad]:
    return StandaloneDetector()(rgb, single)


def evaluate(photo_dir: Path, gt: Dict[str, List[str]], identifier, save_vis: Optional[Path],
             min_score: float, verbose: bool = True, detector=None):
    totals = Counter()
    rows = []
    for fname, true_names in sorted(gt.items()):
        path = photo_dir / fname
        if not path.exists():
            print(f"  missing: {path}")
            continue
        rgb = load_rgb(path)
        t0 = time.perf_counter()
        single = len(true_names) == 1
        if detector is not None:
            quads = detector(rgb, single=False)
        elif hasattr(identifier, 'detect'):
            quads = identifier.detect(rgb, single=False)
        else:
            quads = detect(rgb, single=False)
        preds, used = [], []
        for cq in quads:
            res = identifier.identify_quad(rgb, cq)
            res['det_score'] = cq.score
            accepted = res['accepted'] if 'accepted' in res else \
                (res['score'] >= min_score and res.get('margin', 1.0) >= 0.04)
            if accepted or single:
                preds.append(res)
        if single and preds:
            preds = [max(preds, key=lambda r: r['score'])]
        dt = (time.perf_counter() - t0) * 1000

        want = Counter(normalize_name(n) for n in true_names)
        got = Counter(normalize_name(p['name']) for p in preds)
        correct = sum((want & got).values())
        totals['gt'] += sum(want.values())
        totals['pred'] += sum(got.values())
        totals['correct'] += correct
        totals['images'] += 1
        totals['ms'] += dt
        if single:
            totals['single_images'] += 1
            totals['single_correct'] += int(bool(preds) and normalize_name(preds[0]['name']) in want)
        rows.append((fname, correct, sum(want.values()), sum(got.values()), dt))
        if verbose:
            wrong = list((got - want).elements())
            missed = list((want - got).elements())
            print(f"  {fname:<34} {correct:>3}/{sum(want.values()):<3} found  "
                  f"({sum(got.values())} predicted)  {dt:6.0f}ms"
                  + (f"  wrong={wrong}" if wrong else "") + (f"  missed={missed}" if missed else ""))
        if save_vis is not None:
            save_vis.mkdir(parents=True, exist_ok=True)
            vis_quads = [CardQuad(p['corners'], p['score'], p['variant']) for p in preds]
            labels = [f"{p['name']} {p['score']:.2f}" for p in preds]
            vis = draw_quads(rgb, [CardQuad(np.roll(q.corners, 0, 0), q.score, q.source) for q in vis_quads],
                             labels=labels)
            s = 1400 / max(vis.shape[:2])
            if s < 1:
                vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(save_vis / f"{Path(fname).stem}_eval.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    if totals['images'] == 0:
        print("No images evaluated.")
        return totals
    recall = totals['correct'] / max(totals['gt'], 1)
    precision = totals['correct'] / max(totals['pred'], 1)
    print()
    print("=" * 70)
    print(f"Images: {totals['images']}   cards in ground truth: {totals['gt']}")
    print(f"Recall    (GT cards correctly identified): {recall * 100:5.1f}%")
    print(f"Precision (predictions that are correct):  {precision * 100:5.1f}%")
    if totals['single_images']:
        print(f"Single-card photos, top-1 accuracy:        "
              f"{totals['single_correct'] / totals['single_images'] * 100:5.1f}%  "
              f"({totals['single_correct']}/{totals['single_images']})")
    print(f"Mean time per photo: {totals['ms'] / totals['images']:.0f}ms")
    print("=" * 70)
    return totals


def ground_truth_from_filenames(photo_dir: Path) -> Dict[str, List[str]]:
    gt = {}
    for p in sorted(photo_dir.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS:
            gt[p.name] = [re.sub(r'[\s_-]*\d+$', '', p.stem).replace('_', ' ').strip()]
    return gt


class FullPipelineAdapter:
    """Adapts identify_card.MultiRegionIdentifier to the evaluate() API."""

    def __init__(self, use_ocr: bool = False):
        try:
            import Identify_card as mod
        except ImportError:
            import identify_card as mod
        self.cards, self.pipeline = mod.load_everything(verbose=True, use_ocr=use_ocr)
        self.use_ocr = use_ocr

    def detect(self, rgb: np.ndarray, single: bool) -> List[CardQuad]:
        return self.pipeline.detector.detect(rgb)

    def identify_quad(self, rgb: np.ndarray, cq: CardQuad) -> dict:
        res = self.pipeline.identify_quad(rgb, cq, use_ocr=self.use_ocr)
        top = res['top_k']
        return {'name': self.cards[top[0][0]]['name'], 'score': res['confidence'],
                'margin': res['margin'], 'corners': res['corners'],
                'variant': res.get('variant', 'primary'),
                'accepted': res['quality'] in ('strong', 'good', 'weak')}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('photos', type=Path, help="Folder of real photos")
    ap.add_argument('--gt', type=Path, default=None, help="Ground-truth JSON (default: file names)")
    ap.add_argument('--reference-dir', type=Path, default=None,
                    help="Folder of reference scans for the training-free identifier")
    ap.add_argument('--pipeline', choices=['full'], default=None,
                    help="Use the trained identify_card.py pipeline")
    ap.add_argument('--min-score', type=float, default=0.25,
                    help="Drop multi-card predictions below this confidence")
    ap.add_argument('--save-vis', type=Path, default=None, help="Write overlays here")
    ap.add_argument('--ocr', action='store_true', help="Allow OCR tie-breaks (full pipeline)")
    ap.add_argument('--yolo', type=Path, default=None,
                    help="YOLO-OBB weights for --reference-dir mode (default: classical only)")
    ap.add_argument('--detector', choices=['both', 'classical', 'yolo'], default='both',
                    help="Which detector(s) to use with --reference-dir (needs --yolo for yolo/both)")
    args = ap.parse_args()

    if args.gt is not None:
        gt = json.loads(args.gt.read_text(encoding='utf-8'))
    else:
        gt_file = args.photos / 'ground_truth.json'
        gt = json.loads(gt_file.read_text()) if gt_file.exists() else ground_truth_from_filenames(args.photos)

    detector = None
    if args.reference_dir is not None:
        identifier = ReferenceFolderIdentifier(args.reference_dir)
        detector = StandaloneDetector(args.yolo, args.detector)
    elif args.pipeline == 'full':
        identifier = FullPipelineAdapter(use_ocr=args.ocr)
    else:
        print("Choose --pipeline full or --reference-dir DIR")
        sys.exit(1)

    evaluate(args.photos, gt, identifier, args.save_vis, args.min_score, detector=detector)


if __name__ == '__main__':
    main()
