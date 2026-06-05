"""
Identify a Magic: The Gathering card from an image.

Multi-region pipeline with confidence-aware parallel routing:
  1. Detect + rectify with the fine-tuned YOLO
  2. Predict frame class + confidence (modern / fullart / special)
  3. Route based on confidence:
       confidence >= FRAME_CONFIDENCE_THRESHOLD: use the predicted path
       confidence <  FRAME_CONFIDENCE_THRESHOLD: try BOTH paths, pick winner
  4. For modern cards in close-call situations, set symbol classifier
     and OCR can both contribute as tie-breakers

The parallel-routing branch is the key change for hard-difficulty inference.
The frame classifier hits ~80% on hard photos, but its low-confidence
predictions are roughly 50/50 between right and wrong. Running both paths
and picking the one with the lower top-1 hash distance turns those coin
flips into deterministic correct routing - the right index produces a
much lower hash distance than the wrong one.

Usage:
  python identify_card.py                      # opens a file picker
  python identify_card.py path/to/card.jpg     # identifies that file
  python identify_card.py --no-gui             # forces stdin input
  python identify_card.py --no-ocr             # skip OCR even on close calls
  python identify_card.py --no-parallel        # disable parallel routing
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

from mtg_layout import (
    CARD_W, CARD_H,
    extract_art_crop, extract_whole_card, crop_region,
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

ART_INPUT = 160
WHOLE_INPUT = 224
FRAME_INPUT = 96
SETSYM_INPUT = 64
EMBEDDING_DIM = 256

# Threshold below which the frame classifier's prediction is considered
# uncertain enough to warrant running both routing paths in parallel.
# Tuned to 0.70 based on the empirical observation that the classifier's
# low-confidence predictions on hard images are roughly chance-level.
FRAME_CONFIDENCE_THRESHOLD = 0.70

NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
ART_TRANSFORM = T.Compose([T.Resize((ART_INPUT, ART_INPUT)), T.ToTensor(), NORMALIZE])
WHOLE_TRANSFORM = T.Compose([T.Resize((WHOLE_INPUT, WHOLE_INPUT)), T.ToTensor(), NORMALIZE])
FRAME_TRANSFORM = T.Compose([T.Resize((FRAME_INPUT, FRAME_INPUT)), T.ToTensor(), NORMALIZE])
SETSYM_TRANSFORM = T.Compose([T.Resize((SETSYM_INPUT, SETSYM_INPUT)), T.ToTensor(), NORMALIZE])


# ---------------------------------------------------------------------------
# Models (must match train_identifier.py exactly)

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
# Detector (uses fine-tuned weights)

class CardDetector:
    CARD_ASPECT = CARD_W / CARD_H
    ASPECT_TOLERANCE = 0.05

    def __init__(self, weights_path: Path = YOLO_BEST, verbose: bool = True):
        from ultralytics import YOLO
        if not weights_path.exists():
            print(f"WARNING: {weights_path} not found. Falling back to pretrained YOLO.")
            print("  Run train_detector.py to get a properly fine-tuned detector.")
            self.model = YOLO('yolo11n-obb.pt')
            self.is_finetuned = False
        else:
            self.model = YOLO(str(weights_path))
            self.is_finetuned = True
            if verbose:
                print(f"Loaded fine-tuned detector: {weights_path.name}")

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

        obb = results[0].obb
        if hasattr(obb, 'conf') and len(obb.conf) > 0:
            best_idx = int(obb.conf.argmax())
        else:
            best_idx = 0
        corners = obb.xyxyxyxy[best_idx].cpu().numpy()
        warped = self._warp(image, corners)

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
# Optional OCR

class TitleOCR:
    def __init__(self, card_names):
        import easyocr
        from symspellpy import SymSpell, Verbosity
        self._Verbosity = Verbosity
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        self.symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        self.name_to_idx = defaultdict(list)
        for i, n in enumerate(card_names):
            self.symspell.create_dictionary_entry(n.lower(), 1)
            self.name_to_idx[n.lower()].append(i)

    def candidates(self, card_image, top_k: int = 20):
        h = card_image.shape[0]
        crop = card_image[:int(h * 0.15), :]
        try:
            res = self.reader.readtext(crop)
        except Exception:
            return [], ""
        if not res:
            return [], ""
        title = " ".join(t for _, t, _ in res)
        out = []
        for s in self.symspell.lookup(title.lower(), self._Verbosity.CLOSEST,
                                       max_edit_distance=2)[:top_k]:
            if s.term in self.name_to_idx:
                out.extend(self.name_to_idx[s.term])
        return out[:top_k], title


# ---------------------------------------------------------------------------
# Pipeline

class MultiRegionIdentifier:
    """
    Routes queries by predicted frame class with confidence-aware fallback
    to parallel evaluation when the classifier is uncertain.
    """

    def __init__(self, cards, db, frame_clf, art_reranker, art_emb,
                 whole_reranker, whole_emb, set_clf, set_codes, detector,
                 title_ocr=None, use_parallel: bool = True):
        self.cards = cards
        self.db_indices = db['indices']
        self.art_phash = db['art_phash']
        self.art_dhash = db['art_dhash']
        self.whole_phash = db['whole_phash']
        self.whole_dhash = db['whole_dhash']
        self.frame_class_db = db.get('frame_class', None)

        self.frame_clf = frame_clf
        self.art_reranker = art_reranker
        self.whole_reranker = whole_reranker
        self.set_clf = set_clf
        self.set_codes = set_codes
        self.detector = detector
        self.title_ocr = title_ocr
        self.use_parallel = use_parallel

        self.art_emb = art_emb
        self.whole_emb = whole_emb

    def _classify_frame(self, card_img: np.ndarray) -> Tuple[str, float]:
        """Returns (predicted_class, softmax_confidence)."""
        with torch.no_grad():
            t = FRAME_TRANSFORM(Image.fromarray(card_img)).unsqueeze(0).to(DEVICE)
            logits = self.frame_clf(t)
            probs = F.softmax(logits, dim=1)
            confidence = float(probs.max().item())
            idx = int(probs.argmax(dim=1).item())
        return FRAME_CLASSES[idx], confidence

    def _classify_set(self, card_img: np.ndarray, top_k: int = 10) -> List[str]:
        if self.set_clf is None or not self.set_codes:
            return []
        sym = crop_region(card_img, 'set_symbol')
        with torch.no_grad():
            t = SETSYM_TRANSFORM(Image.fromarray(sym)).unsqueeze(0).to(DEVICE)
            top = self.set_clf(t).topk(top_k, dim=1).indices[0].cpu().tolist()
        return [self.set_codes[i] for i in top if i < len(self.set_codes)]

    def _hash_path(self, card_img: np.ndarray, frame_class: str,
                   hash_candidates: int = 500) -> dict:
        """
        Run the full hash + re-rank pipeline for ONE routing path (either
        modern/art-crop or fullart/whole-card). Returns a dict with the
        ranked candidates and the top-1 hash distance, which is the signal
        used to compare paths in parallel mode.
        """
        if frame_class == 'modern':
            region = extract_art_crop(card_img)
            db_p, db_d = self.art_phash, self.art_dhash
            emb_db = self.art_emb
            reranker_model = self.art_reranker
            transform = ART_TRANSFORM
            region_label = 'art crop'
        else:
            region = extract_whole_card(card_img)
            db_p, db_d = self.whole_phash, self.whole_dhash
            emb_db = self.whole_emb
            reranker_model = self.whole_reranker
            transform = WHOLE_TRANSFORM
            region_label = 'whole card'

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

        k = min(hash_candidates, len(combined) - 1)
        cand_pos = np.argpartition(combined, k)[:hash_candidates]
        cand_pos = cand_pos[np.argsort(combined[cand_pos])]
        cand_card_idx = self.db_indices[cand_pos]
        sorted_combined = combined[cand_pos]
        top1_hash_dist = int(sorted_combined[0])

        # Re-rank
        with torch.no_grad():
            t = transform(Image.fromarray(region)).unsqueeze(0).to(DEVICE)
            q_emb = reranker_model(t).cpu().numpy()
        sims = (q_emb @ emb_db[cand_pos].T).flatten()
        order = np.argsort(sims)[::-1]
        ranked_card_idx = cand_card_idx[order]
        ranked_sims = sims[order]
        ranked_hash = sorted_combined[order]

        return {
            'frame_class': frame_class,
            'region_label': region_label,
            'region_image': region,
            'top1_hash_dist': top1_hash_dist,
            'ranked_card_idx': ranked_card_idx,
            'ranked_sims': ranked_sims,
            'ranked_hash': ranked_hash,
            'sorted_combined': sorted_combined,
            'cand_card_idx': cand_card_idx,
        }

    def identify(self, image_path: str, top_k: int = 5,
                 hash_candidates: int = 500, use_ocr: bool = True):
        timings = {}

        # Stage 1: detect
        t0 = time.perf_counter()
        img = np.array(Image.open(image_path).convert('RGB'))
        card_img, route = self.detector.detect_and_correct(img)
        timings['detect_ms'] = (time.perf_counter() - t0) * 1000

        # Stage 2: predict frame class WITH CONFIDENCE
        t0 = time.perf_counter()
        frame_class, frame_confidence = self._classify_frame(card_img)
        timings['frame_ms'] = (time.perf_counter() - t0) * 1000

        # Stage 3: routing decision based on confidence
        timings['hash_ms'] = 0.0
        timings['rerank_ms'] = 0.0

        if self.use_parallel and frame_confidence < FRAME_CONFIDENCE_THRESHOLD:
            # Uncertain - run BOTH paths, pick whichever has lower hash distance
            t0 = time.perf_counter()
            modern_result = self._hash_path(card_img, 'modern', hash_candidates)
            other_class = 'fullart' if frame_class == 'fullart' else 'fullart'
            # Note: 'fullart' and 'special' both use the whole-card path,
            # so we can use 'fullart' as the canonical "non-modern" label here.
            non_modern_result = self._hash_path(card_img, 'fullart', hash_candidates)
            timings['hash_ms'] = (time.perf_counter() - t0) * 1000

            if modern_result['top1_hash_dist'] <= non_modern_result['top1_hash_dist']:
                winner = modern_result
                loser = non_modern_result
                pipeline_route = 'parallel_modern'
            else:
                winner = non_modern_result
                loser = modern_result
                pipeline_route = 'parallel_whole'

            # The winning path's frame class becomes the "effective" routing
            effective_frame = winner['frame_class']
            # If the winner disagrees with the classifier's prediction, use
            # the winner's class for downstream decisions (set tie-break, etc.)
            losing_top1 = loser['top1_hash_dist']
            winning_top1 = winner['top1_hash_dist']
        else:
            # Confident - just run the predicted path
            t0 = time.perf_counter()
            winner = self._hash_path(card_img, frame_class, hash_candidates)
            timings['hash_ms'] = (time.perf_counter() - t0) * 1000
            pipeline_route = 'single'
            effective_frame = frame_class
            losing_top1 = None
            winning_top1 = winner['top1_hash_dist']

        # Stage 4: fast-hash bypass if the winning path is decisive
        sorted_combined = winner['sorted_combined']
        gap = (sorted_combined[1] - sorted_combined[0]
               if len(sorted_combined) > 1 else 0)
        fast_path = sorted_combined[0] <= 4 and gap >= 8

        if fast_path:
            cand_card_idx = winner['cand_card_idx']
            top = [(int(cand_card_idx[i]),
                    float(1.0 / (1 + sorted_combined[i])),
                    int(sorted_combined[i]))
                   for i in range(min(top_k, len(cand_card_idx)))]
            timings['set_ms'] = 0.0
            timings['ocr_ms'] = 0.0
            timings['total_ms'] = sum(timings.values())
            return {
                'top_k': top,
                'predicted_frame': frame_class,
                'frame_confidence': frame_confidence,
                'effective_frame': effective_frame,
                'detection_route': route,
                'pipeline_route': f'{pipeline_route}+fast_hash',
                'region_label': winner['region_label'],
                'losing_path_hash_dist': losing_top1,
                'winning_path_hash_dist': winning_top1,
                'ocr_title': None,
                'set_predictions': [],
                'timings': timings,
                'card_image': card_img,
                'region_image': winner['region_image'],
            }

        # Stage 5: re-ranker results are already in winner. Tie-breakers next.
        timings['rerank_ms'] = 0.0  # already counted in hash_ms above
        ranked_card_idx = winner['ranked_card_idx']
        ranked_sims = winner['ranked_sims']
        ranked_hash = winner['ranked_hash']

        timings['set_ms'] = 0.0
        timings['ocr_ms'] = 0.0
        set_predictions: List[str] = []
        ocr_title = None

        # Tie-breakers only fire on modern cards with tight margins. Use the
        # effective frame class (from winning path) so that a parallel run
        # that flipped to whole-card path skips the tie-breakers correctly.
        if effective_frame == 'modern' and len(ranked_sims) > 1:
            margin = ranked_sims[0] - ranked_sims[1]

            if margin < 0.05 and self.set_clf is not None:
                t0 = time.perf_counter()
                set_predictions = self._classify_set(card_img, top_k=10)
                timings['set_ms'] = (time.perf_counter() - t0) * 1000
                if set_predictions:
                    set_set = set(set_predictions)
                    boost = np.array([
                        0.10 if self.cards[int(ci)].get('set') in set_set else 0.0
                        for ci in ranked_card_idx
                    ])
                    ranked_sims = ranked_sims + boost
                    order2 = np.argsort(ranked_sims)[::-1]
                    ranked_card_idx = ranked_card_idx[order2]
                    ranked_sims = ranked_sims[order2]
                    ranked_hash = ranked_hash[order2]

            margin = ranked_sims[0] - ranked_sims[1] if len(ranked_sims) > 1 else 1.0
            if use_ocr and margin < 0.05 and self.title_ocr is not None:
                t0 = time.perf_counter()
                ocr_cands, ocr_title = self.title_ocr.candidates(card_img, top_k=20)
                timings['ocr_ms'] = (time.perf_counter() - t0) * 1000
                if ocr_cands:
                    ocr_set = set(ocr_cands)
                    boost = np.array([
                        0.10 if int(ci) in ocr_set else 0.0
                        for ci in ranked_card_idx
                    ])
                    ranked_sims = ranked_sims + boost
                    order2 = np.argsort(ranked_sims)[::-1]
                    ranked_card_idx = ranked_card_idx[order2]
                    ranked_sims = ranked_sims[order2]
                    ranked_hash = ranked_hash[order2]

        timings['total_ms'] = sum(timings.values())
        top = [(int(ranked_card_idx[i]), float(ranked_sims[i]), int(ranked_hash[i]))
               for i in range(min(top_k, len(ranked_card_idx)))]
        return {
            'top_k': top,
            'predicted_frame': frame_class,
            'frame_confidence': frame_confidence,
            'effective_frame': effective_frame,
            'detection_route': route,
            'pipeline_route': pipeline_route,
            'region_label': winner['region_label'],
            'losing_path_hash_dist': losing_top1,
            'winning_path_hash_dist': winning_top1,
            'ocr_title': ocr_title,
            'set_predictions': set_predictions,
            'timings': timings,
            'card_image': card_img,
            'region_image': winner['region_image'],
        }


# ---------------------------------------------------------------------------
# Display

def display_result(image_path, result, cards):
    top = result['top_k']
    timings = result['timings']
    route = result['pipeline_route']
    frame_class = result['predicted_frame']
    confidence = result['frame_confidence']
    effective = result['effective_frame']

    if 'fast_hash' in route:
        confidence_str = f"hash distance {top[0][2]} (lower = better)"
    else:
        confidence_str = f"similarity {top[0][1]:.3f}"

    top1_card = cards[top[0][0]]

    print()
    print("=" * 78)
    print(f"PREDICTION: {top1_card['name']}")
    print("=" * 78)
    if 'set_name' in top1_card:
        print(f"  Set:           {top1_card.get('set_name', '?')} "
              f"({top1_card.get('set', '?').upper()})")
    if 'collector_number' in top1_card:
        print(f"  Collector:     #{top1_card.get('collector_number', '?')}")
    if 'type_line' in top1_card:
        print(f"  Type:          {top1_card.get('type_line', '?')}")
    if 'mana_cost' in top1_card and top1_card.get('mana_cost'):
        print(f"  Mana cost:     {top1_card.get('mana_cost')}")
    print(f"  Confidence:    {confidence_str}")
    if effective != frame_class:
        print(f"  Frame class:   {frame_class} (predicted, {confidence:.2f}) "
              f"-> {effective} (chose by hash distance)")
    else:
        print(f"  Frame class:   {frame_class} ({confidence:.2f} confidence, "
              f"route via {result['region_label']})")
    print(f"  Detection:     {result['detection_route']}")
    print(f"  Pipeline:      {route}")
    if result.get('losing_path_hash_dist') is not None:
        print(f"  Hash dist:     winner={result['winning_path_hash_dist']}, "
              f"loser={result['losing_path_hash_dist']}")
    if result['set_predictions']:
        print(f"  Set predict:   {', '.join(result['set_predictions'][:5])}")
    if result.get('ocr_title'):
        print(f"  OCR title:     '{result['ocr_title']}'")
    print()
    print(f"Top-{len(top)}:")
    for rank, (idx, score, hash_d) in enumerate(top, 1):
        print(f"  {rank}. {cards[idx]['name']:<40} sim={score:.4f}  hash_d={hash_d}")
    print()
    print("Timing:")
    for k in ['detect_ms', 'frame_ms', 'hash_ms', 'rerank_ms', 'set_ms', 'ocr_ms', 'total_ms']:
        if k in timings:
            print(f"  {k:<14} {timings[k]:>6.1f}ms")
    print("=" * 78)

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    query_img = Image.open(image_path)
    axes[0, 0].imshow(query_img)
    axes[0, 0].set_title("Query image", fontsize=10)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(result['card_image'])
    axes[0, 1].set_title(f"After detection ({result['detection_route']})", fontsize=10)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(result['region_image'])
    axes[0, 2].set_title(
        f"{result['region_label']} ({effective}, conf={confidence:.2f})",
        fontsize=10,
    )
    axes[0, 2].axis('off')

    for i in range(3):
        if i < len(top):
            idx, score, hash_d = top[i]
            match_path = IMAGE_DIR / f"{cards[idx]['id']}.jpg"
            if match_path.exists():
                axes[1, i].imshow(Image.open(match_path))
                color = 'green' if i == 0 else 'gray'
                axes[1, i].set_title(
                    f"#{i+1}: {cards[idx]['name'][:30]}\n"
                    f"sim={score:.3f}  hash_d={hash_d}",
                    fontsize=9, color=color,
                )
            else:
                axes[1, i].text(0.5, 0.5, "image\nnot found", ha='center', va='center')
                axes[1, i].set_title(f"#{i+1}: {cards[idx]['name'][:30]}", fontsize=9)
        axes[1, i].axis('off')

    plt.suptitle(f"Predicted: {top1_card['name']}  ({timings['total_ms']:.0f}ms, "
                  f"{route})",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# File picker

def pick_file_gui() -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askopenfilename(
            title="Select a card image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.webp *.bmp"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return path if path else None
    except Exception:
        return None


def pick_file_stdin() -> Optional[str]:
    try:
        path = input("Path to card image (or blank to quit): ").strip().strip('"').strip("'")
    except (EOFError, KeyboardInterrupt):
        return None
    return path if path else None


# ---------------------------------------------------------------------------
# Loading

def _check_artifacts():
    required = {
        HASH_DB_PATH: "hash database",
        ART_RERANKER_PATH: "art re-ranker checkpoint",
        ART_EMB_PATH: "art embeddings",
        WHOLE_RERANKER_PATH: "whole-card re-ranker checkpoint",
        WHOLE_EMB_PATH: "whole-card embeddings",
        FRAME_CLF_PATH: "frame classifier checkpoint",
    }
    missing = [(p, label) for p, label in required.items() if not p.exists()]
    if missing:
        print("ERROR: required artifacts missing. Run train_identifier.py first.")
        for p, label in missing:
            print(f"  missing: {p}  ({label})")
        sys.exit(1)


def load_everything(verbose: bool = True, use_parallel: bool = True):
    if verbose:
        print(f"Device: {DEVICE}")
        print("Loading metadata, hash DB, models, embeddings...")

    _check_artifacts()

    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        cards = json.load(f)

    db = np.load(HASH_DB_PATH)
    required_keys = ['indices', 'art_phash', 'art_dhash', 'whole_phash', 'whole_dhash']
    missing = [k for k in required_keys if k not in db.files]
    if missing:
        print(f"ERROR: hash DB missing keys: {missing}")
        sys.exit(1)

    if verbose:
        print(f"  Cards: {len(cards):,}")
        print(f"  Hash DB: {len(db['indices']):,} entries (art + whole)")

    frame_clf = FrameClassifier(n_classes=len(FRAME_CLASSES)).to(DEVICE)
    frame_clf.load_state_dict(torch.load(FRAME_CLF_PATH, map_location=DEVICE))
    frame_clf.eval()
    if verbose:
        print(f"  Frame classifier loaded")

    art_reranker = MobileNetReranker(EMBEDDING_DIM).to(DEVICE)
    art_reranker.load_state_dict(torch.load(ART_RERANKER_PATH, map_location=DEVICE))
    art_reranker.eval()
    art_emb = np.load(ART_EMB_PATH)
    if verbose:
        print(f"  Art re-ranker loaded ({art_emb.shape} embeddings)")

    whole_reranker = MobileNetReranker(EMBEDDING_DIM).to(DEVICE)
    whole_reranker.load_state_dict(torch.load(WHOLE_RERANKER_PATH, map_location=DEVICE))
    whole_reranker.eval()
    whole_emb = np.load(WHOLE_EMB_PATH)
    if verbose:
        print(f"  Whole-card re-ranker loaded ({whole_emb.shape} embeddings)")

    set_clf = None
    set_codes: List[str] = []
    if SET_CLF_PATH.exists() and SET_CODES_PATH.exists():
        with open(SET_CODES_PATH) as f:
            set_codes = json.load(f)
        set_clf = SetSymbolClassifier(n_classes=len(set_codes)).to(DEVICE)
        set_clf.load_state_dict(torch.load(SET_CLF_PATH, map_location=DEVICE))
        set_clf.eval()
        if verbose:
            print(f"  Set classifier loaded ({len(set_codes)} sets)")
    elif verbose:
        print(f"  Set classifier not found - tie-break will use OCR only")

    detector = CardDetector(weights_path=YOLO_BEST, verbose=verbose)

    title_ocr = None
    try:
        if verbose:
            print("Loading OCR (this can take 10-20s on first run)...")
        title_ocr = TitleOCR([c['name'] for c in cards])
        if verbose:
            print("  OCR ready")
    except Exception as e:
        if verbose:
            print(f"  OCR unavailable ({e}). Pipeline will skip OCR confirmation.")

    if verbose:
        print(f"  Parallel routing: {'ENABLED' if use_parallel else 'DISABLED'} "
              f"(threshold={FRAME_CONFIDENCE_THRESHOLD})")

    pipeline = MultiRegionIdentifier(
        cards, db, frame_clf, art_reranker, art_emb,
        whole_reranker, whole_emb, set_clf, set_codes,
        detector, title_ocr, use_parallel=use_parallel,
    )
    return cards, pipeline


def run_one(pipeline, cards, image_path: str, use_ocr: bool = True):
    if not Path(image_path).exists():
        print(f"File not found: {image_path}")
        return False
    try:
        result = pipeline.identify(image_path, top_k=5, use_ocr=use_ocr)
    except Exception as e:
        print(f"Error during identification: {e}")
        import traceback
        traceback.print_exc()
        return False
    display_result(image_path, result, cards)
    return True


def main():
    parser = argparse.ArgumentParser(description="Identify an MTG card from an image")
    parser.add_argument('image', nargs='?', help="Path to a card image (optional)")
    parser.add_argument('--no-gui', action='store_true',
                        help="Force stdin input instead of file picker")
    parser.add_argument('--once', action='store_true',
                        help="Identify one image and exit")
    parser.add_argument('--no-ocr', action='store_true',
                        help="Skip OCR even when re-ranker is unsure")
    parser.add_argument('--no-parallel', action='store_true',
                        help="Disable parallel routing on low-confidence frame predictions")
    args = parser.parse_args()

    cards, pipeline = load_everything(use_parallel=not args.no_parallel)
    use_ocr = not args.no_ocr

    if args.image:
        run_one(pipeline, cards, args.image, use_ocr=use_ocr)
        return

    print()
    print("Pick a card image to identify (close the dialog or send EOF to quit).")
    print()
    while True:
        path = pick_file_stdin() if args.no_gui else (pick_file_gui() or pick_file_stdin())
        if not path:
            print("Goodbye.")
            break
        if not run_one(pipeline, cards, path, use_ocr=use_ocr):
            continue
        if args.once:
            break
        print()


if __name__ == '__main__':
    main()