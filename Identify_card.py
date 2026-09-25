"""
Identify Magic: The Gathering cards in real photos.

Pipeline:
  1. Detect every card: fine-tuned YOLO proposals + classical quad detection
     (card_detection.py), refined onto the real card edges so perspective,
     rotation and busy / non-white backgrounds don't skew the crop.
  2. Rectify each card to 488x680 and inpaint glare (card_matching.py).
  3. Predict the frame class (modern / fullart / special). When the frame
     classifier is unsure, both extraction routes are tried.
  4. Retrieve candidates from the WHOLE database: pHash/dHash distances plus
     re-ranker embedding similarity over every card - not just the top hash
     hits, which glare can push out of reach. Both 0 and 180 degree
     orientations are scored.
  5. Verify the best candidates with glare-aware local features (ORB +
     RANSAC restricted to near-aligned matches) and masked correlation,
     then fuse with the retrieval score.
  6. Close calls on modern frames still go to the set-symbol classifier and
     OCR, as before.
  7. Each detection also has alternative outlines (inner frame, a card
     inside a toploader, expanded white-border variants); whichever matches
     best wins.

Usage:
  python Identify_card.py                      # opens a file picker
  python Identify_card.py path/to/card.jpg     # identifies that file
  python Identify_card.py photo.jpg --all      # every card in the photo
  python Identify_card.py --batch folder/      # save a result figure per image
  python Identify_card.py --no-gui             # forces stdin input
  python Identify_card.py --no-ocr             # skip OCR even on close calls
  python Identify_card.py --fast               # skip local-feature verification
  python Identify_card.py --yolo-only          # skip classical detection (faster)
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
from PIL import Image, ImageOps

from mtg_layout import (
    CARD_W, CARD_H,
    extract_art_crop, extract_whole_card, crop_region,
    FRAME_CLASSES, FRAME_CLASS_TO_IDX,
    phash_64, dhash_64, hamming_distance_vectorized,
)
from card_detection import (
    CardQuad, find_card_quads, full_image_quad, warp_quad, draw_quads,
)
from card_matching import (
    CandidateVerifier, remove_glare, quad_variants, fuse,
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

# Folder processed when the script is run with no arguments (your photos)
DEFAULT_BATCH_FOLDER = Path(r"C:\Users\Ben Funk\PycharmProjects\DS-Capstone-2\Mtg-Cards")

ART_INPUT = 160
WHOLE_INPUT = 224
FRAME_INPUT = 96
SETSYM_INPUT = 64
EMBEDDING_DIM = 256

# Below this softmax confidence the frame class is treated as uncertain and
# both routes (art crop / whole card) are scored.
FRAME_CONFIDENCE_THRESHOLD = 0.70

# Match quality grades, from the local-feature verification (how much of
# the illustration actually matched) and the margin over the best card with
# a *different name*. The fused score ranks candidates well but isn't a good
# yes/no signal: retrieval similarity is high-ish even for a wrong card when
# the right one isn't in the database. Measured on real photos: true matches
# verify at >= 0.2 (>= ~14 illustration inliers), a card missing from the
# database at < 0.1.
QUALITY_STRONG = 0.45
QUALITY_GOOD = (0.20, 0.05)   # (verify, margin)
QUALITY_WEAK = (0.12, 0.03)
ACCEPTED_QUALITIES = ('strong', 'good', 'weak')

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
    def __init__(self, embedding_dim: int = EMBEDDING_DIM, pretrained: bool = False):
        super().__init__()
        # At inference every weight comes from our checkpoint, so there's no
        # need to download the ImageNet weights first.
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_large(weights=weights)
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
# Detector: YOLO proposals + classical quad detection

class CardDetector:
    """
    Finds cards and returns perspective-correct outlines.

    The fine-tuned YOLO (if present) proposes rotated boxes; the classical
    detector in card_detection.py proposes contour-based quads. YOLO boxes
    are snapped onto the real card edges, both sets compete, and agreement
    between them raises confidence. Without YOLO weights the classical
    detector runs alone - the pretrained aerial-imagery YOLO is never used,
    since it finds art fragments rather than cards.
    """

    def __init__(self, weights_path: Path = YOLO_BEST, verbose: bool = True,
                 use_yolo: bool = True, yolo_conf: float = 0.10, use_classical: bool = True):
        self.model = None
        self.yolo_conf = yolo_conf
        self.use_classical = use_classical
        if use_yolo and weights_path.exists():
            from ultralytics import YOLO
            self.model = YOLO(str(weights_path))
            if verbose:
                print(f"Loaded fine-tuned detector: {weights_path.name}")
        elif verbose:
            print(f"No fine-tuned YOLO at {weights_path} - using classical card detection only.")
        self.is_finetuned = self.model is not None

    def propose(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        if self.model is None:
            return [], []
        results = self.model(image, conf=self.yolo_conf, iou=0.5, verbose=False)
        obb = results[0].obb
        if obb is None or len(obb) == 0:
            return [], []
        corners = obb.xyxyxyxy.cpu().numpy()
        confs = obb.conf.cpu().numpy() if hasattr(obb, 'conf') else np.full(len(corners), 0.5)
        return [c.reshape(4, 2) for c in corners], [float(c) for c in confs]

    def detect(self, image: np.ndarray, max_cards: Optional[int] = None) -> List[CardQuad]:
        yq, yc = self.propose(image)
        quads = find_card_quads(image, extra_quads=yq or None, extra_scores=yc or None,
                                max_cards=max_cards,
                                use_contours=self.use_classical or self.model is None)
        # A pre-cropped photo or clean scan: the card *is* the image
        full = full_image_quad(image)
        covered = any(q.area > 0.8 * image.shape[0] * image.shape[1] for q in quads)
        if not quads:
            quads = [full]
        elif not covered and full.score > 0.4:
            quads[0].alternatives.append(full)
        return quads

    def detect_and_correct(self, image):
        """Backwards-compatible single-card API: (rectified card, route)."""
        quads = self.detect(image, max_cards=1)
        q = quads[0]
        route = 'clean' if q.source == 'full_image' else f'detected_{q.source}'
        return warp_quad(image, q.corners), route


# ---------------------------------------------------------------------------
# Optional OCR

class TitleOCR:
    """EasyOCR + SymSpell name fuzzy match. Constructed lazily."""
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
        h, w = card_image.shape[:2]
        # Title bar only (skip the mana cost on the right)
        crop = card_image[int(h * 0.03):int(h * 0.11), int(w * 0.04):int(w * 0.80)]
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
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
    Detects every card in a photo and identifies each one. Routes by
    predicted frame class; each route has its own hash index, embedding
    model and embedding database.
    """

    def __init__(self, cards, db, frame_clf, art_reranker, art_emb,
                 whole_reranker, whole_emb, set_clf, set_codes, detector,
                 title_ocr=None, use_parallel: bool = True, verify: bool = True,
                 image_dir: Path = IMAGE_DIR):
        self.cards = cards
        self.db_indices = db['indices']
        self.art_phash = db['art_phash']
        self.art_dhash = db['art_dhash']
        self.whole_phash = db['whole_phash']
        self.whole_dhash = db['whole_dhash']
        self.frame_class_db = db['frame_class'] if 'frame_class' in db else None
        self.db_names = np.array([cards[int(i)]['name'] for i in self.db_indices])

        self.frame_clf = frame_clf
        self.art_reranker = art_reranker
        self.whole_reranker = whole_reranker
        self.set_clf = set_clf
        self.set_codes = set_codes  # list[str], index -> set code
        self.detector = detector
        self.title_ocr = title_ocr
        self.use_parallel = use_parallel
        self.verify = verify
        self.image_dir = image_dir

        self.art_emb = art_emb
        self.whole_emb = whole_emb
        self.verifier = CandidateVerifier(self._load_db_image)

    # -- helpers -----------------------------------------------------------

    def _load_db_image(self, pos: int) -> Optional[np.ndarray]:
        card = self.cards[int(self.db_indices[pos])]
        path = self.image_dir / f"{card['id']}.jpg"
        if not path.exists():
            return None
        img = cv2.imread(str(path))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (CARD_H, CARD_W):
            img = cv2.resize(img, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)
        return img

    def _classify_frame(self, card_img: np.ndarray) -> Tuple[str, float]:
        with torch.no_grad():
            t = FRAME_TRANSFORM(Image.fromarray(card_img)).unsqueeze(0).to(DEVICE)
            probs = F.softmax(self.frame_clf(t), dim=1)[0]
        idx = int(probs.argmax().item())
        return FRAME_CLASSES[idx], float(probs[idx].item())

    def _classify_set(self, card_img: np.ndarray, top_k: int = 10) -> List[str]:
        if self.set_clf is None or not self.set_codes:
            return []
        sym = crop_region(card_img, 'set_symbol')
        with torch.no_grad():
            t = SETSYM_TRANSFORM(Image.fromarray(sym)).unsqueeze(0).to(DEVICE)
            top = self.set_clf(t).topk(top_k, dim=1).indices[0].cpu().tolist()
        return [self.set_codes[i] for i in top if i < len(self.set_codes)]

    def _route(self, route: str):
        if route == 'modern':
            return (extract_art_crop, self.art_phash, self.art_dhash, self.art_emb,
                    self.art_reranker, ART_TRANSFORM, 'art crop')
        return (extract_whole_card, self.whole_phash, self.whole_dhash, self.whole_emb,
                self.whole_reranker, WHOLE_TRANSFORM, 'whole card')

    # -- identification of one rectified card --------------------------------

    def identify_card_image(self, card_img: np.ndarray, top_k: int = 5,
                            hash_candidates: int = 500, use_ocr: bool = True,
                            n_candidates: int = 40, n_verify: int = 10,
                            n_verify_max: int = 40) -> dict:
        """
        Identify a rectified 488x680 card (either orientation). Returns the
        ranked candidates plus confidence / margin for the top one.
        """
        timings = defaultdict(float)
        t0 = time.perf_counter()
        clean, glare = remove_glare(card_img)
        timings['glare_ms'] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        frame_class, frame_conf = self._classify_frame(clean)
        timings['frame_ms'] = (time.perf_counter() - t0) * 1000
        routes = [frame_class]
        if self.use_parallel and frame_conf < FRAME_CONFIDENCE_THRESHOLD:
            routes = ['modern', 'fullart']
        routes = ['modern' if r == 'modern' else 'fullart' for r in routes]
        routes = list(dict.fromkeys(routes))

        # Retrieval over the full database, both orientations, each route
        t0 = time.perf_counter()
        hyps = []  # (rot, route, prior array, hash dist array, region)
        for route in routes:
            extractor, db_p, db_d, emb_db, model, transform, _ = self._route(route)
            regions, tensors = [], []
            for rot in (0, 180):
                img = clean if rot == 0 else np.ascontiguousarray(clean[::-1, ::-1])
                region = extractor(img)
                regions.append((rot, img, region))
                tensors.append(transform(Image.fromarray(region)))
            with torch.no_grad():
                q = model(torch.stack(tensors).to(DEVICE)).cpu().numpy()
            q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-8, None)
            sims_all = np.clip(q @ emb_db.T, -1.0, 1.0)  # (2, N)
            for k, (rot, img, region) in enumerate(regions):
                dist = (hamming_distance_vectorized(phash_64(region), db_p)
                        + hamming_distance_vectorized(dhash_64(region), db_d))
                sims = sims_all[k]
                prior = 0.7 * np.clip(sims, 0, 1) + 0.3 * (1.0 - dist / 128.0)
                hyps.append({'rot': rot, 'route': route, 'img': img, 'region': region,
                             'prior': prior, 'dist': dist, 'sims': sims})
        timings['retrieve_ms'] = (time.perf_counter() - t0) * 1000

        # Candidate pool: best by fused prior, plus the best pure-hash and
        # pure-embedding hits of each hypothesis
        pool = {}
        for h_i, h in enumerate(hyps):
            k = min(n_candidates, len(h['prior']) - 1)
            picks = set(np.argpartition(-h['prior'], k)[:n_candidates].tolist())
            kk = min(10, len(h['dist']) - 1)
            picks |= set(np.argpartition(h['dist'], kk)[:10].tolist())
            picks |= set(np.argpartition(-h['sims'], kk)[:10].tolist())
            for pos in picks:
                key = (pos, h['rot'])
                p = float(h['prior'][pos])
                if key not in pool or p > pool[key][0]:
                    pool[key] = (p, h_i)
        ranked = sorted(((p, pos, rot, h_i) for (pos, rot), (p, h_i) in pool.items()),
                        key=lambda t: -t[0])

        # Fast path: hash decisively agrees with the embedding, skip verification
        best_h = hyps[ranked[0][3]]
        d_sorted = np.partition(best_h['dist'], 1)[:2]
        decisive = (d_sorted[0] <= 4 and d_sorted[1] - d_sorted[0] >= 8
                    and int(np.argmin(best_h['dist'])) == ranked[0][1])

        # Verification
        t0 = time.perf_counter()
        scored = []
        query = None
        if self.verify and not decisive:
            query = self.verifier.prepare_query(clean, glare)

        def verify_range(lo, hi):
            for p, pos, rot, h_i in ranked[lo:hi]:
                v = self.verifier.score(query, pos, rotate180=(rot == 180))
                scored.append([fuse(p, v['verify'], w_verify=0.5), pos, rot, h_i, v])

        if query is not None:
            verify_range(0, n_verify)
            # Nothing convincing among the first few: a misaligned crop or
            # heavy glare can push the right card down the retrieval list,
            # so look further before giving up.
            if max(r[4]['verify'] for r in scored) < QUALITY_GOOD[0]:
                verify_range(n_verify, n_verify_max)
            n_done = len(scored)
            scored += [[fuse(p, 0.0, w_verify=0.5), pos, rot, h_i, {}]
                       for p, pos, rot, h_i in ranked[n_done:]]
        else:
            scored = [[p, pos, rot, h_i, {}] for p, pos, rot, h_i in ranked]
        scored.sort(key=lambda t: -t[0])
        timings['verify_ms'] = (time.perf_counter() - t0) * 1000

        top_h = hyps[scored[0][3]]
        effective_frame = top_h['route']
        card_oriented = top_h['img']

        # Tie-breakers for modern frames: set symbol, then OCR
        set_predictions: List[str] = []
        ocr_title = None
        if effective_frame == 'modern' and len(scored) > 1:
            margin = self._name_margin(scored)
            if margin < 0.05 and self.set_clf is not None:
                t0 = time.perf_counter()
                set_predictions = self._classify_set(card_oriented, top_k=10)
                timings['set_ms'] = (time.perf_counter() - t0) * 1000
                if set_predictions:
                    ss = set(set_predictions)
                    for row in scored:
                        if self.cards[int(self.db_indices[row[1]])].get('set') in ss:
                            row[0] += 0.05
                    scored.sort(key=lambda t: -t[0])
            margin = self._name_margin(scored)
            if use_ocr and margin < 0.05 and self.title_ocr is not None:
                t0 = time.perf_counter()
                ocr_cands, ocr_title = self.title_ocr.candidates(card_oriented, top_k=20)
                timings['ocr_ms'] = (time.perf_counter() - t0) * 1000
                if ocr_cands:
                    oc = set(ocr_cands)
                    for row in scored:
                        if int(self.db_indices[row[1]]) in oc:
                            row[0] += 0.10
                    scored.sort(key=lambda t: -t[0])

        top = []
        seen = set()
        for total, pos, rot, h_i, v in scored:
            ci = int(self.db_indices[pos])
            if ci in seen:
                continue
            seen.add(ci)
            top.append((ci, float(total), int(hyps[h_i]['dist'][pos])))
            if len(top) >= top_k:
                break
        return {
            'top_k': top,
            'confidence': float(scored[0][0]),
            'margin': float(self._name_margin(scored)),
            'rotate180': scored[0][2] == 180,
            'predicted_frame': frame_class,
            'frame_confidence': frame_conf,
            'effective_frame': effective_frame,
            'region_label': self._route(effective_frame)[-1],
            'pipeline_route': 'fast_hash' if decisive else ('verified' if query is not None else 'rerank'),
            'verification': scored[0][4],
            'glare_fraction': float((glare > 0).mean()),
            'ocr_title': ocr_title,
            'set_predictions': set_predictions,
            'timings': dict(timings),
            'card_image': card_oriented,
            'region_image': top_h['region'],
        }

    @staticmethod
    def match_quality(res: dict) -> str:
        if res['pipeline_route'] == 'fast_hash':
            return 'strong'
        v = (res.get('verification') or {}).get('verify')
        m = res['margin']
        if v is None:  # verification disabled: margin is all we have
            return 'good' if m >= 0.10 else 'weak' if m >= 0.05 else 'unknown'
        if v >= QUALITY_STRONG and m >= 0.03:
            return 'strong'
        if v >= QUALITY_GOOD[0] and m >= QUALITY_GOOD[1]:
            return 'good'
        if v >= QUALITY_WEAK[0] and m >= QUALITY_WEAK[1]:
            return 'weak'
        return 'unknown'

    def _name_margin(self, scored) -> float:
        """Score gap between the top candidate and the best candidate with a
        *different name* (reprints of the same card aren't competition)."""
        top_name = self.db_names[scored[0][1]]
        for row in scored[1:]:
            if self.db_names[row[1]] != top_name:
                return scored[0][0] - row[0]
        return scored[0][0]

    def identify_quad(self, rgb: np.ndarray, cq: CardQuad, top_k: int = 5,
                      use_ocr: bool = True) -> dict:
        """Try the outline variants of one detection; keep the best match.
        Stops early once a variant matches convincingly."""
        best = None
        for label, corners in quad_variants(cq):
            card = warp_quad(rgb, corners)
            res = self.identify_card_image(card, top_k=top_k, use_ocr=use_ocr)
            res['variant'] = label
            res['corners'] = np.roll(corners, 2, axis=0) if res['rotate180'] else corners
            res['detection_score'] = cq.score
            res['detection_source'] = cq.source
            res['quality'] = self.match_quality(res)
            if best is None or res['confidence'] > best['confidence']:
                best = res
            if best['quality'] in ('strong', 'good') and best['margin'] >= 0.08:
                break
        return best

    # -- public API ----------------------------------------------------------

    @staticmethod
    def load_image(image_path) -> np.ndarray:
        return np.array(ImageOps.exif_transpose(Image.open(image_path)).convert('RGB'))

    def identify_all(self, image, top_k: int = 5, use_ocr: bool = True,
                     max_cards: Optional[int] = None) -> List[dict]:
        """Every confidently identified card in the photo, best first."""
        rgb = self.load_image(image) if not isinstance(image, np.ndarray) else image
        t0 = time.perf_counter()
        quads = self.detector.detect(rgb, max_cards=max_cards)
        detect_ms = (time.perf_counter() - t0) * 1000
        results = []
        for cq in quads:
            res = self.identify_quad(rgb, cq, top_k=top_k, use_ocr=use_ocr)
            res['timings']['detect_ms'] = detect_ms / max(len(quads), 1)
            res['timings']['total_ms'] = sum(res['timings'].values())
            res['detection_route'] = cq.source
            if res['quality'] in ACCEPTED_QUALITIES:
                results.append(res)
        return sorted(results, key=lambda r: -r['confidence'])

    def identify(self, image_path, top_k: int = 5, hash_candidates: int = 500,
                 use_ocr: bool = True, max_detections: int = 3):
        """
        Single-card API (backwards compatible): the most confident card in
        the photo. The top few detections are all identified and the best
        *match* wins, so a card-shaped distraction that happens to score
        highest as a detection doesn't hijack the result.
        """
        rgb = self.load_image(image_path) if not isinstance(image_path, np.ndarray) else image_path
        t0 = time.perf_counter()
        quads = self.detector.detect(rgb)
        detect_ms = (time.perf_counter() - t0) * 1000
        # Prefer large, central detections for single-card photos
        H, W = rgb.shape[:2]

        def prior(q):
            c = q.corners.mean(axis=0)
            off = np.hypot((c[0] - W / 2) / W, (c[1] - H / 2) / H)
            return q.score + 0.3 * np.sqrt(q.area / (H * W)) - 0.3 * off
        quads = sorted(quads, key=lambda q: -prior(q))[:max_detections]
        best = None
        for cq in quads:
            res = self.identify_quad(rgb, cq, top_k=top_k, use_ocr=use_ocr)
            res['detection_route'] = cq.source
            if best is None or res['confidence'] > best['confidence']:
                best = res
        best['timings']['detect_ms'] = detect_ms
        best['timings']['total_ms'] = sum(v for k, v in best['timings'].items() if k != 'total_ms')
        best['n_cards_detected'] = len(quads)
        return best


# ---------------------------------------------------------------------------
# Display

def _describe(result, cards) -> List[str]:
    top = result['top_k']
    c = cards[top[0][0]]
    lines = [f"PREDICTION: {c['name']}"]
    if 'set_name' in c:
        lines.append(f"  Set:           {c.get('set_name', '?')} ({c.get('set', '?').upper()})")
    if 'collector_number' in c:
        lines.append(f"  Collector:     #{c.get('collector_number', '?')}")
    if 'type_line' in c:
        lines.append(f"  Type:          {c.get('type_line', '?')}")
    if c.get('mana_cost'):
        lines.append(f"  Mana cost:     {c.get('mana_cost')}")
    quality = result.get('quality', '?')
    note = {'strong': '', 'good': '', 'weak': '  (check it)',
            'unknown': '  (probably not in the database, or not a card)'}.get(quality, '')
    lines.append(f"  Match:         {quality}{note}")
    lines.append(f"  Confidence:    {result['confidence']:.3f}  (margin {result['margin']:.3f})")
    lines.append(f"  Frame class:   {result['predicted_frame']} ({result['frame_confidence']:.2f})"
                 f" -> route via {result['region_label']}")
    lines.append(f"  Detection:     {result.get('detection_route', '?')} / outline {result.get('variant', '?')}"
                 f"{'  (card upside down)' if result.get('rotate180') else ''}")
    lines.append(f"  Pipeline:      {result['pipeline_route']}")
    if result.get('glare_fraction', 0) > 0.01:
        lines.append(f"  Glare:         {result['glare_fraction'] * 100:.0f}% of card inpainted")
    v = result.get('verification') or {}
    if v:
        lines.append(f"  Verification:  {v.get('art_inliers', 0)} art / {v.get('inliers', 0)} total "
                     f"keypoint matches, correlation {v.get('corr', 0):.2f}")
    if result['set_predictions']:
        lines.append(f"  Set predict:   {', '.join(result['set_predictions'][:5])}")
    if result.get('ocr_title'):
        lines.append(f"  OCR title:     '{result['ocr_title']}'")
    return lines


def display_result(image_path, result, cards, show: bool = True):
    top = result['top_k']
    timings = result['timings']
    print()
    print("=" * 78)
    for line in _describe(result, cards):
        print(line)
    print("=" * 78)
    print(f"Top-{len(top)}:")
    for rank, (idx, score, hash_d) in enumerate(top, 1):
        c = cards[idx]
        print(f"  {rank}. {c['name']:<40} {c.get('set', ''):>6}  score={score:.4f}  hash_d={hash_d}")
    print()
    print("Timing:")
    for k, v in timings.items():
        print(f"  {k:<14} {v:>7.1f}ms")
    print("=" * 78)
    if not show:
        return

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    query_img = MultiRegionIdentifier.load_image(image_path)
    q = CardQuad(result['corners'], result['confidence'], result.get('detection_route', ''))
    axes[0, 0].imshow(draw_quads(query_img, [q], labels=[cards[top[0][0]]['name']]))
    axes[0, 0].set_title("Query image", fontsize=10)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(result['card_image'])
    axes[0, 1].set_title(f"Rectified ({result.get('variant', '')})", fontsize=10)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(result['region_image'])
    axes[0, 2].set_title(f"{result['region_label']} ({result['effective_frame']})", fontsize=10)
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
                    f"score={score:.3f}  hash_d={hash_d}",
                    fontsize=9, color=color,
                )
            else:
                axes[1, i].text(0.5, 0.5, "image\nnot found", ha='center', va='center')
                axes[1, i].set_title(f"#{i+1}: {cards[idx]['name'][:30]}", fontsize=9)
        axes[1, i].axis('off')

    plt.suptitle(f"Predicted: {cards[top[0][0]]['name']}  ({timings.get('total_ms', 0):.0f}ms)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def display_all(image_path, results, cards, show: bool = True, output_path: Optional[Path] = None):
    print()
    print("=" * 78)
    print(f"{len(results)} card(s) identified in {Path(str(image_path)).name}")
    print("=" * 78)
    for i, r in enumerate(results, 1):
        c = cards[r['top_k'][0][0]]
        print(f"  {i:>2}. {c['name']:<40} {c.get('set', '').upper():>6}  "
              f"{r['quality']:<7} conf={r['confidence']:.3f}  margin={r['margin']:.3f}")
    if not (show or output_path):
        return
    rgb = MultiRegionIdentifier.load_image(image_path)
    quads = [CardQuad(r['corners'], r['confidence'], '') for r in results]
    labels = [cards[r['top_k'][0][0]]['name'] for r in results]
    vis = draw_quads(rgb, quads, labels=labels)
    fig = plt.figure(figsize=(12, 9))
    plt.imshow(vis)
    plt.axis('off')
    plt.title(f"{len(results)} card(s)")
    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=120, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


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
    """Verify required artifacts exist before loading anything heavy."""
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


def _load_embeddings(path: Path) -> np.ndarray:
    emb = np.load(path).astype(np.float32)
    # Re-normalise: guards against float32 drift between training-time and
    # inference-time normalisation (similarities > 1.0)
    return emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)


def load_everything(verbose: bool = True, use_parallel: bool = True, verify: bool = True,
                    use_ocr: bool = True, use_yolo: bool = True, use_classical: bool = True):
    if verbose:
        print(f"Device: {DEVICE}")
        print("Loading metadata, hash DB, models, embeddings...")

    _check_artifacts()

    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        cards = json.load(f)

    db = dict(np.load(HASH_DB_PATH))
    required_keys = ['indices', 'art_phash', 'art_dhash', 'whole_phash', 'whole_dhash']
    missing = [k for k in required_keys if k not in db]
    if missing:
        print(f"ERROR: hash DB missing keys: {missing}")
        print("  This identifier needs the v2 hash DB. Re-run train_identifier.py")
        print("  (it rebuilds the DB automatically with the new schema).")
        sys.exit(1)

    if verbose:
        print(f"  Cards: {len(cards):,}")
        print(f"  Hash DB: {len(db['indices']):,} entries (art + whole)")

    frame_clf = FrameClassifier(n_classes=len(FRAME_CLASSES)).to(DEVICE)
    frame_clf.load_state_dict(torch.load(FRAME_CLF_PATH, map_location=DEVICE, weights_only=True))
    frame_clf.eval()
    if verbose:
        print(f"  Frame classifier loaded")

    art_reranker = MobileNetReranker(EMBEDDING_DIM).to(DEVICE)
    art_reranker.load_state_dict(torch.load(ART_RERANKER_PATH, map_location=DEVICE, weights_only=True))
    art_reranker.eval()
    art_emb = _load_embeddings(ART_EMB_PATH)
    whole_reranker = MobileNetReranker(EMBEDDING_DIM).to(DEVICE)
    whole_reranker.load_state_dict(torch.load(WHOLE_RERANKER_PATH, map_location=DEVICE, weights_only=True))
    whole_reranker.eval()
    whole_emb = _load_embeddings(WHOLE_EMB_PATH)
    for name, emb in (('art', art_emb), ('whole-card', whole_emb)):
        if len(emb) != len(db['indices']):
            print(f"ERROR: {name} embeddings ({len(emb)}) don't match the hash DB "
                  f"({len(db['indices'])}). Re-run: python train_identifier.py --skip-train")
            sys.exit(1)
    if verbose:
        print(f"  Re-rankers loaded (art {art_emb.shape}, whole {whole_emb.shape})")

    set_clf = None
    set_codes: List[str] = []
    if SET_CLF_PATH.exists() and SET_CODES_PATH.exists():
        with open(SET_CODES_PATH) as f:
            set_codes = json.load(f)
        set_clf = SetSymbolClassifier(n_classes=len(set_codes)).to(DEVICE)
        set_clf.load_state_dict(torch.load(SET_CLF_PATH, map_location=DEVICE, weights_only=True))
        set_clf.eval()
        if verbose:
            print(f"  Set classifier loaded ({len(set_codes)} sets)")
    elif verbose:
        print(f"  Set classifier not found - tie-break will use OCR only")

    detector = CardDetector(weights_path=YOLO_BEST, verbose=verbose, use_yolo=use_yolo,
                            use_classical=use_classical)

    title_ocr = None
    if use_ocr:
        try:
            if verbose:
                print("Loading OCR (this can take 10-20s on first run)...")
            title_ocr = TitleOCR([c['name'] for c in cards])
            if verbose:
                print("  OCR ready")
        except Exception as e:
            if verbose:
                print(f"  OCR unavailable ({e}). Pipeline will skip OCR confirmation.")

    pipeline = MultiRegionIdentifier(
        cards, db, frame_clf, art_reranker, art_emb,
        whole_reranker, whole_emb, set_clf, set_codes,
        detector, title_ocr, use_parallel=use_parallel, verify=verify,
    )
    return cards, pipeline


def save_result_figure(image_path, result, cards, output_path):
    """Save a comparison figure (original, rectified, predicted) to output_path."""
    top = result['top_k']
    top1_card = cards[top[0][0]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    query_img = MultiRegionIdentifier.load_image(image_path)
    q = CardQuad(result['corners'], result['confidence'], '')
    axes[0].imshow(draw_quads(query_img, [q], labels=[top1_card['name']]))
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(result['card_image'])
    axes[1].set_title(f"Rectified ({result.get('detection_route', '')}, {result.get('variant', '')})",
                      fontsize=11)
    axes[1].axis('off')

    match_path = IMAGE_DIR / f"{top1_card['id']}.jpg"
    if match_path.exists():
        axes[2].imshow(Image.open(match_path))
    else:
        axes[2].text(0.5, 0.5, "image\nnot found", ha='center', va='center')
    axes[2].set_title(f"Predicted: {top1_card['name'][:35]}\n"
                      f"confidence={result['confidence']:.3f} margin={result['margin']:.3f}", fontsize=10)
    axes[2].axis('off')

    plt.suptitle(f"{Path(image_path).name}  →  {top1_card['name']}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_one(pipeline, cards, image_path: str, use_ocr: bool = True, all_cards: bool = False,
            show: bool = True):
    if not Path(image_path).exists():
        print(f"File not found: {image_path}")
        return False
    try:
        if all_cards:
            results = pipeline.identify_all(image_path, top_k=5, use_ocr=use_ocr)
            display_all(image_path, results, cards, show=show)
        else:
            result = pipeline.identify(image_path, top_k=5, use_ocr=use_ocr)
            display_result(image_path, result, cards, show=show)
    except Exception as e:
        print(f"Error during identification: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True


def run_batch(pipeline, cards, input_dir: Path, output_dir: Path, use_ocr: bool = True,
              all_cards: bool = False):
    """Process all images in input_dir, save result figures to output_dir."""
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in extensions
    )
    if not image_files:
        print(f"No images found in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nProcessing {len(image_files)} images from {input_dir}")
    print(f"Saving results to {output_dir}\n")

    summary = []
    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] {img_path.name} ... ", end='', flush=True)
        try:
            out_name = f"{img_path.stem}_result.png"
            if all_cards:
                results = pipeline.identify_all(str(img_path), top_k=5, use_ocr=use_ocr)
                names = [cards[r['top_k'][0][0]]['name'] for r in results]
                display_all(str(img_path), results, cards, show=False, output_path=output_dir / out_name)
                print(f"→ {len(names)} card(s): {', '.join(names)}")
                summary.append({'file': img_path.name, 'cards': names})
            else:
                result = pipeline.identify(str(img_path), top_k=5, use_ocr=use_ocr)
                top1_name = cards[result['top_k'][0][0]]['name']
                save_result_figure(str(img_path), result, cards, output_dir / out_name)
                print(f"→ {top1_name}  ({result['quality']}, conf {result['confidence']:.2f})")
                summary.append({'file': img_path.name, 'card': top1_name,
                                'quality': result['quality'],
                                'confidence': round(result['confidence'], 4),
                                'margin': round(result['margin'], 4)})
        except Exception as e:
            print(f"ERROR: {e}")

    (output_dir / 'results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"\nDone! {len(image_files)} results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Identify MTG cards in a photo")
    parser.add_argument('image', nargs='?', help="Path to a card image (optional)")
    parser.add_argument('--batch', type=str, default=None,
                        help="Process all images in this folder")
    parser.add_argument('--output', type=str, default=None,
                        help="Output folder for batch results (default: <batch>_results)")
    parser.add_argument('--all', action='store_true',
                        help="Identify every card in the photo, not just the main one")
    parser.add_argument('--no-gui', action='store_true',
                        help="Force stdin input instead of file picker")
    parser.add_argument('--once', action='store_true',
                        help="Identify one image and exit")
    parser.add_argument('--no-ocr', action='store_true',
                        help="Skip OCR even when the match is a close call")
    parser.add_argument('--no-parallel', action='store_true',
                        help="Trust the frame classifier even when it's unsure")
    parser.add_argument('--fast', action='store_true',
                        help="Skip local-feature verification (faster, less robust to glare)")
    parser.add_argument('--no-yolo', action='store_true',
                        help="Use only the classical card detector")
    parser.add_argument('--yolo-only', action='store_true',
                        help="Skip the classical detector (faster; needs yolo_card_best.pt)")
    args = parser.parse_args()

    use_ocr = not args.no_ocr
    batch = args.batch
    if not batch and not args.image and DEFAULT_BATCH_FOLDER.is_dir():
        batch = str(DEFAULT_BATCH_FOLDER)
        use_ocr = False

    cards, pipeline = load_everything(use_parallel=not args.no_parallel, verify=not args.fast,
                                      use_ocr=use_ocr, use_yolo=not args.no_yolo,
                                      use_classical=not args.yolo_only)

    if batch:
        input_dir = Path(batch)
        if not input_dir.is_dir():
            print(f"ERROR: {input_dir} is not a directory")
            sys.exit(1)
        output_dir = Path(args.output) if args.output else input_dir.parent / f"{input_dir.name}_results"
        run_batch(pipeline, cards, input_dir, output_dir, use_ocr=use_ocr, all_cards=args.all)
        return

    if args.image:
        run_one(pipeline, cards, args.image, use_ocr=use_ocr, all_cards=args.all)
        return

    print()
    print("Pick a card image to identify (close the dialog or send EOF to quit).")
    print()
    while True:
        path = pick_file_stdin() if args.no_gui else (pick_file_gui() or pick_file_stdin())
        if not path:
            print("Goodbye.")
            break
        if not run_one(pipeline, cards, path, use_ocr=use_ocr, all_cards=args.all):
            continue
        if args.once:
            break
        print()


if __name__ == '__main__':
    main()
