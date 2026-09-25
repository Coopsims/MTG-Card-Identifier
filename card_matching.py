"""
Glare-aware matching of a rectified card photo against reference scans.

The hash + embedding retrieval in identify_card.py compares whole regions,
so anything that changes a large patch of pixels - a glare spot on a sleeve,
a shadow, a finger - moves the query away from its reference. This module
adds signals that degrade gracefully under those conditions:

  * glare_mask() / remove_glare(): find specular highlights (bright,
    desaturated, brighter than their surroundings) and inpaint them before
    hashing / embedding, so a white blob doesn't dominate the descriptor.
  * normalize_lighting(): gray-world white balance + local contrast
    normalisation to cancel colour casts from warm indoor lighting.
  * LocalFeatureVerifier: ORB keypoints matched between the query and a
    candidate reference with RANSAC. Glare and occlusions only remove some
    keypoints; the rest still vote for the right card. Because both images
    are rectified cards, true matches must agree with a near-identity
    homography, which makes the inlier count extremely discriminative.
  * masked_correlation(): zero-mean normalised cross-correlation of
    illumination-normalised images, ignoring glare pixels.

Everything takes RGB uint8 cards at the canonical 488x680 size.
"""
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np

from mtg_layout import CARD_W, CARD_H


# ---------------------------------------------------------------------------
# Glare

def glare_mask(card_rgb: np.ndarray, min_blob_frac: float = 0.0004,
               dilate_px: int = 5) -> np.ndarray:
    """
    Binary mask (uint8 0/255) of specular highlights on a rectified card.

    Hysteresis, like Canny: glare almost always has a clipped core (nearly
    pure white, no colour), so those pixels seed the mask, which then grows
    into the connected halo of bright, weakly saturated pixels that stand out
    from their surroundings. Pale regions without a clipped core - white
    borders, text boxes, snowy art - are left alone even though they are
    bright and unsaturated; inpainting them would erase real content.
    """
    hsv = cv2.cvtColor(card_rgb, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    h, w = v.shape
    seeds = (v >= 248) & (s < 50)
    if not seeds.any():
        return np.zeros((h, w), np.uint8)

    # Local surface brightness, robust to thin dark strokes (text): close
    # small dark gaps first, then open away bright structures smaller than
    # the kernel (glare spots / streaks).
    k = max(15, (min(h, w) // 6) | 1)
    surface = cv2.morphologyEx(v, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    surface = cv2.morphologyEx(surface, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    surface = cv2.blur(surface, (k, k))
    tophat = v.astype(np.int16) - surface.astype(np.int16)
    halo = (v >= 190) & (s < 80) & (tophat > 20)
    # A halo only extends a limited distance from its core; beyond that a
    # bright region is more likely pale card content next to the glare.
    r = max(6, int(0.04 * min(h, w)))
    near = cv2.dilate(seeds.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    candidates = ((halo & near) | seeds).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
    seeded = np.zeros(n, bool)
    seeded[np.unique(labels[seeds])] = True
    seeded[0] = False
    min_px = max(4, int(min_blob_frac * h * w))
    seeded &= stats[:, cv2.CC_STAT_AREA] >= min_px
    mask = (seeded[labels] * 255).astype(np.uint8)
    if dilate_px > 0 and mask.any():
        mask = cv2.dilate(mask, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)))
    return mask


def remove_glare(card_rgb: np.ndarray, mask: Optional[np.ndarray] = None,
                 max_frac: float = 0.35) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inpaint specular highlights. Returns (clean_image, mask), where mask is
    the full glare mask (core + halo) for the verifier to ignore.

    Only the washed-out core is inpainted: there the underlying card is
    gone, and a flat white blob drags pHash / embeddings away from the
    reference. The halo still shows the card through a veil, so it's kept
    (inpainting would replace real content with guesses) and merely masked
    out of the verification. If the core covers more than max_frac of the
    card, only its clipped centre is filled - smearing half a card is worse
    than leaving it.
    """
    if mask is None:
        mask = glare_mask(card_rgb)
    if not mask.any():
        return card_rgb, mask
    hsv = cv2.cvtColor(card_rgb, cv2.COLOR_RGB2HSV)
    washed = (hsv[:, :, 2] >= 240) & (hsv[:, :, 1] < 35) & (mask > 0)
    fill = cv2.dilate(washed.astype(np.uint8) * 255,
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) & mask
    if (fill > 0).mean() > max_frac:
        fill = ((hsv[:, :, 2] >= 250) & (mask > 0)).astype(np.uint8) * 255
    if not fill.any():
        return card_rgb, mask
    # Inpaint at half resolution for speed, then paste back only the filled
    # pixels so everything else keeps full resolution.
    small = cv2.resize(card_rgb, (card_rgb.shape[1] // 2, card_rgb.shape[0] // 2),
                       interpolation=cv2.INTER_AREA)
    small_fill = cv2.resize(fill, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    filled = cv2.inpaint(small, small_fill, 5, cv2.INPAINT_TELEA)
    filled = cv2.resize(filled, (card_rgb.shape[1], card_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    out = card_rgb.copy()
    out[fill > 0] = filled[fill > 0]
    return out, mask


# ---------------------------------------------------------------------------
# Lighting normalisation

def gray_world(rgb: np.ndarray) -> np.ndarray:
    """Remove colour casts (warm bulbs, blue daylight) by equalising the
    mean of each channel. Clipped so a mostly-red card isn't neutralised."""
    f = rgb.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0) + 1e-3
    gain = np.clip(means.mean() / means, 0.75, 1.33)
    return np.clip(f * gain, 0, 255).astype(np.uint8)


def normalize_lighting(rgb: np.ndarray, clip: float = 2.0) -> np.ndarray:
    """Gray-world white balance + CLAHE on luminance."""
    wb = gray_world(rgb)
    lab = cv2.cvtColor(wb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(4, 4))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _local_normalize(gray: np.ndarray, sigma: float) -> np.ndarray:
    g = gray.astype(np.float32)
    mu = cv2.GaussianBlur(g, (0, 0), sigma)
    d = g - mu
    sd = np.sqrt(cv2.GaussianBlur(d * d, (0, 0), sigma)) + 4.0
    return d / sd


# ---------------------------------------------------------------------------
# Masked correlation

CORR_SIZE = (CARD_W // 4, CARD_H // 4)  # 122 x 170


def correlation_features(card_rgb: np.ndarray) -> np.ndarray:
    """Illumination-invariant representation used by masked_correlation:
    locally normalised luminance + chroma at quarter resolution."""
    small = cv2.resize(card_rgb, CORR_SIZE, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
    L = _local_normalize(lab[:, :, 0], 3.0)
    a = _local_normalize(lab[:, :, 1], 6.0) * 0.5
    b = _local_normalize(lab[:, :, 2], 6.0) * 0.5
    return np.stack([L, a, b], axis=2)


def masked_correlation(q_feat: np.ndarray, r_feat: np.ndarray,
                       valid: Optional[np.ndarray] = None, max_shift: int = 2) -> float:
    """Normalised correlation between two correlation_features() maps over
    valid (non-glare) pixels, maximised over small shifts to absorb residual
    rectification error. The features are locally zero-mean already, so the
    per-shift mean subtraction of full ZNCC is skipped."""
    h, w = q_feat.shape[:2]
    if valid is None:
        valid = np.ones((h, w), bool)
    m = max_shift
    vc = valid[m:h - m, m:w - m].astype(np.float32)[..., None]
    if vc.mean() < 0.2:
        return 0.0
    qc = q_feat[m:h - m, m:w - m] * vc
    qn = float(np.sqrt((qc * qc).sum())) + 1e-6
    best = -1.0
    for dy in (-m, 0, m):
        for dx in (-m, 0, m):
            rc = r_feat[m + dy:h - m + dy, m + dx:w - m + dx]
            num = float((qc * rc).sum())
            rn = float(np.sqrt((rc * rc * vc).sum())) + 1e-6
            best = max(best, num / (qn * rn))
    return best


THUMB_SIZE = (CARD_W // 16, CARD_H // 16)  # 30 x 42


def thumbnail_vector(corr_feat: np.ndarray) -> np.ndarray:
    """Flattened low-res correlation features for fast whole-database scans."""
    return cv2.resize(corr_feat, THUMB_SIZE, interpolation=cv2.INTER_AREA).reshape(-1).astype(np.float32)


def masked_cosine_scan(q_vec: np.ndarray, q_valid: np.ndarray, db: np.ndarray,
                       db_sq: np.ndarray) -> np.ndarray:
    """Cosine similarity of a query thumbnail against every row of db,
    restricted to the query's valid (non-glare) entries. db_sq = db**2 is
    precomputed so the per-row masked norms are one matrix product."""
    m = q_valid.astype(np.float32)
    qm = q_vec * m
    num = db @ qm
    den = np.sqrt(db_sq @ m) * (np.linalg.norm(qm) + 1e-6) + 1e-6
    return num / den


# ---------------------------------------------------------------------------
# Local features

# Region that holds the illustration on nearly every frame style (1993
# through modern); full-art cards are all illustration anyway.
ART_ZONE = ((0.06, 0.94), (0.09, 0.58))


class LocalFeatureVerifier:
    """
    ORB + RANSAC verification of a rectified query against candidate
    references. Reference features are computed lazily and cached, so the
    full 90k-card database never needs precomputed keypoints.

    load_reference: callable(card_index) -> RGB image (any size); it's
    resized to the canonical card size internally.
    """

    def __init__(self, load_reference: Callable[[int], Optional[np.ndarray]],
                 n_features: int = 1500, cache_size: int = 2048):
        self.load_reference = load_reference
        self.orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=1.2, nlevels=6,
                                  edgeThreshold=15, patchSize=31, fastThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.cache: 'OrderedDict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]' = OrderedDict()
        self.cache_size = cache_size
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))

    def _prep(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.shape[0] != CARD_H or rgb.shape[1] != CARD_W:
            rgb = cv2.resize(rgb, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)
        return self.clahe.apply(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))

    def features(self, rgb: np.ndarray, mask: Optional[np.ndarray] = None):
        gray = self._prep(rgb)
        m = None
        if mask is not None:
            m = cv2.bitwise_not(mask)
            if m.shape != gray.shape:
                m = cv2.resize(m, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        kps, desc = self.orb.detectAndCompute(gray, m)
        pts = np.array([k.pt for k in kps], np.float32).reshape(-1, 2)
        return pts, desc, gray

    def reference_features(self, idx: int):
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]
        img = self.load_reference(idx)
        if img is None:
            feats = (np.zeros((0, 2), np.float32), None, None)
        else:
            pts, desc, _ = self.features(img)
            feats = (pts, desc, None)  # don't keep the image: ~50 KB per entry instead of ~400
        self.cache[idx] = feats
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return feats

    def verify(self, q_feats, idx: int, rotate180: bool = False) -> Dict[str, float]:
        """
        Returns {'inliers', 'ratio'} for query features vs reference idx.
        rotate180 treats the query as upside down (keypoint coordinates are
        flipped; ORB descriptors are rotation invariant so they're reused).
        """
        q_pts, q_desc, _ = q_feats
        r_pts, r_desc, _ = self.reference_features(idx)
        if q_desc is None or r_desc is None or len(q_pts) < 8 or len(r_pts) < 8:
            return {'inliers': 0, 'art_inliers': 0, 'ratio': 0.0}
        if rotate180:
            q_pts = np.stack([CARD_W - 1 - q_pts[:, 0], CARD_H - 1 - q_pts[:, 1]], axis=1)
        knn = self.matcher.knnMatch(q_desc, r_desc, k=2)
        src, dst = [], []
        max_dist = 0.12 * CARD_W  # rectified cards: matches must be near-aligned
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance > 0.85 * n.distance:
                continue
            p, r = q_pts[m.queryIdx], r_pts[m.trainIdx]
            if abs(p[0] - r[0]) > max_dist or abs(p[1] - r[1]) > max_dist:
                continue
            src.append(p)
            dst.append(r)
        if len(src) < 6:
            return {'inliers': 0, 'art_inliers': 0, 'ratio': 0.0}
        src = np.array(src, np.float32)
        dst = np.array(dst, np.float32)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 6.0, maxIters=500, confidence=0.99)
        if H is None or inl is None:
            return {'inliers': 0, 'art_inliers': 0, 'ratio': 0.0}
        inl = inl.ravel().astype(bool)
        n_inl = int(inl.sum())
        # Frames and text boxes are shared by thousands of cards (every
        # basic land of a set has the same frame), so inliers inside the
        # illustration are what actually tell two cards apart.
        (x0, x1), (y0, y1) = ART_ZONE
        in_art = ((dst[:, 0] >= x0 * CARD_W) & (dst[:, 0] <= x1 * CARD_W)
                  & (dst[:, 1] >= y0 * CARD_H) & (dst[:, 1] <= y1 * CARD_H))
        denom = max(20, min(len(q_pts), len(r_pts)))
        return {'inliers': n_inl, 'art_inliers': int((inl & in_art).sum()),
                'ratio': float(n_inl / denom)}


# ---------------------------------------------------------------------------
# Quad variants
#
# The detected outline isn't always the card's outer edge: white-bordered
# cards on a white table often yield the coloured inner frame, and a card in
# a toploader can yield the toploader. The identifier scores a few variants
# of each detection and keeps whichever matches best.

# Border thickness of a physical card is ~2.5-3.5 mm on a 63 x 88 mm card.
_BORDER_EXPANSIONS = ((1 + 2 * 2.6 / 63, 1 + 2 * 2.6 / 88),
                      (1 + 2 * 4.0 / 63, 1 + 2 * 4.0 / 88))


def expand_quad(corners: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Scale a card quad about its centre *in the card's own frame* (so the
    expansion follows the perspective)."""
    rect = np.array([[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]], np.float32)
    H = cv2.getPerspectiveTransform(rect, corners.astype(np.float32))
    cx, cy = (CARD_W - 1) / 2, (CARD_H - 1) / 2
    ex = np.stack([cx + (rect[:, 0] - cx) * sx, cy + (rect[:, 1] - cy) * sy], axis=1)
    return cv2.perspectiveTransform(ex.reshape(1, -1, 2), H).reshape(-1, 2).astype(np.float32)


def quad_variants(card_quad, include_expansions: bool = True, max_alternatives: int = 2):
    """Yield (label, corners) outlines to try for one detected card, most
    likely first."""
    yield 'primary', card_quad.corners
    for i, alt in enumerate(card_quad.alternatives[:max_alternatives]):
        yield f'alt{i}', alt.corners
    if include_expansions:
        for i, (sx, sy) in enumerate(_BORDER_EXPANSIONS):
            yield f'expand{i}', expand_quad(card_quad.corners, sx, sy)


# ---------------------------------------------------------------------------
# Candidate verification + score fusion

class CandidateVerifier:
    """
    Re-scores retrieval candidates for a rectified query with signals that
    survive glare and occlusion. Reference images are loaded lazily through
    load_reference(card_index) and their features cached.
    """

    def __init__(self, load_reference: Callable[[int], Optional[np.ndarray]],
                 cache_size: int = 1024):
        # ~50 KB of keypoints + ~125 KB of correlation features per cached
        # card, so the default costs ~180 MB at most.
        self.load_reference = load_reference
        self.local = LocalFeatureVerifier(load_reference, cache_size=cache_size)
        self._corr_cache: 'OrderedDict[int, Optional[np.ndarray]]' = OrderedDict()
        self.cache_size = cache_size

    def _ref_corr(self, idx: int) -> Optional[np.ndarray]:
        if idx in self._corr_cache:
            self._corr_cache.move_to_end(idx)
            return self._corr_cache[idx]
        img = self.load_reference(idx)
        feat = None
        if img is not None:
            if img.shape[0] != CARD_H or img.shape[1] != CARD_W:
                img = cv2.resize(img, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)
            feat = correlation_features(img).astype(np.float16)  # halves the cache
        self._corr_cache[idx] = feat
        if len(self._corr_cache) > self.cache_size:
            self._corr_cache.popitem(last=False)
        return feat

    def prepare_query(self, card_rgb: np.ndarray, glare: Optional[np.ndarray] = None) -> dict:
        """Compute query-side features once; reused for every candidate
        and for both orientations."""
        if glare is None:
            glare = glare_mask(card_rgb)
        feats = self.local.features(card_rgb, mask=glare)
        corr = correlation_features(card_rgb)
        valid = cv2.resize(glare, CORR_SIZE, interpolation=cv2.INTER_NEAREST) == 0
        corr180 = corr[::-1, ::-1].copy()
        valid180 = valid[::-1, ::-1].copy()
        return {'orb': feats, 'corr': corr, 'valid': valid,
                'corr180': corr180, 'valid180': valid180,
                'glare_frac': float((glare > 0).mean())}

    def score(self, query: dict, idx: int, rotate180: bool = False) -> Dict[str, float]:
        v = self.local.verify(query['orb'], idx, rotate180=rotate180)
        rf = self._ref_corr(idx)
        corr = 0.0
        if rf is not None:
            rf = rf.astype(np.float32)
            if rotate180:
                corr = masked_correlation(query['corr180'], rf, query['valid180'])
            else:
                corr = masked_correlation(query['corr'], rf, query['valid'])
        # Saturating slowly: tens of art inliers is typical for a true match
        # on a phone photo, hundreds on a clean scan.
        art_score = 1.0 - np.exp(-v['art_inliers'] / 40.0)
        all_score = 1.0 - np.exp(-v['inliers'] / 120.0)
        orb_score = 0.75 * art_score + 0.25 * all_score
        fused = 0.6 * orb_score + 0.4 * max(corr, 0.0)
        return {'inliers': v['inliers'], 'art_inliers': v['art_inliers'], 'corr': corr,
                'orb_score': float(orb_score), 'verify': float(fused)}


def fuse(prior: float, verify: float, w_verify: float = 0.7) -> float:
    """Blend a retrieval prior (hash / embedding similarity mapped to [0,1])
    with the verification score."""
    return float(w_verify * verify + (1 - w_verify) * prior)
