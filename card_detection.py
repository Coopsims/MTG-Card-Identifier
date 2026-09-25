"""
Card localisation for real photos.

The YOLO-OBB detector in train_detector.py is trained on synthetic scenes and
returns a *rotated rectangle*. Real phone photos break it in three ways:

  1. Perspective. A card photographed at an angle is a general quadrilateral,
     not a rectangle, so a rotated-rectangle warp leaves the art crop skewed
     and the hashes miss.
  2. Orientation. The old corner ordering (sum / diff of coordinates) breaks
     for cards lying sideways or rotated near 45 degrees, and squashes
     landscape-oriented cards into a portrait canvas.
  3. Backgrounds. Busy play mats, wood grain, dark cloth and glare were never
     in the synthetic training distribution.

This module fixes all three with classical computer vision that needs no
training data:

  * find_card_quads() generates candidate quadrilaterals from several
    complementary binarisations (colour edges, adaptive threshold in both
    polarities, Otsu, distance-from-background), fits a true 4-sided
    polygon to each contour with robust line fits (so rounded corners and
    partial occlusion by fingers / dice don't pull the corners), scores
    each quad on card-likeness, and resolves nested / duplicate candidates.
  * refine_quad() snaps any rough quad - e.g. a YOLO box - onto the real
    card edges by searching for the strongest gradient along each side's
    normal and RANSAC-fitting a line per side.
  * order_corners_portrait() orders corners so the warp always produces a
    portrait card. The remaining 0/180 degree ambiguity is resolved by the
    identifier, which scores both orientations.

Everything works on RGB uint8 arrays (the rest of the pipeline uses RGB).
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import cv2
import numpy as np

from mtg_layout import CARD_W, CARD_H

# Physical MTG card is 63 x 88 mm
CARD_ASPECT = 63.0 / 88.0

# Working resolution for segmentation. Corners are mapped back to full
# resolution before warping so the rectified card keeps all its detail.
WORK_LONG_SIDE = 1024

# Plausible short/long side ratio of a card quad after perspective. A card
# viewed head-on is 0.716; steep camera angles push it either way.
MIN_QUAD_RATIO = 0.50
MAX_QUAD_RATIO = 0.92

# Smallest card we try to find, as a fraction of the working image's short
# side. Below this the card is too small to identify anyway.
MIN_CARD_SHORT_SIDE_FRAC = 0.06


@dataclass
class CardQuad:
    """A detected card: 4 corners in full-resolution image coordinates,
    ordered TL, TR, BR, BL in the card's own portrait frame."""
    corners: np.ndarray
    score: float
    source: str
    details: dict = field(default_factory=dict)
    # Other plausible outlines for the same physical card (inner frame,
    # a card inside a toploader / slab, ...). The identifier can try them
    # when the primary outline doesn't produce a confident match.
    alternatives: List['CardQuad'] = field(default_factory=list)

    def warp(self, image: np.ndarray, rotate180: bool = False,
             size=(CARD_W, CARD_H)) -> np.ndarray:
        return warp_quad(image, self.corners, rotate180=rotate180, size=size)

    @property
    def area(self) -> float:
        return float(cv2.contourArea(self.corners.astype(np.float32)))


# ---------------------------------------------------------------------------
# Geometry helpers

def order_corners_clockwise(pts: np.ndarray) -> np.ndarray:
    """Order 4 points clockwise (in image coordinates, y down), starting
    from the point closest to the top-left of the quad's bounding box."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]
    start = int(np.argmin(pts[:, 0] + pts[:, 1]))
    return np.roll(pts, -start, axis=0)


def order_corners_portrait(pts: np.ndarray) -> np.ndarray:
    """
    Order corners TL, TR, BR, BL such that TL->TR is a *short* side of the
    card, i.e. warping to a portrait canvas never squashes the card.

    Of the two portrait orderings (0 / 180 degrees apart) we pick the one
    whose top edge is higher in the image, since people mostly photograph
    cards roughly upright. The identifier still checks the flipped
    orientation, so a wrong guess here costs time, not accuracy.
    """
    q = order_corners_clockwise(pts)
    side = lambda i: float(np.linalg.norm(q[(i + 1) % 4] - q[i]))
    horiz = side(0) + side(2)
    vert = side(1) + side(3)
    if horiz > vert:
        q = np.roll(q, -1, axis=0)  # make the short sides top / bottom
    flipped = np.roll(q, 2, axis=0)
    top_y = (q[0, 1] + q[1, 1]) / 2
    flipped_top_y = (flipped[0, 1] + flipped[1, 1]) / 2
    return flipped if flipped_top_y < top_y - 1e-6 else q


def warp_quad(image: np.ndarray, corners: np.ndarray, rotate180: bool = False,
              size=(CARD_W, CARD_H)) -> np.ndarray:
    w, h = size
    src = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    if rotate180:
        src = np.roll(src, 2, axis=0)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    # INTER_AREA is not supported by warpPerspective; when shrinking a large
    # phone photo, pre-blur slightly to avoid aliasing in the hashes.
    scale = max(np.linalg.norm(src[1] - src[0]) / w,
                np.linalg.norm(src[3] - src[0]) / h)
    if scale > 2.0:
        k = int(scale) | 1
        image = cv2.GaussianBlur(image, (k, k), 0)
    return cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def quad_side_ratio(q: np.ndarray) -> float:
    """Short/long ratio from the mean lengths of opposite sides."""
    s = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    a = (s[0] + s[2]) / 2
    b = (s[1] + s[3]) / 2
    return float(min(a, b) / max(a, b, 1e-6))


def rectified_aspect(q: np.ndarray, img_shape, focal: Optional[float] = None) -> float:
    """
    Estimate the true short/long ratio of the physical rectangle whose
    perspective projection is quad q (clockwise corners), following Zhang &
    He, "Whiteboard scanning and image enhancement" (2007).

    A card seen at a steep angle can look almost square in the image; this
    undoes the perspective so we can still check it's 63x88 mm. The focal
    length is estimated from the quad when the perspective is strong enough
    to constrain it, otherwise a typical phone-camera focal length is used.
    """
    h, w = img_shape[:2]
    u0, v0 = w / 2.0, h / 2.0
    q = np.asarray(q, np.float64)
    m1 = np.array([q[0, 0], q[0, 1], 1.0])
    m2 = np.array([q[1, 0], q[1, 1], 1.0])
    m4 = np.array([q[2, 0], q[2, 1], 1.0])
    m3 = np.array([q[3, 0], q[3, 1], 1.0])
    try:
        k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
        k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)
    except FloatingPointError:
        return quad_side_ratio(q.astype(np.float32))
    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1

    def aspect_for(f):
        Ainv = np.array([[1 / f, 0, -u0 / f], [0, 1 / f, -v0 / f], [0, 0, 1]])
        B = Ainv.T @ Ainv
        num = float(n2 @ B @ n2)
        den = float(n3 @ B @ n3)
        if num <= 0 or den <= 0:
            return None
        r = np.sqrt(num / den)
        return float(min(r, 1 / r))

    long_side = max(w, h)
    if focal is not None:
        r = aspect_for(focal)
        return r if r is not None else quad_side_ratio(q.astype(np.float32))
    f2 = None
    if abs(n2[2] * n3[2]) > 1e-12:
        f2 = -((n2[0] * n3[0] - (n2[0] * n3[2] + n2[2] * n3[0]) * u0 + n2[2] * n3[2] * u0 * u0)
               + (n2[1] * n3[1] - (n2[1] * n3[2] + n2[2] * n3[1]) * v0 + n2[2] * n3[2] * v0 * v0)
               ) / (n2[2] * n3[2])
    if f2 is not None and f2 > 0 and 0.5 * long_side <= np.sqrt(f2) <= 3.0 * long_side:
        r = aspect_for(np.sqrt(f2))
        if r is not None:
            return r
    # Focal length not recoverable from this quad (perspective along one
    # axis only, or near-affine): take the most card-like reading over the
    # range of focal lengths real phone photos and crops of them have.
    best = None
    for fm in (0.6, 0.8, 1.0, 1.3, 1.8, 3.0):
        r = aspect_for(fm * long_side)
        if r is not None and (best is None or abs(r - CARD_ASPECT) < abs(best - CARD_ASPECT)):
            best = r
    return best if best is not None else quad_side_ratio(q.astype(np.float32))


def quad_angles_ok(q: np.ndarray, min_deg: float = 50, max_deg: float = 130) -> bool:
    for i in range(4):
        a = q[i - 1] - q[i]
        b = q[(i + 1) % 4] - q[i]
        cosang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        if not (min_deg <= ang <= max_deg):
            return False
    return True


def quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    if inter <= 0:
        return 0.0
    ua = cv2.contourArea(a.astype(np.float32)) + cv2.contourArea(b.astype(np.float32)) - inter
    return float(inter / max(ua, 1e-6))


def quad_containment(inner: np.ndarray, outer: np.ndarray) -> float:
    """Fraction of `inner`'s area that lies inside `outer`."""
    inter, _ = cv2.intersectConvexConvex(inner.astype(np.float32), outer.astype(np.float32))
    return float(inter / max(cv2.contourArea(inner.astype(np.float32)), 1e-6))


def _line_intersection(l1, l2) -> Optional[np.ndarray]:
    # Lines as (vx, vy, x0, y0)
    vx1, vy1, x1, y1 = l1
    vx2, vy2, x2, y2 = l2
    det = vx1 * (-vy2) - vy1 * (-vx2)
    if abs(det) < 1e-9:
        return None
    t = ((x2 - x1) * (-vy2) - (y2 - y1) * (-vx2)) / det
    return np.array([x1 + t * vx1, y1 + t * vy1], dtype=np.float32)


def _long_axis(q: np.ndarray) -> np.ndarray:
    """Unit vector along the quad's long axis."""
    s01 = (q[1] - q[0]) + (q[2] - q[3])
    s12 = (q[2] - q[1]) + (q[3] - q[0])
    v = s01 if np.linalg.norm(s01) > np.linalg.norm(s12) else s12
    return v / (np.linalg.norm(v) + 1e-9)


# ---------------------------------------------------------------------------
# Robust line fitting

def _ransac_line(points: np.ndarray, weights: Optional[np.ndarray] = None,
                 thresh: float = 1.5, iters: int = 60,
                 rng: Optional[np.random.Generator] = None):
    """Fit a line (vx, vy, x0, y0) robust to outliers (occlusions, glare)."""
    n = len(points)
    if n < 2:
        return None, np.zeros(n, dtype=bool)
    if rng is None:
        rng = np.random.default_rng(0)
    pts = points.astype(np.float32)
    w = np.ones(n, np.float32) if weights is None else weights.astype(np.float32)
    # Evaluate all hypotheses at once: iters x n distance matrix
    i = rng.integers(0, n, iters)
    j = rng.integers(0, n, iters)
    d = pts[j] - pts[i]
    nd = np.linalg.norm(d, axis=1)
    ok = nd > 1e-6
    if not ok.any():
        return None, np.zeros(n, dtype=bool)
    i, d, nd = i[ok], d[ok], nd[ok]
    nrm = np.stack([-d[:, 1], d[:, 0]], axis=1) / nd[:, None]
    dist = np.abs(((pts[None, :, :] - pts[i][:, None, :]) * nrm[:, None, :]).sum(axis=2))
    inl_all = dist < thresh
    best_inl = inl_all[int(np.argmax((inl_all * w[None, :]).sum(axis=1)))]
    if best_inl.sum() < 2:
        return None, np.zeros(n, dtype=bool)
    line = cv2.fitLine(points[best_inl].astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    return line.flatten(), best_inl


def fit_quad_to_points(points: np.ndarray, init_quad: np.ndarray,
                       iters: int = 3, corner_skip: float = 0.12) -> Optional[np.ndarray]:
    """
    Fit a quadrilateral to boundary points: assign each point to the nearest
    side of the current quad, robustly fit a line per side (ignoring the
    rounded-corner region near each end), intersect adjacent lines.
    Handles perspective because each side is fitted independently.
    """
    q = order_corners_clockwise(init_quad)
    pts = points.reshape(-1, 2).astype(np.float32)
    for _ in range(iters):
        lines = []
        # Distance of every point to every side (segment)
        dists = np.zeros((len(pts), 4), dtype=np.float32)
        ts = np.zeros((len(pts), 4), dtype=np.float32)
        for k in range(4):
            a, b = q[k], q[(k + 1) % 4]
            ab = b - a
            L2 = float(ab @ ab) + 1e-9
            t = np.clip(((pts - a) @ ab) / L2, 0, 1)
            proj = a + t[:, None] * ab
            dists[:, k] = np.linalg.norm(pts - proj, axis=1)
            ts[:, k] = t
        assign = np.argmin(dists, axis=1)
        for k in range(4):
            sel = (assign == k) & (ts[:, k] > corner_skip) & (ts[:, k] < 1 - corner_skip)
            side_pts = pts[sel]
            if len(side_pts) < 5:
                return None
            side_len = float(np.linalg.norm(q[(k + 1) % 4] - q[k]))
            line, _ = _ransac_line(side_pts, thresh=max(1.5, 0.01 * side_len))
            if line is None:
                return None
            lines.append(line)
        new_q = []
        for k in range(4):
            p = _line_intersection(lines[k - 1], lines[k])
            if p is None:
                return None
            new_q.append(p)
        new_q = np.array(new_q, dtype=np.float32)
        if not np.all(np.isfinite(new_q)):
            return None
        if np.abs(new_q - q).max() < 0.5:
            q = new_q
            break
        q = new_q
    return q


# ---------------------------------------------------------------------------
# Gradient-based refinement of a rough quad (e.g. from YOLO)

def _gradient_images(gray: np.ndarray):
    g = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy


def _color_gradient_images(rgb: np.ndarray):
    """Per-pixel gradient of the Lab channel with the strongest response,
    so a red card on a brown table still has an edge even if luminance
    matches."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = cv2.GaussianBlur(lab, (5, 5), 0)
    best_mag = None
    gx_out = gy_out = None
    for c, wgt in ((0, 1.0), (1, 1.6), (2, 1.6)):
        gx = cv2.Sobel(lab[:, :, c], cv2.CV_32F, 1, 0, ksize=3) * wgt
        gy = cv2.Sobel(lab[:, :, c], cv2.CV_32F, 0, 1, ksize=3) * wgt
        mag = gx * gx + gy * gy
        if best_mag is None:
            best_mag, gx_out, gy_out = mag, gx, gy
        else:
            m = mag > best_mag
            best_mag = np.where(m, mag, best_mag)
            gx_out = np.where(m, gx, gx_out)
            gy_out = np.where(m, gy, gy_out)
    return gx_out, gy_out


def _refine_sides_once(q: np.ndarray, gx: np.ndarray, gy: np.ndarray, search_frac: float,
                       n_samples: int, rng: np.random.Generator):
    """One refinement pass: a line per side, or None where no edge was found."""
    h, w = gx.shape
    center = q.mean(axis=0)
    lines = []
    for k in range(4):
        a, b = q[k], q[(k + 1) % 4]
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 8:
            return None
        tvec = d / L
        nvec = np.array([-tvec[1], tvec[0]], dtype=np.float32)
        if np.dot((a + b) / 2 - center, nvec) < 0:
            nvec = -nvec
        other = float(np.linalg.norm(q[(k + 2) % 4] - q[(k + 1) % 4]))
        span = max(3.0, search_frac * max(L, other))
        offs = np.arange(-span, span + 0.5, 1.0, dtype=np.float32)
        ts = np.linspace(0.1, 0.9, n_samples, dtype=np.float32)
        base = a[None, :] + ts[:, None] * d[None, :]
        sx = (base[:, 0:1] + offs[None, :] * nvec[0]).astype(np.float32)
        sy = (base[:, 1:2] + offs[None, :] * nvec[1]).astype(np.float32)
        mx = cv2.remap(gx, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        my = cv2.remap(gy, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        signed = mx * nvec[0] + my * nvec[1]
        mag = np.abs(signed)
        inside = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
        mag = np.where(inside, mag, 0)
        # every local maximum of the edge response is a candidate edge point
        peak = np.zeros_like(mag, bool)
        peak[:, 1:-1] = (mag[:, 1:-1] >= mag[:, :-2]) & (mag[:, 1:-1] >= mag[:, 2:])
        row_max = mag.max(axis=1)
        strong_level = 0.35 * float(np.percentile(row_max, 80)) + 1e-3
        peak &= mag >= strong_level
        ri, ci = np.nonzero(peak)
        if len(ri) < n_samples * 0.25:
            lines.append(None)
            continue
        pts = np.stack([sx[ri, ci], sy[ri, ci]], axis=1)
        # prefer the edge nearest the current side, not the strongest far one
        prox = np.exp(-(offs[ci] / (0.6 * span)) ** 2)
        wts = mag[ri, ci] * prox
        sgn = np.sign(signed[ri, ci])
        best = None
        for s in (1.0, -1.0):  # never mix edge polarities in one line
            sel = sgn == s
            if sel.sum() < n_samples * 0.25:
                continue
            line, inl = _ransac_line(pts[sel], weights=wts[sel], thresh=max(1.0, 0.006 * L), rng=rng)
            if line is None:
                continue
            support = len(np.unique(ri[sel][inl]))   # distinct sample positions
            score = float(wts[sel][inl].sum())
            if support >= n_samples * 0.3 and (best is None or score > best[0]):
                best = (score, line)
        lines.append(best[1] if best else None)
    return lines


def refine_quad(rgb: np.ndarray, quad: np.ndarray, search_frac: float = 0.05,
                n_samples: int = 48, grads=None, max_shift_frac: float = 0.12,
                iterations: int = 3) -> np.ndarray:
    """
    Snap a rough quad onto the card's real edges.

    For each side we sample points along it, look along the normal for
    (colour) gradient peaks, and RANSAC-fit a line through them - weighted
    toward peaks near the current side, and never mixing edge polarities, so
    a neighbouring card's edge or the card's own inner frame doesn't win.
    Several passes with a shrinking search band let a rotated rectangle
    (YOLO-OBB) converge onto a perspective trapezoid.

    Falls back to the input quad if the refinement looks implausible.
    """
    q0 = order_corners_clockwise(quad)
    gx, gy = grads if grads is not None else _color_gradient_images(rgb)
    rng = np.random.default_rng(0)
    q = q0.copy()
    for it in range(iterations):
        lines = _refine_sides_once(q, gx, gy, search_frac / (1.8 ** it), n_samples, rng)
        if lines is None:
            break
        for k in range(4):  # keep the current side where no edge was found
            if lines[k] is None:
                a, b = q[k], q[(k + 1) % 4]
                d = (b - a) / (np.linalg.norm(b - a) + 1e-9)
                lines[k] = np.array([d[0], d[1], a[0], a[1]], dtype=np.float32)
        new_q = []
        for k in range(4):
            p = _line_intersection(lines[k - 1], lines[k])
            if p is None:
                break
            new_q.append(p)
        if len(new_q) < 4:
            break
        new_q = np.array(new_q, dtype=np.float32)
        if not np.all(np.isfinite(new_q)) or not quad_angles_ok(new_q):
            break
        moved = np.abs(new_q - q).max()
        q = new_q
        if moved < 0.5:
            break
    size = max(np.linalg.norm(q0[2] - q0[0]), np.linalg.norm(q0[3] - q0[1]))
    if np.abs(q - q0).max() > max_shift_frac * size or not quad_angles_ok(q):
        return q0
    return q


# ---------------------------------------------------------------------------
# Candidate generation

def _binarisations(rgb: np.ndarray) -> List[np.ndarray]:
    """
    Several complementary binary maps; each tends to isolate cards on a
    different kind of background. A contour only needs to be clean in one.
    """
    h, w = rgb.shape[:2]
    short = min(h, w)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    Lc = clahe.apply(L)
    Lb = cv2.bilateralFilter(Lc, 7, 40, 7)
    maps = []

    # 1. Colour edges -> regions between edges. Cards separate from any
    # background that differs in luminance OR colour at the card boundary.
    edges = np.zeros((h, w), np.uint8)
    for ch, (lo, hi) in ((Lb, (20, 60)), (cv2.GaussianBlur(lab[:, :, 1], (5, 5), 0), (8, 24)),
                         (cv2.GaussianBlur(lab[:, :, 2], (5, 5), 0), (8, 24))):
        edges |= cv2.Canny(ch, lo, hi, L2gradient=True)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_d = cv2.dilate(edges, k3, iterations=1)
    maps.append(cv2.bitwise_not(edges_d))           # regions enclosed by edges
    maps.append(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k3, iterations=2))  # the edge curves

    # 2. Adaptive threshold, both polarities (dark borders on light tables
    # and light cards on dark mats).
    block = max(15, (short // 12) | 1)
    at = cv2.adaptiveThreshold(Lb, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, block, 4)
    maps.append(at)
    maps.append(cv2.bitwise_not(at))

    # 3. Global Otsu on luminance and saturation, both polarities
    _, ot = cv2.threshold(Lb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    maps.append(ot)
    maps.append(cv2.bitwise_not(ot))
    s = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1]
    s = cv2.GaussianBlur(s, (5, 5), 0)
    _, os_ = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    maps.append(os_)

    # 4. Distance from the background colour estimated at the image border.
    # Works for uniform non-white backgrounds (dark mats, coloured tables).
    border = np.concatenate([lab[:4].reshape(-1, 3), lab[-4:].reshape(-1, 3),
                             lab[:, :4].reshape(-1, 3), lab[:, -4:].reshape(-1, 3)]).astype(np.float32)
    bg = np.median(border, axis=0)
    spread = np.median(np.abs(border - bg), axis=0) + 4.0
    dist = np.sqrt((((lab.astype(np.float32) - bg) / spread) ** 2).sum(axis=2))
    fg = (dist > 3.0).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k3, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k3, iterations=2)
    maps.append(fg)
    return maps


def _edge_support(q: np.ndarray, gmag: np.ndarray, thresh: float, n: int = 40) -> float:
    """Fraction of points along the quad's sides that sit on a strong edge."""
    h, w = gmag.shape
    hits = 0
    total = 0
    for k in range(4):
        a, b = q[k], q[(k + 1) % 4]
        for t in np.linspace(0.1, 0.9, n):
            p = a + t * (b - a)
            x, y = int(round(p[0])), int(round(p[1]))
            if x < 2 or y < 2 or x >= w - 2 or y >= h - 2:
                continue
            total += 1
            if gmag[y - 2:y + 3, x - 2:x + 3].max() > thresh:
                hits += 1
    return hits / max(total, 1)


def _touches_border(q: np.ndarray, shape, margin_frac: float = 0.005) -> bool:
    h, w = shape[:2]
    m = margin_frac * max(h, w)
    return bool((q[:, 0] < m).any() or (q[:, 1] < m).any()
                or (q[:, 0] > w - 1 - m).any() or (q[:, 1] > h - 1 - m).any())


def _dist_to_polygon_edges(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest edge of a closed polygon."""
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a                                       # E x 2
    L2 = (ab * ab).sum(axis=1) + 1e-9                # E
    ap = pts[:, None, :] - a[None, :, :]             # N x E x 2
    t = np.clip((ap * ab[None]).sum(axis=2) / L2[None], 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.sqrt(((pts[:, None, :] - proj) ** 2).sum(axis=2)).min(axis=1)


def _initial_quad(hull: np.ndarray) -> np.ndarray:
    """4-vertex approximation of a convex hull; unlike minAreaRect this
    keeps the trapezoid shape of a card seen in perspective."""
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.04, 0.05, 0.07, 0.09):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
        if len(approx) < 4:
            break
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)


def _quad_from_contour(cnt: np.ndarray, min_area: float, max_area: float):
    area = cv2.contourArea(cnt)
    if area < min_area * 0.5 or area > max_area:
        return None
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area < min_area or hull_area > max_area:
        return None
    solidity = area / max(hull_area, 1)
    if solidity < 0.6:
        return None
    # Only keep contour points on the convex boundary: concave dents come
    # from fingers, dice or glare and shouldn't bend the fitted sides.
    pts = cnt.reshape(-1, 2).astype(np.float32)
    if len(pts) > 400:
        pts = pts[np.linspace(0, len(pts) - 1, 400).astype(int)]
    hull_pts = hull.reshape(-1, 2).astype(np.float32)
    d = _dist_to_polygon_edges(pts, hull_pts)
    boundary = pts[d < max(2.0, 0.01 * np.sqrt(hull_area))]
    if len(boundary) < 20:
        boundary = hull_pts
    init = _initial_quad(hull)
    q = fit_quad_to_points(boundary, init)
    if q is None:
        q = order_corners_clockwise(init)
    quad_area = cv2.contourArea(q)
    if quad_area <= 0:
        return None
    # How well the outline fills its fitted quad (rounded corners lose ~1%)
    fill = hull_area / quad_area
    if not (0.85 <= fill <= 1.08):
        return None
    return q, {'fill': float(fill), 'solidity': float(solidity)}


def _score_quad(q: np.ndarray, gmag: np.ndarray, edge_thresh: float, shape,
                extra: dict) -> Optional[float]:
    if not quad_angles_ok(q, 35, 145):
        return None
    ratio = rectified_aspect(q, shape)
    if not (MIN_QUAD_RATIO <= ratio <= MAX_QUAD_RATIO):
        return None
    aspect_score = float(np.exp(-((ratio - CARD_ASPECT) / 0.08) ** 2))
    support = _edge_support(q, gmag, edge_thresh)
    fill = extra.get('fill', 0.97)
    fill_score = float(np.clip(1.0 - abs(fill - 0.985) / 0.1, 0, 1))
    solid = extra.get('solidity', 0.9)
    score = 0.40 * support + 0.30 * aspect_score + 0.15 * fill_score + 0.15 * min(1.0, solid)
    if _touches_border(q, shape):
        score *= 0.8
    return float(score)


def _same_card_parallel(inner: dict, outer: dict) -> bool:
    return abs(float(np.dot(_long_axis(inner['quad']), _long_axis(outer['quad'])))) > 0.7


def _resolve_candidates(cands: List[dict]) -> List[dict]:
    """
    Turn a soup of overlapping candidate quads into one entry per physical
    card, each with ranked alternative outlines.

      * near-identical quads are merged (best score wins)
      * containers - quads holding two or more separate card candidates
        (a sheet of paper, a play mat, a binder page) - are dropped
      * art / text boxes (perpendicular long axis inside another quad) are
        dropped
      * everything else that overlaps is grouped; within a group the card's
        outer edge is preferred over its inner frame, and remaining members
        become alternatives (e.g. a card inside a slab, or a slab around it)
    """
    cands = sorted(cands, key=lambda c: -c['score'])
    kept: List[dict] = []
    for c in cands:
        if all(quad_iou(c['quad'], k['quad']) <= 0.85 for k in kept):
            kept.append(c)
    n = len(kept)
    if n == 0:
        return []

    # Pairwise containment: contain[i, j] = fraction of i inside j
    contain = np.zeros((n, n), np.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                contain[i, j] = quad_containment(kept[i]['quad'], kept[j]['quad'])

    drop = set()
    # Containers: j holds >= 2 mutually disjoint, strong, similar-sized
    # card candidates (cards on a sheet / mat). Art fragments inside a single
    # card are weaker and vary wildly in size, so they don't trigger this.
    for j in range(n):
        inside = [i for i in range(n) if i != j and contain[i, j] > 0.85
                  and kept[i]['area'] < 0.5 * kept[j]['area'] and kept[i]['score'] >= 0.8]
        disjoint = []
        for i in inside:
            if all(quad_iou(kept[i]['quad'], kept[d]['quad']) < 0.1 for d in disjoint):
                disjoint.append(i)
        areas = sorted(kept[i]['area'] for i in disjoint)
        similar = sum(1 for a in areas if a >= areas[-1] / 2.5) if areas else 0
        if similar >= 2:
            drop.add(j)
    # Art / text boxes: perpendicular quads inside a card. Only a live,
    # reasonably card-like outer quad can explain away an inner one - a
    # weak blob spanning several cards must not swallow them.
    for i in range(n):
        for j in range(n):
            if (i != j and j not in drop and contain[i, j] > 0.85
                    and kept[i]['area'] < 0.7 * kept[j]['area']
                    and kept[j]['score'] >= 0.7 * kept[i]['score']
                    and not _same_card_parallel(kept[i], kept[j])):
                drop.add(i)
    alive = [i for i in range(n) if i not in drop]

    # Greedy grouping by score
    groups: List[List[int]] = []
    assigned = set()
    for i in alive:
        if i in assigned:
            continue
        grp = [i]
        assigned.add(i)
        for j in alive:
            if j in assigned:
                continue
            if contain[i, j] > 0.8 or contain[j, i] > 0.8 or quad_iou(kept[i]['quad'], kept[j]['quad']) > 0.3:
                grp.append(j)
                assigned.add(j)
        groups.append(grp)

    out = []
    for grp in groups:
        primary = grp[0]
        best = kept[primary]['score']
        # Cards contain card-shaped fragments (art, mana symbols) far more
        # often than card-shaped containers contain cards, so default to the
        # largest well-scored member - unless it runs off the image, which
        # is typical of paper, slabs and play mats.
        big = [k for k in grp if kept[k]['score'] >= 0.8 * best and not kept[k].get('touches')]
        if big:
            primary = max(big, key=lambda k: kept[k]['area'])
        # Walk outward: prefer the enclosing outline of the *same* card
        # (outer border edge rather than the coloured inner frame).
        changed = True
        while changed:
            changed = False
            for j in grp:
                if j == primary:
                    continue
                if (contain[primary, j] > 0.85
                        and kept[primary]['area'] > 0.6 * kept[j]['area']
                        and kept[j]['area'] > kept[primary]['area']
                        and _same_card_parallel(kept[primary], kept[j])
                        and kept[j]['score'] >= 0.75 * kept[primary]['score']):
                    primary = j
                    changed = True
                    break
        entry = dict(kept[primary])
        entry['score'] = max(kept[k]['score'] for k in grp)
        alts = sorted((k for k in grp if k != primary), key=lambda k: -kept[k]['score'])
        entry['alternatives'] = [kept[k] for k in alts[:4]]
        out.append(entry)

    # Size consistency: in multi-card photos cards have similar sizes. Drop
    # groups far smaller than the dominant cards (usually art fragments or
    # background pattern that happened to be card-shaped).
    if len(out) > 1:
        strong = [c['area'] for c in out if c['score'] >= 0.7]
        ref_area = np.median(strong) if strong else max(c['area'] for c in out)
        out = [c for c in out if c['area'] >= 0.25 * ref_area]
    return sorted(out, key=lambda c: -c['score'])


class _ContourDeduper:
    """Many binarisations return the same outline. Skip a contour once a
    near-identical one has been fitted successfully (failed fits don't count,
    so a cleaner copy from another binarisation still gets its chance)."""

    def __init__(self, tol: float = 0.03):
        self.tol = tol
        self.done = []

    @staticmethod
    def key(cnt):
        (cx, cy), (rw, rh), _ = cv2.minAreaRect(cnt)
        return (cx, cy, min(rw, rh), max(rw, rh))

    def seen(self, key) -> bool:
        size = max(key[3], 1.0)
        return any(all(abs(s[i] - key[i]) < self.tol * size for i in range(4)) for s in self.done)

    def add(self, key):
        self.done.append(key)


def find_card_quads(rgb: np.ndarray, extra_quads: Optional[Sequence[np.ndarray]] = None,
                    extra_scores: Optional[Sequence[float]] = None,
                    max_cards: Optional[int] = None,
                    min_score: float = 0.5, use_contours: bool = True) -> List[CardQuad]:
    """
    Find cards in a photo. Returns CardQuads sorted by score (best first),
    with corners in full-resolution coordinates in portrait order. Each
    CardQuad carries alternative outlines for the same card.

    extra_quads: optional rough quads from another detector (e.g. YOLO, in
    full-resolution coordinates). They are refined onto real edges and
    compete with the contour candidates; agreement boosts the score.
    use_contours: False skips the classical candidates (only refines and
    resolves extra_quads) - faster, relies entirely on the other detector.
    """
    H, W = rgb.shape[:2]
    s = min(1.0, WORK_LONG_SIDE / max(H, W))
    work = cv2.resize(rgb, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    h, w = work.shape[:2]
    img_area = h * w
    min_side = MIN_CARD_SHORT_SIDE_FRAC * min(h, w)
    min_area = min_side * min_side / CARD_ASPECT
    max_area = 0.98 * img_area

    grads = _color_gradient_images(work)
    gmag = np.sqrt(grads[0] ** 2 + grads[1] ** 2)
    edge_thresh = max(20.0, float(np.percentile(gmag, 90)) * 0.6)

    # Cheap geometric pre-filter, then dedupe across binarisations
    raw = []
    for bmap in (_binarisations(work) if use_contours else []):
        contours, _ = cv2.findContours(bmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if len(cnt) < 20:
                continue
            (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
            if rw * rh < min_area * 0.8 or rw * rh > max_area * 1.05:
                continue
            if min(rw, rh) / max(rw, rh, 1) < 0.3:
                continue
            raw.append(cnt)
    raw.sort(key=lambda c: -cv2.contourArea(c))

    cands: List[dict] = []
    dedupe = _ContourDeduper()
    for cnt in raw:
        key = dedupe.key(cnt)
        if dedupe.seen(key):
            continue
        res = _quad_from_contour(cnt, min_area, max_area)
        if res is None:
            continue
        q, extra = res
        q = refine_quad(work, q, search_frac=0.02, grads=grads, max_shift_frac=0.04)
        sc = _score_quad(q, gmag, edge_thresh, work.shape, extra)
        if sc is None:
            continue
        dedupe.add(key)
        cands.append({'quad': q, 'score': sc, 'source': 'contour',
                      'area': float(cv2.contourArea(q)), 'extra': extra,
                      'touches': _touches_border(q, work.shape)})

    # External proposals (YOLO): refine, score, and boost agreeing contours
    if extra_quads is not None:
        for i, eq in enumerate(extra_quads):
            conf = float(extra_scores[i]) if extra_scores is not None else 0.5
            q0 = order_corners_clockwise(np.asarray(eq, np.float32) * s)
            q = refine_quad(work, q0, search_frac=0.05, grads=grads)
            sc = _score_quad(q, gmag, edge_thresh, work.shape, {})
            if sc is None:
                continue
            sc = 0.6 * sc + 0.4 * conf
            cands.append({'quad': q, 'score': sc, 'source': 'yolo',
                          'area': float(cv2.contourArea(q)), 'extra': {'conf': conf},
                          'touches': _touches_border(q, work.shape)})
            for c in cands:
                if c['source'] == 'contour' and quad_iou(c['quad'], q) > 0.7:
                    c['score'] = min(1.0, c['score'] + 0.15 * conf)

    cands = [c for c in cands if c['score'] >= min_score * 0.8]
    kept = _resolve_candidates(cands)
    kept = [c for c in kept if c['score'] >= min_score]
    # Final NMS between groups (two outlines of one card that the grouping
    # didn't join, e.g. via a chain through a neighbouring card)
    final: List[dict] = []
    for c in kept:
        if all(quad_iou(c['quad'], f['quad']) < 0.5 for f in final):
            final.append(c)
    kept = final
    if max_cards is not None:
        kept = kept[:max_cards]

    def to_cq(c):
        return CardQuad(corners=order_corners_portrait(c['quad'] / s), score=c['score'],
                        source=c['source'], details=c.get('extra', {}))

    out = []
    for c in kept:
        cq = to_cq(c)
        cq.alternatives = [to_cq(a) for a in c.get('alternatives', [])]
        out.append(cq)
    return out


def full_image_quad(rgb: np.ndarray) -> CardQuad:
    """Treat the whole image as the card (clean scans / pre-cropped photos)."""
    H, W = rgb.shape[:2]
    q = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    ratio = min(W, H) / max(W, H)
    score = float(np.exp(-((ratio - CARD_ASPECT) / 0.05) ** 2)) * 0.6
    return CardQuad(corners=order_corners_portrait(q), score=score, source='full_image')


def draw_quads(rgb: np.ndarray, quads: Sequence[CardQuad], labels: Optional[Sequence[str]] = None,
               color=(0, 255, 0)) -> np.ndarray:
    """Debug overlay: quads with the top edge (card's top) highlighted."""
    out = rgb.copy()
    th = max(2, int(max(rgb.shape[:2]) / 400))
    for i, cq in enumerate(quads):
        pts = cq.corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, color, th, cv2.LINE_AA)
        cv2.line(out, tuple(cq.corners[0].astype(int)), tuple(cq.corners[1].astype(int)),
                 (255, 0, 0), th, cv2.LINE_AA)
        text = labels[i] if labels is not None else f"{cq.score:.2f} {cq.source}"
        org = tuple(cq.corners.mean(axis=0).astype(int))
        fs = max(0.5, max(rgb.shape[:2]) / 1600)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
    return out
