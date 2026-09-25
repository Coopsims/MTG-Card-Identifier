"""
Realistic synthetic photos of MTG cards, shared by train_detector.py and
train_identifier.py.

The original synthesiser pasted one flat, sharp-cornered card onto a blurred
noise background, rotated it in-plane, and added a round white blob as
"glare". Real phone photos look quite different, and those differences are
exactly where the trained detector and re-rankers failed:

  * Perspective. Cards are photographed at an angle: the card is rotated
    in 3-D and projected through a camera, so it's a trapezoid, not a
    rotated rectangle.
  * Backgrounds. Play mats are fantasy art, tables are wood grain, people
    shoot on notebook paper, dark cloth, binders. Solid colours and blurred
    noise don't teach the detector to ignore any of that.
  * Glare. Sleeves and foils produce specular highlights: saturated white
    cores with soft halos, long streaks from ceiling lights, broad sheens
    that wash out contrast, rainbow iridescence. These are "screen" blends
    toward white, not additive circles.
  * Clutter. Several cards per photo, overlapping hands, fingers holding
    the card, dice on top, phones / ID cards / paper next to it.
  * Camera. White balance, exposure, defocus and motion blur, sensor
    noise, JPEG.

Everything here works in RGB uint8.
"""
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from mtg_layout import CARD_W, CARD_H

CARD_ASPECT = 63.0 / 88.0


# ===========================================================================
# Backgrounds

def _fractal_noise(h: int, w: int, octaves: int = 5, base: int = 4,
                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        s = base * (2 ** o)
        layer = rng.random((max(2, h * s // max(h, w)), max(2, w * s // max(h, w)))).astype(np.float32)
        out += amp * cv2.resize(layer, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= 0.5
    out /= total
    return (out - out.min()) / (out.max() - out.min() + 1e-6)


def _wood(h, w, rng):
    base = np.array([rng.uniform(90, 190), rng.uniform(60, 140), rng.uniform(30, 90)], np.float32)
    warp = _fractal_noise(h, w, 3, 2, rng) * rng.uniform(20, 60)
    ys = np.arange(h, dtype=np.float32)[:, None] + warp
    grain = np.sin(ys / rng.uniform(3, 9)) * 0.5 + np.sin(ys / rng.uniform(15, 40)) * 0.5
    fine = _fractal_noise(h, w, 6, 16, rng) - 0.5
    shade = 1.0 + 0.12 * grain + 0.10 * fine
    img = base[None, None, :] * shade[..., None]
    if rng.random() < 0.5:
        img = np.ascontiguousarray(np.transpose(img, (1, 0, 2)))
        img = cv2.resize(img, (w, h))
    # planks
    if rng.random() < 0.4:
        n = rng.integers(2, 6)
        for x in np.linspace(0, w, n + 1)[1:-1]:
            img[:, int(x):int(x) + 2] *= 0.6
    return np.clip(img, 0, 255).astype(np.uint8)


def _cloth(h, w, rng):
    """Play-mat rubber / felt: one colour, fine texture, often dark."""
    dark = rng.random() < 0.6
    base = rng.uniform(10, 70, 3) if dark else rng.uniform(40, 200, 3)
    tex = (_fractal_noise(h, w, 4, 32, rng) - 0.5) * rng.uniform(8, 25)
    img = base[None, None, :] + tex[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def _stone(h, w, rng):
    n = _fractal_noise(h, w, 6, 3, rng)
    veins = np.abs(np.sin(n * rng.uniform(8, 20)))
    c1 = rng.uniform(150, 240, 3)
    c2 = c1 * rng.uniform(0.5, 0.9)
    img = c1[None, None] * veins[..., None] + c2[None, None] * (1 - veins[..., None])
    return np.clip(img, 0, 255).astype(np.uint8)


def _paper(h, w, rng):
    """Notebook / printer paper, sometimes with lines and scribbles."""
    base = rng.uniform(215, 250)
    tint = np.array([base, base * rng.uniform(0.97, 1.0), base * rng.uniform(0.92, 1.0)], np.float32)
    img = np.ones((h, w, 3), np.float32) * tint
    img += (_fractal_noise(h, w, 3, 4, rng)[..., None] - 0.5) * 20
    if rng.random() < 0.7:
        gap = rng.integers(max(8, h // 40), max(12, h // 18))
        for y in range(int(rng.integers(0, gap)), h, int(gap)):
            cv2.line(img, (0, y), (w, y), (170, 190, 230), 1)
    if rng.random() < 0.5:
        for _ in range(rng.integers(3, 15)):
            x, y = rng.integers(0, w), rng.integers(0, h)
            pts = np.cumsum(rng.normal(0, 4, (rng.integers(5, 30), 2)), axis=0) + [x, y]
            cv2.polylines(img, [pts.astype(np.int32)], False, (60, 60, 90), 1, cv2.LINE_AA)
    return np.clip(img, 0, 255).astype(np.uint8)


def _gradient(h, w, rng):
    c1 = rng.uniform(0, 255, 3)
    c2 = rng.uniform(0, 255, 3)
    t = np.linspace(0, 1, h if rng.random() < 0.5 else w, dtype=np.float32)
    grad = c1[None] * (1 - t[:, None]) + c2[None] * t[:, None]
    img = np.repeat(grad[:, None, :], w, axis=1) if len(t) == h else np.repeat(grad[None], h, axis=0)
    img = img + (_fractal_noise(h, w, 3, 4, rng)[..., None] - 0.5) * 15
    return np.clip(img, 0, 255).astype(np.uint8)


def _clutter(h, w, rng):
    img = _gradient(h, w, rng)
    for _ in range(rng.integers(3, 12)):
        color = tuple(float(c) for c in rng.uniform(20, 235, 3))
        if rng.random() < 0.5:
            cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                       int(rng.integers(h // 20, h // 4)), color, -1)
        else:
            x, y = rng.integers(0, w), rng.integers(0, h)
            cv2.rectangle(img, (int(x), int(y)), (int(x + rng.integers(w // 10, w // 2)),
                          int(y + rng.integers(h // 10, h // 2))), color, -1)
    return cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 6))


class BackgroundBank:
    """
    Source of realistic backgrounds.

    art_sampler: callable returning a random RGB card image. Crops of card
    illustrations, blown up and softened, look remarkably like play mats -
    which is what most people photograph their cards on.
    photo_dirs: optional folders of real background photos (tables, mats,
    desks) - the best source if you have them.
    """

    def __init__(self, art_sampler: Optional[Callable[[], np.ndarray]] = None,
                 photo_dirs: Sequence[Path] = ()):
        self.art_sampler = art_sampler
        self._cache = {}
        self.cache_per_kind = 6
        self.photos = []
        for d in photo_dirs:
            d = Path(d)
            if d.is_dir():
                self.photos += [p for p in d.iterdir()
                                if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')]

    def _procedural(self, kind: str, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
        """Procedural textures are slow to generate, so each worker keeps a
        small pool per kind and serves random crops / flips / tints of it."""
        pool = self._cache.setdefault(kind, [])
        if len(pool) < self.cache_per_kind or rng.random() < 0.05:
            tex = {'wood': _wood, 'cloth': _cloth, 'paper': _paper, 'stone': _stone,
                   'gradient': _gradient, 'clutter': _clutter}[kind](1024, 1024, rng)
            if len(pool) < self.cache_per_kind:
                pool.append(tex)
            else:
                pool[int(rng.integers(len(pool)))] = tex
        tex = pool[int(rng.integers(len(pool)))]
        s = rng.uniform(0.5, 1.0)
        ch, cw = int(1024 * s * min(1.0, h / w) + 0.5), int(1024 * s * min(1.0, w / h) + 0.5)
        y = int(rng.integers(0, 1024 - ch + 1))
        x = int(rng.integers(0, 1024 - cw + 1))
        crop = tex[y:y + ch, x:x + cw]
        if rng.random() < 0.5:
            crop = crop[:, ::-1]
        out = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        tint = rng.uniform(0.85, 1.15, 3)
        return np.clip(out * tint, 0, 255).astype(np.uint8)

    def sample(self, h: int, w: int, rng: Optional[np.random.Generator] = None,
               kinds: Optional[Sequence[str]] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        options = {'playmat': 0.30 if self.art_sampler else 0.0,
                   'photo': 0.25 if self.photos else 0.0,
                   'wood': 0.15, 'cloth': 0.15, 'paper': 0.08, 'stone': 0.04,
                   'gradient': 0.05, 'clutter': 0.05}
        if kinds is not None:
            options = {k: v for k, v in options.items() if k in kinds and v > 0} or {'gradient': 1.0}
        names = list(options)
        p = np.array([options[n] for n in names], np.float64)
        kind = names[rng.choice(len(names), p=p / p.sum())]
        if kind == 'playmat':
            art = self.art_sampler()
            ah, aw = art.shape[:2]
            # the illustration box of most frames
            art = art[int(ah * 0.11):int(ah * 0.55), int(aw * 0.07):int(aw * 0.93)]
            if rng.random() < 0.5:
                art = art[:, ::-1]
            bg = cv2.resize(art, (w, h), interpolation=cv2.INTER_CUBIC)
            bg = cv2.GaussianBlur(bg, (0, 0), rng.uniform(0.5, 3.0))
            return (bg.astype(np.float32) * rng.uniform(0.5, 1.0)).clip(0, 255).astype(np.uint8)
        if kind == 'photo':
            p = self.photos[rng.integers(len(self.photos))]
            img = cv2.imread(str(p))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ih, iw = img.shape[:2]
                s = max(h / ih, w / iw) * rng.uniform(1.0, 1.6)
                img = cv2.resize(img, (int(iw * s) + 1, int(ih * s) + 1))
                y = rng.integers(0, img.shape[0] - h + 1)
                x = rng.integers(0, img.shape[1] - w + 1)
                return img[y:y + h, x:x + w].copy()
            kind = 'wood'
        return self._procedural(kind, h, w, rng)


# ===========================================================================
# Card appearance

def rounded_corner_mask(h: int, w: int, radius_frac: float = 0.048) -> np.ndarray:
    """Alpha mask with the card's rounded corners (r ~ 3 mm on 63 mm)."""
    r = max(1, int(radius_frac * w))
    m = np.full((h, w), 255, np.uint8)
    for cy, cx in ((r, r), (r, w - r - 1), (h - r - 1, r), (h - r - 1, w - r - 1)):
        y0, x0 = (0 if cy == r else h - r), (0 if cx == r else w - r)
        m[y0:y0 + r, x0:x0 + r] = 0
        cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return m


def add_sleeve(card: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Put the card in a sleeve: a coloured / clear border slightly larger than
    the card and a faint glossy film. Returns (image, alpha, pad_x, pad_y)
    where pad is the sleeve border as a fraction of the card size.
    """
    h, w = card.shape[:2]
    px = rng.uniform(0.012, 0.035)
    py = px * w / h * rng.uniform(0.8, 1.6)
    PX, PY = int(px * w), int(py * h)
    H, W = h + 2 * PY, w + 2 * PX
    if rng.random() < 0.6:  # opaque coloured back showing around the card
        color = rng.choice([[15, 15, 18], [20, 30, 90], [110, 15, 20], [230, 230, 235],
                            list(rng.uniform(0, 255, 3))])
        out = np.empty((H, W, 3), np.float32)
        out[:] = np.array(color, np.float32)
    else:  # clear: the edge region just looks like a light, faint film
        out = np.full((H, W, 3), rng.uniform(150, 230), np.float32)
    out[PY:PY + h, PX:PX + w] = card
    film = rng.uniform(0.0, 0.12)
    out = out * (1 - film) + 255 * film
    alpha = rounded_corner_mask(H, W, 0.04)
    return np.clip(out, 0, 255).astype(np.uint8), alpha, px, py


# ===========================================================================
# Camera geometry

def camera_homography(src_w: float, src_h: float, out_w: int, out_h: int,
                      height_frac: float, center: Tuple[float, float],
                      roll: float, pitch: float, yaw: float,
                      focal_factor: float = 0.9) -> np.ndarray:
    """
    Homography that maps a flat src_w x src_h image (the card) into the
    photo as seen by a pinhole camera: the card is rotated in 3-D by
    pitch / yaw (tilt away from the camera) and roll (in-plane), then
    projected. height_frac sets the card's approximate size in the photo and
    center where its centre lands (pixels).
    """
    f = focal_factor * max(out_w, out_h)
    a, b, c = np.radians([pitch, yaw, roll])
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    # physical size in arbitrary units = pixels of the source image
    Z = f * src_h / (height_frac * out_h)
    P = np.array([[-src_w / 2, -src_h / 2, 0], [src_w / 2, -src_h / 2, 0],
                  [src_w / 2, src_h / 2, 0], [-src_w / 2, src_h / 2, 0]], np.float64)
    X = (R @ P.T).T + np.array([0, 0, Z])
    x = f * X[:, :2] / X[:, 2:3]
    x += np.array(center) - x.mean(axis=0)
    src = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]], np.float32)
    return cv2.getPerspectiveTransform(src, x.astype(np.float32))


# ===========================================================================
# Lighting effects

def _screen(img: np.ndarray, alpha: np.ndarray, color=(255, 255, 255)) -> np.ndarray:
    """Screen-blend toward a light colour: how reflections add to a scene."""
    a = np.clip(alpha, 0, 1)[..., None]
    col = np.array(color, np.float32)[None, None]
    f = img.astype(np.float32)
    return np.clip(f + (col - f) * a, 0, 255).astype(np.uint8)


def add_glare(img: np.ndarray, region: Optional[np.ndarray] = None,
              rng: Optional[np.random.Generator] = None, kind: Optional[str] = None,
              strength: Optional[float] = None) -> np.ndarray:
    """
    Specular highlight inside `region` (uint8 mask of the card / sleeve; the
    whole image if None). Kinds:
      spot     elliptical hot spot: blown-out core + soft halo (lamp / flash)
      streak   long soft band (tube light / window on a sleeve)
      sheen    large, weak wash that lowers contrast (overhead light on gloss)
      rainbow  iridescent band (sleeves, foils)
    """
    rng = rng or np.random.default_rng()
    h, w = img.shape[:2]
    kind = kind or rng.choice(['spot', 'streak', 'sheen', 'rainbow'], p=[0.4, 0.3, 0.2, 0.1])
    strength = strength if strength is not None else rng.uniform(0.5, 1.0)
    if region is not None and region.any():
        ys, xs = np.nonzero(region[::4, ::4])
        x0, x1 = max(0, xs.min() * 4 - 4), min(w, xs.max() * 4 + 8)
        y0, y1 = max(0, ys.min() * 4 - 4), min(h, ys.max() * 4 + 8)
    else:
        x0, x1, y0, y1 = 0, w, 0, h
    rw, rh = max(8, x1 - x0), max(8, y1 - y0)
    # all maths happens inside the region's bounding box
    cx, cy = rng.uniform(0, rw), rng.uniform(0, rh)
    yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float32)
    ang = rng.uniform(0, np.pi)
    u = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
    v = -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)
    size = min(rw, rh)
    if kind == 'spot':
        a_len = size * rng.uniform(0.06, 0.35)
        b_len = a_len * rng.uniform(0.3, 1.0)
        d2 = (u / a_len) ** 2 + (v / b_len) ** 2
        alpha = strength * (np.exp(-d2 * 2.0) * 1.4 + 0.5 * np.exp(-d2 * 0.25))
    elif kind == 'streak':
        width = size * rng.uniform(0.03, 0.18)
        along = np.exp(-(u / (max(rw, rh) * rng.uniform(0.4, 1.2))) ** 2)
        alpha = strength * np.exp(-(v / width) ** 2) * (0.5 + 0.8 * along)
    elif kind == 'sheen':
        a_len = size * rng.uniform(0.4, 1.0)
        alpha = strength * rng.uniform(0.2, 0.45) * np.exp(-((u / a_len) ** 2 + (v / (a_len * 0.7)) ** 2))
    else:  # rainbow
        width = size * rng.uniform(0.08, 0.3)
        alpha = strength * 0.36 * np.exp(-(v / width) ** 2)
    alpha = np.clip(alpha, 0, 1).astype(np.float32)
    if region is not None:
        alpha *= cv2.GaussianBlur(region[y0:y1, x0:x1], (0, 0), 2).astype(np.float32) / 255
    patch = img[y0:y1, x0:x1].astype(np.float32)
    if kind == 'rainbow':
        hue = ((u / (size + 1e-6)) * 90 + rng.uniform(0, 180)) % 180
        hsv = np.stack([hue, np.full_like(hue, 160), np.full_like(hue, 255)], axis=2).astype(np.uint8)
        light = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)
    else:
        light = 255.0
    patch += (light - patch) * alpha[..., None]
    out = img.copy()
    out[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)
    return out


def add_shadow(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Soft shadow of a hand / phone / head across part of the scene."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    n = rng.integers(3, 7)
    pts = np.stack([rng.uniform(-0.2, 1.2, n) * w, rng.uniform(-0.2, 1.2, n) * h], axis=1)
    hull = cv2.convexHull(pts.astype(np.float32)).astype(np.int32)
    cv2.fillConvexPoly(mask, hull, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), rng.uniform(5, 30))
    dark = rng.uniform(0.35, 0.75)
    out = img.astype(np.float32) * (1 - (1 - dark) * mask[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def add_occluders(img: np.ndarray, card_quads: Sequence[np.ndarray],
                  rng: np.random.Generator) -> np.ndarray:
    """Fingers holding a card edge and dice / counters sitting on cards."""
    out = img.copy()
    for q in card_quads:
        r = rng.random()
        side = rng.integers(0, 4)
        a, b = q[side], q[(side + 1) % 4]
        size = float(np.linalg.norm(q[1] - q[0]))
        if r < 0.25:  # finger over an edge
            t = rng.uniform(0.2, 0.8)
            p = a + t * (b - a)
            inward = q.mean(axis=0) - p
            inward /= np.linalg.norm(inward) + 1e-6
            tip = p + inward * size * rng.uniform(0.08, 0.25)
            base = p - inward * size * rng.uniform(0.3, 0.8)
            skin = np.array([rng.uniform(150, 235), rng.uniform(100, 180), rng.uniform(80, 150)])
            skin *= rng.uniform(0.5, 1.0)
            thick = int(size * rng.uniform(0.12, 0.2))
            cv2.line(out, tuple(base.astype(int)), tuple(tip.astype(int)), skin.tolist(), thick, cv2.LINE_AA)
            cv2.circle(out, tuple(tip.astype(int)), thick // 2, (skin * 1.05).clip(0, 255).tolist(), -1, cv2.LINE_AA)
        elif r < 0.40:  # die / counter on the card
            c = q.mean(axis=0) + (q[2] - q[0]) * rng.uniform(-0.3, 0.3)
            s = size * rng.uniform(0.12, 0.25)
            color = rng.uniform(0, 255, 3).tolist()
            box = cv2.boxPoints(((float(c[0]), float(c[1])), (s, s), float(rng.uniform(0, 90))))
            cv2.fillConvexPoly(out, box.astype(np.int32), color, cv2.LINE_AA)
            for _ in range(rng.integers(1, 6)):
                d = c + rng.uniform(-0.3, 0.3, 2) * s
                cv2.circle(out, tuple(d.astype(int)), max(1, int(s * 0.07)),
                           (255, 255, 255) if np.mean(color) < 128 else (20, 20, 20), -1, cv2.LINE_AA)
    return out


def add_distractors(img: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    """Card-like rectangles that are NOT cards: phones, ID / credit cards,
    paper, sticky notes. They teach the detector what not to fire on."""
    h, w = img.shape[:2]
    out = img.copy()
    for _ in range(n):
        kind = rng.choice(['phone', 'idcard', 'paper', 'note'])
        aspect = {'phone': 0.47, 'idcard': 0.63, 'paper': 0.77, 'note': 1.0}[kind]
        sh = min(h, w) * rng.uniform(0.2, 0.7)
        sw = sh * aspect
        c = (rng.uniform(0, w), rng.uniform(0, h))
        box = cv2.boxPoints((c, (sw, sh), float(rng.uniform(0, 180)))).astype(np.int32)
        color = {'phone': [20, 20, 25], 'idcard': rng.uniform(80, 250, 3).tolist(),
                 'paper': [240, 240, 235], 'note': [250, 235, 110]}[kind]
        cv2.fillConvexPoly(out, box, color, cv2.LINE_AA)
        if kind == 'idcard':
            x, y = box.min(axis=0)
            cv2.rectangle(out, (int(x + sw * 0.1), int(y + sh * 0.1)),
                          (int(x + sw * 0.4), int(y + sh * 0.4)), rng.uniform(0, 255, 3).tolist(), -1)
    return out


def camera_effects(img: np.ndarray, rng: np.random.Generator, level: float = 1.0) -> np.ndarray:
    """White balance, exposure, vignette, blur, noise, JPEG."""
    t = rng.uniform(-0.12, 0.18) * level        # colour temperature (warm > 0)
    gains = np.array([1 + t, 1 + t * 0.2, 1 - t]) * rng.uniform(1 - 0.35 * level, 1 + 0.3 * level)
    gamma = rng.uniform(1 - 0.2 * level, 1 + 0.25 * level)
    x = np.arange(256, dtype=np.float32)
    out = np.empty_like(img)
    for c in range(3):  # per-channel lookup table: gain, then gamma
        lut = (255.0 * (np.clip(x * gains[c], 0, 255) / 255.0) ** gamma).astype(np.uint8)
        out[:, :, c] = cv2.LUT(img[:, :, c], lut)
    h, w = img.shape[:2]
    if rng.random() < 0.5 * level:
        yy, xx = np.mgrid[0:16, 0:16].astype(np.float32)
        d = ((xx - 7.5) ** 2 + (yy - 7.5) ** 2) / (2 * 7.5 ** 2)
        vig = cv2.resize(1 - rng.uniform(0.1, 0.4) * d, (w, h), interpolation=cv2.INTER_LINEAR)
        out = (out * vig[..., None]).astype(np.uint8)
    r = rng.random()
    if r < 0.35 * level:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.5, 2.0) * level)
    elif r < 0.5 * level:
        k = int(rng.integers(3, 9))
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0 / k
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(rng.uniform(0, 180)), 1)
        kern = cv2.warpAffine(kern, M, (k, k))
        kern /= kern.sum() + 1e-6
        out = cv2.filter2D(out, -1, kern)
    if rng.random() < 0.7 * level:
        sigma = rng.uniform(1, 7) * level
        noise = np.empty(out.shape, np.int16)
        cv2.randn(noise, 0, sigma)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        q = int(rng.uniform(40, 95))
        ok, buf = cv2.imencode('.jpg', out[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            out = np.ascontiguousarray(cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1])
    return out


# ===========================================================================
# Scenes

DIFFICULTY = {
    #          tilt  roll  glare_p sleeve_p shadow_p occl_p  camera  max_cards
    'easy':   (15,   20,   0.15,   0.2,     0.1,     0.0,    0.4,    2),
    'medium': (35,   180,  0.45,   0.4,     0.3,     0.2,    0.8,    4),
    'hard':   (55,   180,  0.75,   0.5,     0.5,     0.45,   1.0,    7),
}


def _place_cards(card_imgs: List[np.ndarray], out_w: int, out_h: int, diff: str,
                 rng: np.random.Generator):
    """Choose a camera pose per card. Returns list of (H, src_w, src_h)."""
    tilt, roll_range = DIFFICULTY[diff][0], DIFFICULTY[diff][1]
    n = len(card_imgs)
    layout = rng.choice(['scatter', 'grid', 'fan']) if n > 1 else 'single'
    # shared camera tilt for the whole photo (the table is one plane)
    pitch = rng.uniform(-tilt, tilt)
    yaw = rng.uniform(-tilt, tilt) * 0.6
    focal = rng.uniform(0.7, 1.3)
    if layout == 'single':
        size = rng.uniform(0.35, 0.9)
    else:
        size = rng.uniform(0.18, 0.5) / max(1.0, np.sqrt(n / 2))
    base_roll = rng.uniform(-roll_range, roll_range)
    poses = []
    placed = []
    for i, card in enumerate(card_imgs):
        ch, cw = card.shape[:2]
        pose = None
        for _ in range(30):
            if layout == 'grid':
                cols = int(np.ceil(np.sqrt(n)))
                r_, c_ = divmod(i, cols)
                cx = (c_ + 0.5) / cols * out_w + rng.normal(0, out_w * 0.02)
                cy = (r_ + 0.5) / cols * out_h + rng.normal(0, out_h * 0.02)
                roll = base_roll + rng.normal(0, 3)
            elif layout == 'fan':
                cx = out_w * (0.3 + 0.4 * i / max(n - 1, 1)) + rng.normal(0, out_w * 0.02)
                cy = out_h * rng.uniform(0.45, 0.6)
                roll = base_roll + (i - (n - 1) / 2) * rng.uniform(6, 15)
            else:
                cx = rng.uniform(0.15, 0.85) * out_w
                cy = rng.uniform(0.15, 0.85) * out_h
                roll = base_roll + rng.normal(0, 20) if layout == 'scatter' else base_roll
            H = camera_homography(cw, ch, out_w, out_h, size, (cx, cy), roll, pitch, yaw, focal)
            corners = cv2.perspectiveTransform(
                np.array([[[0, 0], [cw, 0], [cw, ch], [0, ch]]], np.float32), H)[0]
            if (corners[:, 0].min() < -0.02 * out_w or corners[:, 0].max() > 1.02 * out_w
                    or corners[:, 1].min() < -0.02 * out_h or corners[:, 1].max() > 1.02 * out_h):
                size *= 0.95
                continue
            if layout == 'scatter' and any(_overlap(corners, p) > 0.1 for p in placed):
                continue
            placed.append(corners)
            pose = (H, corners)
            break
        poses.append(pose)  # None when the card couldn't be placed
    return poses


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    inter, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    return float(inter / max(min(cv2.contourArea(a), cv2.contourArea(b)), 1e-6))


def synthesize_scene(card_imgs: List[np.ndarray], out_size: Tuple[int, int],
                     difficulty: str = 'medium', backgrounds: Optional[BackgroundBank] = None,
                     rng: Optional[np.random.Generator] = None,
                     min_visible: float = 0.6):
    """
    Render a photo containing the given cards.

    Returns (scene RGB, cards) where cards is a list of dicts with
      'corners'  4x2 float32 image coordinates of the CARD (not the sleeve),
                 ordered TL, TR, BR, BL in the card's own frame
      'visible'  fraction of the card not covered by later cards
      'index'    position in card_imgs
    """
    rng = rng or np.random.default_rng()
    out_w, out_h = out_size
    diff = DIFFICULTY[difficulty]
    glare_p, sleeve_p, shadow_p, occl_p, cam_level = diff[2], diff[3], diff[4], diff[5], diff[6]
    backgrounds = backgrounds or BackgroundBank()
    scene = backgrounds.sample(out_h, out_w, rng)
    if rng.random() < 0.3 * cam_level:
        scene = add_distractors(scene, rng, int(rng.integers(1, 3)))

    # Prepare card images (optionally sleeved) and their card-corner offsets
    prepared = []
    for card in card_imgs:
        card = cv2.resize(card, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA) \
            if card.shape[:2] != (CARD_H, CARD_W) else card
        if rng.random() < sleeve_p:
            img, alpha, px, py = add_sleeve(card, rng)
        else:
            img, alpha, px, py = card, rounded_corner_mask(CARD_H, CARD_W), 0.0, 0.0
        prepared.append((img, alpha, px, py))

    poses = _place_cards([p[0] for p in prepared], out_w, out_h, difficulty, rng)
    owner = np.full((out_h, out_w), -1, np.int32)
    full_area = []
    results = []
    card_region = np.zeros((out_h, out_w), np.uint8)
    for i, ((img, alpha, px, py), pose) in enumerate(zip(prepared, poses)):
        if pose is None:
            full_area.append(1.0)
            continue
        H = pose[0]
        warped = cv2.warpPerspective(img, H, (out_w, out_h), flags=cv2.INTER_LINEAR)
        a = cv2.warpPerspective(alpha, H, (out_w, out_h), flags=cv2.INTER_LINEAR).astype(np.float32) / 255
        # soft contact shadow under the card, then alpha-composite
        sf = scene.astype(np.float32)
        if rng.random() < 0.7:
            sh = cv2.GaussianBlur(a, (0, 0), rng.uniform(2, 8))
            off = rng.integers(-4, 5, 2)
            sh = np.roll(sh, tuple(int(o) for o in off), axis=(0, 1))
            sf *= (1 - 0.4 * sh)[..., None]
        a3 = a[..., None]
        scene = (sf + (warped.astype(np.float32) - sf) * a3).astype(np.uint8)
        hard = a > 0.5
        owner[hard] = i
        card_region |= hard.astype(np.uint8) * 255
        ih, iw = img.shape[:2]
        # the card itself sits inside the sleeve padding
        x0, y0 = px * CARD_W, py * CARD_H
        card_pts = np.array([[[x0, y0], [x0 + CARD_W, y0], [x0 + CARD_W, y0 + CARD_H], [x0, y0 + CARD_H]]],
                            np.float32)
        corners = cv2.perspectiveTransform(card_pts, H)[0]
        full_area.append(max(1.0, cv2.contourArea(corners)))
        results.append({'corners': corners, 'index': i})

    occluded_quads = [r['corners'] for r in results]
    if rng.random() < occl_p:
        scene = add_occluders(scene, occluded_quads, rng)
    if rng.random() < shadow_p:
        scene = add_shadow(scene, rng)
    if rng.random() < glare_p and card_region.any():
        for _ in range(int(rng.integers(1, 3))):
            target = card_region
            if results and rng.random() < 0.7:
                # aim at one card
                m = np.zeros_like(card_region)
                cv2.fillConvexPoly(m, results[int(rng.integers(len(results)))]['corners'].astype(np.int32), 255)
                target = m
            scene = add_glare(scene, target, rng)
    scene = camera_effects(scene, rng, cam_level)

    for r in results:
        visible = float((owner == r['index']).sum()) / full_area[r['index']]
        r['visible'] = min(1.0, visible)
    results = [r for r in results if r['visible'] >= min_visible]
    return scene, results


def synthesize_card_photo(card: np.ndarray, difficulty: str = 'medium',
                          backgrounds: Optional[BackgroundBank] = None,
                          distractor_cards: Sequence[np.ndarray] = (),
                          corner_jitter: float = 0.012,
                          rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    A realistic photo of `card`, rectified back to 488x680 the way the
    detector would: through the true corners plus a small localisation
    error. Used to train the re-rankers / frame / set classifiers on what
    they actually see at inference.
    """
    rng = rng or np.random.default_rng()
    size = (int(rng.integers(480, 800)), int(rng.integers(480, 800)))
    cards = [card] + list(distractor_cards)
    for _ in range(3):
        scene, found = synthesize_scene(cards, size, difficulty, backgrounds, rng, min_visible=0.0)
        target = next((f for f in found if f['index'] == 0), None)
        if target is not None:
            break
    if target is None:
        return cv2.resize(card, (CARD_W, CARD_H))
    corners = target['corners'].copy()
    diag = float(np.linalg.norm(corners[2] - corners[0]))
    err = rng.normal(0, corner_jitter, corners.shape) * diag
    if rng.random() < 0.1:
        err *= 3  # occasional bad localisation
    corners = (corners + err).astype(np.float32)
    dst = np.array([[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(scene, M, (CARD_W, CARD_H), borderMode=cv2.BORDER_REPLICATE)
