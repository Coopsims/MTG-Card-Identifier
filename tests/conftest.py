import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtg_layout import CARD_W, CARD_H  # noqa: E402


def make_card(seed: int = 0) -> np.ndarray:
    """A procedural stand-in for a card scan: black border, coloured frame,
    textured 'illustration', pale text box with 'text' strokes."""
    rng = np.random.default_rng(seed)
    card = np.zeros((CARD_H, CARD_W, 3), np.uint8)
    frame = rng.integers(60, 200, 3)
    cv2.rectangle(card, (18, 18), (CARD_W - 19, CARD_H - 19), frame.tolist(), -1)
    art = np.zeros((300, 420, 3), np.uint8)
    art[:] = rng.integers(0, 255, 3)
    for _ in range(60):
        c = rng.integers(0, 255, 3).tolist()
        p = (int(rng.integers(0, 420)), int(rng.integers(0, 300)))
        if rng.random() < 0.5:
            cv2.circle(art, p, int(rng.integers(5, 50)), c, -1)
        else:
            q = (p[0] + int(rng.integers(-80, 80)), p[1] + int(rng.integers(-80, 80)))
            cv2.line(art, p, q, c, int(rng.integers(2, 10)))
    card[70:370, 34:454] = art
    cv2.rectangle(card, (34, 400), (454, 620), (225, 220, 205), -1)
    for y in range(415, 600, 22):
        x = 45
        while x < 430:
            w = int(rng.integers(10, 40))
            cv2.line(card, (x, y), (min(x + w, 440), y), (30, 30, 30), 3)
            x += w + 8
    cv2.rectangle(card, (30, 30), (400, 60), (230, 230, 230), -1)
    return card


@pytest.fixture
def card():
    return make_card(0)


@pytest.fixture
def other_card():
    return make_card(1)
