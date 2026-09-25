import numpy as np

from photo_synthesis import BackgroundBank, synthesize_card_photo, synthesize_scene
from mtg_layout import CARD_W, CARD_H


def test_scene_labels_are_inside_image(card, other_card):
    rng = np.random.default_rng(0)
    for diff in ('easy', 'medium', 'hard'):
        scene, found = synthesize_scene([card, other_card], (640, 480), diff, BackgroundBank(), rng)
        assert scene.shape == (480, 640, 3) and scene.dtype == np.uint8
        for f in found:
            c = f['corners']
            assert c.shape == (4, 2)
            assert (c[:, 0] > -0.05 * 640).all() and (c[:, 0] < 1.05 * 640).all()
            assert f['visible'] >= 0.6


def test_card_photo_is_rectified_card(card):
    rng = np.random.default_rng(0)
    out = synthesize_card_photo(card, 'easy', BackgroundBank(), rng=rng)
    assert out.shape == (CARD_H, CARD_W, 3)
    # the rectified photo should still resemble the card (mean colour close)
    assert np.abs(out.astype(float).mean(axis=(0, 1)) - card.astype(float).mean(axis=(0, 1))).max() < 60
