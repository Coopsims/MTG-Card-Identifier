import cv2
import numpy as np

from card_matching import (CandidateVerifier, expand_quad, glare_mask, remove_glare,
                           masked_correlation, correlation_features)
from photo_synthesis import add_glare
from mtg_layout import CARD_W, CARD_H


def _washed(img, original):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    changed = np.abs(img.astype(int) - original.astype(int)).sum(axis=2) > 100
    return changed & (hsv[:, :, 2] >= 235) & (hsv[:, :, 1] < 50)


def test_glare_mask_finds_specular_spot_not_white_regions(card):
    for seed in range(4):
        glared = add_glare(card, None, np.random.default_rng(seed), kind='spot', strength=1.0)
        mask = glare_mask(glared)
        washed = _washed(glared, card)
        assert washed.any()
        assert (mask[washed] > 0).mean() > 0.9   # the washed-out spot is masked
    clean_mask = glare_mask(card)
    assert (clean_mask > 0).mean() < 0.01        # pale text box / title bar aren't glare


def test_remove_glare_moves_image_back_toward_original(card):
    for seed in range(4):
        glared = add_glare(card, None, np.random.default_rng(seed), kind='spot', strength=1.0)
        cleaned, _ = remove_glare(glared)
        changed = np.abs(cleaned.astype(int) - glared.astype(int)).sum(axis=2) > 0
        assert changed.any()
        before = np.abs(glared.astype(int) - card.astype(int))[changed].mean()
        after = np.abs(cleaned.astype(int) - card.astype(int))[changed].mean()
        assert after < before
    assert np.array_equal(remove_glare(card)[0], card)  # nothing to do on a clean card


def test_expand_quad_identity_and_growth():
    q = np.array([[10, 10], [110, 12], [112, 150], [8, 148]], np.float32)
    assert np.allclose(expand_quad(q, 1.0, 1.0), q, atol=1e-3)
    big = expand_quad(q, 1.1, 1.1)
    assert cv2.contourArea(big) > cv2.contourArea(q) * 1.15


def test_verifier_prefers_true_card_even_with_glare(card, other_card):
    refs = [card, other_card]
    verifier = CandidateVerifier(lambda i: refs[i])
    rng = np.random.default_rng(1)
    photo = add_glare(card, None, rng, kind='streak', strength=1.0)
    photo = cv2.GaussianBlur(photo, (3, 3), 0)
    clean, mask = remove_glare(photo)
    q = verifier.prepare_query(clean, mask)
    s_true = verifier.score(q, 0)
    s_false = verifier.score(q, 1)
    assert s_true['verify'] > s_false['verify'] + 0.2
    # upside-down photo is recognised through the 180 degree hypothesis
    q180 = verifier.prepare_query(np.ascontiguousarray(clean[::-1, ::-1]), mask[::-1, ::-1].copy())
    assert verifier.score(q180, 0, rotate180=True)['verify'] > s_false['verify'] + 0.2


def test_masked_correlation_ignores_masked_pixels(card):
    f = correlation_features(card)
    corrupted = f.copy()
    corrupted[:40] = 0
    valid = np.ones(f.shape[:2], bool)
    valid[:40] = False
    assert masked_correlation(corrupted, f, valid) > 0.95
