import cv2
import numpy as np
import pytest

from card_detection import (CARD_ASPECT, order_corners_portrait,
                            find_card_quads, quad_iou, rectified_aspect, refine_quad,
                            order_corners_clockwise)
from photo_synthesis import camera_homography, rounded_corner_mask
from mtg_layout import CARD_W, CARD_H


def render(card, bg_value=(40, 90, 40), size=(900, 700), pitch=0.0, yaw=0.0, roll=0.0,
           height_frac=0.6, center=None, texture=False, seed=0):
    rng = np.random.default_rng(seed)
    w, h = size
    scene = np.empty((h, w, 3), np.uint8)
    scene[:] = bg_value
    if texture:  # busy background: blobs of random colour
        for _ in range(80):
            cv2.circle(scene, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                       int(rng.integers(10, 60)), rng.integers(0, 255, 3).tolist(), -1)
    center = center or (w / 2, h / 2)
    H = camera_homography(CARD_W, CARD_H, w, h, height_frac, center, roll, pitch, yaw)
    warped = cv2.warpPerspective(card, H, (w, h))
    a = cv2.warpPerspective(rounded_corner_mask(CARD_H, CARD_W), H, (w, h)) > 127
    scene[a] = warped[a]
    corners = cv2.perspectiveTransform(
        np.array([[[0, 0], [CARD_W, 0], [CARD_W, CARD_H], [0, CARD_H]]], np.float32), H)[0]
    return scene, corners


def test_order_corners_portrait_landscape_card():
    # A card lying sideways: long side horizontal
    q = np.array([[100, 100], [100 + 88 * 3, 100], [100 + 88 * 3, 100 + 63 * 3], [100, 100 + 63 * 3]],
                 np.float32)
    o = order_corners_portrait(q[[2, 0, 3, 1]])
    top = np.linalg.norm(o[1] - o[0])
    left = np.linalg.norm(o[3] - o[0])
    assert top < left  # top edge is a short side -> portrait warp won't squash


@pytest.mark.parametrize('angle', [0, 30, 45, 60, 90, 135, 180, 270])
def test_order_corners_is_clockwise_and_portrait(angle):
    rect = cv2.boxPoints(((300, 300), (126, 176), angle)).astype(np.float32)
    o = order_corners_portrait(rect)
    # clockwise in image coordinates -> positive signed area with y down
    area = 0.5 * sum(o[i, 0] * o[(i + 1) % 4, 1] - o[(i + 1) % 4, 0] * o[i, 1] for i in range(4))
    assert area > 0
    assert np.linalg.norm(o[1] - o[0]) < np.linalg.norm(o[3] - o[0])


@pytest.mark.parametrize('pitch,yaw', [(0, 0), (35, 20), (50, -25), (-40, 30)])
def test_rectified_aspect_recovers_card_shape(pitch, yaw):
    _, corners = render(np.zeros((CARD_H, CARD_W, 3), np.uint8), pitch=pitch, yaw=yaw, roll=17)
    r = rectified_aspect(order_corners_clockwise(corners), (700, 900))
    assert abs(r - CARD_ASPECT) < 0.06


@pytest.mark.parametrize('kwargs', [
    dict(),
    dict(bg_value=(235, 235, 235)),                    # white table
    dict(bg_value=(20, 20, 25), pitch=30, yaw=15),     # dark mat, perspective
    dict(texture=True, roll=40),                       # busy play mat
    dict(texture=True, pitch=45, yaw=-20, roll=100),   # busy + steep + sideways
])
def test_find_card_quads_locates_card(card, kwargs):
    scene, gt = render(card, **kwargs)
    quads = find_card_quads(scene)
    assert quads, "no card found"
    best = max(quad_iou(q.corners, gt) for q in [quads[0]] + quads[0].alternatives)
    assert best > 0.9


def test_find_card_quads_multiple_cards(card, other_card):
    scene, gt1 = render(card, size=(1200, 700), height_frac=0.5, center=(330, 350), bg_value=(60, 40, 30))
    s2, gt2 = render(other_card, size=(1200, 700), height_frac=0.5, center=(870, 350), roll=25)
    mask = np.zeros(scene.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, gt2.astype(np.int32), 1)
    scene[mask > 0] = s2[mask > 0]
    quads = find_card_quads(scene)
    for gt in (gt1, gt2):
        assert max(quad_iou(q.corners, gt) for q in quads) > 0.9


def test_refine_quad_snaps_rough_box_to_edges(card):
    scene, gt = render(card, pitch=25, yaw=10, roll=12, bg_value=(200, 190, 170))
    rough = order_corners_clockwise(gt) + np.random.default_rng(0).normal(0, 6, (4, 2)).astype(np.float32)
    refined = refine_quad(scene, rough)
    err_before = np.abs(rough - order_corners_clockwise(gt)).mean()
    err_after = np.abs(order_corners_clockwise(refined) - order_corners_clockwise(gt)).mean()
    assert err_after < err_before * 0.6
