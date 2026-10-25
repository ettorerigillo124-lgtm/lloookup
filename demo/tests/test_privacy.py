"""Test delle regole privacy della demo: la persona non deve mai uscire in chiaro."""
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detector import Persons  # noqa: E402
from privacy import PrivacyGuard  # noqa: E402

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())


def textured(h=480, w=640, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def person_at(frame, box):
    x1, y1, x2, y2 = box
    m = np.zeros(frame.shape[:2], np.uint8)
    m[y1:y2, x1:x2] = 255
    return Persons(m, [np.array(box, float)])


def changed(a, b, box):
    x1, y1, x2, y2 = box
    return np.mean(a[y1:y2, x1:x2] != b[y1:y2, x1:x2])


def test_person_region_is_obscured():
    f = textured()
    box = (200, 100, 360, 460)
    out, mode = PrivacyGuard(CFG).apply(f, person_at(f, box))
    assert mode == "mask"
    assert changed(out, f, box) > 0.9           # quasi tutti i pixel della persona sono alterati
    assert changed(out, f, (0, 0, 120, 120)) == 0  # lo sfondo lontano resta intatto


def test_input_frame_not_modified():
    f = textured()
    copy = f.copy()
    PrivacyGuard(CFG).apply(f, person_at(f, (200, 100, 360, 460)))
    assert np.array_equal(f, copy)


def test_failsafe_full_frame_when_person_lost():
    f = textured()
    g = PrivacyGuard(CFG)
    g.apply(f, person_at(f, (200, 100, 360, 460)))
    out, mode = g.apply(f, Persons(None, []))     # segmentazione fallita nel frame successivo
    assert mode == "full"
    assert changed(out, f, (0, 0, 640, 480)) > 0.9


def test_failsafe_when_hands_without_person():
    f = textured()
    out, mode = PrivacyGuard(CFG).apply(f, Persons(None, []), hands=[np.zeros((21, 2))])
    assert mode == "full"


def test_face_always_obscured_even_if_object_revealed():
    f = textured()
    face = np.array([260, 120, 300, 170], float)
    # oggetto che si sovrappone al volto: non deve essere mostrato in chiaro
    out, _ = PrivacyGuard(CFG).apply(f, person_at(f, (200, 100, 360, 460)), faces=[face],
                                     reveal=[("pen", np.array([250, 130, 320, 160], float))])
    assert changed(out, f, (260, 120, 300, 170)) > 0.9


def test_object_pixels_revealed_away_from_head():
    f = textured()
    obj = (230, 330, 270, 360)  # in basso nella bbox persona (mano), lontano dalla testa
    out, _ = PrivacyGuard(CFG).apply(f, person_at(f, (200, 100, 360, 460)),
                                     reveal=[("pen", np.array(obj, float))])
    assert changed(out, f, obj) == 0


def test_never_reveal_classes_stay_obscured():
    f = textured()
    obj = (230, 330, 270, 360)
    out, _ = PrivacyGuard(CFG).apply(f, person_at(f, (200, 100, 360, 460)),
                                     reveal=[("eyeglasses", np.array(obj, float))])
    assert changed(out, f, obj) > 0.9


def test_no_person_no_blur_after_hold():
    f = textured()
    out, mode = PrivacyGuard(CFG).apply(f, Persons(None, []))
    assert mode == "none" and np.array_equal(out, f)
