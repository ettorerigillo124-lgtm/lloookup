"""Anonimizzazione: la persona e' SEMPRE oscurata. In caso di dubbio si oscura tutto il frame."""
import cv2
import numpy as np


def obscure(img, method="pixelate", block=16, kernel=99):
    """Versione irreversibilmente oscurata dell'intera immagine."""
    h, w = img.shape[:2]
    if method == "blur":
        k = kernel | 1
        small = cv2.resize(img, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
        return cv2.resize(cv2.GaussianBlur(small, (k // 4 | 1, k // 4 | 1), 0), (w, h))
    block = max(2, int(block))
    small = cv2.resize(img, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _clip_box(b, w, h, pad=0.0):
    x1, y1, x2, y2 = [float(v) for v in b]
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    return (int(max(0, x1 - px)), int(max(0, y1 - py)), int(min(w, x2 + px)), int(min(h, y2 + py)))


def _overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


class PrivacyGuard:
    """Applica l'oscuramento e ricorda lo stato tra un frame e l'altro (fail-safe)."""

    def __init__(self, cfg):
        self.p = cfg["privacy"]
        self.person = cfg["person"]
        self.frames_since_person = 10 ** 9
        self.prev_mask = None

    def block_size(self, frame, person_boxes):
        h, w = frame.shape[:2]
        sides = [min(b[2] - b[0], b[3] - b[1]) for b in person_boxes] or [min(h, w)]
        return max(8, int(max(sides) / self.p["pixel_blocks"]))

    def apply(self, frame, persons, faces=(), hands=(), reveal=()):
        """
        frame: BGR originale (non viene modificato).
        persons: detector.Persons. faces: bbox volti. hands: liste di landmark.
        reveal: [(cls, box)] oggetti i cui pixel reali possono stare sopra l'oscuramento.
        Restituisce (frame_anonimizzato, modalita') con modalita' in {"mask", "full", "none"}.
        """
        h, w = frame.shape[:2]
        p = self.p
        block = self.block_size(frame, persons.boxes)
        hidden = obscure(frame, p["method"], block, p["blur_kernel"])
        mask = persons.mask

        if mask is not None and mask.any():
            self.frames_since_person = 0
            k = int(self.person["mask_dilate_px"] * max(h, w) / 640) | 1
            m = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            # copre anche la posizione del frame precedente (movimento veloce, maschera in ritardo)
            if self.prev_mask is not None and self.prev_mask.shape == m.shape:
                m = cv2.bitwise_or(m, self.prev_mask)
            self.prev_mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            for b in persons.boxes:  # nessun pixel della bbox persona deve sfuggire nella zona testa
                x1, y1, x2, y2 = _clip_box(b, w, h)
                m[y1:y1 + int((y2 - y1) * p["head_zone_ratio"]), x1:x2] = 255
            out = np.where(m[..., None] > 0, hidden, frame)
            mode = "mask"
        else:
            self.frames_since_person += 1
            self.prev_mask = None
            recent = self.frames_since_person <= self.person["hold_frames"]
            if p["failsafe_full_frame"] and (recent or len(hands) or len(faces)):
                out, mode = hidden.copy(), "full"
            else:
                out, mode = frame.copy(), "none"

        if mode != "none" and p["show_object_pixels"]:
            head_zones = [_clip_box(f, w, h, p["face_pad"]) for f in faces]
            for b in persons.boxes:
                x1, y1, x2, y2 = _clip_box(b, w, h)
                head_zones.append((x1, y1, x2, y1 + int((y2 - y1) * p["head_zone_ratio"])))
            for cls, box in reveal:
                if cls in p["never_reveal"]:
                    continue
                b = _clip_box(box, w, h)
                if (b[2] - b[0]) * (b[3] - b[1]) > p["reveal_max_frac"] * w * h:
                    continue
                if any(_overlap(b, z) for z in head_zones):
                    continue
                out[b[1]:b[3], b[0]:b[2]] = frame[b[1]:b[3], b[0]:b[2]]

        # i volti rilevati restano sempre oscurati, qualunque cosa sia successa sopra
        for f in faces:
            x1, y1, x2, y2 = _clip_box(f, w, h, p["face_pad"])
            out[y1:y2, x1:x2] = hidden[y1:y2, x1:x2]
        return out, mode
