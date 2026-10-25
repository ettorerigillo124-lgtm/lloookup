"""Disegno: bbox per stato, persona in rosso in allarme, pannello eventi, FPS."""
import time

import cv2
import numpy as np

from state_machine import IN_MANO, OCCULTATO, VICINO_SACCHETTO, VISIBILE

# BGR
COLORS = {
    VISIBILE: (80, 200, 60),
    IN_MANO: (0, 215, 255),
    VICINO_SACCHETTO: (0, 140, 255),
    OCCULTATO: (40, 40, 230),
}
LABELS = {VISIBILE: "visibile", IN_MANO: "in mano", VICINO_SACCHETTO: "vicino sacchetto", OCCULTATO: "OCCULTATO"}
BAG_COLOR = (200, 160, 60)
RED = (40, 40, 230)
PANEL_BG = (32, 28, 26)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, txt, org, scale=0.5, color=(255, 255, 255), thick=1):
    cv2.putText(img, txt, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, FONT, scale, color, thick, cv2.LINE_AA)


def _label(img, txt, x, y, color, scale):
    (tw, th), _ = cv2.getTextSize(txt, FONT, scale, 1)
    y = max(th + 6, int(y))
    cv2.rectangle(img, (int(x), y - th - 6), (int(x) + tw + 8, y), color, -1)
    cv2.putText(img, txt, (int(x) + 4, y - 4), FONT, scale, (20, 20, 20), 1, cv2.LINE_AA)


def draw_scene(img, tracks, bags, hands, person_mask, alert, s, show_hands=True):
    """Disegna sul frame gia' anonimizzato. s = fattore di scala del testo."""
    if alert and person_mask is not None and person_mask.any():
        red = np.zeros_like(img)
        red[:] = RED
        m = person_mask > 0
        img[m] = cv2.addWeighted(img, 0.45, red, 0.55, 0)[m]
        cnts, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, RED, max(3, int(6 * s)), cv2.LINE_AA)
    elif alert:  # nessuna maschera: bordo rosso su tutto il frame
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), RED, max(6, int(12 * s)))

    if show_hands:
        for h in hands:
            for x, y in h[::4]:
                cv2.circle(img, (int(x), int(y)), max(2, int(3 * s)), (255, 220, 120), -1, cv2.LINE_AA)

    for b in bags:
        x1, y1, x2, y2 = b.box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), BAG_COLOR, max(1, int(2 * s)), cv2.LINE_AA)
        _label(img, f"sacchetto {b.conf:.2f}", x1, y1, BAG_COLOR, 0.45 * s)

    for tr in tracks:
        c = COLORS[tr.state]
        x1, y1, x2, y2 = tr.box.astype(int)
        dashed = tr.missing > 0
        th = max(2, int(3 * s))
        if dashed:  # oggetto non visto in questo frame: posizione stimata
            for i in range(x1, x2, 12):
                cv2.line(img, (i, y1), (min(i + 6, x2), y1), c, th)
                cv2.line(img, (i, y2), (min(i + 6, x2), y2), c, th)
            for i in range(y1, y2, 12):
                cv2.line(img, (x1, i), (x1, min(i + 6, y2)), c, th)
                cv2.line(img, (x2, i), (x2, min(i + 6, y2)), c, th)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), c, th, cv2.LINE_AA)
        _label(img, f"{tr.cls} | {LABELS[tr.state]}", x1, y1, c, 0.5 * s)

    if alert:
        h, w = img.shape[:2]
        bh = int(48 * s)
        cv2.rectangle(img, (0, 0), (w, bh), RED, -1)
        _text(img, f"ALERT: occultamento rilevato - {alert}", (int(12 * s), int(bh * 0.68)),
              0.75 * s, (255, 255, 255), max(1, int(2 * s)))


def compose(video, events, fps, cfg, alert, mode, info=""):
    """Affianca il pannello eventi al video e scala alla larghezza della finestra."""
    d = cfg["display"]
    pw = d["panel_width"]
    target_w = d["width"]
    vh, vw = video.shape[:2]
    scale = (target_w - pw) / vw
    vid = cv2.resize(video, (int(vw * scale), int(vh * scale)))
    H = vid.shape[0]
    panel = np.full((H, pw, 3), PANEL_BG, np.uint8)

    _text(panel, "LookUp", (16, 34), 0.85, (255, 255, 255), 2)
    _text(panel, "demo anti-occultamento", (16, 58), 0.45, (180, 180, 180))
    status, col = ("ALLARME", RED) if alert else ("monitoraggio", COLORS[VISIBILE])
    cv2.circle(panel, (22, 86), 7, col, -1, cv2.LINE_AA)
    _text(panel, status, (36, 91), 0.55, col, 1)

    y = 124
    cv2.line(panel, (16, y - 12), (pw - 16, y - 12), (80, 80, 80), 1)
    _text(panel, "EVENTI", (16, y + 4), 0.45, (160, 160, 160))
    y += 28
    for t, txt in events[-d["log_lines"]:][::-1]:
        stamp = t if isinstance(t, str) else "--:--:--"
        color = (120, 120, 255) if "occultat" in txt else (230, 230, 230)
        _text(panel, stamp, (16, y), 0.45, (150, 150, 150))
        _text(panel, txt, (100, y), 0.45, color)
        y += 24
        if y > H - 80:
            break

    cv2.line(panel, (16, H - 64), (pw - 16, H - 64), (80, 80, 80), 1)
    priv = {"mask": "persona oscurata", "full": "frame oscurato (fail-safe)", "none": "nessuna persona"}[mode]
    _text(panel, f"privacy: {priv}", (16, H - 42), 0.42, (170, 220, 170))
    _text(panel, "r reset   f schermo intero   q esci", (16, H - 18), 0.4, (140, 140, 140))

    out = np.hstack([vid, panel])
    _text(out, f"{fps:4.1f} FPS {info}", (out.shape[1] - pw - 150, H - 14), 0.5, (255, 255, 255))
    return out


class FpsMeter:
    def __init__(self, alpha=0.1):
        self.t = time.perf_counter()
        self.fps = 0.0
        self.alpha = alpha

    def tick(self):
        now = time.perf_counter()
        dt, self.t = now - self.t, now
        inst = 1.0 / dt if dt > 0 else 0.0
        self.fps = inst if self.fps == 0 else (1 - self.alpha) * self.fps + self.alpha * inst
        return self.fps
