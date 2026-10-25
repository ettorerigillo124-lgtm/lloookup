"""Tracking semplice per centro + macchina a stati con isteresi per ogni oggetto."""
import numpy as np

VISIBILE, IN_MANO, VICINO_SACCHETTO, OCCULTATO = "VISIBILE", "IN_MANO", "VICINO_SACCHETTO", "OCCULTATO"

EVENT_TEXT = {
    IN_MANO: "presa in mano",
    VICINO_SACCHETTO: "vicino al sacchetto",
    OCCULTATO: "occultata",
    VISIBILE: "di nuovo visibile",
}


def box_point_dist(box, pts):
    """Distanza minima tra una bbox e un insieme di punti (0 se un punto e' dentro)."""
    x1, y1, x2, y2 = box
    dx = np.maximum(np.maximum(x1 - pts[:, 0], 0), pts[:, 0] - x2)
    dy = np.maximum(np.maximum(y1 - pts[:, 1], 0), pts[:, 1] - y2)
    return float(np.min(np.hypot(dx, dy)))


def box_box_dist(a, b):
    dx = max(b[0] - a[2], a[0] - b[2], 0)
    dy = max(b[1] - a[3], a[1] - b[3], 0)
    return float(np.hypot(dx, dy))


class Track:
    _next = 1

    def __init__(self, det, t):
        self.id = Track._next
        Track._next += 1
        self.cls = det.cls
        self.box = det.box.copy()
        self.conf = det.conf
        self.state = VISIBILE
        self.last_seen = t
        self.missing = 0          # frame di detection consecutivi senza questo oggetto
        self.hand_score = 0       # contatori con isteresi
        self.bag_score = 0
        self.armed_until = -1.0   # fino a quando una sparizione conta come occultamento
        self.armed_by_bag = False
        self.seen_streak = 0
        self.hits = 1
        self.hold_pos = None      # ultima posizione nota della mano che teneva l'oggetto


class StateMachine:
    def __init__(self, cfg, bag_classes):
        self.s = cfg["state"]
        self.tr = cfg["tracking"]
        self.bag_classes = set(bag_classes)
        self.tracks = []
        self.events = []          # (t, testo)
        self.alert = None         # testo dell'allarme attivo (latched fino al reset)
        self.bags = []
        self._bag_mem = []
        self.t = 0.0
        self.step = 1             # ogni quanti frame gira la detection (main.py lo aggiorna)
        self.stamp = lambda t: f"{t:6.1f}s"   # main.py lo sostituisce (ora reale o tempo video)

    def reset(self, t):
        self.tracks.clear()
        self.alert = None
        self.events.append((self.stamp(t), "reset operatore"))

    def _emit(self, t, tr, state):
        self.events.append((self.stamp(t), f"{tr.cls} {EVENT_TEXT[state]}"))

    def _set(self, t, tr, state):
        if tr.state != state:
            tr.state = state
            self._emit(t, tr, state)
            if state == OCCULTATO and self.alert is None:
                self.alert = tr.cls
            elif state == OCCULTATO:
                if tr.cls not in self.alert.split(", "):
                    self.alert += f", {tr.cls}"

    def _match(self, dets, t):
        """Associa le detection alle tracce della stessa classe per distanza dei centri."""
        unmatched = list(range(len(dets)))
        pairs = []
        for ti, tr in enumerate(self.tracks):
            for di in unmatched:
                d = dets[di]
                if d.cls != tr.cls:
                    continue
                c1 = (tr.box[:2] + tr.box[2:]) / 2
                c2 = (d.box[:2] + d.box[2:]) / 2
                diag = np.hypot(*(tr.box[2:] - tr.box[:2])) + np.hypot(*(d.box[2:] - d.box[:2]))
                dist = np.hypot(*(c1 - c2)) / max(diag / 2, 1.0)
                pairs.append((dist, ti, di))
        used_t, matched = set(), {}
        for dist, ti, di in sorted(pairs):
            if ti in used_t or di in matched.values():
                continue
            # tracce "armate" (in mano / vicino al sacchetto) accettano un'associazione piu' larga
            lim = self.tr["match_dist"] * (3 if self.tracks[ti].state != VISIBILE else 1)
            if dist <= lim:
                used_t.add(ti)
                matched[ti] = di
        return matched

    def visible_tracks(self):
        """Tracce da mostrare: confermate e viste di recente, oppure in uno stato rilevante."""
        out = []
        for tr in self.tracks:
            if tr.hits < self.tr["min_hits"]:
                continue
            if tr.state != VISIBILE or self.t - tr.last_seen <= self.tr["show_stale_s"]:
                out.append(tr)
        return out

    def _update_bags(self, t, dets):
        for d in dets:
            if d.cls in self.bag_classes:
                self._bag_mem = [(bt, b) for bt, b in self._bag_mem if iou_like(b.box, d.box) < 0.3]
                self._bag_mem.append((t, d))
        self._bag_mem = [(bt, b) for bt, b in self._bag_mem if t - bt <= self.tr["bag_memory_s"]]
        self.bags = [b for _, b in self._bag_mem]

    def _near_bag(self, box, scale):
        return any(box_box_dist(box, b.box) <= self.s["bag_dist_px"] * scale for b in self.bags)

    def update(self, t, dets, hands, scale, sides=None):
        """
        Chiamato solo sui frame dove gira la detection.
        t: secondi. dets: Det (incluse le borse). hands: liste di punti (21,2). sides: lato di ogni mano.
        scale: lato_lungo/640 (le distanze in config sono riferite a 640 px).
        """
        # le soglie in config sono in frame video: con detection a frame alterni si scalano
        s = {k: (max(1, round(v / self.step)) if k.endswith("_frames") else v) for k, v in self.s.items()}
        self.t = t
        sides = sides or ["?"] * len(hands)
        self._update_bags(t, dets)
        objs = [d for d in dets if d.cls not in self.bag_classes]
        matched = self._match(objs, t)
        matched_dets = set(matched.values())
        for ti, tr in enumerate(self.tracks):
            if ti in matched:
                d = objs[matched[ti]]
                tr.box = 0.5 * tr.box + 0.5 * d.box   # smoothing della bbox
                tr.conf = d.conf
                tr.last_seen = t
                tr.missing = 0
                tr.seen_streak += 1
                tr.hits += 1
            else:
                tr.missing += 1
                tr.seen_streak = 0
        for di, d in enumerate(objs):
            if di not in matched_dets:
                self.tracks.append(Track(d, t))

        hand_boxes = [h_box(h) for h in hands]
        for tr in self.tracks:
            if tr.hits < self.tr["min_hits"]:
                continue
            seen = tr.missing == 0
            if tr.state == OCCULTATO:
                # l'allarme resta; se l'oggetto torna stabilmente visibile la traccia torna VISIBILE (log)
                if seen and tr.seen_streak >= s["reappear_frames"] * 2:
                    tr.state = VISIBILE
                    tr.hand_score = tr.bag_score = 0
                    tr.armed_until = -1
                    tr.armed_by_bag = False
                    self._emit(t, tr, VISIBILE)
                continue

            if seen:
                dists = [box_point_dist(tr.box, h) for h in hands]
                close = [i for i, dd in enumerate(dists) if dd <= s["hand_dist_px"] * scale]
                near_hand = bool(close)
                if near_hand:
                    hb = hand_boxes[min(close, key=lambda i: dists[i])]
                    tr.hold_pos = (hb[:2] + hb[2:]) / 2
                near_bag = self._near_bag(tr.box, scale) or any(
                    self._near_bag(hand_boxes[i], scale) for i in close)
                tr.hand_score = min(tr.hand_score + 1, s["in_hand_frames"] + s["release_frames"]) \
                    if near_hand else max(tr.hand_score - 1, 0)
                tr.bag_score = min(tr.bag_score + 1, s["near_bag_frames"] + s["bag_release_frames"]) \
                    if (near_bag and tr.state in (IN_MANO, VICINO_SACCHETTO)) else max(tr.bag_score - 1, 0)

                if tr.state == VISIBILE and tr.hand_score >= s["in_hand_frames"]:
                    self._set(t, tr, IN_MANO)
                if tr.state == IN_MANO:
                    if tr.bag_score >= s["near_bag_frames"]:
                        self._set(t, tr, VICINO_SACCHETTO)
                    elif tr.hand_score == 0:
                        self._set(t, tr, VISIBILE)
                elif tr.state == VICINO_SACCHETTO and tr.bag_score == 0:
                    self._set(t, tr, IN_MANO if tr.hand_score > 0 else VISIBILE)

                if tr.state in (IN_MANO, VICINO_SACCHETTO):
                    tr.armed_until = t + s["armed_window_s"]
                    tr.armed_by_bag = tr.state == VICINO_SACCHETTO
            else:
                if t > tr.armed_until:
                    continue
                # l'oggetto non si vede: seguiamo per posizione la mano che lo teneva (l'etichetta
                # Left/Right di MediaPipe non e' affidabile) e vediamo se arriva al sacchetto
                if s["hand_to_bag"] and tr.hold_pos is not None and hand_boxes:
                    centers = [(hb[:2] + hb[2:]) / 2 for hb in hand_boxes]
                    j = min(range(len(centers)), key=lambda i: np.hypot(*(centers[i] - tr.hold_pos)))
                    if np.hypot(*(centers[j] - tr.hold_pos)) <= s["hand_follow_px"] * scale:
                        tr.hold_pos = centers[j]
                        if not tr.armed_by_bag and self._near_bag(hand_boxes[j], scale):
                            tr.armed_by_bag = True
                            self._set(t, tr, VICINO_SACCHETTO)
                armed = tr.armed_by_bag or not s["require_bag"]
                if armed and tr.missing >= s["missing_frames"]:
                    self._set(t, tr, OCCULTATO)

        # pulizia: tracce perse da troppo tempo e non armate
        self.tracks = [tr for tr in self.tracks
                       if tr.state == OCCULTATO or t - tr.last_seen <= self.tr["max_age_s"]
                       or t <= tr.armed_until]


def iou_like(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / small if small > 0 else 0.0


def h_box(pts):
    return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
