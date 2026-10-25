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


class StateMachine:
    def __init__(self, cfg, bag_classes):
        self.s = cfg["state"]
        self.tr = cfg["tracking"]
        self.bag_classes = set(bag_classes)
        self.tracks = []
        self.events = []          # (t, testo)
        self.alert = None         # testo dell'allarme attivo (latched fino al reset)
        self.bags = []
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

    def update(self, t, dets, hands, scale):
        """
        Chiamato solo sui frame dove gira la detection.
        t: secondi. dets: Det (incluse le borse). hands: liste di punti (21,2). scale: lato_lungo/640.
        """
        s = self.s
        self.bags = [d for d in dets if d.cls in self.bag_classes]
        objs = [d for d in dets if d.cls not in self.bag_classes]
        matched = self._match(objs, t)
        matched_dets = set(matched.values())
        for di, d in enumerate(objs):
            if di not in matched_dets:
                self.tracks.append(Track(d, t))

        hand_pts = list(hands)
        for ti, tr in enumerate(self.tracks):
            if ti in matched:
                d = objs[matched[ti]]
                tr.box = 0.5 * tr.box + 0.5 * d.box   # smoothing della bbox
                tr.conf = d.conf
                tr.last_seen = t
                tr.missing = 0
                tr.seen_streak += 1
                tr.hits += 1
            elif tr.hits > 0 and tr.last_seen < t:
                tr.missing += 1
                tr.seen_streak = 0

        for tr in self.tracks:
            seen = tr.missing == 0
            if tr.state == OCCULTATO:
                # l'allarme resta; se l'oggetto torna stabilmente visibile la traccia torna VISIBILE (log)
                if seen and tr.seen_streak >= s["reappear_frames"] * 2:
                    tr.state = VISIBILE
                    tr.hand_score = tr.bag_score = 0
                    self._emit(t, tr, VISIBILE)
                continue

            if seen:
                near_hand = any(box_point_dist(tr.box, h) <= s["hand_dist_px"] * scale for h in hand_pts)
                ref = [tr.box] + [h_box(h) for h in hand_pts
                                  if box_point_dist(tr.box, h) <= s["hand_dist_px"] * scale]
                near_bag = any(box_box_dist(r, b.box) <= s["bag_dist_px"] * scale
                               for r in ref for b in self.bags)
                tr.hand_score = min(tr.hand_score + 1, s["in_hand_frames"] + s["release_frames"]) if near_hand \
                    else max(tr.hand_score - 1, 0)
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
                    tr.armed_by_bag = tr.state == VICINO_SACCHETTO or (
                        tr.armed_by_bag and tr.armed_until > t)
            else:
                armed = t <= tr.armed_until and (tr.armed_by_bag or not s["require_bag"])
                if armed and tr.missing >= s["missing_frames"]:
                    self._set(t, tr, OCCULTATO)

        # pulizia: tracce perse da troppo tempo e non armate
        self.tracks = [tr for tr in self.tracks
                       if tr.state == OCCULTATO or t - tr.last_seen <= self.tr["max_age_s"]
                       or t <= tr.armed_until]


def h_box(pts):
    return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
