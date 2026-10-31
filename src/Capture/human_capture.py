"""Cattura video e tracking di posa/mani con MediaPipe.

Contenuto:
  - VideoIO         : ingresso (webcam o file) + registrazione opzionale
  - OneEuroFilter   : filtro adattivo anti-jitter / anti-lag
  - PoseTracker     : detection pose + mani -> dict `state` (nessuna GUI)
  - draw_overlay    : disegna landmark e angoli su un frame
  - Skeleton3DPlot  : grafico matplotlib 3D live (solo debug, lento)

Convenzioni MediaPipe:
  - world landmarks in METRI, origine al centro del bacino, y verso il basso;
  - "left"/"right" sono riferiti al soggetto, non all'immagine.
"""
import os
import re
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Topologia scheletro (indici MediaPipe Pose, 33 landmark) -----------------
BONES = {
    'head':  [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10)],
    'torso': [(11, 12), (12, 24), (24, 23), (23, 11)],
    'r_arm': [(12, 14), (14, 16), (16, 18), (16, 20), (16, 22)],
    'l_arm': [(11, 13), (13, 15), (15, 17), (15, 19), (15, 21)],
}
COLORS = {'head': 'yellow', 'torso': 'blue', 'r_arm': 'red', 'l_arm': 'cyan', 'hand': 'green'}
BONE_IDXS = {k: sorted({i for pair in v for i in pair}) for k, v in BONES.items()}


# =============================================================================
# Ingresso / uscita video
# =============================================================================
class VideoIO:
    """Sorgente video (indice webcam o path file) con registrazione opzionale.

    Timestamp restituiti in ms, strettamente crescenti (richiesto da MediaPipe VIDEO mode):
      - file   : dal numero di frame / fps (riproducibile, indipendente dalla velocità di calcolo)
      - webcam : dal tempo reale trascorso dal primo frame
    In registrazione da webcam i frame vengono duplicati/saltati per avere fps costante nel file.
    """

    def __init__(self, src=0, out_dir="./recording", width=640, height=480, fps=30, rec=False):
        self.is_file = isinstance(src, str)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Impossibile aprire la sorgente video: {src!r}")

        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or fps

        self.rec = bool(rec and out_dir)
        self.out_dir = out_dir
        self.writer = None
        self.frames_read = self.frames_written = 0
        self._t0 = None  # istante del primo frame letto

    def read(self):
        """Ritorna (ok, frame, timestamp_ms)."""
        ok, frame = self.cap.read()
        if not ok:
            return False, None, 0
        if self._t0 is None:
            self._t0 = time.time()
        self.frames_read += 1
        if self.is_file:
            ts = int(self.frames_read / self.fps * 1000)
        else:
            ts = int((time.time() - self._t0) * 1000)
        return True, frame, ts

    def _open_writer(self, frame):
        # Dimensione presa dal frame reale: cap.get() può non coincidere con quanto la camera consegna
        os.makedirs(self.out_dir, exist_ok=True)
        ids = [int(m.group(1)) for f in os.listdir(self.out_dir)
               if (m := re.fullmatch(r"Cattura_n(\d+)\.mp4", f))]
        path = os.path.join(self.out_dir, f"Cattura_n{max(ids, default=0) + 1}.mp4")
        h, w = frame.shape[:2]
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))
        print(f"[rec] scrivo su {path}")

    def write(self, frame):
        if not self.rec:
            return
        if self.writer is None:
            self._open_writer(frame)
        if self.is_file:
            self.writer.write(frame)
            self.frames_written += 1
        else:
            target = int((time.time() - self._t0) * self.fps)
            while self.frames_written < target:
                self.writer.write(frame)
                self.frames_written += 1

    def release(self):
        self.cap.release()
        if self.writer:
            self.writer.release()


# =============================================================================
# Filtro
# =============================================================================
class OneEuroFilter:
    """One Euro Filter (Casiez et al., 2012), vettoriale: ogni componente ha il suo cutoff.

    min_cutoff [Hz] : più basso = meno jitter da fermo, ma più lag.
    beta            : quanto il cutoff cresce con la velocità (meno lag nei movimenti rapidi).
    d_cutoff [Hz]   : cutoff del filtro sulla derivata.

    ATTENZIONE alle unità: beta moltiplica una velocità. Con dati in metri (come le world
    landmarks) le velocità sono ~1 m/s, quindi beta ~0.01 non ha praticamente effetto.
    Valori da cui partire per il tuning: min_cutoff ~1, beta ~1-5 (da verificare a occhio).
    """

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        """Dimentica lo stato: il prossimo campione passa invariato (da usare quando il tracking si perde)."""
        self.x_prev = self.dx_prev = None

    def __call__(self, x, dt=1 / 30.0):
        if self.x_prev is None:
            self.x_prev, self.dx_prev = x, np.zeros_like(x)
            return x
        dx = (x - self.x_prev) / dt
        a_d = 1.0 / (1.0 + (1.0 / (2 * np.pi * self.d_cutoff)) / dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = 1.0 / (1.0 + (1.0 / (2 * np.pi * cutoff)) / dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat


# =============================================================================
# Tracking
# =============================================================================
class PoseTracker:
    """Rileva posa e mani e restituisce uno `state` (dict). Non apre finestre e non disegna.

    state = {
        'pose':       (33, 3) world landmarks filtrate [m], oppure None
        'visibility': (33,)   visibilità per landmark in [0, 1], oppure None
        'pose_2d':    landmark normalizzati MediaPipe (per overlay/plot), oppure None
        'angles':     {'l_elbow': deg, 'r_elbow': deg}
        'hands':      [{'side': 'left'|'right', 'pts': (21, 3) [m],
                        'pinch': 0..1 (0 = pollice-indice a contatto, 1 = aperta),
                        'openness': 0..1 (0 = pugno, 1 = mano aperta)}]
        'hands_2d':   landmark normalizzati delle mani, stesso ordine di 'hands'
    }
    """

    def __init__(self, models_dir="models", pose_model="pose_landmarker_lite.task",
                 use_pose=True, use_hands=True, min_cutoff=0.5, beta=0.01):
        md = os.path.abspath(models_dir)
        self.pose_det = self.hand_det = None

        if use_pose:
            opt = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=os.path.join(md, pose_model)),
                running_mode=vision.RunningMode.VIDEO)
            self.pose_det = vision.PoseLandmarker.create_from_options(opt)

        if use_hands:
            opt = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=os.path.join(md, "hand_landmarker.task")),
                running_mode=vision.RunningMode.VIDEO, num_hands=2)
            self.hand_det = vision.HandLandmarker.create_from_options(opt)

        self.pose_filter = OneEuroFilter(min_cutoff, beta)
        self.hand_filters = {s: OneEuroFilter(min_cutoff, beta) for s in ('left', 'right')}
        self._last_ts = -1

    # ---- API pubblica ------------------------------------------------------
    def process(self, frame, ts):
        """frame BGR + timestamp [ms] -> state."""
        state = {'pose': None, 'visibility': None, 'pose_2d': None,
                 'angles': {'l_elbow': 0.0, 'r_elbow': 0.0}, 'hands': [], 'hands_2d': []}
        if frame is None:
            return state

        # MediaPipe VIDEO richiede timestamp strettamente crescenti
        ts = max(int(ts), self._last_ts + 1)
        dt = 1 / 30.0 if self._last_ts < 0 else float(np.clip((ts - self._last_ts) / 1000.0, 1e-3, 0.2))
        self._last_ts = ts

        mpi = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        p_res = self.pose_det.detect_for_video(mpi, ts) if self.pose_det else None
        h_res = self.hand_det.detect_for_video(mpi, ts) if self.hand_det else None

        pts = self._process_pose(p_res, dt, state)
        self._process_hands(h_res, pts, dt, state)
        return state

    def close(self):
        if self.pose_det:
            self.pose_det.close()
        if self.hand_det:
            self.hand_det.close()

    # ---- Interni -----------------------------------------------------------
    def _process_pose(self, p_res, dt, state):
        if not (p_res and p_res.pose_world_landmarks):
            self.pose_filter.reset()  # al ritorno non si trascina uno stato vecchio
            return None
        world = p_res.pose_world_landmarks[0]
        pts = self.pose_filter(np.array([[lm.x, lm.y, lm.z] for lm in world]), dt)
        vis = np.array([1.0 if lm.visibility is None else lm.visibility for lm in world])
        state.update(pose=pts, visibility=vis, pose_2d=p_res.pose_landmarks[0],
                     angles={'l_elbow': self._angle(pts, 11, 13, 15),
                             'r_elbow': self._angle(pts, 12, 14, 16)})
        return pts

    def _process_hands(self, h_res, pts, dt, state):
        seen = set()
        if h_res and h_res.hand_world_landmarks:
            state['hands_2d'] = list(h_res.hand_landmarks)
            for i, (side, wrist) in enumerate(self._assign_sides(h_res, state['pose_2d'])):
                local = np.array([[l.x, l.y, l.z] for l in h_res.hand_world_landmarks[i]])
                # Le world landmarks della mano sono centrate sul suo polso: le si traslano sul polso della posa
                h_pts = local + (pts[wrist] - local[0]) if (pts is not None and wrist is not None) else local
                h_pts = self.hand_filters[side](h_pts, dt)
                state['hands'].append({'side': side, 'pts': h_pts,
                                       'pinch': self._pinch(h_pts),
                                       'openness': self._openness(h_pts)})
                seen.add(side)
        for side, f in self.hand_filters.items():
            if side not in seen:
                f.reset()

    @staticmethod
    def _assign_sides(h_res, p2d):
        """Per ogni mano ritorna (side, indice_polso_posa).

        Con la posa: ogni mano va al polso (15 sx, 16 dx) più vicino in 2D, senza usare lo stesso
        polso due volte. Senza posa: si ripiega sull'handedness di MediaPipe (che assume immagine
        specchiata: può risultare invertita).
        """
        n = len(h_res.hand_landmarks)
        if p2d is None:
            return [(h_res.handedness[i][0].category_name.lower(), None) for i in range(n)]
        pairs = sorted((np.hypot(h[0].x - p2d[w].x, h[0].y - p2d[w].y), i, w)
                       for i, h in enumerate(h_res.hand_landmarks) for w in (15, 16))
        chosen, used = {}, set()
        for _, i, w in pairs:
            if i not in chosen and w not in used:
                chosen[i] = w
                used.add(w)
        return [('left' if chosen[i] == 15 else 'right', chosen[i]) for i in range(n)]

    @staticmethod
    def _angle(pts, a, b, c):
        """Angolo in gradi in `b` tra i segmenti b->a e b->c."""
        v1, v2 = pts[a] - pts[b], pts[c] - pts[b]
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    @staticmethod
    def _pinch(pts):
        """Distanza pollice-indice normalizzata: 0 = a contatto, 1 = aperta (soglie 2 cm / 12 cm)."""
        return float(np.clip((np.linalg.norm(pts[4] - pts[8]) - 0.02) / 0.1, 0, 1))

    @staticmethod
    def _openness(pts, min_r=0.3, max_r=1.5):
        """Apertura della mano: distanza punta-nocca delle 4 dita, normalizzata sulla larghezza del palmo."""
        palm = np.linalg.norm(pts[5] - pts[17])
        if palm == 0:
            return 0.0
        ratios = [np.linalg.norm(pts[t] - pts[m]) / palm for t, m in [(8, 5), (12, 9), (16, 13), (20, 17)]]
        return float(np.clip((np.mean(ratios) - min_r) / (max_r - min_r), 0.0, 1.0))


# =============================================================================
# Visualizzazione
# =============================================================================
def draw_overlay(frame, state):
    """Ritorna una COPIA di `frame` con landmark, angoli dei gomiti e apertura delle mani."""
    out = frame.copy()
    h, w = out.shape[:2]

    p2d = state['pose_2d']
    if p2d is not None:
        for lm in p2d:
            cv2.circle(out, (int(lm.x * w), int(lm.y * h)), 4, (255, 255, 0), -1)
        a = state['angles']
        cv2.putText(out, f"L:{int(a['l_elbow'])} R:{int(a['r_elbow'])}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for hand_2d, hand in zip(state['hands_2d'], state['hands']):
        for lm in hand_2d:
            cv2.circle(out, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)
        cv2.putText(out, f"{hand['side']}: {hand['openness']:.2f}",
                    (int(hand_2d[0].x * w), int(hand_2d[0].y * h) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return out


class Skeleton3DPlot:
    """Scheletro 3D live con matplotlib. Lento (plt.pause): solo per debug.

    La profondità assoluta NON è nota (le world landmarks sono relative al bacino): per avere un
    riferimento nel grafico la si stima con un modello pinhole, Z = f * W / w_px, dove W è la larghezza
    spalle reale e w_px quella misurata nell'immagine. FOCAL_PX dipende da camera e risoluzione.
    """
    SHOULDER_WIDTH_M = 0.4
    FOCAL_PX = 800.0

    def __init__(self, every=3):
        import matplotlib.pyplot as plt  # import locale: matplotlib serve solo con --plot3d
        plt.ion()
        self.plt = plt
        self.ax = plt.figure(figsize=(9, 9)).add_subplot(111, projection='3d')
        self.every, self._n = every, 0

    def update(self, state, frame_shape):
        pts, p2d = state['pose'], state['pose_2d']
        if pts is None:
            return
        self._n += 1
        if self._n % self.every:
            return

        h, w = frame_shape[:2]
        shoulder_px = max(np.hypot((p2d[11].x - p2d[12].x) * w, (p2d[11].y - p2d[12].y) * h), 1e-5)
        z_off = self.SHOULDER_WIDTH_M * self.FOCAL_PX / shoulder_px

        def to_plot(p):  # MediaPipe (x, y, z) -> assi del grafico (X, Z, Y), con Y verso l'alto
            return np.column_stack((p[:, 0], p[:, 2] + z_off, -p[:, 1]))

        ax = self.ax
        ax.clear()
        ax.set(xlim=[-0.6, 0.6], ylim=[z_off - 0.6, z_off + 0.6], zlim=[-0.6, 0.6],
               xlabel='X', ylabel='Z', zlabel='Y')
        p3 = to_plot(pts)
        for k, idx in BONE_IDXS.items():
            ax.scatter(p3[idx, 0], p3[idx, 1], p3[idx, 2], c=COLORS[k], s=25)
            for a, b in BONES[k]:
                ax.plot(p3[[a, b], 0], p3[[a, b], 1], p3[[a, b], 2], c=COLORS[k], lw=2)
        for hand in state['hands']:
            h3 = to_plot(hand['pts'])
            ax.scatter(h3[:, 0], h3[:, 1], h3[:, 2], c=COLORS['hand'], s=20)
        self.plt.pause(0.001)