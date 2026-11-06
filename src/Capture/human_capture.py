"""Cattura video e tracking di posa/mani con MediaPipe.

Contenuto:
  - VideoIO         : ingresso (webcam o file) + registrazione opzionale
  - RealSenseIO     : ingresso Intel RealSense (RGB + depth allineata) con la stessa interfaccia di VideoIO
  - OneEuroFilter   : filtro adattivo anti-jitter / anti-lag
  - BodyLeveler     : compensa l'inclinazione della camera usando l'asse del busto (testa -> bacino)
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

# --- Asse del busto per la compensazione dell'inclinazione della camera -------
LEVEL_HEAD = (7, 8)     # orecchie; con (11, 12) si usano le spalle: meno sensibile al collo piegato
LEVEL_HIPS = (23, 24)   # fianchi
UP = np.array([0.0, -1.0, 0.0])  # "su" nella convenzione MediaPipe (y verso il basso)


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

    def __init__(self, src=1, out_dir="./recording", width=640, height=480, fps=30, rec=False):
        self.is_file = isinstance(src, str)
        print(f"[video] sorgente: {src!r} ({'file' if self.is_file else 'webcam'})")
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


class RealSenseIO(VideoIO):
    """Intel RealSense: RGB + depth allineata al colore. Stessa interfaccia di VideoIO.

    In più espone `points_3d(px, py)`, che converte pixel (u, v) in punti 3D [m] nel sistema della
    camera (x destra, y giù, z avanti) usando la depth misurata e le intrinseche del sensore colore.
    Si registra solo il video a colori (la depth non viene salvata).
    pyrealsense2 è importato qui: serve solo con --realsense.
    """

    def __init__(self, out_dir="./recording", width=640, height=480, fps=30, rec=False):
        import pyrealsense2 as rs
        print("[video] sorgente: RealSense")
        self.is_file = False  # per write(): si comporta come una webcam (fps costante nel file)
        self.fps = fps
        self.rec = bool(rec and out_dir)
        self.out_dir = out_dir
        self.writer = None
        self.frames_read = self.frames_written = 0
        self._t0 = None
        self._depth = None

        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.pipe = rs.pipeline()
        profile = self.pipe.start(cfg)  # solleva RuntimeError se non c'è nessuna camera
        self.align = rs.align(rs.stream.color)  # depth riproiettata nel frame del colore

        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()  # unità depth -> metri
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.fx, self.fy, self.ppx, self.ppy = intr.fx, intr.fy, intr.ppx, intr.ppy

        for _ in range(15):  # scarta i primi frame: l'auto-esposizione deve stabilizzarsi
            self.pipe.wait_for_frames()

    def read(self):
        """Ritorna (ok, frame BGR, timestamp_ms); la depth del frame resta disponibile per points_3d()."""
        try:
            frames = self.align.process(self.pipe.wait_for_frames(5000))
        except RuntimeError:  # timeout: camera scollegata o bloccata
            return False, None, 0
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            return False, None, 0

        # copy(): i buffer appartengono al frame RealSense e possono essere riutilizzati
        frame = np.asanyarray(color.get_data()).copy()
        self._depth = np.asanyarray(depth.get_data()).copy()  # uint16, 0 = nessuna misura

        if self._t0 is None:
            self._t0 = time.time()
        self.frames_read += 1
        return True, frame, int((time.time() - self._t0) * 1000)

    def points_3d(self, px, py, k=2):
        """Pixel -> punti 3D [m] nel frame camera. Ritorna (N, 3), con NaN dove la depth non è valida.

        La profondità è la mediana dei valori non nulli in una finestra (2k+1)x(2k+1): robusta ai buchi.
        """
        h, w = self._depth.shape
        out = np.full((len(px), 3), np.nan)
        for i, (x, y) in enumerate(zip(px, py)):
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < w and 0 <= yi < h):
                continue
            win = self._depth[max(yi - k, 0):yi + k + 1, max(xi - k, 0):xi + k + 1]
            win = win[win > 0]
            if win.size == 0:
                continue
            z = float(np.median(win)) * self.depth_scale
            out[i] = ((x - self.ppx) * z / self.fx, (y - self.ppy) * z / self.fy, z)
        return out

    def release(self):
        self.pipe.stop()
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

    def __call__(self, x, dt=1 / 30.0, valid_mask=None):
        """Filtra `x`; con `valid_mask`, aggiorna solo gli elementi validi e conserva gli altri."""
        x = np.asarray(x)
        mask = (np.ones(x.shape, dtype=bool) if valid_mask is None else
                np.broadcast_to(np.asarray(valid_mask, dtype=bool).reshape(
                    (len(x),) + (1,) * (x.ndim - 1)), x.shape))
        if self.x_prev is None:
            self.x_prev, self.dx_prev = x.copy(), np.zeros_like(x)
            return self.x_prev
        dx = np.where(mask, (x - self.x_prev) / dt, 0.0)
        a_d = 1.0 / (1.0 + (1.0 / (2 * np.pi * self.d_cutoff)) / dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = 1.0 / (1.0 + (1.0 / (2 * np.pi * cutoff)) / dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = np.where(mask, x_hat, self.x_prev)
        self.dx_prev = np.where(mask, dx_hat, self.dx_prev)
        return self.x_prev


# =============================================================================
# Compensazione inclinazione camera
# =============================================================================
class BodyLeveler:
    """Raddrizza posa e mani rispetto all'asse del busto, per compensare una camera inclinata.

    Si misura UNA volta, con la persona dritta e ferma, l'asse bacino -> testa (media su `n_frames`
    frame validi) e si calcola la rotazione minima che lo porta sulla verticale; poi la stessa
    rotazione si applica a ogni frame. Non si ricalcola a ogni frame: altrimenti il sistema ruoterebbe
    con te quando ti pieghi in avanti.

    Compensa pitch e roll della camera. Lo yaw non è definito dall'asse del busto e resta com'è:
    la camera deve inquadrarti più o meno di fronte.

    Parte subito a raccogliere i frame; finché la prima calibrazione non è finita azzera pose e mani,
    così il robot non riceve comandi. `recalibrate()` ne avvia un'altra e nel frattempo continua ad
    applicare la rotazione precedente. `ready` dice quando la rotazione è valida e stabile.
    """

    def __init__(self, n_frames=30, vis_min=0.5):
        self.n_frames = n_frames
        self.vis_min = vis_min
        self.R = np.eye(3)
        self.calibrated = False
        self._buf = []
        self._collecting = True
        print(f"[level] mettiti dritto e fermo davanti alla camera: calibro l'inclinazione su "
              f"{n_frames} frame (premi 'b' per ripetere)")

    @property
    def ready(self):
        """True se esiste una rotazione valida e non è in corso una (ri)calibrazione."""
        return self.calibrated and not self._collecting

    def recalibrate(self):
        self._buf = []
        self._collecting = True
        print("[level] ricalibrazione: resta dritto e fermo")

    def __call__(self, state):
        """Modifica `state` e lo ritorna: 'pose' e 'hands'[i]['pts'] ruotati nel frame raddrizzato."""
        pose = state['pose']
        if self._collecting and pose is not None:
            self._collect(pose, state['visibility'])
        if not self.calibrated:  # prima calibrazione in corso: nessun comando (evita salti del robot)
            state['pose'], state['hands'] = None, []
            return state
        # Nuovi array (niente modifica in place): i filtri One Euro tengono un riferimento ai loro
        if pose is not None:
            state['pose'] = pose @ self.R.T
        for h in state['hands']:
            h['pts'] = h['pts'] @ self.R.T
        return state

    # ---- Interni -----------------------------------------------------------
    def _collect(self, pose, vis):
        idx = list(LEVEL_HEAD + LEVEL_HIPS)
        if vis is not None and vis[idx].min() < self.vis_min:
            return
        axis = pose[list(LEVEL_HEAD)].mean(axis=0) - pose[list(LEVEL_HIPS)].mean(axis=0)
        n = np.linalg.norm(axis)
        if n < 1e-6:
            return
        self._buf.append(axis / n)
        if len(self._buf) >= self.n_frames:
            self._finish()

    def _finish(self):
        a = np.mean(self._buf, axis=0)
        a /= np.linalg.norm(a)
        self.R = self._rot_between(a, UP)
        self.calibrated, self._collecting = True, False
        tilt = float(np.degrees(np.arccos(np.clip(a @ UP, -1.0, 1.0))))
        print(f"[level] calibrato: l'asse del busto risultava inclinato di {tilt:.1f} gradi rispetto alla verticale")
        if tilt > 60:
            print("[level] avviso: inclinazione molto grande, controlla di essere dritto e ben inquadrato")

    @staticmethod
    def _rot_between(a, b):
        """Rotazione minima (3, 3) che porta il versore a sul versore b (formula di Rodrigues)."""
        v = np.cross(a, b)
        c = float(a @ b)
        if c < -0.999:  # asse quasi capovolto: la rotazione minima non è definita
            return np.eye(3)
        K = np.array([[0.0, -v[2], v[1]],
                      [v[2], 0.0, -v[0]],
                      [-v[1], v[0], 0.0]])
        return np.eye(3) + K + K @ K / (1.0 + c)


# =============================================================================
# Tracking
# =============================================================================
class PoseTracker:
    """Rileva posa e mani e restituisce uno `state` (dict). Non apre finestre e non disegna.

    Con `level=True` (default) e la posa attiva, posa e mani sono raddrizzate sull'asse del busto
    (vedi BodyLeveler); nei primi frame di calibrazione 'pose' e 'hands' restano vuoti.

    state = {
        'pose':       (33, 3) landmarks filtrate [m], origine al bacino, oppure None.
                      Con una RealSense la profondità è misurata; altrimenti sono le world landmarks MediaPipe.
                      Raddrizzate sull'asse del busto se `level` è attivo
        'visibility': (33,)   visibilità per landmark in [0, 1], oppure None
        'pose_2d':    landmark normalizzati MediaPipe (per overlay/plot), oppure None
        'depth_ref':  distanza [m] del bacino dalla camera (solo con RealSense), altrimenti None
        'angles':     {'l_elbow': deg, 'r_elbow': deg}
        'hands':      [{'side': 'left'|'right', 'pts': (21, 3) [m],
                        'pinch': 0..1 (0 = pollice-indice a contatto, 1 = aperta),
                        'openness': 0..1 (0 = pugno, 1 = mano aperta)}]
        'hands_2d':   landmark normalizzati delle mani, stesso ordine di 'hands'
    }
    """

    def __init__(self, models_dir="models", pose_model="pose_landmarker_lite.task",
                 use_pose=True, use_hands=True, min_cutoff=0.5, beta=0.01, level=True):
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
        self.leveler = BodyLeveler() if (level and use_pose) else None  # senza posa non c'è asse del busto
        self._last_ts = -1

    # ---- API pubblica ------------------------------------------------------
    def process(self, frame, ts, depth_cam=None):
        """frame BGR + timestamp [ms] -> state.

        depth_cam: oggetto con `points_3d(px, py)` (RealSenseIO) per usare la depth misurata; None = solo MediaPipe.
        """
        state = {'pose': None, 'visibility': None, 'pose_2d': None, 'depth_ref': None,
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

        pts = self._process_pose(p_res, dt, state, frame.shape, depth_cam)
        self._process_hands(h_res, pts, dt, state)
        if self.leveler:
            self.leveler(state)  # per ultimo: filtri, angoli e pinch/openness lavorano sui dati grezzi
        return state

    @property
    def level_ready(self):
        """True se il frame di riferimento è stabile: raddrizzamento spento, oppure calibrato e non in corso."""
        return self.leveler is None or self.leveler.ready

    def recalibrate_level(self):
        """Ripete la calibrazione dell'inclinazione (persona dritta e ferma). Senza effetto se level è spento."""
        if self.leveler:
            self.leveler.recalibrate()

    def close(self):
        if self.pose_det:
            self.pose_det.close()
        if self.hand_det:
            self.hand_det.close()

    # ---- Interni -----------------------------------------------------------
    def _process_pose(self, p_res, dt, state, shape, depth_cam):
        if not (p_res and p_res.pose_world_landmarks):
            return None
        world = p_res.pose_world_landmarks[0]
        raw = np.array([[lm.x, lm.y, lm.z] for lm in world])
        vis = np.array([1.0 if lm.visibility is None else lm.visibility for lm in world])
        lm2d = p_res.pose_landmarks[0]

        depth_ref = None
        if depth_cam is not None:
            raw, depth_ref = self._fuse_depth(raw, vis, lm2d, shape, depth_cam)

        valid = (vis >= 0.5) & np.isfinite(raw).all(axis=1)
        pts = self.pose_filter(raw, dt, valid_mask=valid)
        state.update(pose=pts, visibility=vis, pose_2d=lm2d, depth_ref=depth_ref,
                     angles={'l_elbow': self._angle(pts, 11, 13, 15),
                             'r_elbow': self._angle(pts, 12, 14, 16)})
        return pts

    @staticmethod
    def _fuse_depth(world, vis, lm2d, shape, cam, min_vis=0.5, max_dev=0.3):
        """Sostituisce la z stimata da MediaPipe con la depth misurata dalla camera.

        Ritorna (landmarks (33, 3) [m] centrate sul bacino, distanza del bacino dalla camera [m]).
        Dove la depth manca, il landmark è poco visibile o la misura si discosta più di `max_dev` [m]
        dalla z di MediaPipe (es. polso nascosto che "vede" la parete dietro), si usa la z di MediaPipe
        traslata di un offset (mediana sui landmark validi). Se i landmark validi sono troppo pochi
        si ripiega interamente sulle world landmarks.
        """
        h, w = shape[:2]
        px = np.array([l.x * w for l in lm2d])
        py = np.array([l.y * h for l in lm2d])
        z_meas = cam.points_3d(px, py)[:, 2]  # NaN dove la depth non è valida

        ok = ~np.isnan(z_meas) & (vis > min_vis)
        if ok.sum() < 4:
            return world, None
        offset = np.median(z_meas[ok] - world[ok, 2])  # world z è relativa al bacino, z_meas è assoluta
        ok &= np.abs(z_meas - (world[:, 2] + offset)) < max_dev
        if ok.sum() < 4:
            return world, None

        z = np.where(ok, z_meas, world[:, 2] + offset)
        pts = np.column_stack(((px - cam.ppx) * z / cam.fx, (py - cam.ppy) * z / cam.fy, z))
        hip = pts[[23, 24]].mean(axis=0)  # stessa convenzione di MediaPipe: origine al centro del bacino
        return pts - hip, float(hip[2])

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

    Con la RealSense la distanza dalla camera è misurata (`state['depth_ref']`). Altrimenti la profondità
    assoluta NON è nota (le world landmarks sono relative al bacino): la si stima con un modello pinhole,
    Z = f * W / w_px, dove W è la larghezza spalle reale e w_px quella misurata nell'immagine.
    FOCAL_PX dipende da camera e risoluzione.
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

        z_off = state.get('depth_ref')  # misurata (RealSense)
        if z_off is None:  # stimata (pinhole)
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