import os
import time
import threading
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class ThreadedCamera:
    def __init__(self, src=0, width=640, height=480, fps=30):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.new = threading.Event()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                self.new.set()
            else:
                time.sleep(0.01)

    def read(self, timeout=1.0):
        """Attende un frame NUOVO. Ritorna (False, None) se scade il timeout."""
        if not self.new.wait(timeout):
            return False, None
        self.new.clear()
        return True, self.frame

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = self.dx_prev = None

    def __call__(self, x, dt=1/30.0):
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


class PoseTracker:
    BONES = {
        'head': [(0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8), (9,10)],
        'torso': [(11,12), (12,24), (24,23), (23,11)],
        'r_arm': [(12,14), (14,16), (16,18), (16,20), (16,22)],
        'l_arm': [(11,13), (13,15), (15,17), (15,19), (15,21)]
    }
    COLORS = {'head': 'yellow', 'torso': 'blue', 'r_arm': 'red', 'l_arm': 'cyan', 'hand': 'green'}
    IDXS = {k: list(set(sum(v, ()))) for k, v in BONES.items()}

    def __init__(self, models_dir="models", pose_model="pose_landmarker_lite.task",
                 use_pose=True, use_hands=True, min_cutoff=0.5, beta=0.01, plot_every=3):
        md = os.path.abspath(models_dir)
        self.pose_det = self.hand_det = None

        if use_pose:
            p_opt = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=f"{md}/{pose_model}"),
                running_mode=vision.RunningMode.VIDEO)
            self.pose_det = vision.PoseLandmarker.create_from_options(p_opt)

        if use_hands:
            h_opt = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=f"{md}/hand_landmarker.task"),
                running_mode=vision.RunningMode.VIDEO, num_hands=2)
            self.hand_det = vision.HandLandmarker.create_from_options(h_opt)

        self.filter = OneEuroFilter(min_cutoff, beta)
        self.ax = self.fig = None
        self.plot_every = plot_every
        self._plot_n = 0
        self._last_ts = 0
        self._last_t = None

    def _init_plot(self):
        import matplotlib.pyplot as plt
        plt.ion()
        self.fig = plt.figure(figsize=(9, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.plt = plt

    def _get_angles(self, pts, i1, i2, i3):
        v1, v2 = pts[i1] - pts[i2], pts[i3] - pts[i2]
        return np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)))

    def step(self, frame, plot_3d=False, show_2d=False):
        state = {'pose': None, 'hands': [], 'angles': {'l_elbow': 0, 'r_elbow': 0}}
        if frame is None:
            return state, frame

        h, w = frame.shape[:2]

        # timestamp strettamente crescente (richiesto dalla modalita' VIDEO) e dt reale
        now = time.monotonic()
        ts = max(int(now * 1000), self._last_ts + 1)
        self._last_ts = ts
        dt = 1/30.0 if self._last_t is None else float(np.clip(now - self._last_t, 1e-3, 0.2))
        self._last_t = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mpi = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        p_res = self.pose_det.detect_for_video(mpi, ts) if self.pose_det else None
        h_res = self.hand_det.detect_for_video(mpi, ts) if self.hand_det else None

        pts = p2d = None
        if p_res is not None and p_res.pose_world_landmarks:
            raw = np.array([[lm.x, lm.y, lm.z] for lm in p_res.pose_world_landmarks[0]])
            pts = self.filter(raw, dt)
            p2d = p_res.pose_landmarks[0]
            state['pose'] = pts
            state['angles']['l_elbow'] = self._get_angles(pts, 11, 13, 15)
            state['angles']['r_elbow'] = self._get_angles(pts, 12, 14, 16)

        if h_res is not None and h_res.hand_world_landmarks:
            for i, hw in enumerate(h_res.hand_world_landmarks):
                local = np.array([[l.x, l.y, l.z] for l in hw])
                if pts is not None:
                    # aggancia la mano al polso piu' vicino
                    h2d = h_res.hand_landmarks[i][0]
                    d_l = np.hypot(h2d.x - p2d[15].x, h2d.y - p2d[15].y)
                    d_r = np.hypot(h2d.x - p2d[16].x, h2d.y - p2d[16].y)
                    t_idx = 15 if d_l < d_r else 16
                    h_pts = local + (pts[t_idx] - local[0])
                    side = 'left' if t_idx == 15 else 'right'
                else:
                    # solo mani: nessuna posa a cui agganciarsi, uso l'handedness di MediaPipe
                    h_pts = local
                    side = h_res.handedness[i][0].category_name.lower()
                grip = np.clip((np.linalg.norm(h_pts[4] - h_pts[8]) - 0.02) / 0.1, 0, 1)
                state['hands'].append({'side': side, 'pts': h_pts, 'grip': grip})

        if plot_3d and pts is not None:
            self._plot_n += 1
            if self._plot_n % self.plot_every == 0:
                if not self.ax:
                    self._init_plot()
                z_off = (0.4 * 800) / max(np.hypot((p2d[11].x - p2d[12].x) * w, (p2d[11].y - p2d[12].y) * h), 1e-5)
                elev, azim = self.ax.elev, self.ax.azim
                self.ax.clear()
                self.ax.view_init(elev=elev, azim=azim)
                self.ax.set(xlim=[-0.6, 0.6], ylim=[z_off-0.6, z_off+0.6], zlim=[-0.6, 0.6],
                            xlabel='X', ylabel='Z', zlabel='Y')

                t3d = np.column_stack((pts[:, 0], pts[:, 2] + z_off, -pts[:, 1]))
                for k, idx in self.IDXS.items():
                    self.ax.scatter(t3d[idx, 0], t3d[idx, 1], t3d[idx, 2], c=self.COLORS[k], s=25)
                    for a, b in self.BONES[k]:
                        self.ax.plot(t3d[[a, b], 0], t3d[[a, b], 1], t3d[[a, b], 2], c=self.COLORS[k], lw=2)

                for hnd in state['hands']:
                    ht3d = np.column_stack((hnd['pts'][:, 0], hnd['pts'][:, 2] + z_off, -hnd['pts'][:, 1]))
                    self.ax.scatter(ht3d[:, 0], ht3d[:, 1], ht3d[:, 2], c=self.COLORS['hand'], s=20)
                self.plt.pause(0.001)

        if show_2d:
            if p2d is not None:
                for lm in p2d:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (255, 255, 0), -1)
                cv2.putText(frame, f"L:{int(state['angles']['l_elbow'])} R:{int(state['angles']['r_elbow'])}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if h_res is not None and h_res.hand_landmarks:
                for hnd in h_res.hand_landmarks:
                    for lm in hnd:
                        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)
            cv2.imshow('Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

        return state, frame

    def close(self):
        for det in (self.pose_det, self.hand_det):
            if det is not None:
                det.close()