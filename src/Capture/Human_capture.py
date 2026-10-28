import os
import cv2
import numpy as np
import threading
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class ThreadedCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.ret, self.frame = ret, frame

    def read(self):
        return self.ret, self.frame

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

    def __init__(self, models_dir="models", min_cutoff=0.5, beta=0.01):
        md = os.path.abspath(models_dir)
        p_opt = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=f"{md}/pose_landmarker.task"), running_mode=vision.RunningMode.IMAGE)
        h_opt = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=f"{md}/hand_landmarker.task"), running_mode=vision.RunningMode.IMAGE, num_hands=2)
        self.pose_det = vision.PoseLandmarker.create_from_options(p_opt)
        self.hand_det = vision.HandLandmarker.create_from_options(h_opt)
        self.filter = OneEuroFilter(min_cutoff, beta)
        self.ax = self.fig = None

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
        if frame is None: return {'pose': None, 'hands': [], 'angles': {'l_elbow': 0, 'r_elbow': 0}}, frame
        
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mpi = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        p_res, h_res = self.pose_det.detect(mpi), self.hand_det.detect(mpi)

        state = {'pose': None, 'hands': [], 'angles': {'l_elbow': 0, 'r_elbow': 0}}
        if not p_res.pose_world_landmarks: return state, frame

        raw = np.array([[lm.x, lm.y, lm.z] for lm in p_res.pose_world_landmarks[0]])
        pts = self.filter(raw)
        state['pose'] = pts
        state['angles']['l_elbow'] = self._get_angles(pts, 11, 13, 15)
        state['angles']['r_elbow'] = self._get_angles(pts, 12, 14, 16)

        p2d = p_res.pose_landmarks[0]
        z_off = (0.4 * 800) / max(np.hypot((p2d[11].x - p2d[12].x)*w, (p2d[11].y - p2d[12].y)*h), 1e-5)

        if h_res.hand_world_landmarks:
            for i, hw in enumerate(h_res.hand_world_landmarks):
                h2d = h_res.hand_landmarks[i][0]
                t_idx = 15 if np.hypot(h2d.x - p2d[15].x, h2d.y - p2d[15].y) < np.hypot(h2d.x - p2d[16].x, h2d.y - p2d[16].y) else 16
                dxyz = pts[t_idx] - [hw[0].x, hw[0].y, hw[0].z]
                h_pts = np.array([[l.x+dxyz[0], l.y+dxyz[1], l.z+dxyz[2]] for l in hw])
                grip = np.clip((np.linalg.norm(h_pts[4] - h_pts[8]) - 0.02) / 0.1, 0, 1)
                state['hands'].append({'side': 'left' if t_idx==15 else 'right', 'pts': h_pts, 'grip': grip})

        if plot_3d:
            if not self.ax: self._init_plot()
            e, a = self.ax.elev, self.ax.azim
            self.ax.clear()
            self.ax.view_init(elev=e, azim=a)
            self.ax.set(xlim=[-0.6, 0.6], ylim=[z_off-0.6, z_off+0.6], zlim=[-0.6, 0.6], xlabel='X', ylabel='Z', zlabel='Y')
            
            t3d = np.column_stack((pts[:,0], pts[:,2] + z_off, -pts[:,1]))
            for k, idx in self.IDXS.items():
                self.ax.scatter(t3d[idx,0], t3d[idx,1], t3d[idx,2], c=self.COLORS[k], s=25)
                for s, e in self.BONES[k]: self.ax.plot(t3d[[s,e],0], t3d[[s,e],1], t3d[[s,e],2], c=self.COLORS[k], lw=2)
            
            for hnd in state['hands']:
                ht3d = np.column_stack((hnd['pts'][:,0], hnd['pts'][:,2] + z_off, -hnd['pts'][:,1]))
                self.ax.scatter(ht3d[:,0], ht3d[:,1], ht3d[:,2], c=self.COLORS['hand'], s=20)
            self.plt.pause(0.001)

        if show_2d:
            for lm in p2d: cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, (255,255,0), -1)
            if h_res.hand_landmarks:
                for hnd in h_res.hand_landmarks:
                    for lm in hnd: cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 3, (0,255,0), -1)
            cv2.putText(frame, f"L:{int(state['angles']['l_elbow'])} R:{int(state['angles']['r_elbow'])}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow('Feed', frame)
            cv2.waitKey(1)

        return state, frame