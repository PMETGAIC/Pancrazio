import cv2
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from models.Pose_models import PoseHandDetector

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev, self.dx_prev = None, None

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

def get_3d_point(intrinsics, depth_frame, px_x, px_y, max_w, max_h):
    safe_x = max(0, min(int(px_x), max_w - 1))
    safe_y = max(0, min(int(px_y), max_h - 1))
    dist = depth_frame.get_distance(safe_x, safe_y)
    if dist <= 0 or dist > 3.0: 
        return None
    return rs.rs2_deproject_pixel_to_point(intrinsics, [safe_x, safe_y], dist)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

detector = PoseHandDetector()
pose_filter = OneEuroFilter(min_cutoff=0.5, beta=0.01)

fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
plt.ion()
plt.show()

HEAD_IDX = list(range(0, 11))
TORSO_IDX = [11, 12, 23, 24]
RIGHT_ARM_IDX = [12, 14, 16, 18, 20, 22]
LEFT_ARM_IDX = [11, 13, 15, 17, 19, 21]

BONES = {
    'head': [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10)],
    'torso': [(11, 12), (12, 24), (24, 23), (23, 11)],
    'r_arm': [(12, 14), (14, 16), (16, 18), (16, 20), (16, 22)],
    'l_arm': [(11, 13), (13, 15), (15, 17), (15, 19), (15, 21)]
}

def plot_part_3d(ax, pts, indices, color, bone_key=None):
    part_pts = pts[indices]
    ax.scatter(part_pts[:, 0], part_pts[:, 1], part_pts[:, 2], c=color, s=25)
    if bone_key and bone_key in BONES:
        for s, e in BONES[bone_key]:
            ax.plot([pts[s,0], pts[e,0]], [pts[s,1], pts[e,1]], [pts[s,2], pts[e,2]], c=color, lw=2)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame: continue

        frame = np.asanyarray(color_frame.get_data())
        h, w = frame.shape[:2]
        
        pose_res, hand_res = detector.process(frame)

        ax1.clear()
        ax1.set(xlim=[-1, 1], ylim=[0, 3], zlim=[-1, 1], xlabel='X', ylabel='Z (Profondità)', zlabel='Y', title='Frontale')
        ax1.view_init(elev=0, azim=-90)

        ax2.clear()
        ax2.set(xlim=[-1, 1], ylim=[0, 3], zlim=[-1, 1], xlabel='X', ylabel='Z (Profondità)', zlabel='Y', title='Profilo Laterale')
        ax2.view_init(elev=0, azim=0)

        if pose_res.pose_landmarks:
            raw_3d = []
            for lm in pose_res.pose_landmarks[0]:
                pt = get_3d_point(intrinsics, depth_frame, lm.x * w, lm.y * h, w, h)
                raw_3d.append(pt if pt else [0, 0, 0])
            
            raw_3d = np.array(raw_3d)
            if np.any(np.linalg.norm(raw_3d, axis=1) > 0):
                smooth_pts = pose_filter(raw_3d)
                
                # Mappa le coordinate camera (X, Y_giù, Z_avanti) in coordinate umane standard (X_dx, Y_avanti, Z_su)
                human_3d = np.stack([smooth_pts[:, 0], smooth_pts[:, 2], -smooth_pts[:, 1]], axis=1)
                
                for ax in [ax1, ax2]:
                    plot_part_3d(ax, human_3d, HEAD_IDX, 'yellow', 'head')
                    plot_part_3d(ax, human_3d, TORSO_IDX, 'blue', 'torso')
                    plot_part_3d(ax, human_3d, RIGHT_ARM_IDX, 'red', 'r_arm')
                    plot_part_3d(ax, human_3d, LEFT_ARM_IDX, 'cyan', 'l_arm')

                for idx in TORSO_IDX + RIGHT_ARM_IDX + LEFT_ARM_IDX:
                    lm = pose_res.pose_landmarks[0][idx]
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)

        plt.pause(0.01)
        cv2.imshow('RealSense 2D', frame)
        if cv2.waitKey(1) & 0xFF == 27: break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()