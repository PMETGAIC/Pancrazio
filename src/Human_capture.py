import cv2
import numpy as np
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

def get_gripper_state(hand_landmarks):
    thumb = np.array([hand_landmarks[4].x, hand_landmarks[4].y, hand_landmarks[4].z])
    index = np.array([hand_landmarks[8].x, hand_landmarks[8].y, hand_landmarks[8].z])
    dist = np.linalg.norm(thumb - index)
    return np.clip((dist - 0.02) / 0.1, 0.0, 1.0) 

def get_joint_angles(sh, el, wr):
    v1, v2 = sh - el, wr - el
    return np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)))

detector = PoseHandDetector()
cap = cv2.VideoCapture(0)
pose_filter = OneEuroFilter(min_cutoff=0.5, beta=0.01)

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
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

def get_camera_z(pose_lms, w, h):
    l_sh, r_sh = pose_lms[11], pose_lms[12]
    px_dist = np.hypot((l_sh.x - r_sh.x) * w, (l_sh.y - r_sh.y) * h)
    return (0.4 * 800.0) / px_dist if px_dist > 0 else 2.0

def plot_part_3d(landmarks, indices, color, z_offset=0, bone_key=None):
    pts = np.array([[lm[0], lm[2] + z_offset, -lm[1]] for lm in landmarks])
    part_pts = pts[indices]
    ax.scatter(part_pts[:, 0], part_pts[:, 1], part_pts[:, 2], c=color, s=25)
    if bone_key and bone_key in BONES:
        for start, end in BONES[bone_key]:
            p1, p2 = pts[start], pts[end]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c=color, linewidth=2)

def draw_2d_part(img, landmarks, indices, color):
    h, w, _ = img.shape
    for idx in indices:
        lm = landmarks[idx]
        cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 4, color, -1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    h, w, _ = frame.shape
    pose_res, hand_res = detector.process(frame)

    z_offset = 2.0
    if pose_res.pose_landmarks:
        z_offset = get_camera_z(pose_res.pose_landmarks[0], w, h)

    elev, azim = ax.elev, ax.azim
    ax.clear()
    ax.view_init(elev=elev, azim=azim)
    ax.set(xlim=[-0.6, 0.6], ylim=[z_offset - 0.6, z_offset + 0.6], zlim=[-0.6, 0.6], xlabel='X', ylabel='Z', zlabel='Y')

    if pose_res.pose_world_landmarks:
        raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in pose_res.pose_world_landmarks[0]])
        smooth_pts = pose_filter(raw_pts)
        
        plot_part_3d(smooth_pts, HEAD_IDX, 'yellow', z_offset, 'head')
        plot_part_3d(smooth_pts, TORSO_IDX, 'blue', z_offset, 'torso')
        plot_part_3d(smooth_pts, RIGHT_ARM_IDX, 'red', z_offset, 'r_arm')
        plot_part_3d(smooth_pts, LEFT_ARM_IDX, 'cyan', z_offset, 'l_arm')

        r_el_ang = get_joint_angles(smooth_pts[12], smooth_pts[14], smooth_pts[16])
        l_el_ang = get_joint_angles(smooth_pts[11], smooth_pts[13], smooth_pts[15])
        cv2.putText(frame, f'R_Elbow: {int(r_el_ang)} | L_Elbow: {int(l_el_ang)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if pose_res.pose_landmarks:
        p_lms = pose_res.pose_landmarks[0]
        draw_2d_part(frame, p_lms, HEAD_IDX, (0, 255, 255))
        draw_2d_part(frame, p_lms, TORSO_IDX, (255, 0, 0))
        draw_2d_part(frame, p_lms, RIGHT_ARM_IDX, (0, 0, 255))
        draw_2d_part(frame, p_lms, LEFT_ARM_IDX, (255, 255, 0))

    if hand_res.hand_world_landmarks and pose_res.pose_world_landmarks and pose_res.pose_landmarks:
        l_wrist_2d = np.array([p_lms[15].x, p_lms[15].y])
        r_wrist_2d = np.array([p_lms[16].x, p_lms[16].y])

        for i, hand_wlms in enumerate(hand_res.hand_world_landmarks):
            h_wrist_2d = np.array([hand_res.hand_landmarks[i][0].x, hand_res.hand_landmarks[i][0].y])
            target_idx = 15 if np.linalg.norm(h_wrist_2d - l_wrist_2d) < np.linalg.norm(h_wrist_2d - r_wrist_2d) else 16
            target_w3d = smooth_pts[target_idx]
            
            hw = hand_wlms[0]
            dx, dy, dz = target_w3d[0] - hw.x, target_w3d[1] - hw.y, target_w3d[2] - hw.z
            
            pts = np.array([[lm.x + dx, (lm.z + dz) + z_offset, -(lm.y + dy)] for lm in hand_wlms])
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='g', s=20)
            
            grip = get_gripper_state(hand_wlms)
            cv2.putText(frame, f'Grip {i}: {grip:.2f}', (10, 60 + (i*30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if hand_res.hand_landmarks:
        for hand_lms in hand_res.hand_landmarks:
            for lm in hand_lms:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)

    plt.pause(0.01)
    cv2.imshow('Feed 2D', frame)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()