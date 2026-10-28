import cv2
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from src.Capture.Human_capture import PoseTracker, ThreadedCamera
from src.Remapping.QP_pose import MujocoController

def init_system():
    xml_path = str(Path(__file__).resolve().parents[1] / ".venv" / "share" / "openarm_mujoco" / "v2" / "pedestal" / "openarm_pedestal.xml")
    
    camera = ThreadedCamera(src=0)
    tracker = PoseTracker(models_dir="./models")
    controller = MujocoController(xml_path, ["openarm_left_ee_base_link", "openarm_right_ee_base_link"])
    
    return camera, tracker, controller

def process_targets(state, last_t_l, last_t_r, alpha=0.5):
    if state['pose'] is None:
        return None, None, last_t_l, last_t_r

    sh_r, wr_r = state['pose'][12], state['pose'][16]
    dx_r, dy_r, dz_r = wr_r[0] - sh_r[0], wr_r[1] - sh_r[1], wr_r[2] - sh_r[2]
    target_r = np.array([-dz_r, dx_r, -dy_r]) * 1.5 + np.array([0.2, -0.2, 0.5])

    sh_l, wr_l = state['pose'][11], state['pose'][15]
    dx_l, dy_l, dz_l = wr_l[0] - sh_l[0], wr_l[1] - sh_l[1], wr_l[2] - sh_l[2]
    target_l = np.array([-dz_l, dx_l, -dy_l]) * 1.5 + np.array([0.2, 0.2, 0.5])
    
    target_r = np.clip(target_r, [-0.5, -0.8, 0.1], [0.8, 0.2, 1.2])
    target_l = np.clip(target_l, [-0.5, -0.2, 0.1], [0.8, 0.8, 1.2])

    if last_t_l is not None:
        target_l = alpha * target_l + (1 - alpha) * last_t_l
        target_r = alpha * target_r + (1 - alpha) * last_t_r
    
    return target_l, target_r, target_l, target_r

def run_teleoperation(camera, tracker, controller):
    last_t_l = last_t_r = None
    
    with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
        while camera.running and viewer.is_running():
            ret, frame = camera.read()
            if not ret or frame is None:
                continue

            state, frame = tracker.step(frame, plot_3d=False, show_2d=True)
            
            t_l, t_r, last_t_l, last_t_r = process_targets(state, last_t_l, last_t_r, alpha=0.5)

            if t_l is not None and t_r is not None:
                controller.strategy_qp([t_l, t_r])

            viewer.sync()

def main():
    camera, tracker, controller = init_system()
    try:
        run_teleoperation(camera, tracker, controller)
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()