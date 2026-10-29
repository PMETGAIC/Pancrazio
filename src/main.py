import cv2
import sys
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer
import urllib.request
from pathlib import Path
from src.Capture.Human_capture import PoseTracker, ThreadedCamera
from src.Remapping.QP_pose import MujocoController

# (spalla, gomito, polso) MediaPipe: sinistra, destra. Stesso ordine dei body nel controller.
ARM_LANDMARKS = [(11, 13, 15), (12, 14, 16)]

BASE_URL = "https://storage.googleapis.com/mediapipe-models"
HAND_URL = f"{BASE_URL}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def init_system(args):
    camera = tracker = controller = None

    if not args.model_only:
        models_dir = Path("./models")
        models_dir.mkdir(parents=True, exist_ok=True)

        use_pose = not args.only_hands
        use_hands = not args.only_pose
        pose_file = f"pose_landmarker_{args.pose_model}.task"

        tasks = {}
        if use_pose:
            tasks[pose_file] = f"{BASE_URL}/pose_landmarker/pose_landmarker_{args.pose_model}/float16/1/{pose_file}"
        if use_hands:
            tasks["hand_landmarker.task"] = HAND_URL

        for name, url in tasks.items():
            if not (models_dir / name).exists():
                print(f"Scarico {name}...")
                urllib.request.urlretrieve(url, models_dir / name)

        camera = ThreadedCamera(src=0)
        tracker = PoseTracker(models_dir=str(models_dir.resolve()), pose_model=pose_file,
                              use_pose=use_pose, use_hands=use_hands)

    if not args.cam_only:
        xml_path = str(Path(sys.prefix) / "share" / "openarm_mujoco" / "v2" / "pedestal" / "openarm_pedestal.xml")
        controller = MujocoController(
            xml_path,
            ee_names=["openarm_left_ee_base_link", "openarm_right_ee_base_link"],
            # NOMI DA VERIFICARE sul modello (se sbagliati, l'errore elenca quelli validi)
            shoulder_names=["openarm_left_link3", "openarm_right_link3"],
            elbow_names=["openarm_left_link4", "openarm_right_link4"],
            # gomito piegato a riposo: controlla segno e range di joint4 nel modello
            rest_joints={"openarm_left_joint4": 0.8, "openarm_right_joint4": 0.8},
            debug=True,
        )

    return camera, tracker, controller


def to_robot(v):
    """Vettore MediaPipe world (x dx immagine, y giu', z verso camera negativo) -> frame robot (x avanti, y sinistra, z su)."""
    return np.array([-v[2], v[0], -v[1]])


def process_dirs(state, last, alpha=0.5):
    """Versori braccio/avambraccio nel frame robot, con smoothing. Ritorna (dirs, last)."""
    pose = state['pose']
    if pose is None:
        return None, last

    dirs = []
    for i, (sh, el, wr) in enumerate(ARM_LANDMARKS):
        pair = []
        for j, (a, b) in enumerate(((sh, el), (el, wr))):
            u = to_robot(pose[b] - pose[a])
            n = np.linalg.norm(u)
            if n < 1e-6:
                return None, last
            u = u / n
            if last is not None:
                u = alpha * u + (1 - alpha) * last[i][j]
                u = u / (np.linalg.norm(u) + 1e-9)
            pair.append(u)
        dirs.append(tuple(pair))
    return dirs, dirs


def run_teleoperation(camera, tracker, controller, args):
    if args.cam_only:
        while camera.running:
            ret, frame = camera.read()
            if ret:
                tracker.step(frame, plot_3d=args.plot3d, show_2d=True)
        return

    if args.model_only:
        with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
            while viewer.is_running():
                viewer.sync()
                time.sleep(1 / 60)
        return

    last = None
    acc_tracker = acc_qp = 0.0
    n = 0
    with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
        while camera.running and viewer.is_running():
            ret, frame = camera.read()
            if not ret:
                continue

            t0 = time.perf_counter()
            state, frame = tracker.step(frame, plot_3d=args.plot3d, show_2d=True)
            t1 = time.perf_counter()

            dirs, last = process_dirs(state, last, alpha=0.5)
            if dirs is not None:
                with viewer.lock():
                    controller.track(dirs)
            t2 = time.perf_counter()

            viewer.sync()

            if args.profile:
                acc_tracker += t1 - t0
                acc_qp += t2 - t1
                n += 1
                if n == 60:
                    print(f"tracker: {acc_tracker / n * 1000:.1f} ms | qp+sim: {acc_qp / n * 1000:.1f} ms")
                    acc_tracker = acc_qp = 0.0
                    n = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_only", action="store_true")
    parser.add_argument("--model_only", action="store_true")
    parser.add_argument("--only_pose", action="store_true", help="carica solo il modello della posa (consigliato per il robot)")
    parser.add_argument("--only_hands", action="store_true", help="solo mani (richiede --cam_only)")
    parser.add_argument("--pose_model", choices=["lite", "full", "heavy"], default="lite")
    parser.add_argument("--plot3d", action="store_true", help="grafico matplotlib 3D (pesante)")
    parser.add_argument("--profile", action="store_true", help="stampa i ms medi per tracker e QP+sim")
    args = parser.parse_args()

    if args.only_pose and args.only_hands:
        parser.error("--only_pose e --only_hands sono incompatibili")
    if args.only_hands and not args.cam_only:
        parser.error("--only_hands richiede --cam_only (il controller usa la posa)")

    camera, tracker, controller = init_system(args)

    try:
        run_teleoperation(camera, tracker, controller, args)
    except KeyboardInterrupt:
        pass
    finally:
        if camera:
            camera.release()
        if tracker:
            tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()