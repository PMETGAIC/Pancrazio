"""Teleoperazione di un robot con MediaPipe + MuJoCo: webcam/video -> posa -> retargeting -> simulazione."""
import argparse
import contextlib
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer

from src.Capture.human_capture import PoseTracker, Skeleton3DPlot, VideoIO, draw_overlay
from src.Remapping.retargeting import ArmDirectionExtractor, MujocoController

MODELS_DIR = Path("./models")
BASE_URL = "https://storage.googleapis.com/mediapipe-models"
HAND_URL = f"{BASE_URL}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
DEFAULT_XML = Path(sys.prefix) / "share" / "openarm_mujoco" / "v2" / "pedestal" / "openarm_pedestal.xml"

DIR_ALPHA = 0.5  # smoothing extra sulle direzioni (1.0 = nessuno; la posa è già filtrata da One Euro)


# =============================================================================
# Costruzione dei componenti
# =============================================================================
def download_models(use_pose, use_hands, pose_model):
    """Scarica i modelli .task mancanti (via file temporaneo: niente file corrotti se cade la rete)."""
    wanted = {}
    if use_pose:
        name = f"pose_landmarker_{pose_model}.task"
        wanted[name] = f"{BASE_URL}/pose_landmarker/pose_landmarker_{pose_model}/float16/1/{name}"
    if use_hands:
        wanted["hand_landmarker.task"] = HAND_URL

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in wanted.items():
        dst = MODELS_DIR / name
        if dst.exists():
            continue
        print(f"[models] scarico {name} ...")
        tmp = dst.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dst)


def build_capture(args):
    use_pose = not args.only_hands
    # In teleoperazione le mani non sono ancora usate dal controller: si calcolano solo se richieste
    use_hands = not args.only_pose and (args.hands or args.only_hands or args.cam_only)
    download_models(use_pose, use_hands, args.pose_model)

    src = int(args.src) if str(args.src).isdigit() else args.src
    camera = VideoIO(src=src, out_dir="./recording", fps=args.max_fps, rec=args.rec)
    tracker = PoseTracker(models_dir=str(MODELS_DIR.resolve()),
                          pose_model=f"pose_landmarker_{args.pose_model}.task",
                          use_pose=use_pose, use_hands=use_hands,
                          min_cutoff=args.min_cutoff, beta=args.beta)
    return camera, tracker


def build_controller(args):
    # Ordine sinistro, destro: deve coincidere con ArmDirectionExtractor.ARMS
    return MujocoController(
        str(args.xml),
        ee_names=["openarm_left_ee_base_link", "openarm_right_ee_base_link"],
        shoulder_names=["openarm_left_link3", "openarm_right_link3"],
        elbow_names=["openarm_left_link4", "openarm_right_link4"],
        rest_joints={"openarm_left_joint4": 0.8, "openarm_right_joint4": 0.8},
        debug=args.debug,
    )


# =============================================================================
# Loop
# =============================================================================
def run_model_only(controller):
    """Solo il viewer del modello, senza video."""
    with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(1 / 60)


def run_pipeline(args, camera, tracker, controller):
    """Loop unico: camera -> tracking -> (opz.) retargeting + viewer. Con controller=None è solo cattura."""
    plot = Skeleton3DPlot() if args.plot3d else None
    extract = ArmDirectionExtractor(alpha=DIR_ALPHA)
    acc_tracker = acc_qp = 0.0
    n = 0

    viewer_cm = (mujoco.viewer.launch_passive(controller.model, controller.data)
                 if controller else contextlib.nullcontext())
    with viewer_cm as viewer:
        while viewer is None or viewer.is_running():
            ok, frame, ts = camera.read()
            if not ok:
                break

            t0 = time.perf_counter()
            state = tracker.process(frame, ts)
            t1 = time.perf_counter()

            if plot:
                plot.update(state, frame.shape)

            # Con --rec si registra e si mostra il frame pulito; altrimenti si mostra l'overlay
            camera.write(frame)
            cv2.imshow('Feed', frame if args.rec else draw_overlay(frame, state))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if controller:
                t_qp = time.perf_counter()
                dirs = extract(state)
                if dirs is not None:
                    with viewer.lock():
                        controller.track(dirs)
                viewer.sync()
                acc_qp += time.perf_counter() - t_qp

            if args.profile:
                acc_tracker += t1 - t0
                n += 1
                if n == 60:
                    msg = f"tracker: {acc_tracker / n * 1000:.1f} ms"
                    if controller:
                        msg += f" | retarget+sim+viewer: {acc_qp / n * 1000:.1f} ms"
                    print(msg)
                    acc_tracker = acc_qp = 0.0
                    n = 0


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Teleoperazione robot tramite MediaPipe e MuJoCo.")
    # Sorgente / registrazione
    p.add_argument("--src", default=0, help="Sorgente video: indice webcam (es. 0) o path del file (.mp4)")
    p.add_argument("--rec", action="store_true", help="Salva la registrazione (frame puliti) in ./recording")
    p.add_argument("--max_fps", type=int, default=30, help="FPS per cattura e registrazione (default: 30)")
    # Modalità
    p.add_argument("--cam_only", action="store_true", help="Solo cattura e tracking, senza MuJoCo")
    p.add_argument("--model_only", action="store_true", help="Solo il modello MuJoCo, senza video")
    # Tracking
    p.add_argument("--only_pose", action="store_true", help="Traccia solo la posa del corpo")
    p.add_argument("--only_hands", action="store_true", help="Traccia solo le mani (richiede --cam_only)")
    p.add_argument("--hands", action="store_true",
                   help="Traccia anche le mani in teleoperazione (per ora il controller non le usa)")
    p.add_argument("--pose_model", choices=["lite", "full", "heavy"], default="lite",
                   help="Complessità del modello MediaPipe per la posa")
    p.add_argument("--min_cutoff", type=float, default=0.5, help="One Euro: cutoff minimo [Hz]")
    p.add_argument("--beta", type=float, default=0.01,
                   help="One Euro: sensibilità alla velocità (con dati in metri provare valori ~1-5)")
    # Robot
    p.add_argument("--xml", type=Path, default=DEFAULT_XML, help="Path dell'XML MuJoCo del robot")
    # Debug
    p.add_argument("--plot3d", action="store_true", help="Skeleton 3D live in matplotlib (lento)")
    p.add_argument("--profile", action="store_true", help="Stampa i tempi in ms di tracker e simulazione")
    p.add_argument("--debug", action="store_true", help="Controller: check geometria e log periodico degli errori")
    args = p.parse_args()

    if args.only_pose and args.only_hands:
        p.error("--only_pose e --only_hands sono incompatibili")
    if args.only_hands and not args.cam_only:
        p.error("--only_hands richiede --cam_only")
    if args.cam_only and args.model_only:
        p.error("--cam_only e --model_only sono incompatibili")
    return args


def main():
    args = parse_args()
    camera = tracker = controller = None
    try:
        if not args.model_only:
            camera, tracker = build_capture(args)
        if not args.cam_only:
            controller = build_controller(args)

        if args.model_only:
            run_model_only(controller)
        else:
            run_pipeline(args, camera, tracker, controller)
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