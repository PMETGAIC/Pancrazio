"""Teleoperazione di un robot con MediaPipe + MuJoCo: webcam/video -> posa -> retargeting -> simulazione."""
import argparse
import contextlib
import time
import urllib.request
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer

from src.Capture.human_capture import PoseTracker, Skeleton3DPlot, VideoIO, draw_overlay
from src.Remapping.retargeting import ArmDirectionExtractor, MujocoController
from src.Remapping.robot_profile import RobotProfile
from src.Remapping.validate import validate

MODELS_DIR = Path("./models")
BASE_URL = "https://storage.googleapis.com/mediapipe-models"
HAND_URL = f"{BASE_URL}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
DEFAULT_ROBOT = Path("profiles/openarm.yaml")

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


def build_robot(args):
    """Profilo -> (controller, estrattore di direzioni). L'ordine dei bracci viene dal profilo."""
    profile = RobotProfile.from_yaml(args.robot)
    controller = MujocoController(args.xml or profile.xml, profile.arms, debug=args.debug)
    if not args.no_validate and not validate(controller)[0]:  # stampa già esito, errori e avvisi
        return None, None
    return controller, ArmDirectionExtractor(profile, alpha=DIR_ALPHA)


# =============================================================================
# Loop
# =============================================================================
def run_model_only(controller):
    """Solo il viewer del modello, senza video."""
    with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(1 / 60)


def run_pipeline(args, camera, tracker, controller, extractor):
    """Loop unico: camera -> tracking -> (opz.) retargeting + viewer. Con controller=None è solo cattura."""
    plot = Skeleton3DPlot() if args.plot3d else None
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
                dirs = extractor(state)
                if dirs is not None:
                    with viewer.lock():
                        controller.track(dirs)
                viewer.sync()
                acc_qp += time.perf_counter() - t_qp

            if args.timing:
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
    p.add_argument("--robot", type=Path, default=DEFAULT_ROBOT,
                   help=f"Profilo robot YAML (default: {DEFAULT_ROBOT}); bozza con src.Remapping.discover")
    p.add_argument("--xml", type=Path, default=None, help="Override del path XML MuJoCo indicato nel profilo")
    p.add_argument("--no_validate", action="store_true", help="Salta il controllo delle assunzioni del retargeting")
    # Debug
    p.add_argument("--plot3d", action="store_true", help="Skeleton 3D live in matplotlib (lento)")
    p.add_argument("--timing", action="store_true", help="Stampa i tempi in ms di tracker e simulazione")
    p.add_argument("--debug", action="store_true", help="Controller: log periodico degli errori IK")
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
    camera = tracker = controller = extractor = None
    try:
        if not args.model_only:
            camera, tracker = build_capture(args)
        if not args.cam_only:
            controller, extractor = build_robot(args)
            if controller is None:
                print("Il modello non soddisfa le assunzioni del retargeting: correggi il profilo "
                      "o l'XML (oppure usa --no_validate a tuo rischio).")
                return

        if args.model_only:
            run_model_only(controller)
        else:
            run_pipeline(args, camera, tracker, controller, extractor)
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