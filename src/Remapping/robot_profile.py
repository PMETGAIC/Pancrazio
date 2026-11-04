"""robot_profile.py: descrive un robot per il retargeting (YAML <-> oggetti Python).

Un profilo contiene: path del modello, matrice di frame MediaPipe -> robot e, per ogni braccio,
indici MediaPipe (spalla, gomito, polso), nomi dei body MuJoCo e posa di riposo.
Bozza automatica: `python -m src.Remapping.discover robot.xml > profiles/robot.yaml`.
"""
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

# [x, y, z] MediaPipe -> [-z, x, -y]: "egocentrica", il robot imita come se fosse la persona
DEFAULT_R = [[0, 0, -1], [1, 0, 0], [0, -1, 0]]
PREFIX = "${PREFIX}"  # nei path dello YAML sta per sys.prefix (il venv), così il file è portabile


def _expand(path):
    return Path(str(path).replace(PREFIX, sys.prefix)).expanduser()


def _portable(path):
    p = Path(path).resolve()
    try:
        return f"{PREFIX}/" + p.relative_to(Path(sys.prefix).resolve()).as_posix()
    except ValueError:
        return str(p)


@dataclass
class ArmSpec:
    name: str
    mp_indices: tuple                 # (spalla, gomito, polso) in MediaPipe Pose
    shoulder: str                     # body MuJoCo: centro di rotazione della spalla
    elbow: str                        # body MuJoCo: centro di rotazione del gomito
    ee: str                           # body MuJoCo: polso / end effector
    rest_joints: dict = field(default_factory=dict)


@dataclass
class RobotProfile:
    xml: Path
    arms: list                        # lista di ArmSpec: questo ordine vale ovunque
    R_robot_from_mp: np.ndarray = field(default_factory=lambda: np.array(DEFAULT_R, dtype=float))

    @classmethod
    def from_yaml(cls, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Profilo robot non trovato: {path}\n"
                f"Generane una bozza con: python -m src.Remapping.discover <robot.xml> > {path}")
        cfg = yaml.safe_load(path.read_text())
        arms = [ArmSpec(name=a["name"], mp_indices=tuple(a.get("mp_indices", ())),
                        shoulder=a["shoulder"], elbow=a["elbow"], ee=a["ee"],
                        rest_joints=a.get("rest_joints", {}))
                for a in cfg["arms"]]
        profile = cls(_expand(cfg["xml"]), arms, np.array(cfg.get("R_robot_from_mp", DEFAULT_R), float))
        profile._check()
        return profile

    def _check(self):
        if not self.arms:
            raise ValueError("il profilo non contiene bracci")
        for a in self.arms:
            if len(a.mp_indices) != 3:
                raise ValueError(f"braccio '{a.name}': mp_indices deve avere 3 indici "
                                 f"(spalla, gomito, polso), trovati {a.mp_indices}")
        R = self.R_robot_from_mp
        if R.shape != (3, 3) or not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
            raise ValueError("R_robot_from_mp deve essere una matrice 3x3 ortogonale")

    def to_yaml(self):
        R = [[int(v) if float(v).is_integer() else round(float(v), 6) for v in row]
             for row in self.R_robot_from_mp]
        cfg = {"xml": _portable(self.xml),
               "R_robot_from_mp": R,
               "arms": [{**asdict(a), "mp_indices": list(a.mp_indices)} for a in self.arms]}
        return yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None)