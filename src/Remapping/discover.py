"""discover.py: propone nomi di body e giunti per il retargeting leggendo il modello MuJoCo/URDF.

Uso:
    python -m src.Remapping.discover path/robot.xml > profiles/robot.yaml

Assunzioni (se non valgono, correggi lo YAML a mano):
  - bracci seriali con giunti hinge
  - pinza = 2+ body con giunto che condividono lo stesso genitore e non hanno altri giunti sotto;
    l'ee è il body da cui partono le dita
  - base con y verso sinistra, z su (serve solo per distinguere sinistro e destro)
  - giunti con nome (gli URDF li hanno sempre; negli MJCF vanno nominati)
Non gestisce: umanoidi con giunti di busto in comune ai due bracci, pinze con un solo dito.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from src.Remapping.robot_profile import ArmSpec, RobotProfile

HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
BODY, JOINT = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT
MP_ARMS = {"left": (11, 13, 15), "right": (12, 14, 16)}


@dataclass
class Found:
    arm: ArmSpec
    chain: list      # nomi dei giunti, dalla radice al polso
    gripper: bool
    score: float     # [m] il più corto tra braccio e avambraccio: indica quanto è credibile il gomito
    y: float         # [m] posizione laterale della spalla, per ordinare sinistro/destro


def _name(m, kind, i):
    n = mujoco.mj_id2name(m, kind, i)
    if n is None:
        raise ValueError(f"{kind.name} #{i} senza nome: nominalo nel modello")
    return n


class _JointTree:
    """Il modello visto come albero dei soli body con giunto hinge."""

    def __init__(self, m):
        self.m = m
        self.joints = {}  # body -> giunti hinge che porta
        for j in range(m.njnt):
            if int(m.jnt_type[j]) == HINGE:
                self.joints.setdefault(int(m.jnt_bodyid[j]), []).append(j)
        self.children = {b: [] for b in range(m.nbody)}
        for b in range(1, m.nbody):
            self.children[int(m.body_parentid[b])].append(b)
        self.jchildren = {b: [] for b in self.joints}
        for b in self.joints:
            p = self.jparent(b)
            if p is not None:
                self.jchildren[p].append(b)

    def jparent(self, b):
        """Antenato più vicino che porta un giunto hinge."""
        p = int(self.m.body_parentid[b])
        while p != 0:
            if p in self.joints:
                return p
            p = int(self.m.body_parentid[p])
        return None

    def has_joint_below(self, b):
        return self.m.body_jntnum[b] > 0 or any(self.has_joint_below(c) for c in self.children[b])

    def chain(self, leaf):
        """Giunti dalla radice alla foglia."""
        bodies, b = [], leaf
        while b is not None:
            bodies.append(b)
            b = self.jparent(b)
        return [j for b in reversed(bodies) for j in self.joints[b]]

    def grippers(self):
        """({body con giunto: body ee}, insieme dei body-dito)."""
        ee, fingers = {}, set()
        for p, kids in self.jchildren.items():
            if len(kids) >= 2 and not any(self.jchildren[k] for k in kids):
                parents = {int(self.m.body_parentid[k]) for k in kids}
                ee[p] = parents.pop() if len(parents) == 1 else p
                fingers.update(kids)
        return ee, fingers

    def arm_leaves(self, fingers):
        return [b for b in self.joints
                if b not in fingers and all(k in fingers for k in self.jchildren[b])]

    def descend(self, ee):
        """Senza pinza: si scende finché c'è un solo figlio senza giunti sotto di sé."""
        while len(self.children[ee]) == 1 and not self.has_joint_below(self.children[ee][0]):
            ee = self.children[ee][0]
        return ee


def _pick_elbow(m, xpos, chain, ee):
    """Gomito = giunto della catena che massimizza il più corto tra braccio e avambraccio."""
    best = None
    for k in range(1, len(chain) - 1):
        sh, el = int(m.jnt_bodyid[chain[k - 1]]), int(m.jnt_bodyid[chain[k]])
        score = min(np.linalg.norm(xpos[el] - xpos[sh]), np.linalg.norm(xpos[ee] - xpos[el]))
        if best is None or score > best[0]:
            best = (score, k, sh, el)
    return best  # (score, indice nella catena, body spalla, body gomito)


def _rest_angle(m, j):
    """Gomito piegato dal lato permesso dai limiti (40% del limite); 0.8 rad se senza limiti."""
    if not m.jnt_limited[j]:
        return 0.8
    lo, hi = m.jnt_range[j]
    return round(float(0.4 * (hi if hi >= -lo else lo)), 2)


def discover_arms(model, min_joints=4):
    """Ritorna la lista di `Found`, ordinata sinistro -> destro, con nome e mp_indices assegnati."""
    m = model
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)  # posa qpos0
    tree = _JointTree(m)
    grip_ee, fingers = tree.grippers()

    found = []
    for leaf in tree.arm_leaves(fingers):
        chain = tree.chain(leaf)
        if len(chain) < min_joints:
            continue
        ee = grip_ee.get(leaf) or tree.descend(int(m.jnt_bodyid[chain[-1]]))
        score, k, sh, el = _pick_elbow(m, d.xpos, chain, ee)
        arm = ArmSpec(name="", mp_indices=(),
                      shoulder=_name(m, BODY, sh), elbow=_name(m, BODY, el), ee=_name(m, BODY, ee),
                      rest_joints={_name(m, JOINT, chain[k]): _rest_angle(m, chain[k])})
        found.append(Found(arm, [_name(m, JOINT, c) for c in chain], leaf in grip_ee,
                           float(score), float(d.xpos[sh][1])))

    found.sort(key=lambda f: -f.y)  # y decrescente: sinistro prima del destro
    names = ["left", "right"] if len(found) == 2 else [f"arm{i}" for i in range(len(found))]
    for f, name in zip(found, names):
        f.arm.name, f.arm.mp_indices = name, MP_ARMS.get(name, ())
    return found


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = Path(sys.argv[1])
    found = discover_arms(mujoco.MjModel.from_xml_path(str(path)))
    if not found:
        sys.exit("Nessuna catena con >= 4 giunti hinge trovata")

    # Il report va su stderr, così stdout resta uno YAML pulito
    for f in found:
        print(f"[discover] {f.arm.name}: {' > '.join(f.chain)}{'  (+ pinza)' if f.gripper else ''}\n"
              f"           spalla={f.arm.shoulder} gomito={f.arm.elbow} ee={f.arm.ee} "
              f"(bone più corto {f.score * 100:.1f} cm)", file=sys.stderr)
    if len(found) != 2:
        print(f"[discover] ATTENZIONE: trovati {len(found)} bracci: mp_indices va compilato a mano",
              file=sys.stderr)
    print(RobotProfile(path, [f.arm for f in found]).to_yaml())