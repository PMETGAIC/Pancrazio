"""Retargeting umano -> robot: dalle pose MediaPipe ai comandi di giunto in MuJoCo.

Contenuto:
  - ArmDirectionExtractor : da `state['pose']` ai versori (braccio, avambraccio) per ogni braccio
  - MujocoController      : IK differenziale con QP (OSQP) + simulazione

Idea: del braccio umano si usano solo le DIREZIONI delle ossa. I target di gomito e polso si
costruiscono sul robot con le SUE lunghezze (misurate nella posa di riposo), quindi proporzioni
diverse tra umano e robot non creano errori.
"""
from dataclasses import dataclass

import mujoco
import numpy as np
import scipy.sparse as spa
from qpsolvers import solve_qp

BODY, JOINT = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT


# =============================================================================
# Dalla posa umana ai versori
# =============================================================================
class ArmDirectionExtractor:
    """Estrae, per ogni braccio, un array (2, 3) = [u_braccio, u_avambraccio] nel frame del robot.

    Bracci, ordine e frame vengono dal profilo (`profile.arms`, `profile.R_robot_from_mp`): lo stesso
    ordine è usato dal MujocoController. R è "egocentrica" (il robot si comporta come la persona che
    guarda la camera); per un robot a specchio si cambia R nello YAML.

    Se un braccio ha visibilità sotto `vis_min` (occluso / fuori inquadratura) si tiene l'ultima
    direzione valida invece di inseguire landmark inventati. Ritorna None se non c'è ancora nessuna
    direzione utilizzabile; un braccio senza dati vale None.
    """

    def __init__(self, profile, alpha=0.5, vis_min=0.5):
        self.arms = [list(a.mp_indices) for a in profile.arms]
        self.R = np.asarray(profile.R_robot_from_mp, dtype=float)
        self.alpha = alpha      # 1.0 = nessuno smoothing aggiuntivo (la posa è già filtrata a monte)
        self.vis_min = vis_min
        self.reset()

    def reset(self):
        self.last = [None] * len(self.arms)

    def __call__(self, state):
        pose, vis = state['pose'], state['visibility']
        if pose is None:
            return None
        for i, idx in enumerate(self.arms):
            if vis is not None and vis[idx].min() < self.vis_min:
                continue
            dirs = self._dirs(pose[idx], self.last[i])
            if dirs is not None:
                self.last[i] = dirs
        return list(self.last) if any(d is not None for d in self.last) else None

    def _dirs(self, pts, last):
        """pts: (3, 3) = spalla, gomito, polso -> (2, 3) versori smussati, o None se degeneri."""
        v = (pts[1:] - pts[:-1]) @ self.R.T
        n = np.linalg.norm(v, axis=1, keepdims=True)
        if (n < 1e-6).any():
            return None
        u = v / n
        if last is not None:
            u = self.alpha * u + (1 - self.alpha) * last
            u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-9
        return u


# =============================================================================
# Controller MuJoCo
# =============================================================================
@dataclass
class Arm:
    """Id dei body di un braccio e lunghezze delle ossa (misurate nella posa di riposo)."""
    shoulder: int
    elbow: int
    ee: int
    upper_len: float
    fore_len: float


class MujocoController:
    """IK differenziale per bracci robotici: un QP per frame, poi qualche passo di simulazione.

    arms : lista di ArmSpec (profilo). Il body "spalla" deve essere il CENTRO di rotazione della
           spalla e quello "gomito" il centro di rotazione del gomito: lo verifica `validate`.
    La posa di riposo è l'unione dei `rest_joints` dei bracci. Supporta solo giunti hinge/slide.
    """

    def __init__(self, xml_path, arms, debug=False):
        self.model = m = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(m)
        m.opt.gravity[:] = 0
        self.debug = debug
        self.qp_failures = 0
        self.last_err_cm = None  # ultimo errore [cm] per task: [gomito0, polso0, gomito1, polso1, ...]
        self._calls = 0

        self._read_joints()
        self.q_rest = m.qpos0.copy()
        for spec in arms:
            for name, val in spec.rest_joints.items():
                self.q_rest[m.jnt_qposadr[self._id(JOINT, name)]] = val
        self.reset()
        self.arms = self._measure_arms(arms)

    # ---- Inizializzazione ----------------------------------------------------
    def _read_joints(self):
        """Per ogni dof: indirizzo in qpos, nome e limiti (solo giunti a 1 DoF)."""
        m = self.model
        ok = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
        self.qadr = np.zeros(m.nv, dtype=int)
        self.dof_names = [''] * m.nv
        self.lo = np.full(m.nv, -np.inf)
        self.hi = np.full(m.nv, np.inf)
        for j in range(m.njnt):
            name = mujoco.mj_id2name(m, JOINT, j)
            if int(m.jnt_type[j]) not in ok:
                raise ValueError(f"giunto '{name}': il controller supporta solo hinge/slide "
                                 f"(un freejoint alla base va rimosso dall'XML)")
            d = m.jnt_dofadr[j]
            self.qadr[d], self.dof_names[d] = m.jnt_qposadr[j], name
            if m.jnt_limited[j]:
                self.lo[d], self.hi[d] = m.jnt_range[j]

    def _measure_arms(self, specs):
        x = self.data.xpos
        arms = []
        for s in specs:
            sh, el, ee = (self._id(BODY, n) for n in (s.shoulder, s.elbow, s.ee))
            arms.append(Arm(sh, el, ee,
                            float(np.linalg.norm(x[el] - x[sh])),
                            float(np.linalg.norm(x[ee] - x[el]))))
        print(f"Lunghezze braccio: {np.round([a.upper_len for a in arms], 3)} "
              f"avambraccio: {np.round([a.fore_len for a in arms], 3)}")
        return arms

    def _id(self, kind, name):
        i = mujoco.mj_name2id(self.model, kind, name)
        if i < 0:
            n = self.model.nbody if kind == BODY else self.model.njnt
            names = [mujoco.mj_id2name(self.model, kind, k) for k in range(n)]
            raise ValueError(f"'{name}' non trovato. Disponibili: {names}")
        return i

    # ---- API usata anche da validate -----------------------------------------
    def set_qpos(self, q):
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)

    def random_qpos(self, rng):
        """Configurazione casuale dentro i limiti (i giunti senza limiti restano a riposo)."""
        q = self.q_rest.copy()
        lim = np.isfinite(self.lo) & np.isfinite(self.hi)
        q[self.qadr[lim]] = rng.uniform(self.lo[lim], self.hi[lim])
        return q

    def jac(self, body_id):
        """Jacobiano di posizione (3, nv) dell'origine del body, coerente con data.xpos."""
        J = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, J, None, body_id)
        return J

    def reset(self):
        self.data.qvel[:] = 0
        self._set_ctrl(self.q_rest)
        self.set_qpos(self.q_rest)

    def _set_ctrl(self, q):
        """Imposta i target degli attuatori di posizione dal vettore qpos `q`."""
        m = self.model
        for i in range(m.nu):
            if int(m.actuator_trntype[i]) != int(mujoco.mjtTrn.mjTRN_JOINT):
                continue  # attuatori su tendine/altro (es. gripper): qui non gestiti
            c = q[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
            if m.actuator_ctrllimited[i]:
                c = np.clip(c, *m.actuator_ctrlrange[i])
            self.data.ctrl[i] = c

    # ---- Controllo -----------------------------------------------------------
    def track(self, arm_dirs, w_wrist=1.0, w_elbow=0.7, dq_max=0.2,
              damp=1e-3, posture_weight=1e-3, steps=10):
        """Un passo di IK + simulazione.

        arm_dirs : per ogni braccio (u_braccio, u_avambraccio) versori nel frame del robot,
                   oppure None per un braccio senza target (resta guidato dal solo termine di postura).
        dq_max   : massima variazione per giunto in un passo [rad o m].
        steps    : passi di mj_step per ogni chiamata.
        # TODO gripper: pilotare le dita da state['hands'][i]['openness'] / ['pinch'].
        """
        tasks = self._tasks(arm_dirs, w_elbow, w_wrist)
        if not tasks:
            return
        dq = self._solve(tasks, dq_max, damp, posture_weight)
        if dq is None:
            self.qp_failures += 1
            return
        self._apply(dq, steps)

    def _tasks(self, arm_dirs, w_elbow, w_wrist):
        """Target di gomito e polso: lunghezze del ROBOT, direzioni dell'umano."""
        x = self.data.xpos
        tasks = []  # (body_id, posizione_target, peso)
        for arm, dirs in zip(self.arms, arm_dirs):
            if dirs is None:
                continue
            u_up, u_fore = dirs
            elbow_t = x[arm.shoulder] + arm.upper_len * u_up
            wrist_t = elbow_t + arm.fore_len * u_fore
            tasks += [(arm.elbow, elbow_t, w_elbow), (arm.ee, wrist_t, w_wrist)]
        return tasks

    def _solve(self, tasks, dq_max, damp, posture_weight):
        """QP (diviso per 2, equivalente):
             min  sum_i w_i |J_i dq - e_i|^2 + damp |dq|^2 + posture |q + dq - q_rest|^2
             s.t. limiti dei giunti e |dq| <= dq_max
        """
        d = self.data
        q = d.qpos[self.qadr]
        A = (damp + posture_weight) * np.eye(self.model.nv)
        b = posture_weight * (q - self.q_rest[self.qadr])
        errs = []
        for body, target, w in tasks:
            J = self.jac(body)
            e = target - d.xpos[body]
            errs.append(np.linalg.norm(e))
            A += w * J.T @ J
            b -= w * J.T @ e
        self.last_err_cm = np.array(errs) * 100

        # Limiti sul passo: intersezione tra |dq| <= dq_max e limiti di giunto (sempre contenente 0)
        lb = np.minimum(np.maximum(-dq_max, self.lo - q), 0.0)
        ub = np.maximum(np.minimum(dq_max, self.hi - q), 0.0)
        self._log(q)
        return solve_qp(spa.csc_matrix(A), b, lb=lb, ub=ub, solver="osqp",
                        eps_abs=1e-6, eps_rel=1e-6)

    def _apply(self, dq, steps):
        q_target = self.data.qpos.copy()
        q_target[self.qadr] += dq
        if self.model.nu > 0:
            self._set_ctrl(q_target)
            for _ in range(steps):
                mujoco.mj_step(self.model, self.data)
        else:
            self.set_qpos(q_target)  # modalità cinematica

    def _log(self, q):
        if not self.debug:
            return
        self._calls += 1
        if self._calls % 30:
            return
        at_lim = [n for n, qi, lo, hi in zip(self.dof_names, q, self.lo, self.hi)
                  if qi <= lo + 0.01 or qi >= hi - 0.01]
        print(f"[dbg] err (gomito, polso, ...) {np.round(self.last_err_cm, 1)} cm "
              f"| contatti {self.data.ncon} | qp falliti {self.qp_failures} | ai limiti: {at_lim}")