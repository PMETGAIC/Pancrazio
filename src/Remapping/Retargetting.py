import numpy as np
import mujoco
import scipy.sparse as spa
from qpsolvers import solve_qp


class MujocoController:
    def __init__(self, xml_path, ee_names, elbow_names, shoulder_names, rest_joints=None, debug=False):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.gravity[:] = [0, 0, 0]
        self.debug = debug
        self._calls = 0
        m = self.model

        self.ee_ids = [self._id(mujoco.mjtObj.mjOBJ_BODY, n) for n in ee_names]
        self.elbow_ids = [self._id(mujoco.mjtObj.mjOBJ_BODY, n) for n in elbow_names]
        self.shoulder_ids = [self._id(mujoco.mjtObj.mjOBJ_BODY, n) for n in shoulder_names]

        # dof -> indirizzo qpos, nome e limiti (solo giunti a 1 DoF)
        ok_types = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
        self._qadr = np.zeros(m.nv, dtype=int)
        self._dof_names = [''] * m.nv
        self._lo = np.full(m.nv, -np.inf)
        self._hi = np.full(m.nv, np.inf)
        for j in range(m.njnt):
            if int(m.jnt_type[j]) not in ok_types:
                raise ValueError("Questo controller supporta solo giunti hinge/slide")
            d = m.jnt_dofadr[j]
            self._qadr[d] = m.jnt_qposadr[j]
            self._dof_names[d] = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if m.jnt_limited[j]:
                self._lo[d], self._hi[d] = m.jnt_range[j]

        # posa di riposo con gomito piegato
        self.q_rest = m.qpos0.copy()
        for name, val in (rest_joints or {}).items():
            jid = self._id(mujoco.mjtObj.mjOBJ_JOINT, name)
            self.q_rest[m.jnt_qposadr[jid]] = val
        self.reset()

        # lunghezze delle ossa del robot, misurate nella posa di riposo
        x = self.data.xpos
        self.upper_len = [np.linalg.norm(x[e] - x[s]) for s, e in zip(self.shoulder_ids, self.elbow_ids)]
        self.fore_len = [np.linalg.norm(x[w] - x[e]) for e, w in zip(self.elbow_ids, self.ee_ids)]
        print(f"Lunghezze braccio: {np.round(self.upper_len, 3)} avambraccio: {np.round(self.fore_len, 3)}")

        if self.debug:
            self.check_geometry()

    def _id(self, kind, name):
        i = mujoco.mj_name2id(self.model, kind, name)
        if i < 0:
            n = self.model.nbody if kind == mujoco.mjtObj.mjOBJ_BODY else self.model.njnt
            names = [mujoco.mj_id2name(self.model, kind, k) for k in range(n)]
            raise ValueError(f"'{name}' non trovato. Disponibili: {names}")
        return i

    def _set_ctrl(self, q):
        m = self.model
        for i in range(m.nu):
            jid = m.actuator_trnid[i, 0]
            c = q[m.jnt_qposadr[jid]]
            if m.actuator_ctrllimited[i]:
                c = np.clip(c, *m.actuator_ctrlrange[i])
            self.data.ctrl[i] = c

    def reset(self):
        self.data.qpos[:] = self.q_rest
        self.data.qvel[:] = 0
        self._set_ctrl(self.q_rest)
        mujoco.mj_forward(self.model, self.data)

    def check_geometry(self, n=200, seed=0):
        """Campiona configurazioni casuali dentro i limiti. Se spalla-gomito o gomito-polso
        variano di piu' di ~1 cm, il body scelto non e' il centro di rotazione giusto
        e i target di gomito/polso risultano incoerenti."""
        rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        lim = np.isfinite(self._lo) & np.isfinite(self._hi)
        up, fo = [], []
        for _ in range(n):
            q = self.q_rest.copy()
            q[self._qadr[lim]] = rng.uniform(self._lo[lim], self._hi[lim])
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            x = d.xpos
            up.append([np.linalg.norm(x[e] - x[s]) for s, e in zip(self.shoulder_ids, self.elbow_ids)])
            fo.append([np.linalg.norm(x[w] - x[e]) for e, w in zip(self.elbow_ids, self.ee_ids)])
        up, fo = np.array(up) * 100, np.array(fo) * 100
        print(f"[geom] spalla-gomito (cm)  min {np.round(up.min(0), 1)}  max {np.round(up.max(0), 1)}")
        print(f"[geom] gomito-polso  (cm)  min {np.round(fo.min(0), 1)}  max {np.round(fo.max(0), 1)}")
        self.reset()

    def track(self, arm_dirs, w_wrist=1.0, w_elbow=0.7, dq_max=0.2,
              damp=1e-3, posture_weight=1e-3, steps=10):
        """arm_dirs: per ogni braccio (u_braccio, u_avambraccio), versori nel frame del robot."""
        m, d = self.model, self.data

        tasks = []
        for k, (u_up, u_fore) in enumerate(arm_dirs):
            elbow_t = d.xpos[self.shoulder_ids[k]] + self.upper_len[k] * u_up
            wrist_t = elbow_t + self.fore_len[k] * u_fore
            tasks.append((self.elbow_ids[k], elbow_t, w_elbow))
            tasks.append((self.ee_ids[k], wrist_t, w_wrist))

        # min  sum w|J dq - e|^2 + damp|dq|^2 + posture|q + dq - q_rest|^2
        A = (damp + posture_weight) * np.eye(m.nv)
        b = np.zeros(m.nv)
        errs = []
        for body_id, target, w in tasks:
            J = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, J, None, body_id)  # origine del body, coerente con xpos
            e = target - d.xpos[body_id]
            errs.append(np.linalg.norm(e))
            A += w * J.T @ J
            b -= w * J.T @ e

        q = d.qpos[self._qadr]
        b += posture_weight * (q - self.q_rest[self._qadr])

        lb = np.minimum(np.maximum(-dq_max, self._lo - q), 0.0)
        ub = np.maximum(np.minimum(dq_max, self._hi - q), 0.0)

        if self.debug:
            self._calls += 1
            if self._calls % 30 == 0:
                e_cm = np.array(errs) * 100
                at_lim = [self._dof_names[i] for i in range(m.nv)
                          if q[i] <= self._lo[i] + 0.01 or q[i] >= self._hi[i] - 0.01]
                print(f"[dbg] err gomito {np.round(e_cm[0::2], 1)} cm | polso {np.round(e_cm[1::2], 1)} cm "
                      f"| contatti {d.ncon} | ai limiti: {at_lim}")

        dq = solve_qp(spa.csc_matrix(A), b, lb=lb, ub=ub, solver="osqp",
                      eps_abs=1e-6, eps_rel=1e-6)
        if dq is None:
            return

        q_target = d.qpos.copy()
        q_target[self._qadr] += dq
        if m.nu > 0:
            self._set_ctrl(q_target)
            for _ in range(steps):
                mujoco.mj_step(m, d)
        else:
            d.qpos[:] = q_target
            mujoco.mj_forward(m, d)