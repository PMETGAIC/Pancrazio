"""Retargeting umano -> robot: dalle pose MediaPipe ai comandi di giunto in MuJoCo.

Contenuto:
  - mediapipe_to_robot      : cambio di frame MediaPipe -> robot
  - ArmDirectionExtractor   : da `state['pose']` ai versori (braccio, avambraccio) per ogni braccio
  - MujocoController        : IK differenziale con QP (OSQP) + simulazione

Idea: del braccio umano si usano solo le DIREZIONI delle ossa. I target di gomito e polso si
costruiscono sul robot con le SUE lunghezze (misurate nella posa di riposo), quindi proporzioni
diverse tra umano e robot non creano errori.
"""
import mujoco
import numpy as np
import scipy.sparse as spa
from qpsolvers import solve_qp


# =============================================================================
# Dalla posa umana ai versori
# =============================================================================
def mediapipe_to_robot(v):
    """Vettore nel frame world di MediaPipe -> frame del robot (x avanti, y sinistra, z su).

    Mappatura "egocentrica": il robot si comporta come se fosse la persona che guarda la camera.
    MediaPipe: x verso destra dell'immagine, y verso il basso, z verso l'allontanamento dalla camera;
    la persona guarda verso la camera, quindi il suo "avanti" è -z e la sua sinistra è +x.
    Per un robot che imita "a specchio" basta cambiare questa sola funzione.
    """
    return np.array([-v[2], v[0], -v[1]])


class ArmDirectionExtractor:
    """Estrae, per ogni braccio, (u_braccio, u_avambraccio) nel frame del robot, con smoothing.

    ARMS è l'ordine dei bracci e DEVE coincidere con l'ordine di ee/elbow/shoulder_names del
    MujocoController: indici MediaPipe (spalla, gomito, polso) di sinistro e destro.

    Se un braccio ha visibilità sotto `vis_min` (occluso / fuori inquadratura) si tiene l'ultima
    direzione valida invece di inseguire landmark inventati. Ritorna None se non c'è ancora
    nessuna direzione utilizzabile; un braccio senza dati vale None.
    """
    ARMS = ((11, 13, 15), (12, 14, 16))

    def __init__(self, alpha=0.5, vis_min=0.5):
        self.alpha = alpha      # 1.0 = nessuno smoothing aggiuntivo (la posa è già filtrata a monte)
        self.vis_min = vis_min
        self.last = [None] * len(self.ARMS)

    def reset(self):
        self.last = [None] * len(self.ARMS)

    def __call__(self, state):
        pose, vis = state['pose'], state['visibility']
        if pose is None:
            return None
        for i, idx in enumerate(self.ARMS):
            if vis is not None and vis[list(idx)].min() < self.vis_min:
                continue
            dirs = self._arm_dirs(pose, idx, self.last[i])
            if dirs is not None:
                self.last[i] = dirs
        if all(d is None for d in self.last):
            return None
        return list(self.last)

    def _arm_dirs(self, pose, idx, last):
        sh, el, wr = idx
        out = []
        for j, (a, b) in enumerate(((sh, el), (el, wr))):
            u = mediapipe_to_robot(pose[b] - pose[a])
            n = np.linalg.norm(u)
            if n < 1e-6:
                return None
            u = u / n
            if last is not None:
                u = self.alpha * u + (1 - self.alpha) * last[j]
                u = u / (np.linalg.norm(u) + 1e-9)
            out.append(u)
        return tuple(out)


# =============================================================================
# Controller MuJoCo
# =============================================================================
class MujocoController:
    """IK differenziale per bracci robotici: un QP per frame, poi qualche passo di simulazione.

    ee_names / elbow_names / shoulder_names : body MuJoCo (stesso ordine dei bracci dell'estrattore).
        Il body "spalla" deve essere il CENTRO di rotazione della spalla e quello "gomito" il
        centro di rotazione del gomito: verificalo con debug=True (check_geometry).
    rest_joints : {nome_giunto: valore} della posa di riposo (es. gomito piegato).
    Supporta solo giunti hinge/slide.
    """

    def __init__(self, xml_path, ee_names, elbow_names, shoulder_names, rest_joints=None, debug=False):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.gravity[:] = [0, 0, 0]
        self.debug = debug
        self.qp_failures = 0
        self.last_err_cm = None  # ultimo errore [cm] per task: [gomito0, polso0, gomito1, polso1, ...]
        self._calls = 0
        m = self.model

        self.ee_ids = [self._id(mujoco.mjtObj.mjOBJ_BODY, n) for n in ee_names]
        self.elbow_ids = [self._id(mujoco.mjtObj.mjOBJ_BODY, n) for n in elbow_names]
        self.shoulder_ids = [self._id(mujoco.mjtObj.mjOBJ_BODY, n) for n in shoulder_names]

        # Per ogni dof: indirizzo in qpos, nome e limiti (solo giunti a 1 DoF)
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

        # Posa di riposo (con gomito piegato per evitare la singolarità a braccio teso)
        self.q_rest = m.qpos0.copy()
        for name, val in (rest_joints or {}).items():
            jid = self._id(mujoco.mjtObj.mjOBJ_JOINT, name)
            self.q_rest[m.jnt_qposadr[jid]] = val
        self.reset()

        # Lunghezze delle ossa del robot, misurate nella posa di riposo
        x = self.data.xpos
        self.upper_len = [np.linalg.norm(x[e] - x[s]) for s, e in zip(self.shoulder_ids, self.elbow_ids)]
        self.fore_len = [np.linalg.norm(x[w] - x[e]) for e, w in zip(self.elbow_ids, self.ee_ids)]
        print(f"Lunghezze braccio: {np.round(self.upper_len, 3)} avambraccio: {np.round(self.fore_len, 3)}")

        if self.debug:
            self.check_geometry()

    # ---- Utilità -------------------------------------------------------------
    def _id(self, kind, name):
        i = mujoco.mj_name2id(self.model, kind, name)
        if i < 0:
            n = self.model.nbody if kind == mujoco.mjtObj.mjOBJ_BODY else self.model.njnt
            names = [mujoco.mj_id2name(self.model, kind, k) for k in range(n)]
            raise ValueError(f"'{name}' non trovato. Disponibili: {names}")
        return i

    def _set_ctrl(self, q):
        """Imposta i target degli attuatori di posizione dal vettore qpos `q`."""
        m = self.model
        for i in range(m.nu):
            if int(m.actuator_trntype[i]) != int(mujoco.mjtTrn.mjTRN_JOINT):
                continue  # attuatori su tendine/altro (es. gripper): qui non gestiti
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
        variano di più di ~1 cm, il body scelto non è il centro di rotazione giusto
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

    # ---- Controllo -----------------------------------------------------------
    def track(self, arm_dirs, w_wrist=1.0, w_elbow=0.7, dq_max=0.2,
              damp=1e-3, posture_weight=1e-3, steps=10):
        """Un passo di IK + simulazione.

        arm_dirs : per ogni braccio (u_braccio, u_avambraccio) versori nel frame del robot,
                   oppure None per un braccio senza target (resta guidato solo dal termine di postura).
        dq_max   : massima variazione per giunto in un passo [rad o m].
        steps    : passi di mj_step per ogni chiamata.

        QP (diviso per 2, equivalente):
          min  sum_i w_i |J_i dq - e_i|^2 + damp |dq|^2 + posture |q + dq - q_rest|^2
          s.t. limiti dei giunti e |dq| <= dq_max
        # TODO gripper: pilotare le dita da state['hands'][i]['openness'] / ['pinch'].
        """
        m, d = self.model, self.data

        tasks = []  # (body_id, posizione_target, peso)
        for k, dirs in enumerate(arm_dirs):
            if dirs is None:
                continue
            u_up, u_fore = dirs
            elbow_t = d.xpos[self.shoulder_ids[k]] + self.upper_len[k] * u_up
            wrist_t = elbow_t + self.fore_len[k] * u_fore
            tasks.append((self.elbow_ids[k], elbow_t, w_elbow))
            tasks.append((self.ee_ids[k], wrist_t, w_wrist))
        if not tasks:
            return

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
        self.last_err_cm = np.array(errs) * 100

        q = d.qpos[self._qadr]
        b += posture_weight * (q - self.q_rest[self._qadr])

        # Limiti sul passo: intersezione tra |dq| <= dq_max e limiti di giunto (sempre contenente 0)
        lb = np.minimum(np.maximum(-dq_max, self._lo - q), 0.0)
        ub = np.maximum(np.minimum(dq_max, self._hi - q), 0.0)

        if self.debug:
            self._calls += 1
            if self._calls % 30 == 0:
                e_cm = self.last_err_cm
                at_lim = [self._dof_names[i] for i in range(m.nv)
                          if q[i] <= self._lo[i] + 0.01 or q[i] >= self._hi[i] - 0.01]
                print(f"[dbg] err (gomito, polso, ...) {np.round(e_cm, 1)} cm "
                      f"| contatti {d.ncon} | qp falliti {self.qp_failures} | ai limiti: {at_lim}")

        dq = solve_qp(spa.csc_matrix(A), b, lb=lb, ub=ub, solver="osqp",
                      eps_abs=1e-6, eps_rel=1e-6)
        if dq is None:
            self.qp_failures += 1
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