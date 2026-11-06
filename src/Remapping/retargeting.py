"""Retargeting umano -> robot: dalle pose MediaPipe ai comandi di giunto in MuJoCo.

Contenuto:
  - ArmDirectionExtractor : da `state['pose']` ai versori (braccio, avambraccio) per ogni braccio
  - HandCommandExtractor  : da `state['hands']` all'orientamento del palmo e all'apertura, per ogni braccio
  - MujocoController      : IK differenziale con QP (OSQP) + simulazione

Idea: del braccio umano si usano solo le DIREZIONI delle ossa. I target di gomito e polso si
costruiscono sul robot con le SUE lunghezze (misurate nella posa di riposo), quindi proporzioni
diverse tra umano e robot non creano errori.

Mano: l'orientamento del palmo è usato in modo RELATIVO a una posa di calibrazione, così non serve
conoscere la convenzione degli assi del body end effector. La calibrazione è automatica (media su
alcuni frame con la mano in posizione neutra, `start_auto_calibration`) oppure istantanea
(`calibrate`, tasto 'c'). L'apertura (pinch o openness in [0, 1]) comanda i giunti delle dita,
cioè quelli dei body discendenti dall'end effector.
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np
import scipy.sparse as spa
from qpsolvers import solve_qp
from scipy.spatial.transform import Rotation, Slerp

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
# Dalla mano umana a orientamento e apertura
# =============================================================================
class HandCommandExtractor:
    """Per ogni braccio (stesso ordine di `profile.arms`) estrae orientamento del palmo e apertura.

    Ritorna una lista di dict {'R': (3, 3) o None, 'open': float o None, 'fresh': bool}:
      - 'R'     : orientamento ASSOLUTO del palmo nel frame del robot (assi: x attraverso il palmo da
                  mignolo a indice, y verso le dita, z = x cross y). Il controller lo usa in modo relativo
                  rispetto alla posa di calibrazione.
      - 'open'  : segnale scelto (`pinch` o `openness`) in [0, 1], 0 = chiuso, 1 = aperto.
      - 'fresh' : True se 'R' è stato calcolato in QUESTO frame (mano rilevata), False se è l'ultimo
                  valore tenuto. Serve alla calibrazione automatica, che deve mediare solo misure vere.

    La mano di un braccio è quella il cui lato corrisponde al polso del braccio nel profilo
    (landmark 15 = sinistra del soggetto, 16 = destra). Se la mano non è rilevata si tiene l'ultimo
    valore valido, come fa ArmDirectionExtractor.
    """

    SIDE_OF_WRIST = {15: 'left', 16: 'right'}  # landmark del polso MediaPipe -> lato del soggetto

    def __init__(self, profile, alpha=0.5, signal='pinch', min_palm_sin=0.2):
        self.sides = [self.SIDE_OF_WRIST.get(int(a.mp_indices[2])) for a in profile.arms]
        self.R = np.asarray(profile.R_robot_from_mp, dtype=float)
        self.alpha = alpha  # 1.0 = nessuno smoothing aggiuntivo
        self.signal = signal
        self.min_palm_sin = min_palm_sin  # sotto questo seno l'angolo tra gli assi è degenere
        self.reset()

    def reset(self):
        n = len(self.sides)
        self.R_last = [None] * n
        self.open_last = [None] * n

    def __call__(self, state):
        hands = {h['side']: h for h in state['hands']}
        fresh = [False] * len(self.sides)
        for i, side in enumerate(self.sides):
            h = hands.get(side)
            if h is None:
                continue
            o, prev_o = float(h[self.signal]), self.open_last[i]
            self.open_last[i] = o if prev_o is None else self.alpha * o + (1 - self.alpha) * prev_o

            R = self._palm_frame(h['pts'])
            if R is not None:
                prev_R = self.R_last[i]
                self.R_last[i] = R if prev_R is None else self._slerp(prev_R, R, self.alpha)
                fresh[i] = True
        return [{'R': R, 'open': o, 'fresh': f}
                for R, o, f in zip(self.R_last, self.open_last, fresh)]

    def _palm_frame(self, pts):
        """pts (21, 3) -> matrice (3, 3) con gli assi del palmo come colonne (frame robot), o None."""
        x = (pts[5] - pts[17]) @ self.R.T   # attraverso il palmo: mignolo -> indice
        y = (pts[9] - pts[0]) @ self.R.T    # polso -> nocca del medio: lungo le dita
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        if nx < 1e-6 or ny < 1e-6:
            return None
        x, y = x / nx, y / ny
        z = np.cross(x, y)
        nz = np.linalg.norm(z)
        if nz < self.min_palm_sin:
            return None
        z /= nz
        y = np.cross(z, x)  # ortogonalizza: terna destra x, y, z
        return np.column_stack((x, y, z))

    @staticmethod
    def _slerp(prev, new, a):
        """Interpolazione sferica tra due rotazioni: a = 1 -> new, a = 0 -> prev."""
        rots = Rotation.from_matrix(np.stack([prev, new]))
        return Slerp([0.0, 1.0], rots)([a]).as_matrix()[0]


# =============================================================================
# Controller MuJoCo
# =============================================================================
@dataclass
class Arm:
    """Id dei body di un braccio, lunghezze delle ossa (posa di riposo), dita e calibrazione mano."""
    shoulder: int
    elbow: int
    ee: int
    upper_len: float
    fore_len: float
    grip: list = field(default_factory=list)  # [(indirizzo qpos, lo, hi)] dei giunti delle dita
    R_hand0: np.ndarray = None  # calibrazione: orientamento della mano umana (frame robot) ...
    R_ee0: np.ndarray = None    # ... e dell'end effector (frame mondo) nello stesso istante


class MujocoController:
    """IK differenziale per bracci robotici: un QP per frame, poi qualche passo di simulazione.

    arms        : lista di ArmSpec (profilo). Il body "spalla" deve essere il CENTRO di rotazione della
                  spalla e quello "gomito" il centro di rotazione del gomito: lo verifica `validate`.
    grip_invert : se True l'apertura è invertita (limite basso dei giunti delle dita = aperto).
    La posa di riposo è l'unione dei `rest_joints` dei bracci. Supporta solo giunti hinge/slide.
    """

    def __init__(self, xml_path, arms, debug=False, grip_invert=False):
        self.model = m = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(m)
        m.opt.gravity[:] = 0
        self.debug = debug
        self.grip_invert = grip_invert
        self.qp_failures = 0
        self.last_err_cm = None  # ultimo errore [cm] per task: [gomito0, polso0, gomito1, polso1, ...]
        self.last_rot_err_deg = None  # ultimo errore di orientamento [gradi] per ogni task di orientamento
        self._auto = None  # stato della calibrazione automatica dell'orientamento (None = non attiva)
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
                            float(np.linalg.norm(x[ee] - x[el])),
                            grip=self._gripper_joints(ee)))
        print(f"Lunghezze braccio: {np.round([a.upper_len for a in arms], 3)} "
              f"avambraccio: {np.round([a.fore_len for a in arms], 3)}")
        print(f"Giunti gripper trovati per braccio: {[len(a.grip) for a in arms]}")
        if not all(a.grip for a in arms):
            print("[avviso] nessun giunto con limiti sotto il body 'ee' per almeno un braccio: "
                  "l'apertura non comanderà niente su quel braccio.")
        return arms

    def _gripper_joints(self, ee):
        """Giunti (con limiti) dei body discendenti dall'end effector, cioè le dita: [(adr qpos, lo, hi)]."""
        m = self.model
        out = []
        for j in range(m.njnt):
            b = int(m.jnt_bodyid[j])
            if b == ee or not m.jnt_limited[j]:
                continue
            while b != 0 and b != ee:  # risale l'albero cinematico
                b = int(m.body_parentid[b])
            if b == ee:
                out.append((int(m.jnt_qposadr[j]), float(m.jnt_range[j][0]), float(m.jnt_range[j][1])))
        return out

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

    def jac_rot(self, body_id):
        """Jacobiano di rotazione (3, nv) del body: velocità angolare nel frame mondo."""
        J = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, None, J, body_id)
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

    # ---- Calibrazione orientamento -------------------------------------------
    def calibrate(self, hand_cmds):
        """Calibrazione ISTANTANEA: associa la posa attuale della mano all'orientamento dell'end effector.

        Da chiamare con la mano in posizione neutra: da lì in poi la rotazione della mano rispetto a
        questa posa viene applicata all'end effector. Un braccio senza mano rilevata non viene calibrato.
        Annulla una calibrazione automatica eventualmente in corso.
        """
        self._auto = None
        for arm, cmd in zip(self.arms, hand_cmds or []):
            if cmd is None or cmd['R'] is None:
                continue
            arm.R_hand0 = cmd['R'].copy()
            arm.R_ee0 = self.data.xmat[arm.ee].reshape(3, 3).copy()
        n = sum(a.R_hand0 is not None for a in self.arms)
        print(f"[calib] orientamento calibrato per {n}/{len(self.arms)} bracci")

    @property
    def auto_calibrating(self):
        return self._auto is not None

    def start_auto_calibration(self, n_frames=30, min_samples=10):
        """Avvia la calibrazione AUTOMATICA: media l'orientamento della mano su `n_frames` frame.

        Si alimenta con `feed_auto_calibration` a ogni frame. Contano solo i frame in cui la mano è
        davvero rilevata; i bracci con meno di `min_samples` campioni non vengono calibrati.
        Richiamarla di nuovo riparte da zero.
        """
        self._auto = {'left': n_frames, 'min': min_samples, 'buf': [[] for _ in self.arms]}
        print(f"[calib] tieni la mano in posizione neutra: calibro l'orientamento su {n_frames} frame "
              f"(premi 'c' per calibrare subito a mano)")

    def feed_auto_calibration(self, hand_cmds):
        """Un frame della calibrazione automatica; a fine finestra media le pose e calibra i bracci."""
        a = self._auto
        if a is None:
            return
        for buf, cmd in zip(a['buf'], hand_cmds or []):
            if cmd and cmd.get('fresh') and cmd['R'] is not None:
                buf.append(cmd['R'])
        a['left'] -= 1
        if a['left'] > 0:
            return

        self._auto = None
        n_ok = 0
        for arm, buf in zip(self.arms, a['buf']):
            if len(buf) < a['min']:
                continue
            arm.R_hand0 = Rotation.from_matrix(np.stack(buf)).mean().as_matrix()
            arm.R_ee0 = self.data.xmat[arm.ee].reshape(3, 3).copy()
            n_ok += 1
        msg = f"[calib] orientamento calibrato in automatico per {n_ok}/{len(self.arms)} bracci"
        if n_ok < len(self.arms):
            msg += " (per gli altri mostra la mano e premi 'c')"
        print(msg)

    # ---- Controllo -----------------------------------------------------------
    def track(self, arm_dirs, w_wrist=1.0, w_elbow=0.7, dq_max=0.2,
              damp=1e-3, posture_weight=1e-3, steps=10, hand_cmds=None, w_rot=0.3):
        """Un passo di IK + simulazione.

        arm_dirs  : per ogni braccio (u_braccio, u_avambraccio) versori nel frame del robot,
                    oppure None per un braccio senza target (resta guidato dal solo termine di postura).
        hand_cmds : uscita di HandCommandExtractor (o None): orientamento dell'end effector (solo dopo
                    la calibrazione) e apertura del gripper.
        w_rot     : peso del task di orientamento; tenerlo sotto ai pesi di posizione.
        dq_max    : massima variazione per giunto in un passo [rad o m].
        steps     : passi di mj_step per ogni chiamata.
        """
        tasks = self._tasks(arm_dirs, w_elbow, w_wrist)
        if not tasks:
            return
        rot_tasks = self._rot_tasks(hand_cmds, w_rot)
        dq = self._solve(tasks, dq_max, damp, posture_weight, rot_tasks)
        if dq is None:
            self.qp_failures += 1
            return
        self._apply(dq, steps, self._grip_targets(hand_cmds))

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

    def _rot_tasks(self, hand_cmds, w_rot):
        """Target di orientamento dell'end effector: rotazione relativa della mano applicata a R_ee0."""
        tasks = []  # (body_id, R_target, peso)
        for arm, cmd in zip(self.arms, hand_cmds or []):
            if cmd is None or cmd['R'] is None or arm.R_hand0 is None:
                continue
            tasks.append((arm.ee, cmd['R'] @ arm.R_hand0.T @ arm.R_ee0, w_rot))
        return tasks

    def _grip_targets(self, hand_cmds):
        """{indirizzo qpos: valore} per i giunti delle dita, dall'apertura (0 chiusa, 1 aperta) della mano."""
        out = {}
        for arm, cmd in zip(self.arms, hand_cmds or []):
            if cmd is None or cmd['open'] is None:
                continue
            a = float(np.clip(1.0 - cmd['open'] if self.grip_invert else cmd['open'], 0.0, 1.0))
            for adr, lo, hi in arm.grip:
                out[adr] = lo + a * (hi - lo)
        return out

    def _solve(self, tasks, dq_max, damp, posture_weight, rot_tasks=()):
        """QP (diviso per 2, equivalente):
             min  sum_i w_i |J_i dq - e_i|^2 + sum_k w_k |Jr_k dq - r_k|^2
                  + damp |dq|^2 + posture |q + dq - q_rest|^2
             s.t. limiti dei giunti e |dq| <= dq_max
        con e_i errore di posizione e r_k errore di orientamento (rotation vector nel frame mondo).
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

        rot_errs = []
        for body, R_t, w in rot_tasks:
            Jr = self.jac_rot(body)
            e = Rotation.from_matrix(R_t @ d.xmat[body].reshape(3, 3).T).as_rotvec()
            rot_errs.append(np.linalg.norm(e))
            A += w * Jr.T @ Jr
            b -= w * Jr.T @ e
        self.last_rot_err_deg = np.degrees(rot_errs)

        # Limiti sul passo: intersezione tra |dq| <= dq_max e limiti di giunto (sempre contenente 0)
        lb = np.minimum(np.maximum(-dq_max, self.lo - q), 0.0)
        ub = np.maximum(np.minimum(dq_max, self.hi - q), 0.0)
        self._log(q)
        return solve_qp(spa.csc_matrix(A), b, lb=lb, ub=ub, solver="osqp",
                        eps_abs=1e-6, eps_rel=1e-6)

    def _apply(self, dq, steps, grip=None):
        q_target = self.data.qpos.copy()
        q_target[self.qadr] += dq
        for adr, val in (grip or {}).items():  # le dita seguono l'apertura della mano, non l'IK
            q_target[adr] = val
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
              f"| err rot {np.round(self.last_rot_err_deg, 1)} gradi "
              f"| contatti {self.data.ncon} | qp falliti {self.qp_failures} | ai limiti: {at_lim}")