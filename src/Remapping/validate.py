"""validate.py: controlla se un MujocoController è adatto al metodo di retargeting."""
import mujoco
import numpy as np

# Il metodo insegue 2 direzioni per braccio (2 gradi di libertà ciascuna): i target stanno su sfere
# di raggio fisso, quindi il rango massimo utile del Jacobiano impilato [gomito; polso] è 4, non 6.
REQUIRED_RANK = 4


def validate(ctrl, n=200, seed=0, tol_cm=1.0, verbose=True):
    """Campiona configurazioni casuali e verifica le assunzioni del retargeting.

    Controlli:
      1. spalla-gomito e gomito-polso costanti entro `tol_cm` (i body sono centri di rotazione veri)
      2. DoF che muovono il polso (>= 4) e rango del Jacobiano impilato [gomito; polso]:
         rango 4 = si possono centrare entrambe le direzioni; rango 3 = vicino a una singolarità
      3. posa di riposo dentro i limiti e a rango pieno
      4. tipo di attuatori: _set_ctrl assume attuatori di posizione sui giunti

    Ritorna (ok, errori, avvisi). Gli errori invalidano il metodo, gli avvisi no.
    """
    rng = np.random.default_rng(seed)
    errors, warns = [], []

    samples = [_measure(ctrl, ctrl.random_qpos(rng)) for _ in range(n)]
    rest = _measure(ctrl, ctrl.q_rest)

    for k in range(len(ctrl.arms)):
        up, fo, rank, moved = (np.array([s[k][i] for s in samples]) for i in range(4))
        for label, arr in (("spalla-gomito", up), ("gomito-polso", fo)):
            spread = np.ptp(arr) * 100
            if spread > tol_cm:
                errors.append(f"braccio {k}: {label} varia di {spread:.1f} cm (> {tol_cm} cm): "
                              f"il body scelto non è il centro di rotazione")
        n_dof, r_max = int(moved.any(axis=0).sum()), int(rank.max())
        if n_dof < REQUIRED_RANK or r_max < REQUIRED_RANK:
            errors.append(f"braccio {k}: {n_dof} DoF muovono il polso, rango massimo {r_max}: "
                          f"ne servono {REQUIRED_RANK} per inseguire le due direzioni")
        elif np.median(rank) < REQUIRED_RANK:
            warns.append(f"braccio {k}: rango mediano {int(np.median(rank))} < {REQUIRED_RANK}, "
                         f"molte configurazioni sono vicine a una singolarità")
        if rest[k][2] < REQUIRED_RANK:
            warns.append(f"braccio {k}: la posa di riposo è singolare (rango {rest[k][2]}); "
                         f"prova a piegare di più il gomito")

    errors += _rest_out_of_limits(ctrl)
    e, w = _check_actuators(ctrl.model)
    errors, warns = errors + e, warns + w

    ctrl.reset()
    ok = not errors
    if verbose:
        print(f"[validate] {'OK' if ok else 'NON ADATTO'}")
        for msg in errors:
            print(f"  ERRORE : {msg}")
        for msg in warns:
            print(f"  avviso : {msg}")
    return ok, errors, warns


def _measure(ctrl, q):
    """FK in `q` -> per braccio: (lunghezza braccio, avambraccio, rango J, dof che muovono il polso)."""
    ctrl.set_qpos(q)
    x = ctrl.data.xpos
    out = []
    for a in ctrl.arms:
        Je, Jw = ctrl.jac(a.elbow), ctrl.jac(a.ee)
        out.append((np.linalg.norm(x[a.elbow] - x[a.shoulder]),
                    np.linalg.norm(x[a.ee] - x[a.elbow]),
                    int(np.linalg.matrix_rank(np.vstack([Je, Jw]), tol=1e-6)),
                    np.abs(Jw).max(axis=0) > 1e-9))
    return out


def _rest_out_of_limits(ctrl):
    q0 = ctrl.q_rest[ctrl.qadr]
    out = [name for name, q, lo, hi in zip(ctrl.dof_names, q0, ctrl.lo, ctrl.hi)
           if q < lo - 1e-9 or q > hi + 1e-9]
    return [f"posa di riposo fuori dai limiti: {out}"] if out else []


def _is_position_actuator(m, i):
    """Pattern standard di MuJoCo: gain = kp, bias = [0, -kp, ...]. Euristica: attuatori custom possono sfuggire."""
    return (int(m.actuator_biastype[i]) == int(mujoco.mjtBias.mjBIAS_AFFINE)
            and m.actuator_gainprm[i, 0] > 0
            and abs(m.actuator_gainprm[i, 0] + m.actuator_biasprm[i, 1]) < 1e-9)


def _check_actuators(m):
    errors, warns = [], []
    if m.nu == 0:
        warns.append("nessun attuatore: si userà la modalità cinematica (qpos diretto)")
    for i in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"#{i}"
        if int(m.actuator_trntype[i]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            warns.append(f"attuatore {name}: non è su giunto, _set_ctrl lo ignora")
        elif not _is_position_actuator(m, i):
            errors.append(f"attuatore {name}: non è di posizione, ctrl = angolo non ha senso "
                          f"(coppia/velocità richiedono un altro controllore)")
    return errors, warns