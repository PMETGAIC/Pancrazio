import numpy as np
import mujoco
import matplotlib.pyplot as plt
import scipy.sparse as spa
from qpsolvers import solve_qp

class MujocoController:
    def __init__(self, xml_path, ee_names=None):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.ee_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) if n else self.model.nbody-1 for n in (ee_names or [])]
        self.model.opt.gravity[:] = [0, 0, 0]
        self.fig = self.line = None

    def init_plot(self):
        plt.ion()
        self.fig = plt.figure(figsize=(10, 8))
        ax = self.fig.add_subplot(111, projection='3d')
        self.line, = ax.plot([], [], [], 'bo-', linewidth=2, markersize=5)
        ax.set(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), zlim=(0.0, 1.5))
        ax.view_init(elev=30, azim=30)

    def update_plot(self):
        if not self.line: return
        self.line.set_data(self.data.xpos[:, 0], self.data.xpos[:, 1])
        self.line.set_3d_properties(self.data.xpos[:, 2])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def print_state(self):
        print(f"\rqpos: {np.round(self.data.qpos[:6], 2)}...", end="", flush=True)

    def strategy_qp(self, target_poses, dq_max=0.2, damp=0.05, steps=10):
        J, e = [], []
        for ee_id, t_pos in zip(self.ee_ids, target_poses):
            jacp = np.zeros((3, self.model.nv))
            mujoco.mj_jacBodyCom(self.model, self.data, jacp, None, ee_id)
            J.append(jacp)
            e.append(t_pos - self.data.xpos[ee_id])
            
        J = np.vstack(J)
        e = np.hstack(e)

        P = spa.csc_matrix(J.T @ J + np.eye(self.model.nv) * damp)
        q_qp = -J.T @ e

        lb = np.maximum(-dq_max, self.model.jnt_range[:, 0] - self.data.qpos)
        ub = np.minimum(dq_max, self.model.jnt_range[:, 1] - self.data.qpos)

        G = spa.csc_matrix(np.vstack([np.eye(self.model.nv), -np.eye(self.model.nv)]))
        h = np.hstack([ub, -lb])

        dq = solve_qp(P, q_qp, G, h, solver="osqp")

        if dq is not None:
            q_target = self.data.qpos + dq
            if self.model.nu > 0:
                for i in range(self.model.nu):
                    jid = self.model.actuator_trnid[i, 0]
                    self.data.ctrl[i] = q_target[self.model.jnt_qposadr[jid]]
                for _ in range(steps):
                    mujoco.mj_step(self.model, self.data)
            else:
                self.data.qpos[:] = q_target
                mujoco.mj_forward(self.model, self.data)