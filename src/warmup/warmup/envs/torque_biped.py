import os

import gym
import numpy as np
from gym import spaces
from gym import utils
from gym.envs.mujoco import mujoco_env
from mujoco_py import cymj
from scipy.spatial.transform import Rotation as R

from .monoped_torque import MonopedTorque


class TorqueBiped(MonopedTorque):
    def __init__(self, *args, **kwargs):
        self.nq = 14
        self.model_type = "torque_biped"
        self.random_joints = np.array([[3], [6], [7], [8], [11], [12]])
        super(MonopedTorque, self).__init__(*args, **kwargs)
        self.set_gravity([0, 0, -10])
        # self.render_substep()
        self.has_init = True

    def reinitialise(self, args=None):
        self.need_reinit = 0
        # second one is frameskip
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "xml_files/torque_biped.xml",
        )
        mujoco_env.MujocoEnv.__init__(self, path, self.frameskip)
        utils.EzPickle.__init__(self)
        # limits_low = [-1.207, -2.094, -2.094, -2.094, -2.094, -1.570, -1.570]
        # limits_large = [1.770, 2.094,  2.094, 0.174, 0.174, 1.570, 1.570]

    def step(self, a):
        a_eff = a.copy()
        if hasattr(self, "has_init"):
            a_eff[1:4] = a_eff[4:]
        return super().step(a_eff)

    def reset(self):
        super().reset()
        self.data.qpos[3] = -0.01
        for i in [self.random_joints]:
            self.data.qpos[i[0]] = np.random.uniform(
                0.3 * self.model.jnt_range[i[0], 0], 0.3 * self.model.jnt_range[i[0], 1]
            )
        self.data.qvel[:] = 0.0
        self.data.qpos[self.random_joints[:3]] = self.data.qpos[self.random_joints[3:]]
        for i in range(100):
            self.sim.step()
        return self._get_obs()

    def _get_done(self):
        self.done = 0
        if self.geom_pos_z("hat_skull") < 0.3:
            self.done = 1
        if self.geom_pos_z("pelvis") < 0.2:
            self.done = 1
        if self.geom_pos_z("tibia_r") < 0.12:
            self.done = 1
        if self.geom_pos_z("tibia_l") < 0.12:
            self.done = 1
        if self.torso_angle() > 70 or self.torso_angle() < -50:
            self.done = 1
        return self.done

    def geom_pos_z(self, name):
        return self.data.get_geom_xpos(name)[-1]

    def _scale_vel(self, vel):
        return np.minimum(10.0, vel * 100)

    def _stepper_reward(self, action):
        """COM based reward"""
        return (
            self.rew_weights["hopping_drive"]
            * np.exp(np.maximum(0.0, self._scale_vel(self.com_vel)))
            - self.rew_weights["action_cost"] * np.mean(np.square(action))
            - 1  # keeps the hopping drive close to  0 for small vels
            + self.rew_weights["alive_bonus"] * self.alive_bonus
            + self.rew_weights["leg_pain"] * self.leg_pain
        )

    @property
    def leg_pain(self):
        pains = 0
        for i in self.model.actuator_trnid:
            if np.abs(self.data.qpos[i[0]] - self.model.jnt_range[i[0], 0]) < 0.1:
                pains -= 1
            if np.abs(self.data.qpos[i[0]] - self.model.jnt_range[i[0], 1]) < 0.1:
                pains -= 1
        return pains

    @property
    def alive_bonus(self):
        return 1 if not self.done else 0

    def _get_obs(self):
        """Creates observation for MDP.
        The choice here is to either use normalized com_vel in state and reward or just in reward. I could
        imagine it leading to faster learning when normalized, as larger velocities don't constitute "new"
        state space regions. But it also shifts the learning target.
        Removed adaptive scaling."""
        return np.concatenate(
            [
                self.sim.data.qpos[: self.nq].copy(),
                self.sim.data.qvel[: self.nq].copy(),
                self.sim.data.actuator_length.copy(),
                self.actuator_velocities(),
                self.actuator_forces(),
                self.head_pos(),
                self.pelvis_pos(),
                [self.torso_angle()],
                [self._scale_vel(self.com_vel)],
            ]
        )

    def actuator_velocities(self):
        return np.clip(self.data.actuator_velocity, -100, 100).copy()

    def actuator_forces(self):
        return np.clip(self.data.actuator_force / 1000, -100, 100).copy()

    def head_pos(self):
        return self.data.get_geom_xpos("hat_skull").copy()

    def pelvis_pos(self):
        return self.data.get_geom_xpos("pelvis").copy()

    def torso_angle(self):
        return R.from_matrix(self.data.get_geom_xmat("hat_ribs")).as_euler(
            "zyx", degrees=True
        )[0]
