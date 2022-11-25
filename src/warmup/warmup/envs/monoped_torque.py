import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from .muscle_leg import MuscleLeg


class MonopedTorque(MuscleLeg):
    def __init__(self):
        self.model_type = "monoped_torque"
        self.nq = 5
        super().__init__()
        self.set_gravity([0, 0, -9.81])
        # self.render_substep()
        self._set_cost_weights()
        self.has_init = True

    def reset_model(self):
        # self.apply_muscle_settings()
        self.randomise_init_state(0.5)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[:] = [1, -0.3, 1.15]
        self.viewer.cam.elevation = -30
        self.viewer.cam.azimuth = 150

    def reinitialise(self, args):
        self.need_reinit = 0
        # second one is frameskip
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "xml_factory/generated_xmls/monoped_torque.xml",
        )
        mujoco_env.MujocoEnv.__init__(self, path, self.frameskip)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        """Creates observation for MDP.
        The choice here is to either use normalized com_vel in state and reward or just in reward. I could
        imagine it leading to faster learning when normalized, as larger velocities don't constitute "new"
        state space regions. But it also shifts the learning target.
        Removed adaptive scaling."""
        act = self.data.act
        if act is None:
            act = np.zeros_like(self.sim.data.actuator_length)
        return np.concatenate(
            [
                self.sim.data.qpos[: self.nq],
                self.sim.data.qvel[: self.nq],
                self.sim.data.actuator_length,
                act,
                [self._scale_vel(self.com_vel)],
            ]
        )
