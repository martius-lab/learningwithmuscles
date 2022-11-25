import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from .muscle_env import MuscleEnv
import os


class MuscleBody(MuscleEnv):
    def __init__(self):
        self.tracking_str = "addbrev_r-P1"
        super(MuscleBody, self).__init_()

    def reset_model(self):
        raise NotImplementedError

    def viewer_setup(self):
        raise NotImplementedError

    def step(self, a):
        assert a.shape == self.action_space.shape
        if self.need_reinit:
            raise Exception("Need to call self.reinitialise before stepping")
        # mujoco-py checks step() before __init__() is called
        if not hasattr(self, "target"):
            self.target = np.array([10.0, 10.0, 10.0])
        if hasattr(self, "action_multiplier") and self.action_multiplier != 0:
            a = self.redistribute_action(a)
        self.do_simulation(a, self.frame_skip)
        # com_vel_vertical = self.sim.data.get_site_xvelp(self.tracking_str)
        com_vel_vertical = self.sim.data.get_body_xvelp("pelvis")
        act = self.sim.data.act
        reward = self._get_reward(com_vel_vertical, a)
        done = self._get_done(com_vel_vertical)
        if done:
            reward += 10.0
        return (
            self._get_obs(),
            reward,
            done,
            {"tracking": com_vel_vertical, "activity": act},
        )

    def reset(self):
        return super(MuscleBody, self).reset()

    def _get_obs(self):
        act = self.sim.data.act
        if act is None:
            act = np.zeros_like(self.sim.data.actuator_length)
        return np.concatenate(
            [
                self.sim.data.qpos[: self.nq],
                self.sim.data.qvel[: self.nq],
                self.sim.data.actuator_length,
                act,
                self.target,
                self.sim.data.get_site_xpos(self.tracking_str),
            ]
        )

    def _get_reward(self, com_vel_vertical, action):
        penalty = -30 if self.sim.data.get_geom_xpos("hat_skull")[-1] < 0.1 else 0
        return np.clip(np.exp(-com_vel_vertical[-1] / 10), 0, 50) + penalty

    def _get_done(self, com_vel_vertical):
        done = 1 if self.sim.data.get_geom_xpos("hat_skull")[-1] < 0.1 else 0
        return done

    def set_target(*args, **kwargs):
        pass

    def reinitialise(self, args):
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "xml_factory/generated_xmls/fullbody.xml",
        )
        self.need_reinit = 0
        # second one is frameskip
        mujoco_env.MujocoEnv.__init__(self, path, self.frameskip)
        utils.EzPickle.__init__(self)
        # os.remove(path)
