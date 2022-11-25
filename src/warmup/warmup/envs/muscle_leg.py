import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from .muscle_env import MuscleEnv


class MuscleLeg(MuscleEnv):
    def __init__(self):
        self.tracking_str = "bifemlh_r-P1"
        super().__init__()

    def step(self, a):
        self.update_com()
        self.com_before = self.com.copy()
        self.update_com_vel()
        self.com_vel_before = self.com_vel.copy()
        assert a.shape == self.action_space.shape
        if self.need_reinit:
            raise Exception("Need to call self.reinitialise before stepping")
        # mujoco-py checks step() before __init__() is called
        if not hasattr(self, "target"):
            self.target = np.array([10.0, 10.0, 10.0])
        if hasattr(self, "action_multiplier") and self.action_multiplier != 0:
            a = self.redistribute_action(a)
        self.do_simulation(a, self.frame_skip)
        act = self.sim.data.act
        done = self._get_done()
        reward = self._get_reward(a)
        com_vel_vertical = self.data.get_body_xvelp("pelvis")
        return (
            self._get_obs(),
            reward,
            done,
            {"tracking": com_vel_vertical, "activity": act},
        )

    def reset(self):
        return super().reset()

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

    def set_target(*args, **kwargs):
        pass

    def _set_cost_weights(self, rew_args=None):
        self.rew_weights = {
            "leg_pain": 0,
            "com_deviation": 0,
            "hopping_drive": 1,
            "action_cost": 0,
            "alive_bonus": 1,
        }
        if rew_args is not None:
            for k, v in rew_args.items():
                self.rew_weights[k] = rew_args[k]

    def quick_settings(self, args):
        """Applies correct settings from args.
        Don't change the order if you don't know what you are doing.
        Some settings are used by the mujoco_env initilisation."""
        if hasattr(args, "rew_args"):
            self._set_cost_weights(args.rew_args)
        super().quick_settings(args)

    def _get_reward(self, action):
        self.update_com()
        self.update_com_vel()
        self.update_com_acc()
        return self._stepper_reward(action)
        # return self._floor_lava_reward(action)
        # return self._gym_reward(action)

    def update_com(self):
        self.com = self._com_main_body().copy()

    def update_com_vel(self):
        self.com_vel = (self.com[-1] - self.com_before[-1]).copy()

    def update_com_acc(self):
        self.com_acc = (self.com_vel - self.com_vel_before).copy()

    def _com_main_body(self):
        data = self.unwrapped.data
        model = self.unwrapped.model
        tot_mass = np.sum(model.body_subtreemass)
        com_full = 0
        for idx in range(1, model.nbody):
            com_full += model.body_subtreemass[idx] * data.body_xpos[idx]
        return com_full / tot_mass

    def _stepper_reward(self, action):
        """COM based reward"""
        # TODO use env.data.cvel for com-velocity
        return (
            self.rew_weights["hopping_drive"] * np.exp(self._scale_vel(self.com_vel))
            - self.rew_weights["com_deviation"] * np.square(self.com[1])
            - self.rew_weights["action_cost"] * np.mean(np.square(action))
            - 1  # keeps the hopping drive close to  0 for small vels
            + self.rew_weights["alive_bonus"] * self.alive_bonus
            + self.rew_weights["leg_pain"] * self.leg_pain
        )

    @property
    def leg_pain(self):
        pains = 0
        if self.data.qpos[4] < 0.1:
            pains -= 1
        if np.abs(self.data.qpos[4] - 0.66 * np.pi) < 0.1:
            pains -= 1
        return pains

    @property
    def joint_pain(self):
        return -1 if self.data.qpos[4] < 0.1 else 0

    def _scale_vel(self, vel):
        return np.minimum(2.0, vel / 0.01)

    @property
    def alive_bonus(self):
        return 1 if not self.done else 0

    def _get_done(self):
        self.done = 0
        if self.data.ncon > 0:
            if np.any(
                [self.data.contact[idx].geom2 != 5 for idx in range(self.data.ncon)]
            ):
                self.done = 1
        if np.abs(self._com_main_body()[1]) > 0.5 or np.abs(self.data.qpos[2]) > 1.6:
            self.done = 1
        return self.done
