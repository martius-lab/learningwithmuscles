import numpy as np
from gym import spaces

from .monoped_torque import MonopedTorque
from .muscle_arm import MuscleArm


class AbstractMuscleModel:
    def muscle_forces(self):
        forces = []
        [
            forces.append(
                self.data.userdata[self.model.nuserdata - (4 * self.model.nu) + 2 * i]
                / 1000
            )
            for i in range(self.model.nu)
        ]
        [
            forces.append(
                self.data.userdata[
                    self.model.nuserdata - (4 * self.model.nu) + 2 * i + 1
                ]
                / 1000
            )
            for i in range(self.model.nu)
        ]
        return np.clip(forces, -100, 100).copy()

    def muscle_velocities(self):
        velocities = []
        [
            velocities.append(
                self.data.userdata[self.model.nuserdata - (6 * self.model.nu) + 2 * i]
            )
            for i in range(self.model.nu)
        ]
        [
            velocities.append(
                self.data.userdata[
                    self.model.nuserdata - (6 * self.model.nu) + 2 * i + 1
                ]
            )
            for i in range(self.model.nu)
        ]
        return np.clip(velocities, -100, 100).copy()

    def muscle_activities(self):
        activity = []
        [activity.append(self.data.userdata[4 + 2 * i]) for i in range(self.model.nu)]
        [
            activity.append(self.data.userdata[4 + 2 * i + 1])
            for i in range(self.model.nu)
        ]
        return np.clip(activity, 0, 1).copy()

    @property
    def muscles_dep(self):
        return self.muscle_lengths()

    def muscle_lengths(self):
        length = []
        [
            length.append(
                self.data.userdata[self.model.nuserdata - (2 * self.model.nu) + 2 * i]
            )
            for i in range(self.model.actuator_trnid.shape[0])
        ]
        [
            length.append(
                self.data.userdata[
                    self.model.nuserdata - (2 * self.model.nu) + 2 * i + 1
                ]
            )
            for i in range(self.model.actuator_trnid.shape[0])
        ]
        return np.clip(length, -100, 100).copy()

    def _set_action_space_again(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        new_low = np.array([[x] * 2 for x in low]).reshape(-1)
        new_high = np.array([[x] * 2 for x in high]).reshape(-1)

        self.action_space = spaces.Box(low=new_low, high=new_high, dtype=np.float32)
        self.manually_set_action_space = 1
        return self.action_space

    def do_simulation(self, ctrl, n_frames):
        if np.array(ctrl).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")
        ctrl = np.clip(ctrl, 0.0, 1.0)
        if hasattr(self, "has_init"):
            self.data.userdata[4 + 2 * self.model.nu : 4 + 4 * self.model.nu] = ctrl[:]
        for _ in range(n_frames):
            if self.render_substep_bool:
                # self.render('rgb_array')
                self.render("human")
            self.sim.step()

    def _get_obs(self):
        """Creates observation for MDP.
        The choice here is to either use normalized com_vel in state and reward or just in reward. I could
        imagine it leading to faster learning when normalized, as larger velocities don't constitute "new"
        state space regions. But it also shifts the learning target.
        Removed adaptive scaling."""
        return np.concatenate(
            [
                self.sim.data.qpos[: self.nq],
                self.sim.data.qvel[: self.nq],
                self.muscle_velocities(),
                self.muscle_lengths(),
                self.muscle_forces(),
                self.muscle_activations(),
                self.target,
                self.sim.data.get_site_xpos(self.tracking_str),
            ]
        )


class MuscleModelArm(AbstractMuscleModel, MuscleArm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MuscleModelLeg(AbstractMuscleModel, MonopedTorque):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
