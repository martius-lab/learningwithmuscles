import gym
from gym.envs.registration import register

"""No episode_limits here so we dont have to deal with pesky gym TimeLimit wrapper
that sometimes handles episode_terminations and resets incorrectly."""

register(
    id="muscle_arm-v0", entry_point="warmup.envs:MuscleArmMuJoCo", max_episode_steps=500
)

register(
    id="torque_arm-v0", entry_point="warmup.envs:TorqueArmMuJoCo", max_episode_steps=1000
)

register(
    id="muscle_biped-v0", entry_point="warmup.envs:MuscleBiped", max_episode_steps=1000
)

register(
    id="torque_biped-v0", entry_point="warmup.envs:TorqueBiped", max_episode_steps=500
)
