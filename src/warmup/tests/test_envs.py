import gym
import warmup

print('Testing normal environments')
for env_str in ["muscle_arm-v0", "torque_arm-v0", "muscle_biped-v0", "torque_biped-v0"]:
    print(f"Testing {env_str}")
    env = gym.make(env_str)
    for ep in range(1):
        ep_steps = 0
        state = env.reset()
        while True:
            # repeat actions for some time to demonstrate muscle model
            if not ep_steps % 100:
                action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            env.render()
            # We ignore environment terminations here for testing purposes
            if ep_steps >= 1000:
                break
            ep_steps += 1


print('Testing chaotic load perturbations')
for env_str in ["muscle_arm-v0", "torque_arm-v0"]:
    print(f"Testing {env_str} with chaotic load")
    env = gym.make(env_str)
    env.activate_ball()
    for ep in range(1):
        ep_steps = 0
        state = env.reset()
        while True:
            # repeat actions for some time to demonstrate muscle model
            if not ep_steps % 100:
                action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            env.render()
            # We ignore environment terminations here for testing purposes
            if ep_steps >= 1000:
                break
            ep_steps += 1
