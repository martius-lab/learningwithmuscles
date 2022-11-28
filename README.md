# Learning with Muscles: Benefits for Data-Efficiency and Robustness in Anthropomorphic Tasks

 This repo contains several muscle-driven MuJoCo environments used for the [Learning with Muscles: Benefits for Data-Efficiency
and Robustness in Anthropomorphic Tasks](https://openreview.net/forum?id=Xo3eOibXCQ8) paper, published at CoRL 2022.

The work was performed by Isabell Wochner* and Pierre Schumacher* (equal contribution), Georg Martius, Dieter Büchler, Syn Schmitt* and Daniel F.B. Haeufle* (equal contribution).
## Abstract
Humans are able to outperform robots in terms of robustness, versatility,
and learning of new tasks in a wide variety of movements. We hypothesize that
highly nonlinear muscle dynamics play a large role in providing inherent stability,
which is favorable to learning. While recent advances have been made in applying
modern learning techniques to muscle-actuated systems both in simulation as
well as in robotics, so far, no detailed analysis has been performed to show the
benefits of muscles when learning from scratch. Our study closes this gap and
showcases the potential of muscle actuators for core robotics challenges in terms
of data-efficiency, hyperparameter sensitivity, and robustness

## Installation

To install the package, simply download the repo and install using pip:
```
conda create -n warmup python==3.9
git clone https://github.com/martius-lab/learningwithmuscles
pip install -e ./src/warmup
pip install -r ./src/warmup/requirements.txt
pip install -e ./src/warmup/muscle_mujoco_py/
```
Installing locally ensures that you can easily change environment parameters by modifying the param_files. 

To run baselines, install:

```
git clone https://github.com/fabiopardo/tonic
pip install -e tonic/
```

The build has been tested with: 
```
Python==3.9
mujoco==2.1.0
Ubuntu 20.04 and Ubuntu 22.04
```
MuJoCo refers to the version of the binaries, not the DeepMind python bindings.

## Experiments 

The major experiments (precise and fast reaching, hopping) can be repeated with the config files.
Simply from the *src* folder:
```
python main.py configs/arm_muscle_precise_reaching.yaml
```
to train an agent. Model checkpoints will be saved in the current directory. 
The progress can be monitored with:
```
python -m tonic.plot --path folder/
```
To execute a trained policy, use:
```
python -m tonic.play --path folder/folder/
```

See the [TonicRL](https://github.com/fabiopardo/tonic) documentation for details.

## Environments

The underlying environment (warmup) can be used like any other gym environment:
```
import gym
import warmup

env = gym.make("muscle_arm-v0")
# for arm-based environments, you can test perturbations with
# env.activate_ball()

for ep in range(5):
     ep_steps = 0
     state = env.reset()
     while True:
         next_state, reward, done, info = env.step(env.action_space.sample())
         env.render()
         if done or (ep_steps >= env.max_episode_steps):
             break
         ep_steps += 1

```

Settings can be changed in the respective 
```
warmup/param_files/ 
```
json files. Use the *ball_attached* attribute to enable the chaotic load for perturbations.

The used muscle model can be found in:
```
warmup/muscle_mujoco_py/mujoco_py/mjmuscle_geometry_free.pyx
```

## Citation

Please use the following citation if you make use of our work:

```
@inproceedings{
wochner2022learning,
title={Learning with Muscles: Benefits for Data-Efficiency and Robustness in Anthropomorphic Tasks},
author={Isabell Wochner and Pierre Schumacher and Georg Martius and Dieter B{\"u}chler and Syn Schmitt and Daniel Haeufle},
booktitle={6th Annual Conference on Robot Learning},
year={2022},
url={https://openreview.net/forum?id=Xo3eOibXCQ8}
}
```
