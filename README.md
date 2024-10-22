# CQL-based-Offline-RL

This repository contains PyTorch implementations of **Conservative Q-Learning (CQL)** algorithms integrated with:

- **Soft Actor-Critic (SAC)**: `[CQL_SAC](CQL_SAC.py)`
- **Deep Deterministic Policy Gradient (DDPG)**: `[CQL_DDPG](CQL_DDPG.py)`

**Conservative Q-Learning** is an offline reinforcement learning algorithm designed to address overestimation bias and distributional shift by penalizing Q-values of out-of-distribution actions. By integrating CQL with SAC and DDPG, the aim is to enhance policy reliability and performance in offline RL settings.


## Features

- **Offline Reinforcement Learning**: Designed for learning policies from static datasets without environment interaction.
- **CQL Integration**: Implements the conservative Q-learning loss for both SAC and DDPG.
- **Customizable Hyperparameters**: Easily adjust learning rates, batch sizes, temperature parameters, and more.
- **Support for D4RL Datasets**: Compatible with datasets from the D4RL benchmark suite.
- **GPU Acceleration**: Utilizes PyTorch for efficient computation on GPUs.

## Results

* CQL-SAC on HalfCheetah-medium-v2: Achieved an average return of 4300 after 100 episodes.
* CQL-DDPG on Hopper-medium-v2: Achieved an average return of 2000 after 100 episodes.
