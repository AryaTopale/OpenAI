# Deep Sarsa and PPO on Mountain Car Environment

This repository contains an implementation of the Deep Sarsa algorithm applied to the Mountain Car environment using Google Colab.

## Overview

Deep Sarsa (State-Action-Reward-State-Action) is an on-policy reinforcement learning algorithm. It combines the classic Sarsa algorithm with deep learning techniques to handle large state spaces. In this project, Deep Sarsa is used to train an agent to solve the Mountain Car problem.

## Files

- `DeepSARSA.ipynb`: The main notebook implementing the Deep Sarsa algorithm on the Mountain Car environment.

## Environment

The Mountain Car environment is a classic reinforcement learning problem where an underpowered car must drive up a steep hill. The agent receives a reward of -1 for each time step until it reaches the goal at the top of the mountain.

## Requirements

To run the notebook, you need the following libraries:
- `numpy`
- `gym`
- `torch`
- `matplotlib`

You can install these dependencies using `pip`:

```bash
pip install numpy gym torch matplotlib
