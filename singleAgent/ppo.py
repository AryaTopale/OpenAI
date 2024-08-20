import torch.nn.functional as F  
from initialization import *
from DroneEnv import *
from actor_critic import *
from PPOAgent import *
import matplotlib.pyplot as plt

import torch
import tensorflow

# Environment Setup

    
env = DroneEnv()
agent = PPOAgent(env)
episode_rewards, avg_rewards = agent.train(max_episodes=1000)


# Plotting
plt.figure(figsize=(10, 5))
#plt.plot(episode_rewards, label='Episode Reward')
plt.plot(avg_rewards, label='Average Reward (last 100 episodes)')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward vs Episode')
plt.legend()
plt.grid(True)
plt.savefig('reward_plot.png')
plt.show()