import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
import gym
from gym import spaces
import time
import matplotlib.pyplot as plt


class Drone:
    def __init__(self, position):
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.acceleration = np.zeros(2, dtype=np.float32)

    def update(self, action, dt=1.0):
        self.acceleration = np.array(action, dtype=np.float32)
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt + 0.5 * self.acceleration * dt**2

class Wildfire:
    def __init__(self, position, intensity=1.0, spread_rate=0.1):
        self.position = np.array(position, dtype=np.float32)
        self.intensity = intensity
        self.spread_rate = spread_rate
        self.radius = 0.0

    def spread(self, dt):
        self.radius += self.spread_rate * dt
        self.intensity = max(0, self.intensity / (1 + self.spread_rate * dt))

class DroneEnv(gym.Env):
    def __init__(self):
        self.drone = Drone([0, 0])
        self.wildfire = Wildfire([125, 126])
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def reset(self):
        self.drone = Drone([0, 0])
        self.wildfire = Wildfire([125, 126])
        return self._get_obs()

    def step(self, action):
        self.drone.update(action)
        self.wildfire.spread(1.0)
        
        reward = self._calculate_reward()
        done = False
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return np.concatenate([self.drone.position, self.wildfire.position])

    def _calculate_reward(self):
        distance = np.linalg.norm(self.drone.position - self.wildfire.position)
        if distance < self.wildfire.radius:
            return -3 * (self.wildfire.radius - distance)
        elif distance < 20.0:
            return 1000000
        else:
            return -distance

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.mu = nn.Linear(64, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        x = self.fc(x)
        mu = torch.tanh(self.mu(x))
        std = torch.exp(self.log_std.clamp(-20, 2))
        return mu, std
    
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)

class PPOAgent:
    def __init__(self, env, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.env = env
        self.gamma = gamma
        self.eps_clip = eps_clip

        self.actor = Actor(env.observation_space.shape[0], env.action_space.shape[0])
        self.critic = Critic(env.observation_space.shape[0])
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            mu, std = self.actor(state)
            dist = Normal(mu, std)
            action = dist.sample()
            action_log_prob = dist.log_prob(action).sum(dim=-1)
        return action.numpy().squeeze(), action_log_prob.item()

    def update(self, states, actions, log_probs, rewards, next_states):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(log_probs)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # Compute advantage
        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            advantages = rewards + self.gamma * next_values - values

        # PPO update
        for _ in range(5):  # 5 epochs
            mu, std = self.actor(states)
            dist = Normal(mu, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            ratio = (new_log_probs - old_log_probs).exp()

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = F.mse_loss(self.critic(states).squeeze(), rewards + self.gamma * next_values)

            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer.step()

    def train(self, max_episodes=1000):
        episode_rewards = []
        avg_rewards = []
        
        for episode in range(max_episodes):
            state = self.env.reset()
            episode_reward = 0
            states, actions, log_probs, rewards, next_states = [], [], [], [], []
            episode_accelerations = []

            for t in range(200):  # max 200 steps per episode
                action, log_prob = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                next_states.append(next_state)
                episode_accelerations.append(action)

                state = next_state
                episode_reward += reward

                if done:
                    break

            self.update(states, actions, log_probs, rewards, next_states)

            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards[-100:])  # Moving average of last 100 episodes
            avg_rewards.append(avg_reward)
            
            avg_acceleration = np.mean(episode_accelerations, axis=0)
            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}, Avg Acceleration: {avg_acceleration}")

        print("Training completed.")
        return episode_rewards, avg_rewards
# Main execution
if __name__ == "__main__":
    env = DroneEnv()
    agent = PPOAgent(env)
    episode_rewards, avg_rewards = agent.train(max_episodes=1000)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Episode Reward')
    plt.plot(avg_rewards, label='Average Reward (last 100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward vs Episode')
    plt.legend()
    plt.grid(True)
    plt.savefig('reward_plot.png')
    plt.show()