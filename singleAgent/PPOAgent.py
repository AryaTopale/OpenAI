import torch
import torch.nn as nn
import torch.optim as optim
from initialization import *
from DroneEnv import *
from actor_critic import *
from torch.distributions import Normal

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
        count=0
        
        for episode in range(max_episodes):
            state = self.env.reset()
            episode_reward = 0
            states, actions, log_probs, rewards, next_states = [], [], [], [], []
            episode_accelerations = []

            done = False
            while not done:  # Loop until 'done' is True
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

                #if outOfBounds:
                #    count += 1

            self.update(states, actions, log_probs, rewards, next_states)

            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards[-100:])  # Moving average of last 100 episodes
            avg_rewards.append(avg_reward)
            
            avg_acceleration = np.mean(episode_accelerations, axis=0)
            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}, Avg Acceleration: {avg_acceleration} ")

        print("Training completed.")
        return episode_rewards, avg_rewards
