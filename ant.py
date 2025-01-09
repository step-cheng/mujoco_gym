import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.distributions import Normal
import matplotlib.pyplot as plt

env = gym.make('Ant-v5', render_mode='rgb_array')
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.shape[0]


def sample_video(agent):
    # Parameters for the video
    video_path = "ant_environment_video_no_norm.mp4"
    frames = []

    try:
        # Loop until the episode terminates
        done = False
        truncated = False
        state, info = env.reset()
        while not done and not truncated:
            # Take a random action
            action, log_prob = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)

            # Capture the frame
            frame = env.render()  # Render the environment as an RGB array
            frames.append(frame)
            state = next_state

    finally:
        # Always close the environment to release resources
        env.close()

    # Save the video
    print(f"Saving video to {video_path}...")
    imageio.mimwrite(video_path, frames, fps=5)
    print("Video saved successfully!")


# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        logits = torch.tanh(x)
        return logits

# Value Network
class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class PPOAgent: # currently uses VPG w/ baseline
    def __init__(self, policy_network, value_network, policy_lr, value_lr):
        self.policy_network = policy_network  
        self.value_network = value_network  
        self.policy_optimizer = optim.Adam(policy_network.parameters(), policy_lr)    
        self.value_optimizer = optim.Adam(policy_network.parameters(), value_lr)     
        self.value_criterion = nn.MSELoss()     

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)  
        logits = self.policy_network(state)
        action_mu, action_log_std = logits[:8], logits[8:]
        action_std = torch.exp(action_log_std)

        dist = Normal(action_mu, action_std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action).sum(dim=-1)  # Sum log probs over dimensions
        log_prob -= torch.sum(torch.log(1 - action.pow(2) + 1e-6), dim=-1)
        return action.detach().numpy(), log_prob

    def update(self, rewards, log_probs, states):
        # Calculate returns
        returns = []  # store the returns
        G = 0 # initialize the cumulative return
        gamma = 0.995

        for r in reversed(rewards):
            G =  r + gamma*G # compute the cumulative return
            # store the return
            returns.append(G)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = torch.flip(returns, dims=(0,))
                
        # Calculate value estimates and update value network
        values = []

        for s in states:
            s = torch.tensor(s, dtype=torch.float32)
            values.append(self.value_network(s))

        values = torch.stack(values).flatten()
        value_loss = self.value_criterion(values, returns)        

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Calculate policy loss with baseline
        advantages = returns - values.detach() #
        advantages = (advantages - advantages.mean())/ (advantages.std() + 1e-9)

        policy_loss = [] # - initialize the storage for policy losses
        for log_prob, advantage in zip(log_probs, advantages):
            # - store the policy loss
            policy_loss.append(-log_prob*advantage)
        
        policy_loss = torch.stack(policy_loss).mean() # compute the loss
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

def collect_samples(agent):
    pass

def train(num_episodes):
    policy_network = PolicyNetwork(input_dim, 2*output_dim)
    value_network = ValueNetwork(input_dim)
    agent = PPOAgent(policy_network, value_network, policy_lr=5e-4, value_lr=5e-4)
    
    batch = 2048

    episode_rewards = []
    
    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        episode_reward = 0
        log_probs = []
        rewards = []
        states = []
        num_steps = []
        
        done = False
        truncated = False
        steps = 0
        while not done and not truncated:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            rewards.append(reward)
            states.append(state)
            episode_reward += reward
            log_probs.append(log_prob)
            
            state = next_state
            steps += 1
        
        episode_rewards.append(episode_reward)
        num_steps.append(steps)
        agent.update(rewards, log_probs, states)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Steps: {np.mean(num_steps[-10:])}, Average Reward: {np.mean(episode_rewards[-10:]):.2f}")
    
    return agent, episode_rewards, num_steps


def plot_episodes(results, title):
    mavg_results = []
    window = 5
    for i in range(len(results)):
        avg_r = np.mean(results[max(0,i-window):i+1])
        mavg_results.append(avg_r)

    plt.plot(mavg_results)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(title)
    plt.show()

# Create the environment
num_episodes = 1000
trained_agent, episode_rewards, episode_steps = train(num_episodes)

torch.save(trained_agent.policy_network.state_dict(), "ant_actor.pth")
torch.save(trained_agent.value_network.state_dict(), "ant_critic.pth")

plot_episodes(episode_rewards, "rewards")
plot_episodes(episode_steps, "steps")

sample_video(trained_agent)
