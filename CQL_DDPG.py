import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gym
import d4rl  # Import required to register D4RL environments
import os

# Define the Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Assuming action space is between -1 and 1
        )

    def forward(self, state):
        return self.net(state)

# Define the Critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

# Define the CQL-DDPG Agent
class CQLDDPGAgent:
    def __init__(self, state_dim, action_dim, device='cpu',
                 gamma=0.99, tau=5e-3, actor_lr=3e-4, critic_lr=3e-4,
                 alpha_cql=1.0, target_action_gap=10.0, with_lagrange=False,
                 temperature=1.0, hidden_size=256):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha_cql = alpha_cql
        self.with_lagrange = with_lagrange
        self.target_action_gap = target_action_gap
        self.temperature = temperature

        # Actor and Critic networks
        self.actor = Actor(state_dim, action_dim, hidden_dim=hidden_size).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim=hidden_size).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim, hidden_dim=hidden_size).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim=hidden_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Lagrange multiplier for CQL (if using adaptive alpha)
        if self.with_lagrange:
            self.log_alpha_cql = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_cql_optimizer = optim.Adam([self.log_alpha_cql], lr=critic_lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        return np.clip(action, -1, 1)

    def train(self, replay_buffer, batch_size=256):
        # Sample a batch of transitions
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # Convert to tensors
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # ---------------------- Critic Update ---------------------- #
        with torch.no_grad():
            # Compute target actions
            next_action = self.actor_target(next_state)
            # Compute target Q-value
            target_Q = reward + self.gamma * (1 - done) * self.critic_target(next_state, next_action)

        # Compute current Q-value
        current_Q = self.critic(state, action)

        # Standard critic loss
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        # ---------------------- CQL Loss Component ---------------------- #
        # Sample actions from the policy
        with torch.no_grad():
            policy_actions = self.actor(state)

        # Q-values for actions from the policy
        current_Q_pi = self.critic(state, policy_actions)

        # Sample random actions uniformly
        random_actions = torch.FloatTensor(batch_size, action.shape[1]).uniform_(-1, 1).to(self.device)
        current_Q_rand = self.critic(state, random_actions)

        # Compute CQL loss
        cat_Q = torch.cat([current_Q_rand, current_Q_pi], dim=0)
        cql_loss = (torch.logsumexp(cat_Q / self.temperature, dim=0).mean() * self.temperature - current_Q.mean()) * self.alpha_cql

        # Adjust CQL alpha if using Lagrange multiplier
        if self.with_lagrange:
            cql_loss_alpha = (cql_loss - self.target_action_gap) * self.alpha_cql
            self.alpha_cql_optimizer.zero_grad()
            (-cql_loss_alpha).backward(retain_graph=True)
            self.alpha_cql_optimizer.step()
            self.alpha_cql = torch.clamp(self.log_alpha_cql.exp(), min=0.0, max=1e6).item()

        # Total critic loss
        total_critic_loss = critic_loss + cql_loss

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------- Actor Update ---------------------- #
        # Freeze critic parameters during actor update
        for param in self.critic.parameters():
            param.requires_grad = False

        # Actor loss (maximize Q-values)
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic parameters
        for param in self.critic.parameters():
            param.requires_grad = True

        # ---------------------- Soft Updates ---------------------- #
        self.soft_update(self.critic_target, self.critic)
        self.soft_update(self.actor_target, self.actor)

        return actor_loss.item(), critic_loss.item(), cql_loss.item()

    def soft_update(self, target_net, source_net):
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)


# Define the Replay Buffer
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=1e6):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, action_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.next_state = np.zeros((self.max_size, state_dim))
        self.done = np.zeros((self.max_size, 1))

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def load_dataset(self, dataset):
        num_samples = min(len(dataset['observations']), self.max_size)
        self.state[:num_samples] = dataset['observations'][:num_samples]
        self.action[:num_samples] = dataset['actions'][:num_samples]
        self.reward[:num_samples] = dataset['rewards'][:num_samples].reshape(-1, 1)
        self.next_state[:num_samples] = dataset['next_observations'][:num_samples]
        self.done[:num_samples] = dataset['terminals'][:num_samples].reshape(-1, 1)

        self.size = num_samples
        self.ptr = num_samples % self.max_size

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            self.state[ind],
            self.action[ind],
            self.reward[ind],
            self.next_state[ind],
            self.done[ind]
        )

# Define the training function
def train(
    run_name="CQL",
    env_name="halfcheetah-medium-v2",
    episodes=100,
    seed=1,
    log_video=0,
    save_every=100,
    batch_size=512,
    hidden_size=256,
    learning_rate=3e-4,
    temperature=1.0,
    cql_weight=1.0,
    target_action_gap=10.0,
    with_lagrange=0,
    tau=5e-3,
    eval_every=1
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Create environment
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)

    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize agent
    agent = CQLDDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        tau=tau,
        actor_lr=learning_rate,
        critic_lr=learning_rate,
        alpha_cql=cql_weight,
        target_action_gap=target_action_gap,
        with_lagrange=bool(with_lagrange),
        temperature=temperature,
        hidden_size=hidden_size
    )

    # Initialize replay buffer and load dataset
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    dataset = env.get_dataset()
    replay_buffer.load_dataset(dataset)

    print(f"Dataset size: {replay_buffer.size}")

    # Training loop
    for episode in range(episodes):
        # Since it's offline RL, we don't interact with the environment during training
        actor_loss_list = []
        critic_loss_list = []
        cql_loss_list = []

        for _ in range(1000):  # Adjust the number of training steps per episode
            actor_loss, critic_loss, cql_loss = agent.train(replay_buffer, batch_size=batch_size)
            actor_loss_list.append(actor_loss)
            critic_loss_list.append(critic_loss)
            cql_loss_list.append(cql_loss)

        # Logging
        if (episode + 1) % eval_every == 0:
            eval_reward = evaluate_policy(agent, env)
            print(f"Episode: {episode + 1}, Eval Reward: {eval_reward:.2f}, "
                  f"Actor Loss: {np.mean(actor_loss_list):.4f}, "
                  f"Critic Loss: {np.mean(critic_loss_list):.4f}, "
                  f"CQL Loss: {np.mean(cql_loss_list):.4f}")

        # Save the model
        if (episode + 1) % save_every == 0:
            save_path = f"{run_name}_episode_{episode + 1}.pth"
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
            }, save_path)
            print(f"Model saved to {save_path}")

    print("Training complete.")

def evaluate_policy(agent, env, eval_episodes=10):
    avg_reward = 0.0
    for _ in range(eval_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward

# Example usage
if __name__ == "__main__":
    train(
        run_name="CQL",
        env_name="halfcheetah-medium-v2",
        episodes=100,
        seed=1,
        batch_size=512,
        hidden_size=256,
        learning_rate=3e-4,
        temperature=1.0,
        cql_weight=1.0,
        target_action_gap=10.0,
        with_lagrange=0,
        tau=5e-3,
        eval_every=1
    )
