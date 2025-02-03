import torch
import torch.nn as nn
import torch.nn.functional as F
from environment import get_env, env_start


class ActorCritic(nn.Module):
    def __init__(self, hidden_dim, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        # Actor component
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # Critic component
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # Forward pass for actor
        action_probs = F.softmax(self.actor(x), dim=-1)
        # Forward pass for critic
        value = self.critic(x)
        return action_probs, value
    

    

class PPOAgent:
    def __init__(self, agent, env, hidden_dim, input_dim, output_dim, gamma=0.99, lambda_=0.95,
                  clip_epsilon=0.2, epochs=4, lr=3e-4, is_meta_training= True):
        self.meta_model = agent.meta_model
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.rollout_length = 128
        self.variables = agent.variables
        self.equal_input_output = agent.equal_input_output
        self.meta_actor = agent.meta_actor
        self.meta_critic = agent.meta_critic
        self.optimizer = None
        self.MODEL_FILE = agent.MODEL_FILE
        self.device = agent.device 
        self.is_meta_training = is_meta_training
        self.loss_fn = nn.MSELoss()

        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.env = get_env(env, variables= self.variables,is_meta_training=self.is_meta_training)
        num_actions = self.env.action_space.n
        num_states = self.env.observation_space.shape[0]
        if self.equal_input_output:
            if self.meta_model is None:
                self.meta_model = ActorCritic(hidden_dim, num_states, num_actions).to(self.device)
                self.meta_actor = self.meta_model.actor
                self.meta_critic = self.meta_model.critic

            self.model = ActorCritic(hidden_dim, num_states, num_actions).to(self.device)
            self.actor = self.model.actor
            self.actor.load_state_dict(self.meta_actor.state_dict())
            self.critic = self.model.critic
            self.critic.load_state_dict(self.meta_critic.state_dict())
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)
        else:
            pass

        self.actor.train()
        self.critic.train()






    def append_rollout(self, stuff):
        self.states.append(stuff[0])
        self.actions.append(stuff[1])
        self.rewards.append(stuff[2])
        self.dones.append(torch.tensor(stuff[3], dtype=torch.float32))


    def clear_rollout(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def train(self, num_steps, stuff):
        self.append_rollout(stuff)

        if len(self.states) > self.rollout_length:
            self.optimize(stuff, self.dones)
            self.clear_rollout()


    def compute_gae(self, rewards, values, dones):
        gae = 0
        advantages = []
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lambda_ * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return torch.tensor(advantages, dtype=torch.float32)


    def optimize(self, stuff, dones):
        # Compute values and log probabilities
        values = self.critic(self.states).squeeze()
        probs = self.actor(self.states)
        dist = torch.distributions.Categorical(probs)
        log_probs_old = dist.log_prob(self.actions).detach()

        # Compute GAE
        advantages = self.compute_gae(self.rewards, values, dones)
        returns = advantages + values[:-1]

        # Update Actor and Critic for multiple epochs
        for _ in range(self.epochs):
            # Compute new log probabilities and values
            probs = self.actor(self.states)
            dist = torch.distributions.Categorical(probs)
            log_probs_new = dist.log_prob(self.actions)
            values_new = self.critic(self.states).squeeze()

            # Compute ratios and clipped objective
            ratios = torch.exp(log_probs_new - log_probs_old)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Compute Critic loss
            critic_loss = nn.MSELoss()(values_new[:-1], returns)

            # Update networks
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()



