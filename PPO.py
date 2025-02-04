import torch
import torch.nn as nn
import torch.nn.functional as F
from environment import get_env, env_start
import numpy as np
import torch.distributions as distributions


class Actor(nn.Module):
    def __init__(self, hidden_dim, input_dim, output_dim):
        super(Actor, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.hidden(x)
    
class Critic(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(Critic, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.hidden(x)



class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super(ActorCritic, self).__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, x):
        action_probs = self.actor(x)
        value = self.critic(x)
        return action_probs, value
    

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

    

class PPOAgent:
    def __init__(self, agent, env, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=5, n_epochs=5, steps=20, is_meta_training= True):
        self.meta_model = agent.meta_model
        self.optimizer_actor = None
        self.optimizer_critic = None
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.alpha = alpha
        self.policy_clip = policy_clip
        self.epochs = n_epochs
        self.batch_size = batch_size
        self.variables = agent.variables
        self.equal_input_output = agent.equal_input_output
        self.meta_actor = agent.meta_actor
        self.meta_critic = agent.meta_critic
        self.optimizer = None
        self.MODEL_FILE = agent.MODEL_FILE
        self.device = agent.device 
        self.is_meta_training = is_meta_training
        self.loss_fn = nn.MSELoss()
        self.hidden_dim = agent.hidden_dim
        self.steps = steps
        self.lr = agent.learning_rate_a
        self.envs = agent.envs
        self.test_envs = agent.test_envs
        
        self.memory = PPOMemory(self.batch_size)

        self.env = get_env(self, env, is_meta_training=self.is_meta_training)
        num_actions = self.env.action_space.n
        num_states = self.env.observation_space.shape[0]
        if self.equal_input_output:
            if self.meta_model is None:
                self.meta_actor = Actor(self.hidden_dim, num_states, num_actions).to(self.device)
                self.meta_critic = Critic(self.hidden_dim, num_states).to(self.device)
                self.meta_model = ActorCritic(self.meta_actor, self.meta_critic).to(self.device)

            self.model = ActorCritic(self.meta_actor, self.meta_critic).to(self.device)
            self.model.load_state_dict(self.meta_model.state_dict())
            self.actor = self.model.actor
            self.critic = self.model.critic
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        else:
            pass

        self.actor.train()
        self.critic.train()


    def remember(self, state, action, probs, vals, reward, done):
            self.memory.store_memory(state, action, probs, vals, reward, done)


    def get_action(self, env, observation):
        state = torch.tensor(observation, dtype=torch.float).to(self.device)

        dist = distributions.Categorical(self.actor(state))
        value = self.critic(state)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action)
        value = torch.squeeze(value).item()

        return action, probs, value



    def learn(self, step_count, stuff):
        state = stuff[0]
        action = stuff[1].item()
        reward = stuff[2].item()
        done = stuff[4]
        probs = stuff[5]
        vals = stuff[6]
        self.remember(state, action, probs, vals, reward, done)
        if step_count % self.steps == 0:
            for _ in range(self.epochs):
                state_arr, action_arr, old_prob_arr, vals_arr,\
                reward_arr, dones_arr, batches = \
                        self.memory.generate_batches()

                values = vals_arr
                advantage = np.zeros(len(reward_arr), dtype=np.float32)

                for t in range(len(reward_arr)-1):
                    discount = 1
                    a_t = 0
                    for k in range(t, len(reward_arr)-1):
                        a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                                (1-int(dones_arr[k])) - values[k])
                        discount *= self.gamma*self.gae_lambda
                    advantage[t] = a_t
                advantage = torch.tensor(advantage).to(self.device)

                values = torch.tensor(values).to(self.device)
                for batch in batches:
                    states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                    old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                    actions = torch.tensor(action_arr[batch]).to(self.device)

                    dist = distributions.Categorical(self.actor(states))
                    critic_value = self.critic(states)

                    critic_value = torch.squeeze(critic_value)

                    new_probs = dist.log_prob(actions)
                    prob_ratio = new_probs.exp() / old_probs.exp()
                    #prob_ratio = (new_probs - old_probs).exp()
                    weighted_probs = advantage[batch] * prob_ratio
                    weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                            1+self.policy_clip)*advantage[batch]
                    actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                    returns = advantage[batch] + values[batch]
                    critic_loss = (returns-critic_value)**2
                    critic_loss = critic_loss.mean()

                    total_loss = actor_loss + 0.5*critic_loss
                    self.optimizer_actor.zero_grad()
                    self.optimizer_critic.zero_grad()
                    total_loss.backward()
                    self.optimizer_actor.step()
                    self.optimizer_critic.step()

        self.memory.clear_memory()               