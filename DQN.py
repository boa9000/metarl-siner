from torch import nn
import torch
import torch.nn.functional as F 
from experience_replay import ReplayMemory
import random
from environment import get_env, env_start

class DQNnetwork(nn.Module):
    def __init__(self, hidden_dim, input_dim, output_dim):
        super(DQNnetwork, self).__init__()
        self.shared = nn.Sequential(         
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.shared(x)
    

class TaskModel(nn.Module):
    def __init__(self, meta_model, input_dim, output_dim):
        super(TaskModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, meta_model.hidden[0].in_features)
        self.meta_model = meta_model
        self.output_layer = nn.Linear(meta_model.hidden[-2].out_features, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.meta_model(x)
        x = self.output_layer(x)
        return x
    


class DQNAgent:
    def __init__(self, agent, env, is_meta_training= True):

        self.epsilon_initial = agent.epsilon_initial
        self.epsilon_final = agent.epsilon_final
        self.epsilon_decay = agent.epsilon_decay
        self.epsilon_min = agent.epsilon_min
        self.lr_inner = agent.learning_rate_a
        self.lr_outer = agent.lr_outer
        self.hidden_dim = agent.hidden_dim
        self.replay_buffer_size = agent.replay_buffer_size
        self.equal_input_output = agent.equal_input_output
        self.meta_model = agent.meta_model
        self.optimizer = None
        self.MODEL_FILE = agent.MODEL_FILE
        self.device = agent.device 
        self.is_meta_training = is_meta_training
        self.mini_batch_size = agent.mini_batch_size
        self.session_length = agent.session_length
        self.loss_fn = nn.MSELoss()


        self.env = env_start(env, is_meta_training= self.is_meta_training)
        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]
        if self.equal_input_output:
            if self.meta_model is None:
                self.meta_model = DQNnetwork(self.hidden_dim, num_states, num_actions).to(self.device)

            self.model = DQNnetwork(self.hidden_dim, num_states, num_actions).to(self.device)
            self.model.load_state_dict(self.meta_model.state_dict())
        else:
            self.model = TaskModel(self.meta_model, num_states, num_actions).to(self.device)
        self.model.train()
        self.epsilon = self.epsilon_initial
        self.memory = ReplayMemory(self.replay_buffer_size)
        if self.equal_input_output:
            self.target_model = DQNnetwork(self.hidden_dim, num_states, num_actions).to(self.device)
        else:
            self.target_model = TaskModel(self.meta_model, num_states, num_actions).to(self.device)
        #target_model = DQNnetwrok(num_states, num_actions, self.hidden_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_inner)
        self.step_count = 0
        self.rewards_per_session = []


    def get_action(self, env, state):
        if random.random() < self.epsilon:
            action = env.action_space.sample()
            action = torch.tensor(action, dtype=torch.int64, device=self.device)
        else:
            with torch.no_grad():
                action = self.model(state.unsqueeze(dim=0)).squeeze().argmax()
        return action
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def train(self):
        if (len(self.memory) > self.minibatch_size) and (self.step_count % 2 == 0):
            minibatch = self.memory.sample(self.minibatch_size)
            self.optimize(minibatch, self.model, self.target_model)

            self.update_epsilon()

        if self.step_count % self.session_length == 0:
            self.rewards_per_session.append(session_reward)
            session_reward = 0.0


    def optimize(self, minibatch, model, target_model):
        states, actions, rewards, next_states, terminations = zip(*minibatch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states).float()
        terminations = torch.tensor(terminations, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.discount_factor_g * target_model(next_states).max(dim=1)[0]

        current_q = model(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #for param in model.parameters():
        #    param.data -= self.lr_inner * param.grad.data