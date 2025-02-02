import argparse
import os
import random
from datetime import datetime, timedelta
import logging

import gymnasium as gym
import sinergym
from sinergym.utils.wrappers import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.autograd import grad

from experience_replay import ReplayMemory
from models import DQN, MetaModel, TaskModel

DATE_FORMAT = "%m-%d %H:%M:%S"
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
matplotlib.use('Agg')

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"

seed = np.random.randint(0,17022025)
#np.random.seed(seed=seed)
#random.seed(seed)

class Agent():
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', "r") as file:
            all_hyperparameters = yaml.safe_load(file)
            hyperparameters = all_hyperparameters[hyperparameter_set]

            self.hyperparameter_set = hyperparameter_set
            self.env_ids = hyperparameters['env_ids']
            self.test_ids = hyperparameters['test_ids']
            self.variables = hyperparameters['variables']
            self.replay_buffer_size = hyperparameters["replay_buffer_size"]
            self.minibatch_size = hyperparameters["minibatch_size"]
            self.epsilon_initial = hyperparameters["epsilon_initial"]
            self.epsilon_decay = hyperparameters["epsilon_decay"]
            self.epsilon_min = hyperparameters["epsilon_min"]
            self.network_sync_rate = hyperparameters["network_sync_rate"]
            self.learning_rate_a = hyperparameters["learning_rate_a"]
            self.discount_factor_g = hyperparameters["discount_factor_g"]
            self.hidden_dim = hyperparameters["hidden_dim"]
            self.env_make_params = hyperparameters.get('env_make_params', {})
            self.inner_iterations = hyperparameters["inner_iterations"]
            self.outer_iterations = hyperparameters["outer_iterations"]
            self.lr_outer = hyperparameters["lr_outer"]
            self.test_inner_iterations = hyperparameters["test_inner_iterations"]
            self.meta_algorithm = hyperparameters["meta_algorithm"]

            self.meta_model = MetaModel(self.hidden_dim).to(device)
            self.loss_fn = nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.meta_model.parameters(), lr=self.lr_outer)

            self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}.log")
            self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}.pt")
            self.MODEL_FILE_LATEST = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}_latest.pt")
            self.PLOT_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}.png")
            self.META_PLOT_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}_meta.png")

    def run_inner(self, env, is_training=True, is_meta_training=True):
        env = self.get_env(env=env, is_meta_training=is_meta_training)
        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]
        model = TaskModel(self.meta_model, num_states, num_actions).to(device)
        #model = DQN(num_states, num_actions, self.hidden_dim).to(device)

        rewards_per_episode = []
        epsilon_history = []
        last_graph_update_time = datetime.now()

        if is_training:
            model.train()
            epsilon = self.epsilon_initial
            memory = ReplayMemory(self.replay_buffer_size)
            target_model = TaskModel(self.meta_model, num_states, num_actions).to(device)
            #target_model = DQN(num_states, num_actions, self.hidden_dim).to(device)
            target_model.load_state_dict(model.state_dict())
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate_a)
            epsilon_history = []
            step_count = 0
            best_reward = -1e9
        else:
            model.load_state_dict(torch.load(self.MODEL_FILE))
            model.eval()

        num_of_iterations = self.inner_iterations if is_meta_training else self.test_inner_iterations
        lost_episodes = 0

        
        for episode in range(num_of_iterations + lost_episodes):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            done = False
            episode_reward = 0.0

            while (not done):
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = model(state.unsqueeze(dim=0)).squeeze().argmax()
                        

                new_state, reward, terminated, turnicated, info = env.step(action.item())
                #print(f"action: {action.item()}, state: {state}, info: {info} reward: {reward}")
                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float32).to(device)
                reward = torch.tensor(reward, dtype=torch.float32).to(device)

                if is_training:
                    memory.append((state, action, reward, new_state, terminated))
                    

                state = new_state

                done = turnicated or terminated

                step_count += 1
                if is_training:
                    if (len(memory) > self.minibatch_size) and (step_count % 4 == 0):
                        minibatch = memory.sample(self.minibatch_size)
                        self.optimize(minibatch, model, target_model)
                        
                        #epsilon = max(self.epsilon_min, self.epsilon_min + (self.epsilon_initial - self.epsilon_min) * (episode / (num_of_iterations * 0.7)))
                        epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                        epsilon_history.append(epsilon)
                        #print("updated!")

                        if step_count >= self.network_sync_rate:
                            target_model.load_state_dict(model.state_dict())
                            step_count = 0
                            #print("Target Netowrk Updated")
                    else:
                        lost_episodes += 1
                

            rewards_per_episode.append(episode_reward)

        #rewards_per_episode = np.array([reward.cpu().numpy() for reward in rewards_per_episode])
        env.close()
        print(max(rewards_per_episode))
        print(rewards_per_episode)
        return model, rewards_per_episode

    def get_env(self, env, is_meta_training= True):
        if is_meta_training == True:
            day = np.random.randint(1,28-2)
            month = np.random.randint(1,13)
            year = 1999
            period = (day, month, year, day+2, month, year)
            timestepph = 1
            extra_params={'runperiod': period}
            env = gym.make(env, config_params = extra_params, variables = self.variables)
            #env = gym.make(env, variables = self.variables)
            env = DiscreteIncrementalWrapper(
                env, initial_values=[15.0, 30.0], delta_temp=1, step_temp=1)
            env = NormalizeObservation(
                env=env)
        else:
            np.random.seed(seed=seed)
            random.seed(seed)
            day = np.random.randint(1,(28-2))
            month = np.random.randint(1,12)
            year = 1999
            period = (day, month, year, day+2, month, year)
            timestepph = 1
            extra_params={'runperiod': period}
            env = gym.make(env, config_params = extra_params, variables = self.variables)
            #env = gym.make(env, variables = self.variables)
            env = DiscreteIncrementalWrapper(
                env, initial_values=[15.0, 30.0], delta_temp=1, step_temp=1)
            env = NormalizeObservation(
                env=env)
            
        return env

    def update_outer_reptile(self, meta_model, inner_model):
        for meta_param, inner_param in zip(meta_model.parameters(), inner_model.meta_model.parameters()):
            meta_param.data.add_(inner_param.data - meta_param.data, alpha=self.lr_outer)

    def update_outer_maml(self, meta_model, models):
        meta_model.train()
        for name, param in meta_model.named_parameters():
            param.grad = torch.zeros_like(param.data)

        for env in models:
            model = models[env]
            loss = self.compute_loss(model, env)
            loss_tensor = torch.tensor(loss, requires_grad=True)
            grads = torch.autograd.grad(loss_tensor, meta_model.parameters(), create_graph=True, allow_unused=True)
            for param, grad in zip(meta_model.parameters(), grads):
                if grad is not None:
                    param.grad += grad / len(models)
                else:
                    param.grad += torch.zeros_like(param.data)

        self.optimizer.step()

    def compute_loss(self, model, env):
        # Compute the loss for a given model and environment
        env = gym.make(env)
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        terminated = False
        total_loss = 0.0
        step_count = 0

        while not terminated:
            action = model(state.unsqueeze(dim=0)).squeeze().argmax()
            new_state, reward, terminated, truncated, _ = env.step(action.item())
            new_state = torch.tensor(new_state, dtype=torch.float32).to(device)
            reward = torch.tensor(reward).to(device)

            with torch.no_grad():
                target_q = reward + (1 - terminated) * self.discount_factor_g * model(new_state.unsqueeze(dim=0)).max(dim=1)[0]

            current_q = model(state.unsqueeze(dim=0)).gather(dim=1, index=action.unsqueeze(dim=0).unsqueeze(dim=1)).squeeze()
            loss = self.loss_fn(current_q, target_q)
            total_loss += loss.item()
            step_count += 1

            state = new_state
            if truncated:
                break

        return total_loss / step_count

    def run(self, is_meta_training=True):
        iterations = []
        iteration_reward = []
        max_reward_avg = -1e9
        models = {env: None for env in self.env_ids}
        rewards = {env: None for env in self.env_ids}
        max_rs = {env: None for env in self.env_ids}

        if is_meta_training:
            for iteration in range(self.outer_iterations):
                for env in self.env_ids:
                    model, r_per_ep = self.run_inner(env, True, is_meta_training=True)
                    models[env] = model
                    rewards[env] = np.mean(r_per_ep)
                    max_rs[env] = np.max(r_per_ep)

                    if self.meta_algorithm == 'reptile':
                        self.update_outer_reptile(self.meta_model, model)
                    elif self.meta_algorithm == 'maml':
                        self.update_outer_maml(self.meta_model, model)

                rewards_list = [float(reward) for reward in rewards.values()]
                avg_reward = np.mean(list(rewards.values()))
                print(f"Iteration {iteration}, rewards: {rewards_list}, avg reward: {avg_reward:.3f}, best reward avg: {max_reward_avg:.3f}")
                if avg_reward > max_reward_avg:
                    print(f"New best model found with avg reward: {avg_reward}")
                    max_reward_avg = avg_reward
                    torch.save(self.meta_model.state_dict(), self.MODEL_FILE)
                iterations.append(iteration)
                iteration_reward.append(np.mean(list(rewards.values())))
                self.save_graph(iteration_reward, iterations)
                torch.save(self.meta_model.state_dict(), self.MODEL_FILE_LATEST)
        else:
            rewards_dict_meta = {env: None for env in self.test_ids}
            rewards_dict_metaless = {env: None for env in self.test_ids}
            self.meta_model.load_state_dict(torch.load(self.MODEL_FILE_LATEST, weights_only=True))
            print(f"with loaded model and {self.test_inner_iterations} episodes")
            for env in self.test_ids:
                model, r_per_ep_meta = self.run_inner(env, is_training=True, is_meta_training=False)
                print(f"Env: {env}, Mean Reward: {np.mean(r_per_ep_meta)}, Max Reward: {np.max(r_per_ep_meta)}")
                rewards_dict_meta[env] = r_per_ep_meta

            self.meta_model = MetaModel(self.hidden_dim).to(device)
            print(f"without loaded model and {self.test_inner_iterations} episodes")
            for env in self.test_ids:
                model, r_per_ep_metaless = self.run_inner(env, is_training=True, is_meta_training=False)
                print(f"Env: {env}, Mean Reward: {np.mean(r_per_ep_metaless)}, Max Reward: {np.max(r_per_ep_metaless)}")
                rewards_dict_metaless[env] = r_per_ep_metaless

            self.save_meta_graph(rewards_dict_meta, rewards_dict_metaless)

    def optimize(self, minibatch, model, target_model):
        states, actions, rewards, next_states, terminations = zip(*minibatch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states).float()
        terminations = torch.tensor(terminations, dtype=torch.float32).to(device)

        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.discount_factor_g * target_model(next_states).max(dim=1)[0]

        current_q = model(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #for param in model.parameters():
        #    param.data -= self.learning_rate_a * param.grad.data

    def save_graph(self, avg_rewards, iteration):
        fig = plt.figure()
        plt.subplot(111)
        plt.ylabel("avg_Reward")
        plt.plot(avg_rewards)
        fig.savefig(self.PLOT_FILE)
        plt.close(fig)

    def save_meta_graph(self, rewards_with_model, rewards_without_model):
        # Create subplots
        num_envs = len(self.test_ids)
        fig, axs = plt.subplots(num_envs, 1, figsize=(12, 6 * num_envs))

        # If there's only one environment, wrap axs in a list
        if num_envs == 1:
            axs = [axs]

        # Plot rewards for each environment
        for i, env in enumerate(self.test_ids):
            axs[i].plot(rewards_with_model[env], label='With Meta Model')
            axs[i].plot(rewards_without_model[env], label='Without Meta Model')
            axs[i].set_title(f'Rewards for {env}')
            axs[i].set_ylabel('Reward')
            axs[i].legend()

        # Save the figure
        plt.tight_layout()
        fig.savefig(self.META_PLOT_FILE)
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test DQN agent on CartPole-v0')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    rep = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        rep.run(is_meta_training=True)
    else:
        rep.run(is_meta_training=False)
