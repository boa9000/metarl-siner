import argparse
import os
import shutil
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
from models import *
import DQN
import PPO
from organize_tools import delete_env_dir
from visualization_tools import GraphSaver
from environment import adjust_state, obtain_weather, obtain_period

DATE_FORMAT = "%m-%d %H:%M:%S"
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
matplotlib.use('Agg')

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"

#seed = np.random.randint(0,17022025)
#np.random.seed(seed=seed)
#random.seed(seed)

class Agent():
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', "r") as file:
            all_hyperparameters = yaml.safe_load(file)
            hyperparameters = all_hyperparameters[hyperparameter_set]

            self.hyperparameter_set = hyperparameter_set
            self.envs = hyperparameters['env_ids']
            self.env_ids = list(self.envs.keys())

            self.test_envs = hyperparameters['test_ids']
            self.test_ids = list(self.test_envs.keys())

            self.test_id_weather = "Eplus-5zone-mixed-continuous-v1"

            self.replay_buffer_size = hyperparameters["replay_buffer_size"]
            self.minibatch_size = hyperparameters["minibatch_size"]
            self.session_length = hyperparameters["session_length"]
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
            self.equal_input_output = hyperparameters["equal_input_output"]
            self.inner_algo = hyperparameters["inner_algo"]
            self.inner_algo = "DQN"
            self.dummy = False

            self.train_samples = 3
            self.validate_samples = 1


            #self.meta_model = MetaModel(self.hidden_dim).to(device)
            self.meta_model = None
            self.meta_actor = None
            self.meta_critic = None
            self.loss_fn = nn.MSELoss()
            #self.optimizer = torch.optim.Adam(self.meta_model.parameters(), lr=self.lr_outer)

            self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}.log")
            self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}.pt")
            self.MODEL_FILE_LATEST = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}_latest.pt")
            self.MODEL_FILE_LATEST_BUILDING = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}_latest_building.pt")
            self.MODEL_FILE_LATEST_WEATHER = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}_latest_weather.pt")
            self.MODEL_FILE_BEST_BUILDING = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}_best_building.pt")
            self.MODEL_FILE_BEST_WEATHER = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}_best_weather.pt")
            self.PLOT_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}.png")
            self.META_PLOT_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameter_set}_meta.png")
            self.device = device

            self.weather_train, self.weather_validate, self.weather_test = obtain_weather()
            self.weather_default = "USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw"

            self.period = obtain_period()

    def run_inner(self, env, weather, is_training=True, is_meta_training=True, validate = False):
        print(weather)
        self.inner_algo = 'DQN'
        if self.inner_algo == 'DQN':
            algo = DQN.DQNAgent(self, self.meta_model, env, weather, is_meta_training = is_meta_training)
            if self.meta_model is None:
                self.meta_model = algo.meta_model
            if validate:
                algo = DQN.DQNAgent(self, None, env, weather, is_meta_training = is_meta_training)

        if self.inner_algo == 'PPO':
            algo = PPO.PPOAgent(self, self.meta_model, env, is_meta_training = is_meta_training)
            if self.meta_model is None:
                self.meta_model = algo.meta_model
                self.meta_actor = self.meta_model.actor
                self.meta_critic = self.meta_model.critic
            pass
        
        rewards_per_session = []
        step_count = 0
        num_of_iterations = self.inner_iterations if is_meta_training else self.test_inner_iterations

        
        for episode in range(num_of_iterations):
            state, _ = algo.env.reset()
            state = adjust_state(env, state)
            state = torch.tensor(state, dtype=torch.float, device=device)
            done = False
            session_reward = 0.0

            while (not done):
                action, probs, value = algo.get_action(algo.env, state)
                if self.dummy:
                    action = torch.tensor(0).to(device)
                
                new_state, reward, terminated, turnicated, info = algo.env.step(action.item())
                #print(f"action: {action.item()}, state: {state}, info: {info} reward: {reward}")
                session_reward += reward

                new_state = adjust_state(env, new_state)
                new_state = torch.tensor(new_state, dtype=torch.float32).to(device)
                reward = torch.tensor(reward, dtype=torch.float32).to(device)

                stuff = (state, action, reward, new_state, terminated, probs, value)
                state = new_state
                done = turnicated or terminated
                step_count += 1

                algo.learn(step_count,stuff)

                if step_count % self.session_length == 0:
                    rewards_per_session.append(session_reward)
                    session_reward = 0.0

                
        #rewards_per_episode = np.array([reward.cpu().numpy() for reward in rewards_per_episode])
        dir = algo.env.get_wrapper_attr('workspace_path')
        algo.env.close()
        print(max(rewards_per_session))
        print(rewards_per_session)
        delete_env_dir(dir)
        model = algo.model
        return model, rewards_per_session

    def update_outer_reptile(self, meta_model, inner_model):
        if self.inner_algo == 'DQN':
            if self.equal_input_output:
                for meta_param, inner_param in zip(meta_model.shared.parameters(), inner_model.shared.parameters()):
                    meta_param.data.add_(inner_param.data - meta_param.data, alpha=self.lr_outer)
            else:
                for meta_param, inner_param in zip(meta_model.parameters(), inner_model.meta_model.parameters()):
                    meta_param.data.add_(inner_param.data - meta_param.data, alpha=self.lr_outer)
        if self.inner_algo == 'PPO':
            if self.equal_input_output:
                for meta_param, inner_param in zip(meta_model.actor.parameters(), inner_model.actor.parameters()):
                    meta_param.data.add_(inner_param.data - meta_param.data, alpha=self.lr_outer)
                for meta_param, inner_param in zip(meta_model.critic.parameters(), inner_model.critic.parameters()):
                    meta_param.data.add_(inner_param.data - meta_param.data, alpha=self.lr_outer)
            else:
                pass

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
        gs = GraphSaver(self.PLOT_FILE, self.META_PLOT_FILE, self.test_ids, self.env_ids)
        validation_curve = []
        difference_old = -100

        if is_meta_training:
            for iteration in range(self.outer_iterations):
                self.period = obtain_period()
                if iteration % 5 == 0 and iteration != 0:
                    vali_avg_rewards_meta = []
                    vali_avg_rewards_metaless = []
                    print(f"Validation on iteration {iteration}")
                                        
                    for env in self.test_ids:
                        model, r_per_ep = self.run_inner(env, self.weather_default, True, is_meta_training=True)
                        vali_avg_rewards_meta.append(np.mean(r_per_ep))
                    for env in self.test_ids:
                        model, r_per_ep = self.run_inner(env, self.weather_default, True, is_meta_training=True, validate = True)
                        vali_avg_rewards_metaless.append(np.mean(r_per_ep))
                    difference = np.mean(vali_avg_rewards_meta) - np.mean(vali_avg_rewards_metaless)
                    print(f"Difference between averages: {difference}")
                    validation_curve.append(difference)
                    
                    if difference > difference_old:
                        print("New best model found")
                        torch.save(self.meta_model.state_dict(), self.MODEL_FILE_BEST_BUILDING)
                    difference_old = difference

                    gs.save_validation_curve(validation_curve, iteration, "building")


                for env in self.env_ids:
                    model, r_per_ep = self.run_inner(env, self.weather_default, True, is_meta_training=True)
                    models[env] = model
                    rewards[env] = self.relative_improvement(r_per_ep)
                    max_rs[env] = np.max(r_per_ep)

                    if self.meta_algorithm == 'reptile':
                        self.update_outer_reptile(self.meta_model, model)
                    elif self.meta_algorithm == 'maml':
                        self.update_outer_maml(self.meta_model, model)

                rewards_list = [float(reward) for reward in rewards.values()]
                avg_reward = np.mean(list(rewards.values()))
                print(f"Iteration {iteration}, rewards: {rewards_list}, avg reward: {avg_reward:.3f}, best reward avg: {max_reward_avg:.3f}")
                '''
                if avg_reward > max_reward_avg:
                    print(f"New best model found with avg reward: {avg_reward}")
                    max_reward_avg = avg_reward
                    torch.save(self.meta_model.state_dict(), self.MODEL_FILE)
                '''
                iterations.append(iteration)
                iteration_reward.append(np.mean(list(rewards.values())))
                gs.save_graph(iteration_reward, iterations)
                torch.save(self.meta_model.state_dict(), self.MODEL_FILE_LATEST_BUILDING)
        else:
            self.period = obtain_period(test=True)
            rewards_dict_meta = {env: None for env in self.test_ids}
            rewards_dict_metaless = {env: None for env in self.test_ids}
            rewards_dict_dummy = {env: None for env in self.test_ids}
            rewards_avg_meta = []
            rewards_avg_metaless = []
            rewards_avg_dummy = []

            if self.inner_algo == 'DQN':
                self.meta_model = DQN.DQNnetwork(self.hidden_dim, 15, 5).to(device)
                self.meta_model.load_state_dict(torch.load(self.MODEL_FILE_LATEST_BUILDING, weights_only=True))
            if self.inner_algo == 'PPO':
                self.meta_model = PPO.ActorCritic(PPO.Actor(self.hidden_dim, 15, 5).to(device), PPO.Critic(self.hidden_dim, 11).to(device)).to(device)
                self.meta_actor = self.meta_model.actor
                self.meta_critic = self.meta_model.critic
            print(f"with loaded model and {self.test_inner_iterations} episodes")
            for env in self.test_ids:
                model, r_per_ep_meta = self.run_inner(env, self.weather_default, is_training=True, is_meta_training=False)
                print(f"Env: {env}, Mean Reward: {np.mean(r_per_ep_meta)}, Max Reward: {np.max(r_per_ep_meta)}")
                rewards_dict_meta[env] = r_per_ep_meta
                rewards_avg_meta.append(np.mean(r_per_ep_meta))

            del self.meta_model
            del self.meta_actor
            del self.meta_critic
            self.meta_model = None
            self.meta_actor = None
            self.meta_critic = None
            print(f"without loaded model and {self.test_inner_iterations} episodes")
            #self.dummy = True
            for env in self.test_ids:
                model, r_per_ep_metaless = self.run_inner(env, self.weather_default, is_training=True, is_meta_training=False)
                print(f"Env: {env}, Mean Reward: {np.mean(r_per_ep_metaless)}, Max Reward: {np.max(r_per_ep_metaless)}")
                rewards_dict_metaless[env] = r_per_ep_metaless
                rewards_avg_metaless.append(np.mean(r_per_ep_metaless))

            for env in self.test_ids:
                self.dummy = True
                model, r_per_ep_dummy = self.run_inner(env, self.weather_default, is_training=True, is_meta_training=False)
                print(f"Env: {env}, Mean Reward: {np.mean(r_per_ep_dummy)}, Max Reward: {np.max(r_per_ep_dummy)}")
                rewards_dict_dummy[env] = r_per_ep_dummy
                rewards_avg_dummy.append(np.mean(r_per_ep_dummy))

            gs.save_meta_graph(rewards_dict_meta, rewards_dict_metaless,rewards_dict_dummy)

            overall_avg_meta = np.mean(rewards_avg_meta)
            overall_avg_metaless = np.mean(rewards_avg_metaless)
            overall_avg_dummy = np.mean(rewards_avg_dummy)

            print(f"Averages for meta model: {rewards_avg_meta}")
            print(f"Overall average for meta model: {overall_avg_meta}\n")

            print(f"Averages for metaless model: {rewards_avg_metaless}")
            print(f"Overall average for metaless model: {overall_avg_metaless}\n")

            print(f"Averages for dummy model: {rewards_avg_dummy}")
            print(f"Overall average for dummy model: {overall_avg_dummy}\n")


    def run_weather(self, is_meta_training=True):
        iterations = []
        iteration_reward = []
        max_reward_avg = -1e9
        rewards = {weather: 0 for weather in self.weather_train}
        max_rs = {weather: 0 for weather in self.weather_train}
        gs = GraphSaver(self.PLOT_FILE, self.META_PLOT_FILE, self.weather_test, self.weather_train)
        validation_curve = []
        env = self.env_ids[0]
        difference_old = -100

        if is_meta_training:
            for iteration in range(self.outer_iterations):
                self.period = obtain_period()
                if iteration % 5 == 0 and iteration != 0:
                    vali_avg_rewards_meta = []
                    vali_avg_rewards_metaless = []
                    print(f"Validation on iteration {iteration}")
                    
                    sample_weather_validate = random.sample(self.weather_validate, self.validate_samples)
                    
                    for weather in sample_weather_validate:
                        model, r_per_ep = self.run_inner(env, weather, True, is_meta_training=True)
                        vali_avg_rewards_meta.append(np.mean(r_per_ep))
                    for weather in sample_weather_validate:
                        model, r_per_ep = self.run_inner(env, weather, True, is_meta_training=True, validate = True)
                        vali_avg_rewards_metaless.append(np.mean(r_per_ep))
                    difference = np.mean(vali_avg_rewards_meta) - np.mean(vali_avg_rewards_metaless)
                    print(f"Difference between averages: {difference}")
                    validation_curve.append(difference)

                    if difference > difference_old:
                        print("New best model found")
                        torch.save(self.meta_model.state_dict(), self.MODEL_FILE_BEST_WEATHER)

                    difference_old = difference
                    gs.save_validation_curve(validation_curve, iteration, "weather")


                sample_weather_train = random.sample(self.weather_train, self.train_samples)
                for weather in sample_weather_train:
                    model, r_per_ep = self.run_inner(env, weather, True, is_meta_training=True)
                    rewards[weather] = self.relative_improvement(r_per_ep)
                    max_rs[weather] = np.max(r_per_ep)

                    if self.meta_algorithm == 'reptile':
                        self.update_outer_reptile(self.meta_model, model)
                    elif self.meta_algorithm == 'maml':
                        self.update_outer_maml(self.meta_model, model)

                #rewards_list = [float(reward) for reward in rewards.values()]
                #avg_reward = np.mean(list(rewards.values()))
                #print(f"Iteration {iteration}, rewards: {rewards_list}, avg reward: {avg_reward:.3f}, best reward avg: {max_reward_avg:.3f}")
                
                '''
                if avg_reward > max_reward_avg:
                    print(f"New best model found with avg reward: {avg_reward}")
                    max_reward_avg = avg_reward
                    torch.save(self.meta_model.state_dict(), self.MODEL_FILE)
                '''

                #iterations.append(iteration)
                #iteration_reward.append(np.mean(list(rewards.values())))
                #gs.save_graph(iteration_reward, iterations)
                torch.save(self.meta_model.state_dict(), self.MODEL_FILE_LATEST_WEATHER)
        else:
            self.period = obtain_period(test=True)
            rewards_dict_meta = {weather: None for weather in self.weather_test}
            rewards_dict_metaless = {weather: None for weather in self.weather_test}
            rewards_dict_dummy = {weather: None for weather in self.weather_test}
            rewards_avg_meta = []
            rewards_avg_metaless = []
            rewards_avg_dummy = []

            if self.inner_algo == 'DQN':
                self.meta_model = DQN.DQNnetwork(self.hidden_dim, 15, 5).to(device)
                self.meta_model.load_state_dict(torch.load(self.MODEL_FILE_LATEST_WEATHER, weights_only=True))
            if self.inner_algo == 'PPO':
                self.meta_model = PPO.ActorCritic(PPO.Actor(self.hidden_dim, 15, 5).to(device), PPO.Critic(self.hidden_dim, 11).to(device)).to(device)
                self.meta_actor = self.meta_model.actor
                self.meta_critic = self.meta_model.critic
            print(f"with loaded model and {self.test_inner_iterations} episodes")
            for weather in self.weather_test:
                model, r_per_ep_meta = self.run_inner(env, weather, is_training=True, is_meta_training=False)
                print(f"weather: {weather}, Mean Reward: {np.mean(r_per_ep_meta)}, Max Reward: {np.max(r_per_ep_meta)}")
                rewards_dict_meta[weather] = r_per_ep_meta
                rewards_avg_meta.append(np.mean(r_per_ep_meta))
            del self.meta_model
            del self.meta_actor
            del self.meta_critic
            self.meta_model = None
            self.meta_actor = None
            self.meta_critic = None
            print(f"without loaded model and {self.test_inner_iterations} episodes")
            #self.dummy = True
            for weather in self.weather_test:
                model, r_per_ep_metaless = self.run_inner(env, weather, is_training=True, is_meta_training=False)
                print(f"weather: {weather}, Mean Reward: {np.mean(r_per_ep_metaless)}, Max Reward: {np.max(r_per_ep_metaless)}")
                rewards_dict_metaless[weather] = r_per_ep_metaless
                rewards_avg_metaless.append(np.mean(r_per_ep_metaless))

            for weather in self.weather_test:
                self.dummy = True
                model, r_per_ep_dummy = self.run_inner(env, weather, is_training=True, is_meta_training=False)
                print(f"Env: {weather}, Mean Reward: {np.mean(r_per_ep_dummy)}, Max Reward: {np.max(r_per_ep_dummy)}")
                rewards_dict_dummy[weather] = r_per_ep_dummy
                rewards_avg_dummy.append(np.mean(r_per_ep_dummy))

            gs.save_meta_graph(rewards_dict_meta, rewards_dict_metaless,rewards_dict_dummy)
            
            overall_avg_meta = np.mean(rewards_avg_meta)
            overall_avg_metaless = np.mean(rewards_avg_metaless)
            overall_avg_dummy = np.mean(rewards_avg_dummy)

            print(f"Averages for meta model: {rewards_avg_meta}")
            print(f"Overall average for meta model: {overall_avg_meta}\n")

            print(f"Averages for metaless model: {rewards_avg_metaless}")
            print(f"Overall average for metaless model: {overall_avg_metaless}\n")

            print(f"Averages for dummy model: {rewards_avg_dummy}")
            print(f"Overall average for dummy model: {overall_avg_dummy}\n")
            
    def relative_improvement(self, rewards):
        initial_reward = rewards[0]
        final_reward = rewards[-1]
        return (final_reward - initial_reward) / (abs(initial_reward) + 1e-8)  # Avoid division by zero

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test the agent')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Meta Train', action='store_true')
   
    parser.add_argument(
        "--mode",
        choices=["building", "weather"], 
        required=True,  
        help="Select mode: 'building' for building-based training, 'weather' for weather-based training."
    )
   
    args = parser.parse_args()

    rep = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        if args.mode == "weather":
            rep.run_weather(is_meta_training=True)
        elif args.mode == "building":
            rep.run(is_meta_training=True)
    else:
        if args.mode == "weather":
            rep.test_envs = rep.envs
            rep.test_ids = rep.test_id_weather
            rep.run_weather(is_meta_training=False)
        elif args.mode == "building":
            rep.run(is_meta_training=False)
