import gymnasium as gym
import sinergym
from sinergym.utils.wrappers import *
import numpy as np
import os
import random


#seed = np.random.randint(0,17022025)

seed = 12465441

def get_env(agent, env, period, weather, is_meta_training= True):
    if is_meta_training == True:
        variables = agent.envs[env]['variables']
        meters = {}

        timestepph = 1
        extra_params={'runperiod': period}
        env = gym.make(env, weather_files = weather, config_params = extra_params, variables = variables, meters = meters)
        #env = gym.make(env, variables = variables, meters = meters)
        env = DiscreteIncrementalWrapper(
            env, initial_values=[20.0, 25.0], delta_temp=1, step_temp=1)
        env = NormalizeObservation(
            env=env)
    else:
        variables = agent.test_envs[env]['variables']
        meters = {}

        np.random.seed(seed=seed)
        random.seed(seed)
        day = np.random.randint(1,(28-0))
        month = np.random.randint(1,13-3)
        year = 1999
        period = (day, month, year, day+0, month+3, year)
        timestepph = 1
        extra_params={'runperiod': period}
        env = gym.make(env, weather_files = weather, config_params = extra_params, variables = variables, meters = meters)
        #env = gym.make(env, variables = variables, meters = meters)
        env = DiscreteIncrementalWrapper(
            env, initial_values=[20.0, 25.0], delta_temp=1, step_temp=1)
        env = NormalizeObservation(
            env=env)
        
    return env

def env_start(env):
    state, _ = env.reset()
    done = False
    return state, done



def adjust_state(env, state):
    temp = 0
    humidity = 0
    people = 0
    new_state = []
    htg = 0
    clg = 0
    if '5zone' in str(env):
        new_state = state
    elif 'shop' in str(env):
        temp = np.mean(state[11:16])
        humidity = np.mean(state[16:21])
        people = np.mean(state[21:26])
        new_elements = np.array([temp, humidity, people])
        new_state = np.concatenate((state[:11], new_elements, state[26:]), axis=0)
    elif 'datacenter' in str(env):
        htg = np.mean(state[9:11])
        clg = np.mean((state[11:13]+state[19:21]))
        temp = np.mean(state[13:15])
        humidity = np.mean(state[15:17])
        people = np.mean(state[17:19])
        new_elements = np.array([htg, clg, temp, humidity, people])
        new_state = np.concatenate((state[:9], new_elements, state[21:]), axis=0)
    else:
        raise ValueError('Environment not found')

    #print(env)
    #print(new_state)
    #print(len(new_state))
    return new_state

def obtain_weather():
    weather_dir = os.path.join(sinergym.__path__[0], 'data', 'weather')
    weather_files = [f for f in os.listdir(weather_dir) if f.endswith('.epw')]
    #random.shuffle(weather_files)

    
    trainidx = int(len(weather_files) * 0.60)
    valididx = int(len(weather_files) * 0.90)
    weather_train = weather_files[:trainidx]
    weather_validate = weather_files[trainidx:valididx]
    weather_test = weather_files[valididx:]

    return weather_train, weather_validate, weather_test


def obtain_period():
    diff = 7
    day = np.random.randint(1,28-diff)
    month = np.random.randint(1,13)
    year = 1999
    #eriods = [(d, m, year, d + diff, m, year) for d, m in zip(days, months)]
    period = (day, month, year, day+diff, month, year)

    return period
