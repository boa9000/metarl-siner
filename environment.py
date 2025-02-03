import gymnasium as gym
import sinergym
from sinergym.utils.wrappers import *
import numpy as np


def get_env(env, variables, is_meta_training= True, seed=111,):
    if is_meta_training == True:
        day = np.random.randint(1,28-6)
        month = np.random.randint(1,13)
        year = 1999
        period = (day, month, year, day+6, month, year)
        timestepph = 1
        extra_params={'runperiod': period}
        env = gym.make(env, config_params = extra_params, variables = variables)
        #env = gym.make(env, variables = self.variables)
        env = DiscreteIncrementalWrapper(
            env, initial_values=[15.0, 30.0], delta_temp=1, step_temp=1)
        env = NormalizeObservation(
            env=env)
    else:
        np.random.seed(seed=seed)
        random.seed(seed)
        day = np.random.randint(1,(28-6))
        month = np.random.randint(1,13)
        year = 1999
        period = (day, month, year, day+6, month, year)
        timestepph = 1
        extra_params={'runperiod': period}
        env = gym.make(env, config_params = extra_params, variables = variables)
        #env = gym.make(env, variables = self.variables)
        env = DiscreteIncrementalWrapper(
            env, initial_values=[15.0, 30.0], delta_temp=1, step_temp=1)
        env = NormalizeObservation(
            env=env)
        
    return env

def env_start(env):
    state, _ = env.reset()
    done = False
    return state, done




