from utils_v0 import *
from fitted_Q_v0 import *

import gymnasium as gym
import sklearn
import numpy as np
import pickle

from tqdm import tqdm
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt


def show_agent(envName, ag):
    env = gym.make(envName,render_mode = "human")
    
    obs = env.reset()                                           # reset the environment for the start
    obs = obs[0]                                                # first observables
    endBool = False

    while not endBool:
        action = ag.choose_action(obs)
        obs, reward, endBool, info,e = env.step(action)
        obs = obs

        env.render()
    return env


if __name__ == '__main__':
    est = pickle.load(open('model_end_double.sav', 'rb'))[0]
    actionSetDisc = np.linspace(-3, 3, 300)
    ag_test = estAgent(est,actionSetDisc)
    env = show_agent('InvertedDoublePendulum-v4', ag_test)