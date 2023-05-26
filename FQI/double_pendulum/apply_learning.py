from utils_v0 import *
from fitted_Q_v0 import *

import gymnasium as gym
import sklearn
import numpy as np
import pickle

from tqdm import tqdm
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt

if __name__ == '__main__':
    est = FQI_loop('InvertedDoublePendulum-v4')