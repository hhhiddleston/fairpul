from __future__ import division
import os, sys
import math
import numpy as np
import matplotlib.pyplot as plt  # for plotting stuff
from random import seed, shuffle
from scipy.stats import multivariate_normal  # generating synthetic data
from sklearn.linear_model import LogisticRegression

SEED = 1122334455
seed(SEED)  # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)

def generate_toy_data(n_samples, n_samples_low, n_dimensions):
    np.random.seed(0)
    varA = 0.8
    aveApos = [-1.0] * n_dimensions
    aveAneg = [1.0] * n_dimensions
    varB = 0.5
    # aveBpos = [0.5] * int(n_dimensions / 2) + [-0.5] * int(n_dimensions / 2 + n_dimensions % 2)
    aveBpos = [-0.5] * n_dimensions
    aveBneg = [0.5] * n_dimensions

    X = np.random.multivariate_normal(aveApos, np.diag([varA] * n_dimensions), n_samples)
    X = np.vstack([X, np.random.multivariate_normal(aveAneg, np.diag([varA] * n_dimensions), n_samples)])
    X = np.vstack([X, np.random.multivariate_normal(aveBpos, np.diag([varB] * n_dimensions), n_samples_low)])
    X = np.vstack([X, np.random.multivariate_normal(aveBneg, np.diag([varB] * n_dimensions), n_samples)])
    sensible_feature = [1] * (n_samples * 2) + [0] * (n_samples + n_samples_low)
    sensible_feature = np.array(sensible_feature)
    sensible_feature.shape = (len(sensible_feature), 1)
    X = np.hstack([X, sensible_feature])
    y = [1] * n_samples + [0] * n_samples + [1] * n_samples_low + [0] * n_samples
    y = np.array(y)
   
    sensible_feature = sensible_feature.reshape(-1)
    X_new = np.hstack((X, np.power(X[:, 0], 2).reshape(-1, 1)))
    X = np.hstack((X_new, np.power(X[:, 1], 2).reshape(-1, 1)))
    # assert X.shape[1] == 4
    return X, y, sensible_feature

