import gymnasium as gym
import sklearn
import numpy as np

from tqdm import tqdm
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt

import pickle
from utils_v0 import *
from agent import *


def getMeasure(envName, est, nbGen = 100):
    rewMean = []
    rewStd = []
    
    iterations = np.arange(0,59, 4)
    iterations = np.hstack((iterations, [59]))
    for n in tqdm(iterations):
        print(n)
        rew = None

        actionSet = np.linspace(-3,3,300)
        ag = estAgent(est, actionSet)


        for i in range(nbGen):
            traj = simulate(envName, ag, nb = 100000)

            #print(traj)

            r = np.array([traj[i][2] for i in range(len(traj))])

            if rew is None:
                rew = r
            else:
                rew = np.hstack((rew, r))


        rewMean.append(np.mean(rew))
        print(rewMean)
        rewStd.append(np.std(rew))

    return rewMean, rewStd


def getInitDataset(S):
    """
    Creates the the dataset used for the initialization of FQI

    Args:
    -----
    - `S`: set of one-step transitions

    Output:
    -------
    - `X`: np.array with the states of the system (angle, speed_cart, angle_speed, action)
    - `y`: np.array with the associated reward
    """

    ## init of the dataset
    X = np.array([])
    y = []

    for step in tqdm(S):            # loop over all one-step transitions

        ## adding the instances to the dataset
        if X.shape[0] == 0:
            X = np.array([*step[0], step[1]])
        else:
            X = np.vstack((X, np.array([*step[0], step[1]])))
        
        y.append(step[2])

    y = np.array(y)

    return X, y



def createInput(actionSet, X, traj):
    """
    Creates a dictionnary associating each next state with a possible action
    (used in the maximum of fitted Q iterations)

    Args:
    -----
    - `actionSet`: discretized action set (list of possible actions)
    - `X`: initial dataset (just to have the number of one-step transitions in shape[0]) => could be replaced
    - `traj`: set of one-step transitions

    Output:
    -------
    A dictionnary
    """

    ## Init of the dictionnary
    d = {}

    for action in tqdm(actionSet):
        temp_arr = []                       # temporary array for the states-action of the current action

        for k in range(X.shape[0]):
            temp_arr.append([*traj[k][-1], action])       # loop over all the states

        d[action] = np.array(temp_arr)

    return d


def extrRandTrees(X, y, n_estimators = 15):                                                         # predefined args
    """
    Function that defines and train extremely randomized trees

    Args:
    -----
    - `X`: input of the model
    - `y`: ground truths
    - `n_estimators`: parameter of ExtraTreesRegressor from sklearn

    Returns:
    --------
    The trained model
    """

    est = ExtraTreesRegressor(n_estimators=n_estimators).fit(X, y)
    return est

def saveModel(est, name):
    pickle.dump(est, open(f'{name}.sav', 'wb'))



def fitted_Q(N, initialDataset, traj, actionSet, estFunction, gamma = 0.99, saveBool=0):                       # predefined args
    """
    Allows to apply fitted-Q iteration (second implementation)

    Args:
    -----
    - `N`: defines the Q_N that will be reached at the end
    - `initialDataset`: list of two np.array: X, y
        X contains the actions-states of the system (angle, speed_cart, angle_speed, action)
        y contains the associated rewards
    - `traj`: contains the set of all one-step transitions
    - `estFunction`: function that defines and train the model. Returns the model trained 
    - `gamma`: the gamma for the discounted reward (init =1)

    Returns:
    --------
    - the estimator trained on Q_N
    - the updated datasets 
    """

    X = initialDataset[0]                       # X remains the same

    d = createInput(actionSet, X, traj)         # create the dictionnary with the set of next states and the possible actions associated to each action


    for i in tqdm(range(N)):
        if i == 0:
            y = initialDataset[1]       # if first run, only keep the reward for the ouput
        else:
            
            maxQ = np.ones(X.shape[0])*float('-inf')        # init the maximal Q values at -infinity
            
            for action in actionSet:
                
                Q_current = est.predict(d[action])          # predict the Q values
                #maxAct = [action if Q_current[i] >= maxQ[i] else maxAct[i] for i, action in enumerate(maxAct)]
                maxQ = np.maximum(maxQ, Q_current)          # only take the maximal Q values
            
            
            ## update y
            y = initialDataset[1] + gamma*maxQ


        y = np.array(y).reshape(-1)
        est = estFunction(X, y)         # train the estimator
        
        if saveBool:
            saveModel(est, f"est_FQI_{i}")


    return est, X, y




def FQI_part(envName, nbIt = 60, nbTraj = 500):
    """
    Function to put everything together for FQI
    """
    
    np.random.seed(42)

    ## discretize the action set
    actionSetDisc = np.linspace(-3, 3, 300)

    ## Generate trajectory
    print('Generate trajectory')
    ag = randAgent()                                    # define agent to generate trajectories
    trajData = getTrajectoriesSet(envName, ag, nbTraj)  # get set of 1-step transitions

    print('2')
    print(trajData)

    X, y = getInitDataset(trajData)                     # create initial dataset for FQI

    plt.hist(y)                                         # check distribution fo rewards
    plt.show()  

    ## Choose model
    print('hoosing model')
    model_chosen = extrRandTrees

    ## Learn

    print('4')
    est, X, y = fitted_Q(nbIt, [X, y], trajData, actionSetDisc, model_chosen)

    return est

    

def FQI_loop(envName, nbLoop = 3, nbIt = 180, nbTraj = 1000):
    """
    Function to put everything together for FQI
    """
    
    np.random.seed(42)
    est = None
    XFull = None
    yFull = None

    ## discretize the action set
    actionSetDisc = np.linspace(-3, 3, 300)
    trajData = []

    for k in range(nbLoop):
        ## Generate trajectory 
        agRand = randAgent() 
        
        if k ==0:  
            ag = randAgent()                                    # define agent to generate trajectories
        else:
            agBase = estAgent(est, actionSetDisc)
            ag = nomalEpsilonAgent(agBase, alpha = 0.75)

        #trajData.extend(getTrajectoriesSet(envName, ag, nbTraj, display = 0))  # get set of 1-step transitions
        trajData.extend(getTrajectoriesSet(envName, ag, nbTraj, nbStepsPerTraj=10000))
        trajData.extend(getTrajectoriesSet(envName, agRand, 100, nbStepsPerTraj=10000))

        X, y = getInitDataset(trajData)                     # create initial dataset for FQI
        

        print(X.shape)

        ## Choose model
        model_chosen = extrRandTrees

        ## Learn

        if k == (nbLoop-1):
            est_test, X, y = fitted_Q(nbIt, [X, y], trajData, actionSetDisc, model_chosen, saveBool=1)
        else:
            est_test, X, y = fitted_Q(nbIt, [X, y], trajData, actionSetDisc, model_chosen)

        XFull = X
        yFull = y
        if est is None:
            est = est_test

            
        est = est.fit(XFull, yFull)
            
        saveModel(est, f"model_evol_{k}")
        
                
        np.save(f"XFull_{k}.npy", XFull)
        np.save(f'yFull_{k}', yFull)

    return est, XFull, yFull


def updateFQI(envName,alpha = 0.5, nbEpisode = 500, nbIt = 170, nbTraj = 500):
    """
    Function to put everything together for FQI
    """
    
    np.random.seed(42)

    actionSetDisc = np.linspace(-3, 3, 300)
    ag1 = randAgent()                                    # define agent to generate trajectories

    for k in tqdm(range(nbEpisode)):

        if k == 0:
            ag2 = randAgent()
        else:
            ag2 = estAgent(est, actionSetDisc)

        trajData = epsilonDataset(envName, ag1, ag2, alpha, nbTraj)

        X, y = getInitDataset(trajData)

        model_chosen = extrRandTrees

        est, X, y = fitted_Q(nbIt, [X, y], trajData, actionSetDisc, model_chosen)



