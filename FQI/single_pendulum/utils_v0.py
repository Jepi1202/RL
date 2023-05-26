import gymnasium as gym
import sklearn
import numpy as np

from tqdm import tqdm
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt



def simulate(envName, agent,  nb = 200, threeObse = 1):                                # predefined args
    """
    Allows to simulate a trajectory of size nb

    Args:
    -----
    - `envName`: name of the gym environment
    - `agent`: agent used to generate trajectory (see agents.py)
    - `nb`: length of the trajectory

    Output:
    -------
    - a list with the different steps
    A step = ((observables), action, reward, (next_observables))
    """

    #env = gym.make(envName, render_mode = "human")
    env = gym.make(envName)                                     # creating gym environment
    obs = env.reset()                                           # reset the environment for the start
    obs = obs[0]                                                # first observables
    traj = []                                                   # initialization of the list

    for c in range(nb):
        if threeObse:
            act = agent.choose_action(obs)
        else:
            act = agent.choose_action(obs)

        nextObs, reward, done, info, _ = env.step(act)          # step in the environment

        ## force the reward to 0 when a terminal state is reached
        if done:
            reward = 0
            traj.append((obs, act[0], reward, nextObs))
            traj.append((obs, act[0], reward, nextObs))
            traj.append((obs, act[0], reward, nextObs))
            return traj

        
        traj.append((obs, act[0], reward, nextObs))
        obs = nextObs

    return traj



def getTrajectoriesSet(envName, agent, nbTraj = 1000, nbStepsPerTraj = 75, display = 1):              # predefined args
    """
    Allows to simulate nbTraj trajectory of size nbStepsPerTraj

    Args:
    -----
    - `envName`: name of the gym environment
    - `agent`: agent used to generate trajectory (see agents.py)
    - `nbTraj`: number of trajectories
    - `nbStepsPerTraj`: length of the trajectories

    Output:
    -------
    - a list with the different one step transitions of all trajectoriess
    A step = ((observables), action, reward, (next_observables))
    """

    S = []
    for c in tqdm(range(nbTraj)):
        traj = simulate(envName, agent,  nb = nbStepsPerTraj)
        S.extend(traj)

        if display:
            if c == 0:
                fig, axes = plotTraj(traj, display = 0)
            else:
                fig, axes = plotTraj(traj, fig, axes, display = 0)

    if display:
        plt.show()
    return S



"""
First implementation of the function to get the policy

def showPolicy(speedCart, angle, angleSpeed, est):
    policyMat = np.zeros((len(angle), len(speedCart), len(angleSpeed)))

    for i in tqdm(range(len(angle))):
        for j in range(len(speedCart)):
            for k in range(len(angleSpeed)):
                if est.predict(np.array([angle[i], speedCart[j], angleSpeed[k], 3]).reshape(1, -1)) > est.predict(np.array([angle[i], speedCart[j], angleSpeed[k], -3]).reshape(1, -1)):
                    policyMat[i, j] = 3  
                else:
                    policyMat[i, j] = -3

    return policyMat
"""


def getPolicyMat(angle, speedCart, angleSpeed, actionSet, est):
    """
    Allows to get the policy matrix 

    Args:
    -----
    - `speedCart`: vector of the discretized speeds of the cart
    - `angle`: vector of the discretized angle of the pendulum
    - `angleSpeed`: vector of the discretized angular velocities of the pendulum
    - `actionSet`: disretization of the continuous action set
    - `est`: estimator to predict the values 

    Output:
    -------
    - the policy matrix
    - the matrix with the maximal Q_N functions
    """
    
    ## init of state mat
    stateMat = np.zeros((len(angle), len(speedCart), len(angleSpeed), 4))
    for i in tqdm(range(len(angle))):
        for j in range(len(speedCart)):
            for k in range(len(angleSpeed)):
                stateMat[i, j, k, :] = [angle[i], speedCart[j], angleSpeed[k], 0]

    ## init of the input in the estimator
    s = np.copy(stateMat)
    s = s.reshape(-1, 4)

    ## init of policyMat

    policyMat = np.zeros((len(angle), len(speedCart), len(angleSpeed)))

    ## Q mat initialization
    Qmat = np.ones((len(angle), len(speedCart), len(angleSpeed))) * float('-inf')

    for action in tqdm(actionSet):
        s[..., -1] = action
        Qmat_cur = est.predict(s)
        Qmat_cur = Qmat_cur.reshape(len(angle), len(speedCart), len(angleSpeed))

        max_arr = np.maximum(Qmat, Qmat_cur)
        max_indices = np.where(max_arr == Qmat, 1, 2)

        greater_than = (max_indices == 2)

        idx = np.nonzero(greater_than)

        policyMat[idx[0], idx[1], idx[2]] = action
        Qmat = max_arr

    # nb: could also use this technique to efficiently compute the Qmat of a specific action
    return policyMat, Qmat



def getPolicyMat_full(posCart, angle, speedCart, angleSpeed, actionSet, est):
    """
    Allows to get the policy matrix 

    Args:
    -----
    - `posCart`: vector of the discretized positions of the cart
    - `speedCart`: vector of the discretized speeds of the cart
    - `angle`: vector of the discretized angle of the pendulum
    - `angleSpeed`: vector of the discretized angular velocities of the pendulum
    - `actionSet`: disretization of the continuous action set
    - `est`: estimator to predict the values 

    Output:
    -------
    - the policy matrix
    - the matrix with the maximal Q_N functions
    """
    
    ## init of state mat
    stateMat = np.zeros((len(posCart), len(angle), len(speedCart), len(angleSpeed), 5))
    for i in tqdm(range(len(posCart))):
        for j in range(len(angle)):
            for k in range(len(speedCart)):
                for l in range(len(angleSpeed)):
                    stateMat[i, j, k, l, :] = [posCart[i], angle[j], speedCart[k], angleSpeed[l], 0]

    ## init of the input in the estimator
    s = np.copy(stateMat)
    s = s.reshape(-1, 5)

    ## init of policyMat

    policyMat = np.zeros((len(angle), len(speedCart), len(angleSpeed)))

    ## Q mat initialization
    Qmat = np.ones((len(posCart), len(angle), len(speedCart), len(angleSpeed))) * float('-inf')

    for action in tqdm(actionSet):
        s[..., -1] = action
        Qmat_cur = est.predict(s)
        Qmat_cur = Qmat_cur.reshape(len(posCart), len(angle), len(speedCart), len(angleSpeed))

        max_arr = np.maximum(Qmat, Qmat_cur)
        max_indices = np.where(max_arr == Qmat, 1, 2)

        greater_than = (max_indices == 2)

        idx = np.nonzero(greater_than)

        policyMat[idx[0], idx[1], idx[2]] = action
        Qmat = max_arr

    # nb: could also use this technique to efficiently compute the Qmat of a specific action
    return policyMat, Qmat

##################""
# ro debug

def plot_3Dmatrix(matrix, minVal=3, maxVal=3, alpha = 0.4, show = 1):
    """
    Allows to plot a 3D matrix (easier with plotly btw)

    Args:
    -----
    - `matrix`: matrix to plot
    - `minVal`: minimal value of the normalization
    - `maxVal`: maximal value of the normalization
    - `alpha`: alpha of the plot
    - `show`: boolean to say if the plot should be shown
    
    Output:
    -------
    - fig and ax of plt
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    norm = plt.Normalize(minVal, maxVal)
    colors = plt.cm.viridis(norm(matrix))

    ax.voxels(matrix, facecolors=colors, alpha = alpha)
    if show:
        plt.show()

    return fig, ax


def plot_inter_2D(matrix, x, y, display = 1):
        
    fig = plt.figure()
    ax = plt.axes()

    ax.pcolormesh(x, y, matrix.T, cmap='RdBu')

    if display:
        plt.show()

    return fig, ax


def plotTraj(traj, fig=None, axes=None, display=1):

    if (fig is None) or (axes is None):
        fig, axes = plt.subplots(5, figsize=(10, 10))
        fig.tight_layout(pad=4.0)

    inds = np.arange(len(traj))
    for j in range(len(traj[0][0])):

        elems = np.array([traj[i][0][j] for i in range(len(traj))])
        axes[j].scatter(inds, elems, marker='o', color='blue')
        axes[j].plot(inds, elems, 'b--')

        axes[j].grid()

        axes[j].set_xlabel("step")
        axes[j].set_ylabel("Value")  
        #if j == 0:
        #    print(f'{j} ==> [{min(elems)}, {max(elems)}]')

    elems = np.array([traj[i][2] for i in range(len(traj))]) 
    axes[4].scatter(inds, elems, marker='o', color='red')
    axes[4].plot(inds, elems, 'r--')   
    axes[4].grid()

    axes[4].set_xlabel("step")
    axes[4].set_ylabel("Reward") 

    if display:
        plt.show()

    return fig, axes



def estExpResults(envName, ag, gamma, nb = 50):
    """
    Compute the expected return of a policy embodied in the agent

    Args:
    -----
    - `envName`: name of the environment
    - `ag`: agent
    - `gamma`: gamma of the model
    - `nb`: number of times the experience is repeated

    Output:
    -------
    the expected cummulative return and the corresponding standard deviation
    """

    rewardList = []

    print("Estimating expected result")
    for _ in tqdm(range(nb)):
        env = gym.make(envName)
        obs = env.reset()
        obs = obs[0]          

        terminatedBool = 0
        reward_c = 0
        t = 0
        while terminatedBool == 0:
            act = ag.choose_action(obs)
            nextObs, reward, done, info, _ = env.step(act)

            if done:
                reward = 0
                terminatedBool = 1


            reward_c += (gamma**t) * reward
            obs = nextObs
            t+=1

        rewardList.append(reward_c)

    return np.mean(rewardList), np.std(rewardList)




def show_agent(envName, ag):
    env = gym.make(envName,render_mode = "human")
    
    obs = env.reset()                                           # reset the environment for the start
    obs = obs[0]                                                # first observables
    endBool = False

    while not endBool:
        action = ag.choose_action(obs)
        obs, reward, endBool, info,e = env.step(action)
        print(obs)
        obs = obs

        env.render()
    return env



def epsilonSimulate(envName, ag1, ag2, alpha,  nb = 200):                                # predefined args
    """
    Allows to simulate a trajectory of size nb

    Args:
    -----
    - `envName`: name of the gym environment
    - `agent`: agent used to generate trajectory (see agents.py)
    - `nb`: length of the trajectory

    Output:
    -------
    - a list with the different steps
    A step = ((observables), action, reward, (next_observables))
    """

    #env = gym.make(envName, render_mode = "human")
    env = gym.make(envName)                                     # creating gym environment
    obs = env.reset()                                           # reset the environment for the start
    obs = obs[0]                                                # first observables
    traj = []                                                   # initialization of the list

    for c in range(nb):

        if np.random.uniform(0,1) < alpha:
            act = ag1.choose_action(obs)
        else:
            act = ag2.choose_action(obs)

        nextObs, reward, done, info, _ = env.step(act)          # step in the environment

        ## force the reward to 0 when a terminal state is reached
        if done:
            reward = 0
            traj.append((obs, act[0], reward, nextObs))
            traj.append((obs, act[0], reward, nextObs))
            traj.append((obs, act[0], reward, nextObs))
            return traj

        
        traj.append((obs, act[0], reward, nextObs))
        obs = nextObs

    return traj



def epsilonDataset(envName, ag1, ag2, alpha, nbTraj = 1000, nbStepsPerTraj = 75):              # predefined args
    """
    Allows to simulate nbTraj trajectory of size nbStepsPerTraj

    Args:
    -----
    - `envName`: name of the gym environment
    - `agent`: agent used to generate trajectory (see agents.py)
    - `nbTraj`: number of trajectories
    - `nbStepsPerTraj`: length of the trajectories

    Output:
    -------
    - a list with the different one step transitions of all trajectoriess
    A step = ((observables), action, reward, (next_observables))
    """

    S = []
    for c in range(nbTraj):
        traj = simulate(envName, ag1, ag2, alpha,  nb = nbStepsPerTraj)
        S.extend(traj)

    return S
