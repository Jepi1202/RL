import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gymnasium as gym
import copy
from tqdm import tqdm
from numpy import random
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
import torch.nn.functional as F
from torch.distributions.normal import Normal
import sys

def history_to_dataset(hist):
    y = np.array([hist[k][2] for k in range(len(hist))])
    x = np.array([np.hstack((hist[k][0],hist[k][1])) for k in range(len(hist))])
    z = np.array([hist[k][3] for k in range(len(hist))])
    return [x,y,z]

def create_batch(x,u_vect):
    n_act = u_vect.shape[0]
    x_change = np.repeat(x,n_act,0)
    u_change = np.tile(u_vect,x.shape[0]).reshape(-1,1)
    return np.hstack((x_change,u_change))

def copy_weights(target,source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class TrajBuffer:
    def __init__(self,st_dim):
        self.buffer = {}
        self.buffer['state'] = []
        self.buffer['action'] = []
        self.buffer['reward'] = []
        self.buffer['next_state'] = []
        self.buffer['vt'] = []
        self.st_dim = st_dim

    def push(self,*tuple):
        self.buffer['state'].append(torch.tensor(tuple[0]).reshape(1,self.st_dim))
        self.buffer['action'].append(torch.tensor(tuple[1]).reshape(1,-1))
        self.buffer['reward'].append(torch.tensor(tuple[2]).reshape(1,-1))
        self.buffer['next_state'].append(torch.tensor(tuple[3]).reshape(1,self.st_dim))

    def process_traj(self,gamma):
        r_tens = torch.tensor(self.buffer['reward']).reshape(-1)
        self.buffer['vt'] = [torch.sum(torch.tensor([gamma**(k-t) * r_tens[k] for k in range(t,r_tens.shape[0])])) for t in range(r_tens.shape[0])]
    
    def pop_data(self):
        if len(self.buffer['state']) == 0:
            return 0,(None,None,None,None,None)
        st = self.buffer['state'].pop(0)
        ac = self.buffer['action'].pop(0)
        rew = self.buffer['reward'].pop(0)
        nx_st = self.buffer['next_state'].pop(0)
        vt = self.buffer['vt'].pop(0)
        return 1,(st,ac,rew,nx_st,vt)

class ReinforceMLP(nn.Module):
    def __init__(self,in_layer,hid_layer,act):
        super(ReinforceMLP,self).__init__()
        self.l1 = nn.Linear(in_layer,hid_layer)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hid_layer,hid_layer)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(hid_layer,1)
        self.l3_ = nn.Linear(hid_layer,1)
        self.act = act

    def forward(self,x):
        h = self.l1(x)
        h = self.relu1(h)
        h = self.l2(h)
        h = self.relu2(h)
        mu = F.tanh(self.l3(h)) * self.act
        log_sig = self.l3_(h)
        return mu,log_sig
    
class Environment:
    def __init__(self,environment_name = "InvertedPendulum-v4", rend = None):
        self.env = gym.make(environment_name,render_mode = rend)
        self.env_name = environment_name
    def get_action_space(self):
        """
        return: action space - Force applied on the cart [-3,3]
        """
        return self.env.action_space
    
    def get_obs_space(self):
        """
        return: observation space - [0]: position x of the cart, [1]: vertical angle of the pole, [2]:linear velocity of the cart, [3]:angular velocity of the pole
        """
        return self.env.observation_space
    
    def run_random(self):
        obs = self.env.reset()
        print(f"The initial state is: {obs}")
        done = False
        while not done:
            random_action = self.env.action_space.sample()

            new_obs, reward, done, truncated, info = self.env.step(random_action)

            self.env.render()

    def close(self):
        return self.env.close()

    def run_Reinforce(self,n_steps,ag,gamma=1,obs_dim=4,act=3):
        J_tab = []
        ag.policy_est = ReinforceMLP(obs_dim,64,act=act)
        optimizer = torch.optim.Adam(ag.policy_est.parameters(),lr=1e-3)
        steps = 0
        n_traj = 0
        traj_buff = TrajBuffer(obs_dim)
        pbar = tqdm(total=n_steps)
        while steps < n_steps:
            n_traj += 1
            obs,_ = self.env.reset()
            done = False
            curr_step = 0
            while (not done) and curr_step < 2000:
                steps += 1
                pbar.update()
                if steps < 10:
                    action = ag.choose_action(obs,randomly=True)
                else:
                    action = ag.choose_action(obs)

                next_obs, reward, done, _, _ = self.env.step(action.reshape(-1))
                if done:
                    reward = 0
                traj_buff.push(obs,action,reward,next_obs)
                if not done:
                    obs = next_obs
                curr_step += 1

                if steps % 100000 == 0:
                    plot_img(J_tab,steps)

            traj_buff.process_traj(gamma)

            t = 0
            while True:
                cont,(st,ac,rew,nx_st,vt) = traj_buff.pop_data()
                if cont == 0:
                    break
                optimizer.zero_grad()
                loss = - gamma**t * vt * ag.log_prb(st,ac)
                loss.backward()
                optimizer.step()
                t += 1
            if steps > 0 and n_traj%10==0:
                env2 = Environment(environment_name=self.env_name)
                current_J = []
                for i in range(10):
                    current_J.append(env2.run_test(ag))
                print(np.mean(current_J))
                J_tab.append(current_J)

        return J_tab
    def process_traj(self,traj,ag):
        #traj: [Nsteps][Nfeat]
        s = torch.tensor([traj[k][0] for k in range(len(traj))]).reshape(-1,4)
        r = torch.tensor([traj[k][2] for k in range(len(traj))]).reshape(-1)
        a = torch.tensor([traj[k][1] for k in range(len(traj))]).reshape(-1)

        return s,r,a

    def process_r(self,r):
        vt = []
        for t in range(len(r)):
            vt.append(torch.sum(r[t:]))
        return torch.tensor(vt).reshape(-1,1)


    def run_test(self,ag):
        
        done = False

        obs,_ = self.env.reset()
        J = 0
        while (not done) and J < 1000:
            best_act = ag.choose_action(obs)
            obs,reward,done,_,_ = self.env.step(np.array([best_act]).reshape(-1,))
            J += 1
        return J

class Agent:
    def __init__(self,est = None,act=3,obs_dim=4):
        self.act_space = (-act,act)
        self.policy_est = est
        self.obs_dim = obs_dim

    def choose_action(self,state,training = False,randomly = False):
        with torch.no_grad():
            if randomly:
                return random.uniform(self.act_space[0],self.act_space[1],size=(1,))
            mu,log_sig = self.policy_est(torch.tensor(state,dtype=torch.float32).reshape(-1,self.obs_dim))
            action = torch.normal(mean=mu,std=torch.exp(log_sig)).numpy()
            return action
    
    def log_prb(self,state,action):
        mu,log_sig = self.policy_est(torch.tensor(state,dtype=torch.float32).reshape(1,-1))
        sig = torch.exp(log_sig)
        nor = Normal(mu,sig)
        a = nor.log_prob(action)
        return a

def plot_img(J_tab,s):
    J_tab = np.array(J_tab)
    means = np.mean(J_tab,axis=1)
    stds = np.std(J_tab,axis=1)    
    x_ax = np.linspace(100,s,means.shape[0])
    fig_mean = plt.figure()
    plt.plot(x_ax,means)
    plt.fill_between(x_ax,(means-stds),(means+stds),alpha=0.3)
    plt.grid()
    plt.title("Evolution of the expected return")
    plt.xlabel("number of transitions")
    plt.ylabel("expected return")
    plt.show()

if __name__ == "__main__":
    
    if sys.argv[1] == "simple":
        env = Environment(environment_name = "InvertedPendulum-v4")
        obs_dim = 4
        act_max = 3
    if sys.argv[1] == "double":
        env = Environment(environment_name = "InvertedDoublePendulum-v4")
        obs_dim = 11
        act_max = 1
    
    if len(sys.argv) > 2:
        if sys.argv[1] == "simple":
            model = ReinforceMLP(4,64,3)
            model.load_state_dict(torch.load("simple_reinforce.pt"))
            ag = Agent(act=3,obs_dim=4)
            ag.policy_est = model
            env = Environment(environment_name="InvertedPendulum-v4",rend="human")
            for i in range(10):
                env.run_test(ag)

            exit()
        if sys.argv[1] == "double":
            model = ReinforceMLP(11,64,3)
            model.load_state_dict(torch.load("reinforce_double.pt"))
            ag = Agent(act=1,obs_dim=11)
            ag.policy_est = model
            env = Environment(environment_name="InvertedDoublePendulum-v4",rend="human")
            for i in range(10):
                env.run_test(ag)

            exit()

    ag = Agent(act=act_max,obs_dim=obs_dim)
    J_tab = env.run_Reinforce(100000,ag,obs_dim=obs_dim,act=act_max)
    J_tab = np.array(J_tab)
    print('starting to save')
    torch.save(ag.policy_est.state_dict(),"reinforce_simple_good.pt")
    print('model should be saved')
    means = np.mean(J_tab,axis=1)
    stds = np.std(J_tab,axis=1)

    x_ax = np.linspace(100,100000,means.shape[0])


    fig_mean = plt.figure()
    plt.plot(x_ax,means)
    plt.fill_between(x_ax,(means-stds),(means+stds),alpha=0.3)
    plt.grid()
    plt.title("Evolution of the expected return")
    plt.xlabel("number of transitions")
    plt.ylabel("expected return")
    plt.show()

    exit()