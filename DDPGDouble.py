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

class Q_network(nn.Module):
    
    def __init__(self,in_dim,hid_dim):
        super(Q_network,self).__init__()
        self.l1 = nn.Linear(in_dim,hid_dim)
        self.l2 = nn.Linear(hid_dim,hid_dim)
        self.l3 = nn.Linear(hid_dim,1)
    
    def forward(self,x,a):
        z = torch.cat((x,a),1)
        h = F.relu(self.l1(z))
        #h = F.relu(self.l2(h))
        return F.relu(self.l3(h))

class P_network(nn.Module):
    
    def __init__(self,in_dim,hid_dim,act_max = 3):
        super(P_network,self).__init__()
        self.l1 = nn.Linear(in_dim,hid_dim)
        self.l2 = nn.Linear(hid_dim,hid_dim)
        self.l3_mu = nn.Linear(hid_dim,1)
        self.act_max = act_max
    def forward(self,x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        mu = F.tanh(self.l3_mu(h))*self.act_max
        return mu

class ReplayBuffer:
    
    def __init__(self,size,st_dim):
        self.size=size
        self.buffer = {}
        self.buffer['state'] = []
        self.buffer['action'] = []
        self.buffer['reward'] = []
        self.buffer['next_state'] = []
        self.buffer['dones'] = []
        self.st_dim = st_dim
    
    def push(self,*tuple):
        self.buffer['state'].append(tuple[0])
        self.buffer['action'].append(tuple[1])
        self.buffer['reward'].append(tuple[2])
        self.buffer['next_state'].append(tuple[3])
        self.buffer['dones'].append(tuple[4])

        while len(self.buffer['state']) > self.size:
            self.buffer['state'].pop(0)
            self.buffer['action'].pop(0)
            self.buffer['reward'].pop(0)
            self.buffer['next_state'].pop(0)
            self.buffer['dones'].pop(0)

    def get_batch(self,batch_size):
        maxi = len(self.buffer['state'])
        samples = random.randint(0,maxi,size=min(maxi,batch_size))
        st = torch.tensor(np.array(self.buffer['state'])[samples],dtype=torch.float32).reshape(-1,self.st_dim)
        #print(self.buffer['action'])
        ac = torch.tensor(np.array(self.buffer['action'])[samples],dtype=torch.float32).reshape(-1,1)
        rew = torch.tensor(np.array(self.buffer['reward'])[samples],dtype=torch.float32).reshape(-1,1)
        nx_st = torch.tensor(np.array(self.buffer['next_state'])[samples],dtype=torch.float32).reshape(-1,self.st_dim)
        ds = torch.tensor(np.array(self.buffer['dones'])[samples],dtype=torch.float32).reshape(-1,1)
        return st,ac,rew,nx_st,ds

class Environment:

    def __init__(self,environment_name = "InvertedPendulum-v4", rend = None, act_max = 3):
        self.env = gym.make(environment_name,render_mode = rend)
        self.act_max = act_max
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

    def close(self):
        return self.env.close()
    
    def run_DDPG(self,ag,obs_dim=4,gamma=0.99,n_steps=5000):
        
        J_tab = []
        ag.policy_est = P_network(obs_dim,64,self.act_max)
        target_p = P_network(obs_dim,64,self.act_max)
        q_net = Q_network((obs_dim+1),64)
        target_q_net = Q_network((obs_dim+1),64)
        copy_weights(target_p,ag.policy_est)
        copy_weights(target_q_net,q_net)
        
        opti_pol = torch.optim.Adam(ag.policy_est.parameters(),lr=1e-3)
        opti_q = torch.optim.Adam(q_net.parameters(),lr=1e-3)

        mse_loss = nn.MSELoss()
        rb = ReplayBuffer(10000,obs_dim)
        step = 0
        pbar = tqdm(total = n_steps)
        up_step = 1
        while step < n_steps:
            state,_ = self.env.reset()
            done = False
            pbar.update(up_step)
            up_step=0
            while not done:
                up_step += 1
                step += 1
                if step > n_steps:
                    break
                if step < 1000:
                    action = ag.choose_action(state,randomly=True)
                    next_state, reward, done, _, _ = self.env.step(action.reshape(-1))
                    if done:
                        reward = 0
                    rb.push(state,action,reward,next_state,done)
                    if not done:
                        state = next_state

                else:
                    action = ag.choose_action(state,training=True)
                    next_state, reward, done, _, _ = self.env.step(action.reshape(-1))
                    if done:
                        reward = 0
                    rb.push(state,action.reshape(-1),reward,next_state,done)
                    if not done:
                        state = next_state

                if step >= 100 and step%10 == 0:
                    for iterat in range(100):
                        states, actions, rewards, next_states,dones = rb.get_batch(64)
                        target_Q = target_q_net(next_states,target_p(next_states))

                        target_Q = rewards + (gamma * (1-dones) * target_Q).detach()
                        
                        current_Q = q_net(states,actions)
                        
                        critic_loss = F.mse_loss(current_Q,target_Q)
                        
                        opti_q.zero_grad()
                        critic_loss.backward()
                        opti_q.step()

                        actor_loss = -q_net(states,ag.policy_est(states)).mean()
                        
                        opti_pol.zero_grad()
                        actor_loss.backward()
                        opti_pol.step()

                        for target_param, param in zip(target_q_net.parameters(), q_net.parameters()):
                            target_param.data.copy_(param.data * (1.0 - 0.995) + target_param.data * 0.995)
                        
                        for target_param, param in zip(target_p.parameters(), ag.policy_est.parameters()):
                            target_param.data.copy_(param.data * (1.0 - 0.995) + target_param.data * 0.995)
                    
                if step%1000 == 0:
                    current_J = []
                    env2 = Environment(environment_name=self.env_name)
                    for i in tqdm(range(10)):
                        current_J.append(env2.run_testDD(ag))
                    print(np.mean(current_J))
                    J_tab.append(current_J)
                    #env2.close()
        
        return J_tab

    def run_testDD(self,ag):
        
        done = False

        obs,_ = self.env.reset()
        J = 0
        while (not done) and J < 1000:
            best_act = ag.choose_action(obs)
            obs,reward,done,_,_ = self.env.step(np.array([best_act]).reshape(-1,))
            J += reward
        #self.env.render() 
        return J

class Agent:
    def __init__(self,est = None,sig = 1,act=3):
        self.act_space = (-3,3)
        self.discreteAction = np.linspace(-3,3,120)
        self.policy_est = est
        self.sig = sig
        self.act = act

    def choose_action(self,state,training = False,randomly = False):
        with torch.no_grad():
            if randomly:
                return random.uniform(self.act_space[0],self.act_space[1],size=(1,))
            norm_res = torch.normal(torch.tensor(0,dtype=torch.float32),torch.tensor(self.sig,dtype=torch.float32))
            a = torch.clamp(self.policy_est(torch.tensor(state,dtype=torch.float32).reshape(1,-1)) + training*norm_res,-self.act,self.act).numpy()
            return a

if __name__ == "__main__":
    
    if sys.argv[1] == "simple":
        env = Environment(environment_name = "InvertedPendulum-v4",act_max=3)
        obs_dim = 4
        act_max = 3
    if sys.argv[1] == "double":
        env = Environment(environment_name = "InvertedDoublePendulum-v4",act_max=1)
        obs_dim = 11
        act_max = 1

    if len(sys.argv) > 2:
        if sys.argv[1] == "simple":
            env = Environment(environment_name = "InvertedPendulum-v4",rend="human",act_max=3)
            ag = Agent()
            model = P_network(4,64,3)
            model.load_state_dict(torch.load("ddpg_simple.pt")) 
            ag.policy_est = model
            for i in range(10):
                env.run_testDD(ag)

            exit() 

        if sys.argv[1] == "double":
            env = Environment(environment_name = "InvertedDoublePendulum-v4",rend="human",act_max=1)
            ag = Agent()
            model = P_network(11,64,1)
            model.load_state_dict(torch.load("ddpg_double.pt")) 
            ag.policy_est = model
            for i in range(10):
                env.run_testDD(ag)
            exit() 

    ag = Agent(sig=act_max/3)
    J_tab = env.run_DDPG(ag,obs_dim=obs_dim,n_steps=10000)
    J_tab = np.array(J_tab)
    print('starting to save')
    torch.save(ag.policy_est.state_dict(),"ddpg_double.pt")
    print('model should be saved')
    means = np.mean(J_tab,axis=1)
    stds = np.std(J_tab,axis=1)

    x_ax = np.arange(1000,10001,1000)


    fig_mean = plt.figure()
    plt.plot(x_ax,means)
    plt.fill_between(x_ax,(means-stds),(means+stds),alpha=0.3)
    plt.grid()
    plt.title("Evolution of the expected return")
    plt.xlabel("number of transitions")
    plt.ylabel("expected return")
    plt.show()

    exit()