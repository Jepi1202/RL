import numpy as np
import torch

class randAgent():
    """
    Agent that chooses with a uniform random in [-3, 3] the action to follow
    """
    def __init__(self):
        self.obs = None
        self.cumReward = 0

    def choose_action(self, state):
        return [np.random.uniform(-3, 3)]
    

########
# may need debug (rien n'a été test en dessous)

class policyAgent():
    """
    Agent that uses a policyMat in order to decide which action should be taken
    """

    def __init__(self, policyMat):
        self.obs = None
        self.cumReward = 0  
        self.policyMat = policyMat  

    def choose_action(self, state):
        return [self.policyMat[state]]
    

class estAgent():
    """
    Agent that uses a trained estimator in order to decide which action should be taken
    """

    def __init__(self, est, actionSet, pytorchBool = 0):
        self.obs = None
        self.cumReward = 0  
        self.est = est  
        self.actionSet = actionSet
        self.pytorchBool = pytorchBool

    def choose_action(self, state):

        L = len(self.actionSet)
        S = len(state)                  # 3 if no position, 4 otherwise
        Tlength = S+1                   # also consider the action

        if self.pytorchBool:
            self.est.eval()             # eval + no_grad 

        with torch.no_grad():

            possTuples = np.zeros((L, Tlength))                     #

            for i in range(L):
                possTuples[i] = [*state, self.actionSet[i]]

            if self.pytorchBool:
                res = self.est(possTuples)
            else:
                res = self.est.predict(possTuples)

            ind = np.argmax(res)
            
            return [self.actionSet[ind]]
    

class stochaAgent():
    def __init__(self, model):
        self.obs = None
        self.cumReward = 0  
        self.model = model  
    

    def choose_action(self, state):
        self.model.estimator.eval()
        
        with torch.no_grad():
            action_predicted = self.model.get_mean_logstd_action(state)[-1]

        self.model.estimator.train()
        return [action_predicted]
    

class epsilonAgent():
    def __init__(self, agBase, alpha = 0.5):
        self.obs = None
        self.cumReward = 0  
        self.agentBase = agBase
        self.randAgent = randAgent()
        self.alpha = alpha

    def choose_action(self, state):
        if(np.random.uniform(0,1) <= self.alpha):
            return self.agentBase.choose_action(state)
        else:
            return self.randAgent.choose_action(state)

        
        
class nomalEpsilonAgent():
    def __init__(self, agBase, alpha = 0.5):
        self.obs = None
        self.cumReward = 0  
        self.agentBase = agBase
        self.alpha = alpha

    def choose_action(self, state):
        out = self.agentBase.choose_action(state)
        if(np.random.uniform(0,1) <= self.alpha):
            return out
        else:
            out = out[0] + 0.5 * np.random.normal()
            
            if out < -3:
                return [-3]
            if out > 3:
                return [3]
            else:
                return [out]



