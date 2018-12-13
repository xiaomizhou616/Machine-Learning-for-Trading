"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.s = 0
        self.a = 0
        
        self.dyna = dyna
        self.radr = radr
        self.rar = rar
        self.gamma = gamma
        self.alpha = alpha
        
        self.Q = np.random.uniform(-1., 1., (num_states, num_actions)).astype('float32')
#        self.Q = np.zeros((self.num_states, self.num_actions), dtype = 'float32')
        self.T = dict()
        self.R = -1. * np.ones((self.num_states, self.num_actions), dtype = 'float32')
        
    def author(self):
        return 'xhan306'
        
    def choose_action(self, s):
        dice = rand.random()
        if dice < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s, :])
        
        return action   

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        action = self.choose_action(s)
        self.a = action
        
        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        # Update Q
#        best_a_prime = np.argmax(self.Q[s_prime,:])
        self.Q[self.s][self.a] = (1. - self.alpha) * self.Q[self.s][self.a] + self.alpha * (r + self.gamma * np.max(self.Q[s_prime,:]))
        
        # Dyna-Q
        if self.dyna != 0:
            R_prev_estimate = (1. - self.alpha) * self.R[self.s][self.a]
            R_impro_estimate = self.alpha * r
            self.R[self.s][self.a] = R_prev_estimate + R_impro_estimate
            if self.T.get((self.s, self.a)) is None:
                self.T[(self.s, self.a)] = [(s_prime)]
            else:
                self.T[(self.s, self.a)].append(s_prime)
                
            for ii in range(self.dyna):
                s_ = rand.randint(0, self.num_states-1)
                a_ = rand.randint(0, self.num_actions-1)
                if self.T.get((s_, a_)) is not None:
                    s_prime_ = rand.choice(self.T[(s_, a_)])
                else:
                    s_prime_ = rand.randint(0, self.num_states-1)
                    
#                best_a_prime = np.argmax(self.Q[s_prime_, :])
                self.Q[s_][a_] = (1. - self.alpha) * self.Q[s_][a_] + self.alpha * (self.R[s_][a_] + self.gamma * np.max(self.Q[s_prime_,:]))
            
        # Update attributes
        action = self.choose_action(s_prime)
        self.a = action
        self.s = s_prime
        self.rar *= self.radr
        
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return action
        
     

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
