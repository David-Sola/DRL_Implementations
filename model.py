# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 20:32:12 2021

@author: David Sola

Description:
    Genral class for creating a neural network for Actor critic DRL 
    with the PyTorch Framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_space, action_space, out_fcn=nn.Tanh(), fc1_units=35, fc2_units=35):
        '''
        :param state_space: The observation or state space of the environment
        :param action_space: The action space of the environment
        :param hidden_layers: The hidden layers to create the neural network
        '''
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_space, fc1_units)
        self.fc1.to(device)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc2.to(device)
        self.fc3 = nn.Linear(fc2_units, action_space)
        self.fc3.to(device)
        self.fcn = out_fcn
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fcn(self.fc3(x))

    def evaluate(self, x):
        action = self.forward(x)
        dist = Normal(0, 1)
        log_prob = dist.log_prob(action)
        real_log_prob = log_prob - torch.log(1-action.pow(2) + 1e-7)

        return real_log_prob



class Critic(nn.Module):
    def __init__(self, state_space, action_space, out_fcn=nn.Tanh(), fc1_units=35, fc2_units=35):
        '''
        :param state_space: The observation or state space of the environment
        :param action_space: The action space of the environment
        :param hidden_layers: The hidden layers to create the neural network
        '''
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.fc1 = nn.Linear(state_space + action_space, fc1_units)
        self.fc1.to(device)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc2.to(device)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.fc3.to(device)
        
        # Q2 architecture
        self.fc4 = nn.Linear(state_space + action_space, fc1_units)
        self.fc4.to(device)
        self.fc5 = nn.Linear(fc1_units, fc2_units)
        self.fc5.to(device)
        self.fc6 = nn.Linear(fc2_units, 1)
        self.fc6.to(device)
        self.fcn = out_fcn
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
        self.fc4.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc5.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc6.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        
        xs = torch.relu(self.fc1(state_action))
        x1 = torch.relu(self.fc2(xs))
        q1 = self.fc3(x1)
        
        xs = torch.relu(self.fc4(state_action))
        x2 = torch.relu(self.fc5(xs)) 
        q2 = self.fc6(x2)
        return q1, q2
    
    def Q1(self, state, action):
        
        state_action = torch.cat([state, action], 1)
        
        xs = torch.relu(self.fc1(state_action))
        x1 = torch.relu(self.fc2(xs))
        q1 = self.fc3(x1)
        return q1

class State_predictor(nn.Module):
    def __init__(self, input_space, output_space, fc1_units=512, fc2_units=256):
        super(State_predictor, self).__init__()
        self.fc1 = nn.Linear(input_space, fc1_units)
        self.fc1.to(device)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc2.to(device)
        self.fc3 = nn.Linear(fc2_units, output_space)
        self.fc3.to(device)
        #self.fcn = nn.Linear()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)





    
        