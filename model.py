# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 20:32:12 2021

@author: David Sola

Description:
    Genral class for creating a neural network for Actor critic DRL 
    with the PyTorch Framework
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import torch.optim as optim

init_alpha      = 0.1
lr_alpha        = 0.0001



def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, lr, state_space, action_space, max_action=1, out_fcn=nn.Tanh(), fc1_units=512, fc2_units=256, name='actor', chkpt_dir='tmp/sac'):
        '''
        :param state_space: The observation or state space of the environment
        :param action_space: The action space of the environment
        :param hidden_layers: The hidden layers to create the neural network
        '''
        super(Actor, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.name = name
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_SAC')
        self.reparam_noise = 1e-6
        self.max_action = max_action

        self.fc1 = nn.Linear(state_space, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc_mu = nn.Linear(fc2_units,action_space)
        self.fc_sigma  = nn.Linear(fc2_units,action_space)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        prob = F.relu(self.fc1(state))
        prob = F.relu(self.fc2(prob))
        
        mu = self.fc_mu(prob)
        sigma = self.fc_sigma(prob)

        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)
        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = torch.tanh(actions)#*torch.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum()

        return action, log_probs

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))




class Critic(nn.Module):
    def __init__(self, state_space, action_space, name='Critic',  out_fcn=nn.Tanh(), fc1_units=512, fc2_units=256, lr=0.0003, chkpt_dir='tmp/sac'):
        '''
        :param state_space: The observation or state space of the environment
        :param action_space: The action space of the environment
        :param hidden_layers: The hidden layers to create the neural network
        '''
        super(Critic, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.name = name
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_SAC')
        
        # Q1 architecture
        self.fc1 = nn.Linear(state_space + action_space, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        
        xs = F.relu(self.fc1(state_action))
        x1 = F.relu(self.fc2(xs))
        q = self.fc3(x1)

        return q
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class ValueNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_units=512, fc2_units=256, name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.name = name
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_SAC')

        self.fc1 = nn.Linear(input_dims, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.v = nn.Linear(fc2_units, 1)

        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device) 

    def forward(self, state):
        state_value = F.relu(self.fc1(state))
        state_value = F.relu(self.fc2(state_value))

        v = self.v(state_value)

        return v   

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))  






    
        