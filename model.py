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
        self.fc2 = nn.Linear(state_space + action_space, fc2_units)
        self.fc2.to(device)
        self.lstm1 = nn.LSTM(fc1_units, fc2_units)
        self.fc3 = nn.Linear(2*fc2_units, fc2_units)
        self.fc3.to(device)
        self.fc4 = nn.Linear(fc2_units, action_space)
        self.fc4.to(device)
        self.fcn = out_fcn
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, last_action, hidden_in):
        state = state.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        activation = F.relu

        #1st branch
        fc_branch = activation(self.fc1(state))

        #2nd branch
        lstm_branch = torch.cat([state, last_action], -1)
        lstm_branch = activation(self.fc2(lstm_branch))

        lstm_branch, lstm_hidden = self.lstm1(lstm_branch, hidden_in)

        #merged
        merged_branch = torch.cat([fc_branch, lstm_branch], -1)
        x = activation(self.fc3(merged_branch))
        x = F.tanh(self.fc4(x))
        x = x.permute(1,0,2)

        return x, lstm_hidden


class Critic(nn.Module):
    def __init__(self, state_space, action_space, out_fcn=nn.Tanh(), fc1_units=35, fc2_units=35):
        '''
        :param state_space: The observation or state space of the environment
        :param action_space: The action space of the environment
        :param hidden_layers: The hidden layers to create the neural network
        '''
        super(Critic, self).__init__()
        
        self.activation = F.relu
        self.fc1 = nn.Linear(state_space + action_space, fc1_units)
        self.fc1.to(device)
        self.fc2 = nn.Linear(state_space + action_space, fc2_units)
        self.fc2.to(device)
        self.lstm1 = nn.LSTM(fc1_units, fc2_units)
        self.fc3 = nn.Linear(2*fc2_units, fc2_units)
        self.fc3.to(device)
        self.fc4 = nn.Linear(fc2_units, 1)
        self.fc4.to(device)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
        self.fc4.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc5.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc6.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action, last_action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for lstm needs to be permuted as: (sequence_length, batch_size, state_dim)
        """
        state = state.permute(1,0,2)
        action = action.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        # branch 1
        fc_branch = torch.cat([state, action], -1) 
        fc_branch = self.activation(self.fc1(fc_branch))
        # branch 2
        lstm_branch = torch.cat([state, last_action], -1) 
        lstm_branch = self.activation(self.fc2(lstm_branch))  # linear layer for 3d input only applied on the last dim
        lstm_branch, lstm_hidden = self.lstm1(lstm_branch, hidden_in)  # no activation after lstm
        # merged
        merged_branch=torch.cat([fc_branch, lstm_branch], -1) 

        x = self.activation(self.fc3(merged_branch))
        x = self.fc4(x)
        x = x.permute(1,0,2)  # back to same axes as input    
        return x, lstm_hidden 
    
    def Q1(self, state, action):
        
        state_action = torch.cat([state, action], 1)
        
        xs = F.relu(self.fc1(state_action))
        x1 = torch.relu(self.fc2(xs))
        q1 = self.fc3(x1)
        return q1



    
        