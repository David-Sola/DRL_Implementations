# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 22:25:21 2020

@author: sld8fe

Description:
    Training script for the Bipedal walker environment of OpenAI Gym
"""

import gym
import numpy as np
from TD3_Agent import Agent
import matplotlib.pyplot as plt
from model import State_predictor
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Change betweend Hardcore or non Hardcore version
hc = 1

if hc==1:
    env = gym.make('BipedalWalkerHardcore-v3')
else:
    env = gym.make('BipedalWalker-v3')


# Helper variables

# Variable to display accumulated reward for each episode
accumulated_reward = 0

# Maximum number of epsiodes
max_episodes = 10000

# Maximum number of timesteps per Episode
episode_range = 1000

# How many episodes random actions shall be taken
n_rand_actions = 100

# Best reward from all episodes
best_reward = -999

# Empty lists for x and y values to be able to plot the accumulated reward after
# the episodes
x = []
y = []

# Set seeds for reproducible results
seed = 0
env.seed(seed)

# Sigma value for exploration
sigma = 0.4
sigma_decay = 0.95
increase_sigma_counter = 0

# Counter for number of interactions withe nvironemtn
total_int = 0
accumulated_loss = 0
nr_of_its = 1

update_predictor = 1

# Creation of the agent which shall be trained
agent = Agent(24, 4, random_seed=2)
agent.load_network(own_path=0, act_loc='HC_best_checkpoint_actor_loc_mem.pth', act_tar='HC_best_checkpoint_actor_tar_mem.pth', cr_loc='HC_best_checkpoint_critic_loc_mem.pth', cr_tar='HC_best_checkpoint_critic_tar_mem.pth')
   

''' START OF THE WHOLE TRAINING LOOP '''
for i_episode in range(max_episodes):
    

    state = env.reset()
    accumulated_reward = 0
    
    
    ''' START OF THE TEST LOOP FOR EACH EPISODE '''
    for t in range(episode_range):
        env.render()
        action = agent.act(state)    
        
        # Get the feedback from the environment by performing the action
        next_state, reward, done, info = env.step(action)
        accumulated_reward += reward

        # Assign the next state to the current state
        state = next_state
        
        # If the episode is done finish the loop
        if done:
            break
            
        
    print('Accumulated reward was: ', accumulated_reward)
    if accumulated_reward > best_reward:
        best_reward = accumulated_reward
        
        if best_reward > 50:
            episode_range = 1000
        
        
    ''' START AN EVALUATION OF THE CURRENT POLICY AFTER 100 EPISODES 
        FOR 10 EPISODES '''
    if i_episode%100==0 and i_episode != 0:
        average_rew = 0
        nr_eval_episodes = 10
        for eval_episodes in range(nr_eval_episodes):
            state, done = env.reset(), False
            while not done:
                action = agent.act(state) 
                state, reward, done, info = env.step(action)
                average_rew += reward
        average_rew /= nr_eval_episodes
        
        if average_rew>250:
            agent.save_network() 
            plt.pause(0.1)
            plt.plot(x,y)
            plt.title('Reward')
            plt.pause(0.1)
            plt.savefig('Reward.png')
        else:
            plt.close()
        print("---------------------------------------")
        print('Evaluation over ', nr_eval_episodes, ' episodes. Average reward: ', average_rew, ' Hardcore activated: ', hc, ' Total Interactions: ', total_int)
        print("---------------------------------------")
        

env.close()