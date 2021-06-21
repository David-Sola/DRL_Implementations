# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 22:25:21 2020

@author: sld8fe

Description:
    Training script for the Bipedal walker environment of OpenAI Gym
"""

import gym
import numpy as np
from SAC_Agent import Agent
from TD3_Agent import Agent as TD3_Agent
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Change betweend Hardcore or non Hardcore version
hc = 0
imitation_learnen = 0

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
episode_range = 500

# How many episodes random actions shall be taken
n_rand_actions = 100

# Best reward from all episodes
best_reward = -999

# Empty lists for x and y values to be able to plot the accumulated reward after
# the episodes
x = []
y = []
last_100_average = deque(100*[0], 100)
add_to_100_average = 0
plot_count = 1

# Set seeds for reproducible results
seed = 0
env.seed(seed)

# Sigma value for exploration
sigma = 0.01
sigma_decay = 0.95
increase_sigma_counter = 0
imitation_counter = 1

# Counter for number of interactions withe nvironemtn
total_int = 0
accumulated_loss = 0
nr_of_its = 1
tau = 2

update_predictor = 1


# Creation of the agent which shall be trained
agent = Agent(24, 4, random_seed=2)
agent_to_imitate_from = TD3_Agent(24, 4, random_seed=2)
agent_to_imitate_from.load_network(own_path=0, act_loc='HC_best_checkpoint_actor_loc_mem.pth', act_tar='HC_best_checkpoint_actor_tar_mem.pth', cr_loc='HC_best_checkpoint_critic_loc_mem.pth', cr_tar='HC_best_checkpoint_critic_tar_mem.pth')

optimizer_agent = optim.Adam(agent.actor.parameters(), lr=0.005)

def imitation_learnen(agent, agent_to_imitate_from, optimizer_agent):
    loss_fcn = nn.MSELoss()
    if agent.memory_imitation.mem_cntr > 100:
        states, actions,_, _,_ = \
                agent.memory_imitation.sample_buffer(64)

        states = torch.tensor(states, dtype=torch.float).to(agent.actor.device)
        actions = torch.tensor(actions, dtype=torch.float).to(agent.actor.device)

        actions_policy, _ = agent.actor.sample_normal(states, reparameterize=True)
        loss = loss_fcn(actions_policy, actions)
        optimizer_agent.zero_grad()
        loss.backward()
        optimizer_agent.step()
        return loss.cpu().detach().numpy()

''' START OF THE WHOLE TRAINING LOOP '''
for i_episode in range(max_episodes):
    '''
    dist_reward_accumulated = 0
    sigma = max(sigma*sigma_decay, 0.01)

    if (accumulated_loss/nr_of_its) < 0.05:
        increase_sigma_counter += 1
    
    if increase_sigma_counter >= 10:
        sigma = sigma + 0.3/(i_episode/100)
        sigma = min(sigma, 1)
        increase_sigma_counter = 0
    '''
    if add_to_100_average == 1:
        last_100_average.appendleft(accumulated_reward)
        #print(last_100_average)
        x.append(plot_count)
        y.append(sum(last_100_average)/100)
        plot_count+=1

    accumulated_reward = 0

    # Get the first state from the environment
    state = env.reset()

    # Print some usefull stuff
    print("100 Episode average: ", sum(last_100_average)/100)
    print("Best reward: ", best_reward)
    print("Episode: ", i_episode)
    print("Tau: ", tau)
    print("loss: ", accumulated_loss/nr_of_its)
    tau = max(tau*0.99, 0.05)
    nr_of_its = 1
    accumulated_loss = 0

    
    
    ''' START OF THE TRAINING LOOP FOR EACH EPISODE '''
    for t in range(episode_range):
        
        #First 20 Episodes fill up the imitation learning memory
        if i_episode < 20 and imitation_learnen==1:
            imitation_learning=1
            action_to_imitate = agent_to_imitate_from.act(state)
            # Get the feedback from the environment by performing the action
            next_state, reward, done, info = env.step(action_to_imitate)
            #accumulated_reward += reward
            agent.add_memory_imitation(state, action_to_imitate, reward, next_state, done)
            imitation_learnen(agent, agent_to_imitate_from, optimizer_agent)
            state = next_state
            accumulated_reward = accumulated_reward + reward

            if done:
                break


        else:
            imitation_learning=0
            nr_of_its += 1
            total_int += 1
            add_to_100_average = 1
            # Take an action with the agent with added noise in the current state
            # For the first n episodes take random actions
            if i_episode < n_rand_actions:
                action = env.action_space.sample()
            else:
                action = agent.act(state) 

            # Get the feedback from the environment by performing the action
            next_state, reward, done, info = env.step(action)
            accumulated_reward = accumulated_reward + reward
            agent.add_memory(state, action, reward, next_state, done)

            # If the episode is done finish the loop
            if done:
                break
   
            # Take a step with the agent in order to learn but collect first 
            # sufficient amount of data
            if i_episode > n_rand_actions:
                #imitation_learnen(agent, agent_to_imitate_from, optimizer_agent)
                agent.step()
            

            state = next_state

    if imitation_learning:
        print("Accumulated_reward during imitation learning: ", accumulated_reward)
        accumulated_reward = 0
    else: 
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
        
        #if average_rew>250:
        #agent.save_network() 
        plt.pause(0.1)
        plt.plot(x,y)
        plt.title('Reward')
        plt.pause(0.1)
        plt.savefig('Reward.png')
        #else:
            #plt.close()
        print("---------------------------------------")
        print('Evaluation over ', nr_eval_episodes, ' episodes. Average reward: ', average_rew, ' Hardcore activated: ', hc, ' Total Interactions: ', total_int)
        print("---------------------------------------")
        

env.close()