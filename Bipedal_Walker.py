# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 22:25:21 2020

@author: sld8fe

Description:
    Training script for the Bipedal walker environment of OpenAI Gym
"""

import gym
from TD3_Agent import Agent
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import random

def normalize(data, min_value, max_value):

        return ((data - min_value) / (max_value - min_value))
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
episode_range = 300

# How many episodes random actions shall be taken
n_rand_actions = 25

# Best reward from all episodes
best_reward = -999

max_pos_reward = 1e-7
max_neg_reward = 1e-7

# Empty lists for x and y values to be able to plot the accumulated reward after
# the episodes
x = []
y = []
last_100_average = deque(100*[0], 100)

# Set seeds for reproducible results
seed = 0
env.seed(seed)

# Sigma value for exploration
sigma = 0.7
sigma_decay = 0.997

# Counter for number of interactions withe nvironemtn
total_int = 0

# Creation of the agent which shall be trained
agent = Agent(24, 4, random_seed=2)

''' START OF THE WHOLE TRAINING LOOP '''
for i_episode in range(max_episodes):
    sigma = max(sigma*sigma_decay, 0.0)
    accumulated_reward = 0
    last_action = env.action_space.sample()
    # Get the first state from the environment
    state = env.reset()
    last_state = state
    nr_of_its = 0
    target_actor = random.uniform(0,1)
    ''' START OF THE TRAINING LOOP FOR EACH EPISODE '''
    for t in range(episode_range):
        nr_of_its = t
        total_int += 1
        # Take an action with the agent with added noise in the current state
        # For the first n episodes take random actions
        if i_episode < n_rand_actions:
            action = env.action_space.sample()
        else:
            action = agent.act_noise(state, last_state, last_action, sigma, target_actor)    
        # Get the feedback from the environment by performing the action
        next_state, reward, done, info = env.step(action)
        accumulated_reward = accumulated_reward + reward
        if np.sign(reward)==1:
            if reward > max_pos_reward:
                corr_factor = reward/max_pos_reward
                print('Pos_CorR_Factor: ', corr_factor)
                for i in range(min(agent.memory.mem_cntr, agent.batch_size)):
                    if agent.memory.reward_memory[i] > 0:
                        agent.memory.reward_memory[i] /= corr_factor 
                max_pos_reward = reward
            reward = normalize(reward, 0, max_pos_reward)
        else:
            if np.abs(reward) > max_neg_reward:
                corr_factor = np.abs(reward)/max_neg_reward
                print('Neg_CorR_Factor: ', corr_factor)
                for i in range(min(agent.memory.mem_cntr, agent.batch_size)):
                    if agent.memory.reward_memory[i] > 0:
                        agent.memory.reward_memory[i] /= corr_factor
                max_neg_reward = np.abs(reward)
            reward = -normalize(np.abs(reward), 0, max_neg_reward) * 10
           
        # Accumulate the reward to get the reward at the end of the episode
        
        
        # Add the state, action, next_state, reward transition into the replay buffer
        agent.add_memory(state, action, reward, next_state, done, last_action, last_state)
        
        # Take a step with the agent in order to learn but collect first 
        # sufficient amount of data
        if i_episode > n_rand_actions:
            agent.learn()
        
        # Assign the next state to the current state
        last_state = state
        state = next_state
        last_action = action
        
        # If the episode is done finish the loop
        if done:
            break
            
    accumulated_reward = max(accumulated_reward, -500)
    # Print some usefull stuff
    print('Nr of ints: ', total_int, ' Episode ', i_episode, ' reward: ', '{:.2f}'.format(accumulated_reward), \
                                        ' 100 episode avrg: ',  '{:.2f}'.format(sum(last_100_average)/100), \
                                        'Best reward: ' , '{:.2f}'.format(best_reward), \
                                        'Sigma: ', '{:.2f}'.format(sigma))
    last_100_average.appendleft(accumulated_reward)
    #x.append(i_episode)
    #y.append(agent.accumulated_loss/nr_of_its)
    agent.accumulated_loss = 0
    if accumulated_reward > best_reward:
        best_reward = accumulated_reward
        agent.best_act_available = 1
        if target_actor > 0.5:
            agent.soft_update(agent.actor_local, agent.best_actor, 1)
        else:
            agent.soft_update(agent.actor_local2, agent.best_actor, 1)
        
        if best_reward > 50:
            episode_range = 1000
        agent.save_network() 
        
    ''' START AN EVALUATION OF THE CURRENT POLICY AFTER 100 EPISODES 
        FOR 10 EPISODES '''
    if i_episode%100==0 and i_episode != 0:
        average_rew = 0
        nr_eval_episodes = 10
        for eval_episodes in range(nr_eval_episodes):
            state, done = env.reset(), False
            last_action = env.action_space.sample()
            last_state = state
            while not done:
                action = agent.act(state, last_state, last_action) 
                next_state, reward, done, info = env.step(action)
                last_state= state
                state = next_state
                average_rew += reward
                last_action = action
        average_rew /= nr_eval_episodes
        
        #plt.pause(0.1)
        #plt.plot(x,y)
        #plt.title('Average 100 Episode Reward, Hardcore activated: '+ str(hc))
        #plt.pause(0.1)
        #plt.savefig('Reward.png')

        print("---------------------------------------")
        print('Evaluation over ', nr_eval_episodes, ' episodes. Average reward: ', average_rew, ' Hardcore activated: ', hc, ' Total Interactions: ', total_int)
        print("---------------------------------------")
        

env.close()


