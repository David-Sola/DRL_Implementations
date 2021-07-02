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


# Change betweend Hardcore or non Hardcore version
hc = 0

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
n_rand_actions = 100

# Best reward from all episodes
best_reward = -999

# Empty lists for x and y values to be able to plot the accumulated reward after
# the episodes
x = []
y = []
last_100_average = deque(100*[0], 100)

# Set seeds for reproducible results
seed = 0
env.seed(seed)

# Sigma value for exploration
sigma = 0.1

# Counter for number of interactions withe nvironemtn
total_int = 0

# Creation of the agent which shall be trained
agent = Agent(24, 4, random_seed=2)

''' START OF THE WHOLE TRAINING LOOP '''
for i_episode in range(max_episodes):

    accumulated_reward = 0

    # Get the first state from the environment
    state = env.reset()
   
    ''' START OF THE TRAINING LOOP FOR EACH EPISODE '''
    for t in range(episode_range):
        
        total_int += 1
        # Take an action with the agent with added noise in the current state
        # For the first n episodes take random actions
        if i_episode < n_rand_actions:
            action = env.action_space.sample()
        else:
            action = agent.act_noise(state, 0)    
        
        # Get the feedback from the environment by performing the action
        next_state, reward, done, info = env.step(action)
           
        # Accumulate the reward to get the reward at the end of the episode
        accumulated_reward = accumulated_reward + reward
        
        # Add the state, action, next_state, reward transition into the replay buffer
        agent.add_memory(state, action, reward, next_state, done)
        
        # Take a step with the agent in order to learn but collect first 
        # sufficient amount of data
        if i_episode > n_rand_actions:
            agent.learn()
        
        # Assign the next state to the current state
        state = next_state
        
        # If the episode is done finish the loop
        if done:
            break
            
    
    # Print some usefull stuff
    print('Nr of ints: ', total_int, ' Episode ', i_episode, ' reward: ', '{:.2f}'.format(accumulated_reward), \
                                        ' 100 episode avrg: ',  '{:.2f}'.format(sum(last_100_average)/100), \
                                        'Best reward: ' , '{:.2f}'.format(best_reward))
    last_100_average.appendleft(accumulated_reward)
    x.append(i_episode)
    y.append(sum(last_100_average)/100)
    if accumulated_reward > best_reward:
        best_reward = accumulated_reward
        
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
            while not done:
                action = agent.act(state) 
                state, reward, done, info = env.step(action)
                average_rew += reward
        average_rew /= nr_eval_episodes
        
        plt.pause(0.1)
        plt.plot(x,y)
        plt.title('Average 100 Episode Reward, Hardcore activated: '+ str(hc))
        plt.pause(0.1)
        plt.savefig('Reward.png')

        print("---------------------------------------")
        print('Evaluation over ', nr_eval_episodes, ' episodes. Average reward: ', average_rew, ' Hardcore activated: ', hc, ' Total Interactions: ', total_int)
        print("---------------------------------------")
        

env.close()