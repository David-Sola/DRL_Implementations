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

env = gym.make('BipedalWalker-v3')

# Helper variables

# Variable to display accumulated reward for each episode
accumulated_reward = 0

# Maximum number of epsiodes
max_episodes = 10000

# Maximum number of timesteps per Episode
episode_range = 1000

# How many episodes random actions shall be taken
n_rand_actions = 30

# Best reward from all episodes
best_reward = -999

# Empty lists for x and y values to be able to plot the accumulated reward after
# the episodes
x = []
y = []


''' TAKEOVER TO THE AGENT!!!!!'''
actor_local = 'best_checkpoint_actor_loc_mem_imp_add_reinit_both_200_var11121211112111121212112121121.pth'
actor_target = 'best_checkpoint_actor_tar_mem_imp_add_reinit_both_200_var11121211112111121212112121121.pth'
critic_local = 'best_checkpoint_critic_loc_mem_imp_add_reinit_both_200_var11121211112111121212112121121.pth'
critic_target = 'best_checkpoint_critic_tar_mem_imp_add_reinit_both_200_var11121211112111121212112121121.pth'
''' TAKEOVER TO THE AGENT!!!!!'''
actor_local_policy = 'best_checkpoint_actor_loc_mem_imp_add_reinit_both_200_var11121211112111121212112121121.pth'
actor_target_policy = 'best_checkpoint_actor_tar_mem_imp_add_reinit_both_200_var11121211112111121212112121121.pth'
critic_local_policy = 'best_checkpoint_critic_loc_mem_imp_add_reinit_both_200_var11121211112111121212112121121.pth'
critic_target_policy = 'best_checkpoint_critic_tar_mem_imp_add_reinit_both_200_var11121211112111121212112121121.pth'

# Creation of the agent which shall be trained
agent = Agent(24, 4, random_seed=2)

''' START OF THE WHOLE TRAINING LOOP '''
for i_episode in range(max_episodes):
    
       
    x.append(i_episode)
    y.append(accumulated_reward)
    accumulated_reward = 0

    # Plot the current status of progress every 30 episodes and close it after 2 
    # episodes
    if i_episode%30==0:
        plt.pause(0.1)
        plt.plot(x,y)
        plt.title('Predicted next state lower cirtic LR, Best memory with additional learning and best actor, action noise, additional memory with positive memories and adding very positive, reinit mem, 2 critics with small actor loss adaption, only Q1, rewrite main memory and reset best mem,, no actor delay learning, best policy actor')
        plt.pause(0.1)
    elif i_episode%2==0:
        plt.close()

    # Get the first state from the environment
    state = env.reset()

    # Print some usefull stuff
    print("Best reward: ", best_reward)
    print("Episode: ", i_episode)
    
    
    ''' START OF THE LOOP FOR EACH EPISODE '''
    for t in range(episode_range):

        # Take an action with the agent in the current state
        # For the first n episodes take random actions
        if i_episode < n_rand_actions:
            action = env.action_space.sample()
        else:
            action = agent.act(state)    
        
        # Get the feedback from the environment by performing the action
        next_state, reward, done, info = env.step(action)
           
        # Accumulate the reward to get the reward at the end of the episode
        accumulated_reward = accumulated_reward + reward
        
        # Add the state, action, next_state, reward transition into the replay buffer
        agent.add_memory(state, action, reward, next_state, done)
        
        # Take a step with the agent in order to learn but collect first 
        # sufficient amount of data
        agent.step()
        
        # Assign the next state to the current state
        state = next_state
        
        # If the episode is done finish the loop
        if done:
            break
            
        
    print('Accumulated reward was: ', accumulated_reward)
    if accumulated_reward > best_reward:
        best_reward = accumulated_reward
        agent.save_network()      
        


env.close()