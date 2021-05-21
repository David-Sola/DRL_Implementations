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
episode_range = 500

# How many episodes random actions shall be taken
n_rand_actions = 10

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
sigma = 0.01
sigma_decay = 0.95
increase_sigma_counter = 0
imitation_counter = 1

# Counter for number of interactions withe nvironemtn
total_int = 0
accumulated_loss = 0
nr_of_its = 1

update_predictor = 1

# Creation of the agent which shall be trained
agent_imitate = Agent(24, 4, random_seed=2)
agent_imitate.load_network(own_path=0, act_loc='HC_best_checkpoint_actor_loc_mem.pth', act_tar='HC_best_checkpoint_actor_tar_mem.pth', cr_loc='HC_best_checkpoint_critic_loc_mem.pth', cr_tar='HC_best_checkpoint_critic_tar_mem.pth')



# Creation of the agent which shall be trained
agent = Agent(24, 4, random_seed=2)
optimizer_imitation = optim.Adam(agent.actor_local.parameters(), lr=0.005)
state_predictor = State_predictor(28, 24)
next_state_predictor = State_predictor(24, 24)
optimizer = optim.Adam(state_predictor.parameters(), lr=0.01)
loss_fcn = nn.MSELoss()

def imitation_learnen(agent, agent_imitate, optimizer_imitation, loss_fcn):

    if agent.memory.get_len() > 2000:

        experiences = agent_imitate.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        actions_policy = agent.actor_local(states)
        loss = loss_fcn(actions_policy, actions)
        optimizer_imitation.zero_grad()
        loss.backward()
        optimizer_imitation.step()
        return loss.cpu().detach().numpy()

        

def curiosity_by_state_prediction(i_episode, agent, state_predictor, state, action, next_state, optimizer, loss_fcn, update_predictor):

    if agent.memory.get_len() > 200:

        experiences = agent.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        #feature_states = next_state_predictor(states)
        #next_feture_states = next_state_predictor(next_states)
        state_action = torch.cat([next_states, actions], 1)
        predicted_states = state_predictor(state_action)
        loss = loss_fcn(predicted_states, next_states)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #state_action = np.append(next_state, action)
    #state_action = torch.from_numpy(state_action).float().to(device)
    #next_state = torch.from_numpy(next_state).float().to(device)
    #predicted_state = state_predictor(state_action)
    prio = 0 #loss_fcn(predicted_state, next_state)
    #prio.backward()
    #optimizer.step()
 
    return prio#.cpu().detach().numpy()

    

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
    
    x.append(i_episode)
    y.append(accumulated_reward)
    accumulated_reward = 0

    # Get the first state from the environment
    state = env.reset()

    # Print some usefull stuff
    print("Best reward: ", best_reward)
    print("Episode: ", i_episode)
    print("Accumulated Loss: ", accumulated_loss/nr_of_its)
    print("Sigma: ", sigma)
    nr_of_its = 1
    accumulated_loss = 0

    if i_episode%30==0:
        imitation_counter += 1
        imitation_counter = min(imitation_counter, 20)
    
    
    ''' START OF THE TRAINING LOOP FOR EACH EPISODE '''
    for t in range(episode_range):

        if i_episode%imitation_counter!=0:
            nr_of_its += 1
            total_int += 1
            # Take an action with the agent with added noise in the current state
            # For the first n episodes take random actions
            if i_episode < n_rand_actions:
                action = env.action_space.sample()
            else:
                action = agent.act_noise(state, sigma)    
            
            # Get the feedback from the environment by performing the action
            next_state, reward, done, info = env.step(action)
            curiosity_by_state_prediction(i_episode, agent, state_predictor, state, action, next_state, optimizer, loss_fcn, update_predictor)

            #if i_episode > n_rand_actions:
            #    reward += min(curiosity_by_state_prediction(i_episode, agent, state_predictor, state, action, next_state, optimizer, loss_fcn, update_predictor), 1)
            
            # Accumulate the reward to get the reward at the end of the episode
            agent.add_memory(state, action, reward, next_state, done, 0.2)
            accumulated_reward = accumulated_reward + reward
            
            '''
            priority_for_current_memory = curiosity_by_state_prediction(i_episode, agent, state_predictor, state, action, next_state, optimizer, loss_fcn, update_predictor)
            accumulated_loss += priority_for_current_memory
            if reward == -100:
                reward = -100     
            elif reward < 0:
                reward = 0
            if i_episode < 200:
                reward = reward
            else:
                reward = reward + min(priority_for_current_memory, 0.5)
            accumulated_reward = accumulated_reward + reward
            # Add the state, action, next_state, reward transition into the replay buffer
            if i_episode < 200:
                agent.add_memory(state, action, reward, next_state, done, 0.2)
            elif priority_for_current_memory > 0.01:
                
                agent.add_memory(state, action, reward, next_state, done, priority_for_current_memory)
            '''
            
            # Take a step with the agent in order to learn but collect first 
            # sufficient amount of data
            if i_episode > n_rand_actions:
                
                agent.step(state_predictor)

                if i_episode%2==0:
                    loss = imitation_learnen(agent, agent_imitate, optimizer_imitation, loss_fcn)
                #accumulated_loss += loss
            
            # Assign the next state to the current state
            state = next_state
            
            # If the episode is done finish the loop
            if done:
                break

        else:
            nr_of_its += 1
            total_int += 1


            action_to_imitate = agent_imitate.act(state)
            # Get the feedback from the environment by performing the action
            next_state, reward, done, info = env.step(action_to_imitate)
            #accumulated_reward += reward
            agent_imitate.add_memory(state, action_to_imitate, reward, next_state, done, 0.2)
            if i_episode > n_rand_actions:
                loss = imitation_learnen(agent, agent_imitate, optimizer_imitation, loss_fcn)
                #agent.step(state_predictor)
                #accumulated_loss += loss

            #print(loss)

            # Assign the next state to the current state
            state = next_state

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