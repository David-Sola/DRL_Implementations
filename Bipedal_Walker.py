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
from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
episode_range = 500

# How many episodes random actions shall be taken
n_rand_actions = 20

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




# Creation of the agent which shall be trained
agent = Agent(24, 4, random_seed=2)

state_predictor = State_predictor(28, 24)
next_state_predictor = State_predictor(24, 24)
optimizer = optim.Adam(state_predictor.parameters(), lr=0.01)
loss_fcn = nn.MSELoss()
loss_fcn_critic = nn.MSELoss()

def soft_network_update(agent_to_imitate_from, agent, tau=0.995):
    
    #agent.soft_update_directly(agent_to_imitate_from.actor_local, agent.actor_local, tau)
    agent.soft_update_directly(agent_to_imitate_from.actor_local, agent.actor_target, tau)
    agent.soft_update_directly(agent_to_imitate_from.critic_local, agent.critic_local, tau)
    #agent.soft_update_directly(agent_to_imitate_from.critic_local, agent.critic_target, tau)


def imitation_learnen(agent_to_imitate_from, agent_imitate, optimizer_imitation, loss_fcn):

    if agent.memory.get_len() > 2000:

        experiences = agent_imitate.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        actions_policy = agent_to_imitate_from.actor_local(states)
        loss = loss_fcn(actions_policy, actions)
        optimizer_imitation.zero_grad()
        loss.backward()
        optimizer_imitation.step()
        return loss.cpu().detach().numpy()

def imitation_learnen_policy(agent, agent_imitate, optimizer_agent, loss_fcn):

    if agent.memory.get_len() > 2000:

        experiences = agent_imitate.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        actions_policy = agent.actor_local(states)
        loss = loss_fcn(actions_policy, actions)
        optimizer_agent.zero_grad()
        loss.backward()
        optimizer_agent.step()
        return loss.cpu().detach().numpy()

def imitation_learnen_critic_target(agent_to_imitate_from, agent_imitate, optimizer_imitation_critic, loss_fcn_critic):

    if agent.memory.get_len() > 2000:

        experiences = agent_imitate.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        with torch.no_grad():
            noise = (torch.randn_like(actions) * 0.1).clamp(-0.5, 0.5)  
            next_actions = (agent_imitate.actor_local(next_states) + noise).clamp(-1,1)

            
            target_pol_Q1, target_pol_Q2 = agent_to_imitate_from.critic_target(next_states, next_actions)
            target_pol = torch.min(target_pol_Q1, target_pol_Q2)
            target_pol = rewards + (0.98 * target_pol * (1-dones))

        current_pol_1, current_pol_2 = agent_to_imitate_from.critic_local(states, actions)
        loss = F.mse_loss(current_pol_1, target_pol) + F.mse_loss(current_pol_2, target_pol)
        optimizer_imitation_critic.zero_grad()
        loss.backward()
        optimizer_imitation_critic.step()
        agent_to_imitate_from.soft_update(agent_to_imitate_from.critic_local, agent_to_imitate_from.critic_target, 0.99)
        #agent.soft_update(agent_imitate.actor_local, agent.actor_local, 0.001)
        #agent.soft_update(agent_imitate.actor_local, agent.actor_target, 0.0001)
        #optimizer_target_imitation.step()
        #optimizer_local_imitation.step()

        

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
    print("Internal Tau: ", agent.actor_local.log_alpha)
    print("loss: ", accumulated_loss/nr_of_its)
    tau = max(tau*0.99, 0.05)
    nr_of_its = 1
    accumulated_loss = 0

    if i_episode%30==0:
        imitation_counter += 1
        imitation_counter = min(imitation_counter, 20)
    
    
    ''' START OF THE TRAINING LOOP FOR EACH EPISODE '''
    for t in range(episode_range):

        if i_episode%1==0:
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

            curiosity_by_state_prediction(i_episode, agent, state_predictor, state, action, next_state, optimizer, loss_fcn, update_predictor)
           
            accumulated_reward = accumulated_reward + reward
            
            agent.add_memory(state, action, reward, next_state, done, 0.2)

            # If the episode is done finish the loop
            if done:
                break

            
            # Take a step with the agent in order to learn but collect first 
            # sufficient amount of data
            if i_episode > n_rand_actions:

                agent.step(state_predictor)


               
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
        agent.save_network() 
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