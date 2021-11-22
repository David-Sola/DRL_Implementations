# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 22:25:21 2020

@author: sld8fe

Description:
    Implementation of an TD3 Agent with 
"""

import numpy as np
import random
from collections import namedtuple, deque
from model import Actor, Critic, ICM_module

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import datetime
import random



BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 100      # minibatch size
GAMMA = 0.98            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 6e-4         # learning rate of the actor
LR_CRITIC = 6e-4       # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
POLICY_FREQ = 2
NOISE_CLIP = 0.5
POLICY_NOISE = 0.3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    '''
    The agent who interacts with the environment and learns it
    '''

    def __init__(self, state_space, action_space, out_fcn=nn.Tanh(), fc1_units=256, fc2_units=256, random_seed=2):

        self.state_space = state_space
        self.action_space = action_space
        self.seed = random.seed(random_seed)
        self.lr_actor = LR_ACTOR
        self.lr_critic = LR_CRITIC
        self.total_it = 0
        self.sigma = POLICY_NOISE
        self.tau=TAU
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.accumulated_loss = 0

        # The actor network
        self.actor_local = Actor(state_space, action_space, out_fcn, fc1_units, fc2_units).to(device)
        self.actor_target = Actor(state_space, action_space, out_fcn, fc1_units, fc2_units).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        self.actor_local2 = Actor(state_space, action_space, out_fcn, fc1_units, fc2_units).to(device)
        self.actor_target2 = Actor(state_space, action_space, out_fcn, fc1_units, fc2_units).to(device)
        self.actor_optimizer2 = optim.Adam(self.actor_local2.parameters(), lr=self.lr_actor)

        self.best_actor = Actor(state_space, action_space, out_fcn, fc1_units, fc2_units).to(device)
        self.best_act_available = 0

        # The critic network
        self.critic_local = Critic(state_space, action_space, out_fcn, fc1_units, fc2_units).to(device)
        self.critic_target = Critic(state_space, action_space, out_fcn, fc1_units, fc2_units).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=WEIGHT_DECAY)

        #The ICM module
        self.reward_predictor = ICM_module(state_space, action_space, out_fcn, fc1_units, fc2_units).to(device)
        self.reward_predictor_optimizer = optim.Adam(self.reward_predictor.parameters(), lr=1e-5)
        
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, [state_space], action_space)

        # Noise process
        self.noise = OUNoise(action_space, random_seed)
        
        self.act_time = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        self.actor_local_path = self.act_time +'_best_checkpoint_actor_loc_mem.pth' 
        self.actor_target_path = self.act_time +'_best_checkpoint_actor_tar_mem.pth' 
        self.critic_local_path = self.act_time +'_best_checkpoint_critic_loc_mem.pth'
        self.critic_target_path = self.act_time +'_best_checkpoint_critic_tar_mem.pth'

        self.t_step = 0

        self.actor_list = []
        self.critic_list = []

    def add_memory(self, state, action, reward, next_state, done, last_action, last_state):
        '''
        Add memories to the replay Bugger
        '''
        self.memory.store_transition(state, action, reward, next_state, done, last_action, last_state)
        
    def save_network(self):

        torch.save(self.actor_local.state_dict(), self.actor_local_path)
        torch.save(self.actor_target.state_dict(), self.actor_target_path)
        torch.save(self.critic_local.state_dict(), self.critic_local_path)
        torch.save(self.critic_target.state_dict(), self.critic_target_path)  
        
    def load_network(self, own_path=1, act_loc='', act_tar='', cr_loc='', cr_tar=''):
        
        if own_path==1:
            self.actor_local.load_state_dict(torch.load(self.actor_local_path))
            self.actor_target.load_state_dict(torch.load(self.actor_target_path))
            self.critic_local.load_state_dict(torch.load(self.critic_local_path))
            self.critic_target.load_state_dict(torch.load(self.critic_target_path))
        else:
            self.actor_local.load_state_dict(torch.load(act_loc))
            self.actor_target.load_state_dict(torch.load(act_tar))
            self.critic_local.load_state_dict(torch.load(cr_loc))
            self.critic_target.load_state_dict(torch.load(cr_tar))

    def act(self, state, last_state, last_action):
        '''
        Functionality to determine the action in the current state.
        Additionally to the current action which is determined by the actor a normal distribution will be added to the
        action space to have enough exploration at least in the beginning of the training. The normal distribution will
        get smaller with time expressed through the parameter sigma
        :param state: Current state of the environment
        :param sigma: Parameter which decays over time to make the normal distribution smaller
        :return: action which shall be performed in the environment
        '''
        state = torch.from_numpy(state).float().to(device)
        last_action = torch.from_numpy(last_action).float().to(device)
        last_state = torch.from_numpy(last_state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local.evaluate(state, last_state, last_action).cpu().data.numpy()
        self.actor_local.train()
        
        return np.clip(action, -1, 1)

    def act_noise(self, state, last_state, last_action, sigma, target_actor):
        '''
        Functionality to determine the action in the current state.
        Additionally to the current action which is determined by the actor a normal distribution will be added to the
        action space to have enough exploration at least in the beginning of the training. The normal distribution will
        get smaller with time expressed through the parameter sigma
        :param state: Current state of the environment
        :param sigma: Parameter which decays over time to make the normal distribution smaller
        :return: action which shall be performed in the environment
        '''

        noise = np.random.normal(0, sigma, 4)
        state = torch.from_numpy(state).float().to(device)
        last_action = torch.from_numpy(last_action).float().to(device)
        last_state = torch.from_numpy(last_state).float().to(device)
        self.actor_local.eval()
        if target_actor > 0.5:
            with torch.no_grad():
                action = self.actor_local.evaluate(state, last_state, last_action).cpu().data.numpy()
            self.actor_local.train()
        else:
            with torch.no_grad():
                action = self.actor_local2.evaluate(state, last_state, last_action).cpu().data.numpy()
            self.actor_local2.train()

        action = action + noise
        
        return np.clip(action, -1, 1)
        
        
    
    def act_ou(self, state, sigma,  add_noise=False):
        '''
        Functionality to determine the action in the current state.
        Additionally to the current action which is determined by the actor a normal distribution will be added to the
        action space to have enough exploration at least in the beginning of the training. The normal distribution will
        get smaller with time expressed through the parameter sigma
        :param state: Current state of the environment
        :param sigma: Parameter which decays over time to make the normal distribution smaller
        :return: action which shall be performed in the environment
        '''
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += sigma * self.noise.sample()
 
        return np.clip(action, -1, 1)

    
    def reset(self):
        self.noise.reset()
        

    def learn(self):
        '''
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r+ gamma * critic_target(next_state, actor_target(next_state)
            actor_target(state) --> action
            critic_target(state, action) --> Q-value
        Also update the the actor with gradient ascent by comparing the loss between the actor and the critiv actions.
        Perform the learning multiple times expressed by the parameter UPDATE_NUMBER

        IMPORTANT TRICK:
        A important trick has been introduced to make the learning more stable.
        The learning rate decays over time. After every learning step the learning rate will be decayed.
        This makes the agent in the beginning more aggressive and more passive the longer it trains
        The function for this is called exp_lr_scheduler
        :param experiences: A sample of states, actions, rewards, next_states, dones tuples to learn from
        :param gamma: Value to determine the reward discount
        :return: -
        '''
        if self.memory.mem_cntr < self.batch_size:
            return
        self.total_it += 1
        state, action, reward, new_state, done, last_action, last_state = \
                self.memory.sample_buffer(self.batch_size)

        if random.uniform(0,1) > 0.5:
            train_first = 1
        else:
            train_first = 0

        rewards = torch.tensor(reward, dtype=torch.float).to(device)
        dones = torch.tensor(done, dtype=torch.float).to(device)
        next_states = torch.tensor(new_state, dtype=torch.float).to(device)
        states = torch.tensor(state, dtype=torch.float).to(device)
        actions = torch.tensor(action, dtype=torch.float).to(device)
        last_actions = torch.tensor(last_action, dtype=torch.float).to(device)
        last_states = torch.tensor(last_state, dtype=torch.float).to(device)

        predicted_rewards = self.reward_predictor.predict_reward(states, actions)
        prediction_loss = F.mse_loss(predicted_rewards.view(-1), rewards)
        self.reward_predictor_optimizer.zero_grad()
        prediction_loss.backward()
        self.reward_predictor_optimizer.step()

        
        pol_loss = 0
        pol_loss_ = 0
        if self.total_it%2==0 and self.best_act_available==1:
            best_actions = self.best_actor(states, last_states, actions)
            policy_actions = self.actor_local(states, last_states, actions)

            pol_loss = F.mse_loss(policy_actions, best_actions) * 30
        
        with torch.no_grad():  
            # Get intrinsic reward

            # Select action according to policy and add clipped noise
            if self.total_it%2==0 and self.best_act_available==1:
                noise = (torch.randn_like(actions) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
                next_actions = (self.best_actor(next_states, states, actions) + noise).clamp(-1,1)
                target_Q1, target_Q2 = self.critic_target(next_states, next_actions, actions, last_states)
                target_q = torch.min(target_Q1.view(-1), target_Q2.view(-1))

            else:
                noise = (torch.randn_like(actions) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
                next_actions_first = (self.actor_target(next_states, states, actions) + noise).clamp(-1,1)
                next_actions_second = (self.actor_target2(next_states, states, actions) + noise).clamp(-1,1)
                
                # Compute the target Q value
                target_Q1_first, target_Q2_first = self.critic_target(next_states, next_actions_first, actions, last_states)
                if train_first == 1:
                    target_q_first = torch.min(target_Q1_first.view(-1), target_Q2_first.view(-1))
                else:
                    target_q_first = torch.max(target_Q1_first.view(-1), target_Q2_first.view(-1))
                target_Q1_second, target_Q2_second = self.critic_target(next_states, next_actions_second, actions, last_states)
                if train_first == 1:
                    target_q_second = torch.min(target_Q1_second.view(-1), target_Q2_second.view(-1))
                else:
                    target_q_second = torch.max(target_Q1_second.view(-1), target_Q2_second.view(-1))

                target_q = torch.max(target_q_first, target_q_second)
                


            target_q = rewards +(self.gamma * target_q * (1 - dones))
             
        current_Q1, current_Q2 = self.critic_local(states, actions, last_actions, last_states)
        critic_loss = F.mse_loss(current_Q1.view(-1), target_q) + F.mse_loss(current_Q2.view(-1), target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        critic_copy = copy.deepcopy(self.critic_local)
        critic_copy_random = copy.deepcopy(self.critic_local)
        self.soft_update_random(critic_copy_random, critic_copy_random, self.tau)
        self.critic_list.append(critic_copy)
        self.critic_list.append(critic_copy_random)
        if len(self.critic_list) > 20:
            self.sample_best_critic()
        '''
        predicted_state = self.icm.predict_state(states, actions)
        icm_loss = F.mse_loss(next_states, predicted_state)
        icm_loss_actor = icm_loss.detach().clone()
        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        self.icm_optimizer.step()
        '''

        actor_loss = 0
        if self.total_it % POLICY_FREQ == 0:
            #Compute actor loss
            if train_first == 1:
                actor_loss = -self.critic_local.Q1(states, self.actor_local(states, last_states, last_actions), last_actions, last_states).mean() + pol_loss
                    #Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.soft_update(self.critic_local, self.critic_target, self.tau)
                self.soft_update(self.actor_local, self.actor_target, self.tau)

                actor_copy = copy.deepcopy(self.actor_local)
                actor_copy_random = copy.deepcopy(self.actor_local)
                self.soft_update_random(actor_copy, actor_copy_random, self.tau)

                self.actor_list.append(actor_copy)
                self.actor_list.append(actor_copy_random)
            else:
                actor_loss = -self.critic_local.Q1(states, self.actor_local2(states, last_states, last_actions), last_actions, last_states).mean() + pol_loss
                #Optimize the actor
                self.actor_optimizer2.zero_grad()
                actor_loss.backward()
                self.actor_optimizer2.step()
                self.soft_update(self.critic_local, self.critic_target, self.tau)
                self.soft_update(self.actor_local2, self.actor_target2, self.tau)

                actor_copy = copy.deepcopy(self.actor_local2)
                actor_copy_random = copy.deepcopy(self.actor_local2)
                self.soft_update_random(actor_copy, actor_copy_random, self.tau)

                self.actor_list.append(actor_copy)
                self.actor_list.append(actor_copy_random)

        
        if len(self.actor_list) > 20:
            self.sample_best_actor()
           

        
        self.accumulated_loss += actor_loss

    def sample_best_actor(self):
        if self.memory.mem_cntr < self.batch_size*5:
            return
        state, _, _, new_state, _, _, last_state = \
                self.memory.sample_buffer(self.batch_size*5)

        next_states = torch.tensor(new_state, dtype=torch.float).to(device)
        states = torch.tensor(state, dtype=torch.float).to(device)
        last_states = torch.tensor(last_state, dtype=torch.float).to(device)

        best_reward = -999
        number_of_winner = 0
        for i in range(len(self.actor_list)):
            current_action = self.actor_list[i].forward(states, last_states)
            next_action = self.actor_list[i].forward(next_states, states)
            current_reward = self.reward_predictor.predict_reward(states, current_action)
            next_reward = self.reward_predictor.predict_reward(next_states, next_action)
            target_Q1_first, target_Q2_first = self.critic_target(next_states, next_action, current_action, last_states)
            target_q_first = torch.min(target_Q1_first.view(-1), target_Q2_first.view(-1))
            target_q = current_reward +(self.gamma * target_q_first)

            reward_sum = torch.sum(target_q) + torch.sum(next_reward)

            if reward_sum > best_reward:
                number_of_winner = i
                best_reward = reward_sum

        #print('Winner was: ', number_of_winner)
        best_actor = self.actor_list[number_of_winner]
        self.soft_update(self.actor_list[number_of_winner], self.actor_local2, 0.01)
        self.soft_update(self.actor_list[number_of_winner], self.actor_local, 0.01)
        self.soft_update(self.actor_list[number_of_winner], self.actor_target, 0.005)
        self.soft_update(self.actor_list[number_of_winner], best_actor, 1)
        self.actor_list = []
        #self.actor_list.append(best_actor)

    def sample_best_critic(self):
        if self.memory.mem_cntr < self.batch_size*5:
            return
        state, action, reward, new_state, done, last_action, last_state = \
                self.memory.sample_buffer(self.batch_size*5)

        next_states = torch.tensor(new_state, dtype=torch.float).to(device)
        states = torch.tensor(state, dtype=torch.float).to(device)
        last_states = torch.tensor(last_state, dtype=torch.float).to(device)
        actions = torch.tensor(action, dtype=torch.float).to(device)
        last_actions = torch.tensor(last_action, dtype=torch.float).to(device)

        best_reward = -999
        for i in range(len(self.critic_list)):
            current_Q1, current_Q2 = self.critic_list[i].forward(states, actions, last_actions, last_states)
            reward_sum = torch.sum(torch.max(current_Q1, current_Q2))

            if reward_sum > best_reward:
                number_of_winner = i
                best_reward = reward_sum

        #print('Winner was: ', number_of_winner)
        best_critic = self.critic_list[number_of_winner]
        self.soft_update(self.critic_list[number_of_winner], self.critic_local, 0.01)
        self.soft_update(self.critic_list[number_of_winner], self.critic_target, 0.05)
        self.critic_list = []
        #self.critic_list.append(best_critic)



    def exp_lr_scheduler(self, optimizer, decayed_lr):
        '''
        Set the learning rate to a decayed learning rate, without initializing the optimizer from scratch
        :param optimizer: the optimizer in which the learning rate shall be adjusted
        :param decayed_lr: the decaed learning rate to be set
        :return: optimizer with new learning rate
        '''

        for param_group in optimizer.param_groups:
            param_group['lr'] = decayed_lr
        return optimizer

    def soft_update(self, local_model, target_model, tau):
        '''
        θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_mode:
        :param target_model:
        :param tau:
        :return:
        '''
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)

    def soft_update_random(self, local_model, target_model, tau):
        '''
        θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_mode:
        :param target_model:
        :param tau:
        :return:
        '''
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data + 0.001*torch.randn(local_param.data.size()).to(device))

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size)
        self.last_action = np.zeros((self.mem_size, n_actions))
        self.last_state = np.zeros((self.mem_size, *input_shape))

    def store_transition(self, state, action, reward, state_, done, last_action, last_state):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.last_action[index] = last_action
        self.last_state[index] = last_state

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        last_action = self.last_action[batch]
        last_state = self.last_state[batch]

        return states, actions, rewards, states_, dones, last_action, last_state
    
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state













































