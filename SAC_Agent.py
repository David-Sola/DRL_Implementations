# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 22:25:21 2020

@author: sld8fe

Description:
    Implementation of an TD3 Agent with 
"""

import os
import numpy as np
import random
from collections import namedtuple, deque

from torch._C import Value
from model import Actor, Critic, ValueNetwork
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import datetime



BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128     # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.005              # for soft update of target parameters
LR_ACTOR = 0.0003        # learning rate of the actor
LR_CRITIC = 0.0003       # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
POLICY_FREQ = 2
NOISE_CLIP = 0.5
POLICY_NOISE = 0.2
ENTROPY_SCALE = 0.4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    '''
    The agent who interacts with the environment and learns it
    '''

    def __init__(self, state_space, action_space, out_fcn=nn.Tanh(), fc1_units=512, fc2_units=256, random_seed=2):

        self.state_space = state_space
        self.action_space = action_space
        self.seed = random.seed(random_seed)
        self.lr_actor = LR_ACTOR
        self.lr_critic = LR_CRITIC
        self.total_it = 0
        self.sigma = POLICY_NOISE
        self.tau=TAU
        self.tau_noise_curiosity = 1

        # The actor network
        self.actor = Actor(self.lr_actor, state_space, action_space, out_fcn, fc1_units, fc2_units).to(device)

        # The critic network
        self.critic_1 = Critic(state_space, action_space).to(device)
        self.critic_2 = Critic(state_space, action_space).to(device)

        # Value Networt
        self.value = ValueNetwork(self.lr_critic, state_space, name='value')
        self.target_value = ValueNetwork(self.lr_critic, state_space, name='target_value')
        self.update_network_parameters(1)

        self.scale = ENTROPY_SCALE
        
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, state_space ,action_space)
        self.batch_size = BATCH_SIZE
        self.memory_imitation = ReplayBuffer(BUFFER_SIZE, state_space ,action_space)

        # Noise process
        self.noise = OUNoise(action_space, random_seed)
        
        self.act_time = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        self.actor_local_path = self.act_time +'_best_checkpoint_actor_loc_mem.pth' 
        self.actor_target_path = self.act_time +'_best_checkpoint_actor_tar_mem.pth' 
        self.critic_local_path = self.act_time +'_best_checkpoint_critic_loc_mem.pth'
        self.critic_target_path = self.act_time +'_best_checkpoint_critic_tar_mem.pth'

        self.t_step = 0

        self.highest_loss = 0
        self.loss = []

    def add_memory(self, state, action, reward, next_state, done):
        '''
        Add memories to the replay Bugger
        '''
        self.memory.store_transition(state, action, reward, next_state, done)

    def add_memory_imitation(self, state, action, reward, next_state, done):
        '''
        Add memories to the replay Bugger
        '''
        self.memory_imitation.store_transition(state, action, reward, next_state, done)
        
    def save_network(self):

        print('.....Saving Models.....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

        
    def load_network(self):
        print('.....Saving Models.....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)
        
    def step(self):
        '''
        Get the current state, action, reward, next_state and done tuple and store it in the replay buffer.
        Also check if already enough samples have been collected in order to start a training step
        :param state: Current state of the environment
        :param action: Action that has been chosen in current state
        :param reward: The reward that has been received in current state
        :param next_state: The next state that has been reached due to the current action
        :param done: Parameter to see if an episode has finished
        :return: -
        '''

        # Learn only if enough samples have already been collected
        if self.memory.mem_cntr > self.batch_size:
            self.learn(GAMMA)

    def act(self, state):
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
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()


    def learn(self, gamma):
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
        self.total_it += 1
        '''
        if self.total_it%5000==0:
            x = list(range(0, len(self.loss)))
            plt.close()
            plt.pause(0.1)
            plt.plot(x, self.loss, 'y*')
            plt.pause(0.1)
        '''

        states, actions, rewards, new_states, dones = \
                self.memory.sample_buffer(self.batch_size)

        rewards = torch.tensor(rewards, dtype=torch.float).to(self.actor.device)
        dones = torch.tensor(dones).to(self.actor.device)
        next_states = torch.tensor(new_states, dtype=torch.float).to(self.actor.device)
        states = torch.tensor(states, dtype=torch.float).to(self.actor.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.actor.device)

        value = self.value(states).view(-1)
        value_ = self.target_value(next_states).view(-1)
        value_[dones] = 0.0

        actions_policy, log_prob = self.actor.sample_normal(states, reparameterize=False)
        log_prob = log_prob.view(-1)
        q1_new_policy = self.critic_1.forward(states, actions_policy)
        q2_new_policy = self.critic_2.forward(states, actions_policy)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_prob
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions_policy, log_prob = self.actor.sample_normal(states, reparameterize=True) 
        log_prob = log_prob.view(-1)
        q1_new_policy = self.critic_1.forward(states, actions_policy)
        q2_new_policy = self.critic_2.forward(states, actions_policy)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_prob - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*rewards + gamma*value_
        q1_old_policy = self.critic_1.forward(states, actions).view(-1)
        q2_old_policy = self.critic_2.forward(states, actions).view(-1)
        critic_1_loss = 0.5*F.mse_loss(q1_old_policy,  q_hat)
        critic_2_loss = 0.5*F.mse_loss(q2_old_policy,  q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters(self.tau)

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

    def soft_update_directly(self, local_model, target_model, tau):
        '''
        θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_mode:
        :param target_model:
        :param tau:
        :return:
        '''
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)


class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones
    
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













































