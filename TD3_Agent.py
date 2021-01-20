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
from model import Actor, Critic

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import pickle



BUFFER_SIZE = int(1e7)  # replay buffer size
BATCH_SIZE = 100      # minibatch size
GAMMA = 0.98            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 5e-4       # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
POLICY_FREQ = 2
NOISE_CLIP = 0.5
POLICY_NOISE = 0.2

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

        # The actor network
        self.actor_local = Actor(state_space, action_space, out_fcn, fc1_units, fc2_units).to(device)
        self.actor_target = Actor(state_space, action_space, out_fcn, fc1_units, fc2_units).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # The critic network
        self.critic_local = Critic(state_space, action_space, out_fcn, fc1_units, fc2_units).to(device)
        self.critic_target = Critic(state_space, action_space, out_fcn, fc1_units, fc2_units).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=WEIGHT_DECAY)
        
        # Replay memory
        self.memory = ReplayBuffer(action_space, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # Noise process
        self.noise = OUNoise(action_space, random_seed)
        
        self.actor_local_path = 'best_checkpoint_actor_loc_mem.pth'
        self.actor_target_path = 'best_checkpoint_actor_tar_mem.pth'
        self.critic_local_path = 'best_checkpoint_critic_loc_mem.pth'
        self.critic_target_path = 'best_checkpoint_critic_tar_mem.pth'

        self.t_step = 0

    def add_memory(self, state, action, reward, next_state, done):
        '''
        Add memories to the replay Bugger
        '''
        self.memory.add(state, action, reward, next_state, done)
        
    def save_network(self):

        torch.save(self.actor_local.state_dict(), self.actor_local_path)
        torch.save(self.actor_target.state_dict(), self.actor_target_path)
        torch.save(self.critic_local.state_dict(), self.critic_local_path)
        torch.save(self.critic_target.state_dict(), self.critic_target_path)  
        
    def load_network(self):

        torch.save(self.actor_local.state_dict(), self.actor_local_path)
        torch.save(self.actor_target.state_dict(), self.actor_target_path)
        torch.save(self.critic_local.state_dict(), self.critic_local_path)
        torch.save(self.critic_target.state_dict(), self.critic_target_path)
        
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
        if self.memory.get_len() > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, sigma):
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
        
        return np.clip(action, -1, 1)

    def act_noise(self, state, sigma):
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
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

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
        

    def learn(self, experiences, gamma):
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
        states, actions, rewards, next_states, dones = experiences
        
        with torch.no_grad():  
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(actions) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1,1)
             
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_Q1, target_Q2)
            target_q = rewards + (gamma * target_q * (1 - dones))
             
        current_Q1, current_Q2 = self.critic_local(states, actions)
        critic_loss = F.mse_loss(current_Q1, target_q) + F.mse_loss(current_Q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        ''' GRADIENT CLIPPING TO BE EVALUATED!!'''
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(),1)
        self.critic_optimizer.step()
        
        if self.total_it % POLICY_FREQ == 0:
            #Compute actor loss
            actor_loss = -self.critic_local.Q1(states, self.actor_local(states)).mean()
            
            #Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.soft_update(self.critic_local, self.critic_target, self.tau)
            self.soft_update(self.actor_local, self.actor_target, self.tau)
        

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

class ReplayBuffer():
    def __init__(self, action_size, buffer_size, batch_size, seed):
        '''
        Initialize replay buffer
        :param action_size: action size of environment
        :param buffer_size: buffer size for replay buffer
        :param batch_size: batch size to learn from
        :param seed: random seed
        '''

        self.action_space = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "rewards", "next_states", "dones"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        '''
        Adding a nre state, action, reward, nect_state, done tuplt to the replay memory
        :param state: Current state
        :param action: Action taken in current state
        :param reward: Reward that has been granted
        :param next_state: Next state reached
        :param done: Information if environment has finished
        :return: -
        '''
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        '''
        Radnomly sample a batch
        :return: A random selected batch of the memory
        '''
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_states for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.dones for e in experiences if e is not None])).float().to(device)

        return (states, actions, rewards, next_states, dones)
    
    def get_len(self):
        return len(self.memory)

    #def __len__(self):
     #   return len(self.memory)
    
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













































