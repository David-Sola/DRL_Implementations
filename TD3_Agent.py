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
import datetime



BUFFER_SIZE = int(1e7)  # replay buffer size
BATCH_SIZE = 10      # minibatch size
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
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE

        # The actor network
        self.actor_local = Actor(state_space, action_space, out_fcn, fc1_units, fc2_units).to(device)
        self.actor_target = Actor(state_space, action_space, out_fcn, fc1_units, fc2_units).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # The critic networks
        self.critic_local1 = Critic(state_space, action_space, out_fcn, fc1_units, fc2_units).to(device)
        self.critic_target1 = Critic(state_space, action_space, out_fcn, fc1_units, fc2_units).to(device)
        self.critic_optimizer1 = optim.Adam(self.critic_local1.parameters(), lr=self.lr_critic, weight_decay=WEIGHT_DECAY)
        
        self.critic_local2 = Critic(state_space, action_space, out_fcn, fc1_units, fc2_units).to(device)
        self.critic_target2 = Critic(state_space, action_space, out_fcn, fc1_units, fc2_units).to(device)
        self.critic_optimizer2 = optim.Adam(self.critic_local2.parameters(), lr=self.lr_critic, weight_decay=WEIGHT_DECAY)
        

        # Replay memory
        self.memory = ReplayBufferLSTM2(BUFFER_SIZE)

        # Noise process
        self.noise = OUNoise(action_space, random_seed)
        
        self.act_time = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        self.actor_local_path = self.act_time +'_best_checkpoint_actor_loc_mem.pth' 
        self.actor_target_path = self.act_time +'_best_checkpoint_actor_tar_mem.pth' 
        self.critic_local_path = self.act_time +'_best_checkpoint_critic_loc_mem.pth'
        self.critic_target_path = self.act_time +'_best_checkpoint_critic_tar_mem.pth'

        self.t_step = 0

    def add_memory(self, ini_hidden_in, ini_hidden_out, episode_state, episode_action, episode_last_action,
                episode_reward, episode_next_state, episode_done):
        '''
        Add memories to the replay Bugger
        '''
        self.memory.push(ini_hidden_in, ini_hidden_out, episode_state, episode_action, episode_last_action, \
                episode_reward, episode_next_state, episode_done)
        
    def save_network(self):

        torch.save(self.actor_local.state_dict(), self.actor_local_path)
        torch.save(self.actor_target.state_dict(), self.actor_target_path)
        torch.save(self.critic_local1.state_dict(), self.critic_local_path)
        torch.save(self.critic_target1.state_dict(), self.critic_target_path)  
        torch.save(self.critic_local2.state_dict(), self.critic_local_path)
        torch.save(self.critic_target2.state_dict(), self.critic_target_path)
        
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

    def act(self, state, last_action, hidden_in):
        '''
        Functionality to determine the action in the current state.
        Additionally to the current action which is determined by the actor a normal distribution will be added to the
        action space to have enough exploration at least in the beginning of the training. The normal distribution will
        get smaller with time expressed through the parameter sigma
        :param state: Current state of the environment
        :param sigma: Parameter which decays over time to make the normal distribution smaller
        :return: action which shall be performed in the environment
        '''
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).cuda()
        last_action = torch.FloatTensor(last_action).unsqueeze(0).unsqueeze(0).cuda()
        self.actor_local.eval()
        with torch.no_grad():
            action, hidden_out = self.actor_local(state, last_action, hidden_in)
        self.actor_local.train()
        
        return action.detach().cpu().numpy()[0][0], hidden_out

    def act_noise(self, state, last_action, hidden_in, sigma):
        '''
        Functionality to determine the action in the current state.
        Additionally to the current action which is determined by the actor a normal distribution will be added to the
        action space to have enough exploration at least in the beginning of the training. The normal distribution will
        get smaller with time expressed through the parameter sigma
        :param state: Current state of the environment
        :param sigma: Parameter which decays over time to make the normal distribution smaller
        :return: action which shall be performed in the environment
        '''
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).cuda()
        last_action = torch.FloatTensor(last_action).unsqueeze(0).unsqueeze(0).cuda()
        self.actor_local.eval()
        with torch.no_grad():
            action, hidden_out = self.actor_local(state, last_action, hidden_in)
        self.actor_local.train()
        noise = np.random.normal(0, sigma, 4)
        action = action.detach().cpu().numpy()[0][0] + noise
        
        return np.clip(action, -1, 1) , hidden_out
        
        
    
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
        if self.memory.get_length() < self.batch_size:
            return
        self.total_it += 1
        hidden_in, hidden_out, state, action, last_action, reward, next_state, done =\
             self.memory.sample(self.batch_size)

        states          = torch.FloatTensor(state).to(device)
        next_states     = torch.FloatTensor(next_state).to(device)
        actions         = torch.FloatTensor(action).to(device)
        last_actions    = torch.FloatTensor(last_action).to(device)
        rewards         = torch.FloatTensor(reward).unsqueeze(-1).to(device)  
        dones           = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(device)
        
        with torch.no_grad():  
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(actions) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            new_next_action, _ = self.actor_target(next_states, actions, hidden_out)
            new_next_action = (new_next_action+noise).clamp(-1,1)
            new_action, _ = self.actor_local(states, last_actions, hidden_in)
             
            # Compute the target Q value
            target_Q1, _ = self.critic_target1(next_states, new_next_action, actions, hidden_out)
            target_Q2, _ = self.critic_target2(next_states, new_next_action, actions, hidden_out)
            target_q = torch.min(target_Q1, target_Q2)
            target_q = rewards + (self.gamma * target_q * (1 - dones))
             
        current_Q1, _ = self.critic_local1(states, actions, last_actions, hidden_in)
        current_Q2, _ = self.critic_local2(states, actions, last_actions, hidden_in)
        critic_loss1 =((current_Q1 - target_q.detach())**2).mean() 
        critic_loss2 =((current_Q2 - target_q.detach())**2).mean() 

        self.critic_optimizer1.zero_grad()
        critic_loss1.backward()
        self.critic_optimizer1.step()

        self.critic_optimizer2.zero_grad()
        critic_loss2.backward()
        self.critic_optimizer2.step()
        
        if self.total_it % POLICY_FREQ == 0:
            #Compute actor loss
            predicted_new_q_value, _ = self.critic_local1(states, new_action, last_actions, hidden_in)
            actor_loss = -predicted_new_q_value.mean()
            
            #Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.soft_update(self.critic_local1, self.critic_target1, self.tau)
            self.soft_update(self.critic_local2, self.critic_target2, self.tau)
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

class ReplayBufferLSTM2:
    """ 
    Replay buffer for agent with LSTM network additionally storing previous action, 
    initial input hidden state and output hidden state of LSTM.
    And each sample contains the whole episode instead of a single step.
    'hidden_in' and 'hidden_out' are only the initial hidden state for each episode, for LSTM initialization.

    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, hidden_in, hidden_out, state, action, last_action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (hidden_in, hidden_out, state, action, last_action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst, ci_lst, ho_lst, co_lst, d_lst=[],[],[],[],[],[],[],[],[],[]
        batch = random.sample(self.buffer, batch_size)
        for sample in batch:
            (h_in, c_in), (h_out, c_out), state, action, last_action, reward, next_state, done = sample
            s_lst.append(state) 
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)
            hi_lst.append(h_in)  # h_in: (1, batch_size=1, hidden_size)
            ci_lst.append(c_in)
            ho_lst.append(h_out)
            co_lst.append(c_out)
        hi_lst = torch.cat(hi_lst, dim=-2).detach() # cat along the batch dim
        ho_lst = torch.cat(ho_lst, dim=-2).detach()
        ci_lst = torch.cat(ci_lst, dim=-2).detach()
        co_lst = torch.cat(co_lst, dim=-2).detach()

        hidden_in = (hi_lst, ci_lst)
        hidden_out = (ho_lst, co_lst)

        return hidden_in, hidden_out, s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)
    
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













































