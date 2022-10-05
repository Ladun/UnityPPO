
import copy
import numpy as np
import torch

import gym
from gym.spaces.discrete import Discrete

from src.environment.base_env import BaseEnvironment


class UnityLikeStepOutput:
    def __init__(self):
        
        self.obs = [[]]
        self.reward = []
        self.agent_id = []
        
        self.agent_id_to_index = {}
        
    def add(self, id, next_state, reward):
        self.obs[0].append(next_state)
        self.reward.append(reward)
        
        self.agent_id_to_index[id] = len(self.agent_id)
        self.agent_id.append(id)
        
    def __len__(self):
        return len(self.agent_id)
        

class OpenAIGymEnvWrap(BaseEnvironment):
    def __init__(self, env_name):

        self.env = gym.make(env_name)
        self.env.seed(42)

        self.agent_n = 1
        self.action_size = self.env.action_space.n
        self.state_size = np.prod(self.env.observation_space.shape)

    def reset(self):
        state = self.env.reset()
        
        # np array
        # (1, obs_len)
        return copy.deepcopy(state[np.newaxis, :])

    def step(self, action):            
        
        if self.get_action_type() == 'discrete':
            action = action[0]
        
        next_state, reward, done, _ = self.env.step(action)
        
        dec = UnityLikeStepOutput()
        term = UnityLikeStepOutput()
        
        if done:
            term.add(0, next_state, reward)
        else: 
            dec.add(0, next_state, reward)
        
        return dec, term
        

    def get_num_agents(self):
        return 1
    
    def empty_action(self, n_agents : int = 1):
        # _continuous = np.zeros((n_agents, self.action_size), dtype=np.float32)
        #_discrete = np.zeros((n_agents, self.discrete_size), dtype=np.int32)
        # return ActionTuple(continuous=_continuous)
        _continuous = self.env.action_space.sample()
        
        return _continuous
    
    def get_action_type(self):
        if isinstance(self.env.action_space, Discrete):
            return 'discrete'
        else:
            return 'continuous'
