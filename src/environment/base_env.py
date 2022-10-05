
from abc import *
import copy

import numpy as np
from mlagents_envs.environment import ActionTuple, BaseEnv


class BaseEnvironment(ABC):        

    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def step(self, action):
        pass
    
    @abstractmethod
    def empty_action(self, n_agents : int):
        pass
    
    @abstractmethod
    def get_action_type(self):
        pass