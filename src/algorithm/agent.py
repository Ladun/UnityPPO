
from abc import *
import torch


OPTIMIZER_NAME = "optimizer.pt"
ARG_NAME = "train_args.bin"
TRAINING_NAME = "train_values.bin"

class Agent(ABC):
    
    def __init__(self, args):
        self.is_training = False
        self.debug = args.debug
    
    @abstractmethod
    def save_checkpoint(self, args, checkpoint_dir, episode_len):
        pass
        
    @abstractmethod
    def load_checkpoint(self, args):
        pass
    
    @abstractmethod
    def inference(self):
        pass
        
    @abstractmethod
    def _collect_trajectories(self):
        pass
    
    @abstractmethod
    def step(self):
        pass 
    
    def _to_tensor(self, s, dtype=torch.float32):
        return torch.tensor(s, dtype=dtype, device=self.device)
    
    @abstractmethod
    def logging(self, cur_episode_len, logger):
        pass