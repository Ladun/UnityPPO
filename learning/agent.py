
import torch

class Agent:
    
    def __init__(self, args):
        self.is_training = False
        self.debug = args.debug
    
    def save_checkpoint(self, args, episode_len):
        pass
        
    def load_checkpoint(self, args):
        pass
        
    def _collect_trajectories(self):
        raise NotImplementedError("learning function is not implemented")
    
    def step(self):
        raise NotImplementedError("learning function is not implemented")    
    
    def _to_tensor(self, s, dtype=torch.float32):
        return torch.tensor(s, dtype=dtype, device=self.device)
    
    def logging(self, cur_episode_len, logger):
        pass