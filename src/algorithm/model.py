
from abc import *
from torch import nn

class BaseActor(ABC, nn.Module):
    def __init__(self, action_type) -> None:
        super(BaseActor, self).__init__()
        
        # discrete or continuous
        self.action_type = action_type
        
    @abstractmethod
    def get_action(self, state, *args, **kwargs):
        pass