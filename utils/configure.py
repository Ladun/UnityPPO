
from dataclasses import dataclass, field
from typing import List

import torch


@dataclass 
class BaseConfig:
    
    algo: str = field(
        default="ppo",
        metadata={"help": "'ppo', 'sac'"}
    )
    seed: int = field(
        default=2022,
        metadata={"help": "random seed"}
    )
    no_cuda: bool = field(
        default=False,
    )
    env_name: str = field(
        default=None
    )
    no_graphics: bool = field(
        default=False,
    )
    time_scale: int = field(
        default=1,
    )
    checkpoint_dir: str =field(
        default=None,
        metadata={'help': 'checkpoint dir'}
    )
    n_episode: int = field(
        default=1000
    )
    save_steps: int = field(
        default=2
    )
    max_save_limits: int = field(
        default=2
    )
    debug:bool = field(
        default=False
    )
    
    def __post_init__(self):        
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
        self.n_gpu = torch.cuda.device_count()

@dataclass
class PPOConfig(BaseConfig):
    K_epoch: int = field(
        default=3,
        metadata={"help":". (Recommended Range: 3 to 30)"}
    )
    batch_size: int = field(
        default=1024,
        metadata={"help":"batch size of sampling. (Recommended Range: 4 to 4096)"}
    )
    buffer_size: int = field(
        default=30,
        metadata={"help":"min no of batches needed in the memory before learning"}
    )
    gamma :float = field(
        default=0.99,
        metadata={"help":"discount factor. (Recommended Range: 0.8 to 0.997)"}
    )
    lmbda :float = field(
        default=0.95,
        metadata={"help":"value control how much agent rely on current estimate. (Recommended Range: 0.9 to 1)"}
    )
    eps_clip :float = field(
        default=0.1,
        metadata={"help":"eps for ratio clip 1+eps, 1-eps. (Recommended Range: 0.1, 0.2, 0.3)"}
    )
    T: int = field(
        default=512,
        metadata={"help":"max number of time step for collecting trajectory"}
    )
    T_EPS: int = field(
        default=int(3e4),
        metadata={"help":"max number of time step for collecting trajectory"}
    )
    learning_rate :float = field(
        default=1e-4,
        metadata={"help":"learning rate. (Recommended Range: 3e-3 to 5e-6)"}
    )
    critic_loss_weight :float = field(
        default=1.0,
        metadata={"help":"mean square error term weight. (Recommended Range: 0.5, 1)"}
    )
    nan_penalty :float = field(
        default=-5.0,
        metadata={"help":"penalty for actions that resulted in nan reward"}
    )
    loss_type :str = field(
        default="clip",
        metadata={"help":"loss_type, 'clip', 'kl', 'none'"}
    )
    
    # Model 
    actor_hidden_layers: List[int] = field(
        default_factory=list,
        # default=[1024, 1024, 512],
        metadata={"help":"actor model hidden layers"}
    )
    critic_hidden_layers: List[int] = field(
        default_factory=list,
        # default=[1024, 1024, 512],
        metadata= {"help":"actor model hidden layers"}
    )
    normalize: bool = field(
        default=False,
    )
    
    # Entropy
    entropy_weight :float = field(
        default=0.1,
        metadata={"help":"weight of entropy added. (Recommended Range: 0 to 0.01)"}
    )
    entropy_decay :float = field(
        default=0.995,
        metadata={"help":"decay of entropy per 'step'"}
    )
    
    # action std scale
    std_scale_init :float = field(
        default=1.0,
        metadata={"help":"initial value of std scale for action resampling"}
    )
    std_scale_decay :float = field(
        default=0.995,
        metadata={"help":" scale decay of std"}
    )

@dataclass
class SACConfig(BaseConfig):
    batch_size: int = field(
        default=1024,
        metadata={"help":"batch size of sampling. (Recommended Range: 4 to 4096)"}
    )
    T: int = field(
        default=512,
        metadata={"help":"max number of time step for collecting trajectory"}
    )
    T_EPS: int = field(
        default=int(3e4),
        metadata={"help":"max number of time step for collecting trajectory"}
    )
    learning_rate :float = field(
        default=1e-4,
        metadata={"help":"learning rate. (Recommended Range: 3e-3 to 5e-6)"}
    )
    tau : float = field(
        default=0.01,
        metadata={"help": "for target network soft update"}
    )
    gamma : float = field(
        default=0.98,
        metadata={"help": ""}
    )
    # Model 
    actor_hidden_layers: List[int] = field(
        default_factory=list,
        # default=[1024, 1024, 512],
        metadata={"help":"actor model hidden layers"}
    )
    critic_hidden_layers: List[int] = field(
        default_factory=list,
        # default=[1024, 1024, 512],
        metadata= {"help":"actor model hidden layers"}
    )
    normalize: bool = field(
        default=False,
    
    )



CONFIG_CLASS = {
    'sac': SACConfig,
    'ppo': PPOConfig
}
