
import numpy as np

import torch.nn as nn
import torch
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self, 
                 state_size, 
                 action_size,
                 hidden_layers, 
                 normalize=False):
        super().__init__()
        
        # parametric relu
        self.prelu = nn.ReLU()
        
        in_channels = state_size
        self.blocks = nn.Sequential()
        for idx, hidden_layer in enumerate(hidden_layers):
            if normalize:
                self.blocks.add_module(f"bn{idx}", nn.BatchNorm1d(in_channels))
            self.blocks.add_module(f"linear{idx}", nn.Linear(in_channels, hidden_layer))
            self.blocks.add_module(f"prelu{idx}", self.prelu)
            in_channels = hidden_layer
            
        self.linear_mu = nn.Linear(in_channels, action_size)
        self.linear_std = nn.Linear(in_channels, action_size)
        
    def forward(self, state):
        hidden = self.blocks(state)
        
        mu = self.linear_mu(hidden)
        std = F.softplus(self.linear_std(hidden))
        
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()
        
        log_prob = dist.log_prob(action)
        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        return real_action, real_log_prob

        
class QNet(nn.Module):
    def __init__(self, 
                 state_size,
                 action_size ,
                 hidden_layers,
                 normalize=False):
        super().__init__()
        
        # parametric relu
        self.prelu = nn.ReLU()
        
        self.state_embedding = nn.Linear(state_size, hidden_layers[0] // 2)
        self.action_embedding = nn.Linear(action_size, hidden_layers[0] // 2)

        # Define layers
        in_channels = hidden_layers[0]
        self.blocks = nn.Sequential()
        for idx, hidden_layer in enumerate(hidden_layers):
            if normalize:
                self.blocks.add_module(f"bn{idx}", nn.BatchNorm1d(in_channels))
            self.blocks.add_module(f"linear{idx}", nn.Linear(in_channels, hidden_layer))
            self.blocks.add_module(f"prelu{idx}", self.prelu)
            in_channels = hidden_layer
        if normalize:
            self.blocks.add_module(f"bn_last", nn.BatchNorm1d(in_channels))
        self.blocks.add_module(f"linear_last", nn.Linear(in_channels, 1))

    def forward(self, state, action):
        state_emb  = self.state_embedding(state)
        action_emb = self.action_embedding(action)
        
        hidden = torch.cat((state_emb, action_emb), dim=-1)
        
        v = self.blocks(hidden)

        return v

        
        