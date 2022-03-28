
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def weights_init_lim(layer):
    # similar to Xavier initialization except it's
    # dimension of the input layer
    input_dim = layer.weight.data.size()[0]
    lim = 1./np.sqrt(input_dim)
    return (-lim, lim)


class PPOActor(nn.Module):
    """
        Actor: input state (array), convert into action. Based on that
               action create a prob distribution. Based on that distribution
               resample another action. Output the resampled action and prob dist.
               Lastly an entropy term is created for exploration.
    """
    def __init__(self, state_size, action_size, hidden_layers):
        super(PPOActor, self).__init__()

        # input size: batch_size or num_agents x state_size

        # parametric relu
        self.prelu = nn.PReLU()

        # Define layers
        in_channels = state_size
        self.blocks = nn.Sequential()
        for idx, hidden_layer in enumerate(hidden_layers):
            self.blocks.add_module(f"bn{idx}", nn.BatchNorm1d(in_channels))
            self.blocks.add_module(f"linear{idx}", nn.Linear(in_channels, hidden_layer))
            self.blocks.add_module(f"prelu{idx}", self.prelu)
            in_channels = hidden_layer
        self.blocks.add_module(f"bn_last", nn.BatchNorm1d(in_channels))
        self.blocks.add_module(f"linear_last", nn.Linear(in_channels, action_size))
        self.blocks.add_module(f"activation", nn.Tanh())

        # std of the distribution for the resampled action
        self.std = nn.Parameter(torch.ones(1, action_size)*0.15)

    def reset_parameters(self):
        # initialize the values
        for name, layer in self.blocks.named_children():
            if name.startswith("linear"):
                if name.endswith("last"):
                    layer.weight.data.uniform_(-1e-3, 1e-3)
                else:
                    layer.weight.data.uniform_(*weights_init_lim(layer))

    def forward(self, s, resampled_action=None, std_scale=1.0):
        action_mean = self.blocks(s)

        # action_mean: proposed action, we will then use this action as
        # mean to generate a prob distribution to output log_prob
        # base on the action as mean create a distribution with zero std...
        dist = torch.distributions.Normal(action_mean, F.hardtanh(self.std,
                                                                  min_val=0.05*std_scale,
                                                                  max_val=0.5*std_scale))

        # sample from the prob distribution just generated again
        if resampled_action is None:
            # num_agent/batch_size x action_size
            resampled_action = dist.sample()

        # then we have log( p(resampled_action | state) ): batch_size, 1
        log_prob = dist.log_prob(resampled_action)
        # entropy for noise
        entropy = dist.entropy().mean()

        # final output
        return {
            "log_prob": log_prob,
            "action_mean": action_mean,
            "action": resampled_action,
            "entropy": entropy
        }


class PPOCritic(nn.Module):
    
    def __init__(self, state_size, action_size, hidden_layers):
        super(PPOCritic, self).__init__()

        # parametric relu
        self.prelu = nn.PReLU()

        # Define layers
        in_channels = state_size
        self.blocks = nn.Sequential()
        for idx, hidden_layer in enumerate(hidden_layers):
            self.blocks.add_module(f"bn{idx}", nn.BatchNorm1d(in_channels))
            self.blocks.add_module(f"linear{idx}", nn.Linear(in_channels, hidden_layer))
            self.blocks.add_module(f"prelu{idx}", self.prelu)
            in_channels = hidden_layer
        self.blocks.add_module(f"bn_last", nn.BatchNorm1d(in_channels))
        self.blocks.add_module(f"linear_last", nn.Linear(in_channels, 1))
        self.blocks.add_module(f"activation", nn.Tanh())

    def reset_parameters(self):
        # initialize the values
        for name, layer in self.blocks.named_children():
            if name.startswith("linear"):
                if name.endswith("last"):
                    layer.weight.data.uniform_(-1e-3, 1e-3)
                else:
                    layer.weight.data.uniform_(*weights_init_lim(layer))

    def forward(self, state):
        v = self.blocks(state)

        return v


class PPOActorCritic(nn.Module):
    def __init__(self,
                 state_size, action_size,
                 actor_hidden_layers, critic_hidden_layers):
        super(PPOActorCritic, self).__init__()
        self._actor = PPOActor(state_size, action_size, actor_hidden_layers)
        self._critic = PPOCritic(state_size, action_size, critic_hidden_layers)

    def actor(self, state, resampled_action=None, std_scale=1.0):
        return self._actor(state, resampled_action, std_scale)

    def critic(self, state):
        return self._critic(state)

