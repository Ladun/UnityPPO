
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, batch_size):
        self.memory = []
        self.batch_size = batch_size

    def add(self, single_trajectory):
        pass

    def make_batch(self, device):
        pass

    def reset(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)
