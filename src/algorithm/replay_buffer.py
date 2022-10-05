
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


# class ReplayBuffer:
#     def __init__(self, 
#                  batch_size, 
#                  data_keys, 
#                  preprocess_func=None):
#         self.data_keys = data_keys
#         self.memory = {key: [] for key in data_keys}
#         self.batch_size = batch_size
#         self.preprocess_func = preprocess_func

#     def add(self, single_trajectory):
        
#         for key in self.data_keys:
#             self.memory[key].extend(single_trajectory[key])

#     def make_batch(self, device):
        
#         processed_memory = self.preprocess_func(self.memory)
        
#         indices = np.arange(len(processed_memory[self.data_keys[0]]))
#         np.random.shuffle(indices)
#         indices = [indices[div * self.batch_size: (div + 1 ) * self.batch_size]
#                    for div in range(len(indices) // self.batch_size + 1)]

#         batch = []
#         for batch_no, sample_ind in enumerate(indices):
#             if len(sample_ind) >= self.batch_size / 2:
#                 values = {key: [] for key in self.data_keys}

#                 i = 0
#                 while i < len(sample_ind):
#                     for key in self.data_keys:
#                         values[key].append(processed_memory[key][sample_ind[i]])
#                     i += 1

#                 # change the format to tensor and make sure dims are correct for calculation
#                 for key in self.data_keys:
#                     values[key] = torch.stack(values[key]).to(device)
                
#                 batch.append([values[key] for key in self.data_keys])

#         return batch
            
#     def reset(self):
#         self.memory = []

#     def __len__(self):
#         return len(self.memory[self.data_keys[0]])
