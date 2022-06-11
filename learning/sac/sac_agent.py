
import os
import numpy as np
from collections import deque, defaultdict

import torch
import torch.nn.functional as F
import torch.optim

from learning.sac.sac_model import PolicyNet, QNet
from learning.replay_buffer import ReplayBuffer
from learning.agent import *


class SACReplayBuffer(ReplayBuffer):
    def __init__(self, batch_size):
        super().__init__(batch_size)

    def add(self, single_trajectory):        
        (_s, _a, _r, _n_s, _d) = single_trajectory

        for s, a, r, n_s, d in zip(_s, _a, _r, _n_s, _d):
            self.memory.append((s, a, r, n_s, d))

    def make_batch(self, device):
        (all_s, all_a, all_r, all_n_s, all_d) = list(zip(*self.memory))
        assert (len(all_s) == len(self.memory))

        indices = np.arange(len(self.memory))
        np.random.shuffle(indices)
        indices = [indices[div * self.batch_size: (div + 1) * self.batch_size]
                   for div in range(len(indices) // self.batch_size + 1)]

        batch = []
        for batch_no, sample_ind in enumerate(indices):
            if len(sample_ind) >= self.batch_size / 2:
                s_s, s_a, s_r, s_n_s, s_d = ([] for _ in range(5))

                i = 0
                while i < len(sample_ind):
                    s_s.append(all_s[sample_ind[i]])
                    s_a.append(all_a[sample_ind[i]])
                    s_r.append(all_r[sample_ind[i]])
                    s_n_s.append(all_n_s[sample_ind[i]])
                    s_d.append(all_d[sample_ind[i]])
                    i += 1

                # change the format to tensor and make sure dims are correct for calculation
                s_s = torch.stack(s_s).to(device)
                s_a = torch.stack(s_a).to(device)
                s_r = torch.stack(s_r).to(device)
                s_n_r = torch.stack(s_n_s).to(device)
                s_d = torch.tensor(s_d).to(device)

                batch.append((s_s, s_a, s_r, s_n_r, s_d))

        return batch
    

class SACAgent(Agent):
    
    def __init__(self, args, env, 
                 q1: QNet, q1_target: QNet,
                 q2: QNet, q2_target: QNet,
                 pi: PolicyNet):
        super().__init__(args)
        
        self.env = env
        self.buffer = SACReplayBuffer(args.batch_size)
        # Needed states
        # cur_state, action, reward, next_state, done
        
        self.q1 = q1
        self.q2 = q2
        self.q1_target = q1_target
        self.q2_target = q2_target
        self.pi = pi
        
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=0.001)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=0.001)
        self.pi_optimizer = torch.optim.Adam(self.pi.parameters(), lr=0.001)
        
        self.log_alpha = torch.tensor(np.log(0.01))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=0.001)
        
        self.device = args.device
        
        self.gamma = args.gamma
        self.tau = args.tau
        self.T = args.T
        self.T_EPS = args.T_EPS
        
        # for tracking
        self.episodic_rewards = deque(maxlen=1000)
        self.total_steps = deque(maxlen=100)
        self.running_rewards = np.zeros(self.env.agent_n)
        
        self.losses = defaultdict(lambda: deque(maxlen=1000))
        
    def save_checkpoint(self, args, checkpoint_dir, episode_len):
        # save model
        torch.save(self.pi.state_dict(), os.path.join(checkpoint_dir, "actor.pt"))
        critic_dict = {
            "q1": self.q1.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2": self.q2.state_dict(),
            "q2_target": self.q2_target.state_dict(),    
            "log_alpha": self.log_alpha    
        }
        torch.save(critic_dict, os.path.join(checkpoint_dir, "critic.pt"))
        
        # save optimizer
        optimizers = {
            "q1_optimizer": self.q1_optimizer.state_dict(),
            "q2_optimizer": self.q2_optimizer.state_dict(),
            "pi_optimizer": self.pi_optimizer.state_dict(),
            "log_alpha_optimizer": self.log_alpha_optimizer.state_dict(),
        }
        torch.save(optimizers, os.path.join(checkpoint_dir, OPTIMIZER_NAME))
        
        # save args
        torch.save(args, os.path.join(checkpoint_dir, ARG_NAME))
        
        # save training values        
        losses = {k: list(self.losses[k]) for k in self.losses}
        torch.save({
            "episode_len": episode_len,
            "losses": losses
        }, os.path.join(checkpoint_dir, TRAINING_NAME))
        
        
    def load_checkpoint(self, path):
        
        # load model
        self.pi.load_state_dict(torch.load(os.path.join(path, "actor.pt")))
        critic_dict = torch.load(os.path.join(path, "critic.pt"))
        self.q1.load_state_dict(critic_dict['q1'])
        self.q1_target.load_state_dict(critic_dict['q1_target'])
        self.q2.load_state_dict(critic_dict['q2'])
        self.q2_target.load_state_dict(critic_dict['q2_target'])
        self.log_alpha = critic_dict['log_alpha']
        self.log_alpha.requires_grad = True
        
        # load optimizer
        optimizers = torch.load(os.path.join(path, OPTIMIZER_NAME))
        self.q1_optimizer.load_state_dict(optimizers['q1_optimizer'])
        self.q2_optimizer.load_state_dict(optimizers['q2_optimizer'])
        self.pi_optimizer.load_state_dict(optimizers['pi_optimizer'])
        self.log_alpha_optimizer.load_state_dict(optimizers['log_alpha_optimizer'])
        
        # load training values
        checkpoint = torch.load(os.path.join(path, TRAINING_NAME))
        losses = checkpoint['losses']
        for k in losses:
            self.losses[k] = deque(losses[k], maxlen=1000)
        return checkpoint['episode_len']
        
    def inference(self):
        state = self.env.reset()
        while True:
            action, _ = self.pi(torch.from_numpy(state).to(self.device))
            action = np.clip(action.detach().cpu().numpy(), -1., 1.)
            
            # environment step
            dec, term = self.env.step(action)
            
            next_state = np.zeros_like(state, dtype=np.float32)
            
            # Decision steps
            if len(dec) > 0:
                for _id in dec.agent_id:
                    idx = dec.agent_id_to_index[_id]
                    next_state[_id] = dec.obs[0][idx]      
            else:
                # Skip the terminal steps without decision steps
                while self.env.get_num_agents() == 0:
                    empty_action = self.env.empty_action(0)
                    dec, term = self.env.step(empty_action)  
                                         
                    # set next_state
                    for _id in dec.agent_id:
                        idx = dec.agent_id_to_index[_id]
                        next_state[_id] = dec.obs[0][idx]         

            # Change state
            state = next_state
        
    def _collect_trajectories(self): 
        def process_term_steps(traj_for_agent, term, id, _state, _action):
            idx = term.agent_id_to_index[id]
                
            _r = term.reward[idx]
            _n_s = term.obs[0][idx]
            
            # set reward and done
            reward[id] = _r
            done[id] = 1
            
            # Collect trajectories
            if is_collecting:
                traj_for_agent[id].append((_state[id], _action[id], _r, _n_s, 1))
        
        def process_dec_steps(traj_for_agent, dec, id,_state, _action):
            idx = dec.agent_id_to_index[id]
            
            _r = dec.reward[idx]
            _n_s = dec.obs[0][idx]            
            
            # set reward and done
            reward[id] = _r
            done[id] = 0
                        
            # set next_state
            next_state[id] = _n_s
            if id not in term.agent_id:
                # Collect trajectories
                if is_collecting:
                    traj_for_agent[id].append((_state[id], _action[id], _r, _n_s, 0))
                
        
        state = self.env.reset()
        self.running_rewards = np.zeros((self.env.agent_n, 1))

        episode_len = 0
        is_collecting = True
        
        trajectory_for_agents = defaultdict(list)

        temp = True

        while True:
            
            action, dis_action = self.pi(torch.from_numpy(state).to(self.device))
            action = np.clip(action.detach().cpu().numpy(), -1., 1.)
            
            # environment step
            dec, term = self.env.step(action)
            
            reward = np.zeros((self.env.agent_n, 1), dtype=np.float32)
            done = np.zeros((self.env.agent_n, 1), dtype=np.int32)
            next_state = np.zeros_like(state, dtype=np.float32)
            
            # Terminate steps
            for _id in term.agent_id:
                process_term_steps(trajectory_for_agents, term, _id, state, action)
                
            # Decision steps
            if len(dec) > 0:
                for _id in dec.agent_id:
                    process_dec_steps(trajectory_for_agents, dec, _id, state, action)
            else:
                # Skip the terminal steps without decision steps
                while self.env.get_num_agents() == 0:
                    empty_action = self.env.empty_action(0)
                    dec, term = self.env.step(empty_action)   
                    
                    for _id in term.agent_id:
                        process_term_steps(trajectory_for_agents, term, _id, state, action)
                    
                    # set next_state
                    for _id in dec.agent_id:
                        idx = dec.agent_id_to_index[_id]
                        next_state[_id] = dec.obs[0][idx]  
                        
                
            # Collect rewards for tracking
            if not np.any(np.isnan(reward)):
                self.running_rewards += reward
            else:
                self.running_rewards += self.nan_penalty
                print("nan reward encountered!")

            # Change state
            state = next_state

            if episode_len >= self.T:
                if is_collecting:
                    is_collecting = False

                if np.any(done) or episode_len >= self.T_EPS:
                    agents_mean_eps_reward = np.nanmean(self.running_rewards + 1e-10)
                    self.episodic_rewards.append(agents_mean_eps_reward)
                    self.total_steps.append(episode_len)
                    break

            episode_len += 1

        return trajectory_for_agents
    
    def _learn(self):
        def train_pi(pi, q1, q2, batch):
            s, _, _, _, _ = batch
            a, log_prob = pi(s)
            entropy = -self.log_alpha.exp() * log_prob

            q1_val, q2_val = q1(s,a), q2(s,a)
            q1_q2 = torch.cat([q1_val, q2_val], dim=1)
            min_q = torch.min(q1_q2, 1, keepdim=True)[0]

            loss = -min_q - entropy # for gradient ascent
            self.pi_optimizer.zero_grad()
            loss.mean().backward()
            self.pi_optimizer.step()

            self.log_alpha_optimizer.zero_grad()
            alpha_loss = -(self.log_alpha.exp() * (log_prob + -1.0).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        
        def train_Q(q, optimizer, td_target, batch):
            s, a, r, s_prime, done = batch
            loss = F.smooth_l1_loss(q(s, a) , td_target)
            
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
        
        def soft_update_Q(q, q_target):
            for param_target, param in zip(q_target.parameters(), q.parameters()):
                param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
        
        def calc_target(pi, q1_target, q2_target, batch):
            s, a, r, s_prime, done = batch
            
            with torch.no_grad():
                a_prime, log_prob = pi(s_prime)
                entropy = -self.log_alpha.exp() * log_prob
                q1_val, q2_val = q1_target(s_prime, a_prime), q2_target(s_prime, a_prime)
                q1_q2 = torch.cat([q1_val, q2_val], dim=1)
                min_q = torch.min(q1_q2, 1, keepdim=True)[0]
                # Temp code
                entropy = torch.sum(entropy, dim=-1)
                target = r + self.gamma * done * (min_q + entropy)
                
            return target
    
        batches = self.buffer.make_batch(self.device)
        
        for batch in batches:
            td_target = calc_target(self.pi, self.q1_target, self.q2_target, batch)
            
            train_Q(self.q1, self.q1_optimizer, td_target, batch)
            train_Q(self.q2, self.q2_optimizer, td_target, batch)
            
            train_pi(self.pi, self.q1, self.q2, batch)
            
            soft_update_Q(self.q1, self.q1_target)
            soft_update_Q(self.q2, self.q2_target)
            
    
    def step(self):
        
        trajectories = self._collect_trajectories()
        for id in trajectories:            
            trajectory = [
                self._to_tensor(data) if not isinstance(data[0], torch.Tensor)
                else torch.stack(data)
                for data in list(zip(*trajectories[id]))]
            self.buffer.add(trajectory)
        
        if len(self.buffer) > 10000:
            if not self.is_training:
                print("Prefetch completed. Training starts! \r")
                print("Number of Agents: ", self.env.agent_n)
                print("Device: ", self.device)
                self.is_training = True
                
            self._learn()
            
            self.buffer.reset()
            
    def logging(self, cur_episode_len, logger):
        logger.info("e: {}  score: {:.2f} "
                    "steps: {}  \n\t\t\t\t".format(
                        cur_episode_len + 1,
                        np.mean(self.episodic_rewards),
                        int(np.mean(self.total_steps)),
                        )
                    )
            

def construct_agent(args, env):
    
    q1_model = QNet(env.state_size, env.action_size, args.critic_hidden_layers, args.normalize)
    q1_target = QNet(env.state_size, env.action_size, args.critic_hidden_layers, args.normalize)
    q2_model = QNet(env.state_size, env.action_size, args.critic_hidden_layers, args.normalize)
    q2_target = QNet(env.state_size, env.action_size, args.critic_hidden_layers, args.normalize)
    
    q1_target.load_state_dict(q1_model.state_dict())
    q2_target.load_state_dict(q2_model.state_dict())
    
    pi_model = PolicyNet(env.state_size, env.action_size, args.actor_hidden_layers, 
                         args.normalize)
    
    q1_model = q1_model.to(args.device)
    q1_target = q1_target.to(args.device)
    q2_model = q2_model.to(args.device)
    q2_target = q2_target.to(args.device)
    pi_model = pi_model.to(args.device)
    
    agent = SACAgent(args, env=env,
                     q1=q1_model, q1_target=q1_target,
                     q2=q2_model, q2_target=q2_target,
                     pi=pi_model)   
    
    return agent