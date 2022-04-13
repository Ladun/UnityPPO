import numpy as np
from collections import deque, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from environment_wrapper import WrapEnvironment


class PPOAgent:
    def __init__(self,
                 args,
                 env: WrapEnvironment,
                 model):
        self.env = env
        self.model = model
        self.buffer = ReplayBuffer(args.batch_size, self.env.agent_n)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)

        self.device = args.device

        self.K_epoch = args.K_epoch
        self.T = args.T
        self.T_EPS = args.T_EPS
        self.gamma = args.gamma
        self.lmbda = args.lmbda
        self.nan_penalty = args.nan_penalty
        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size
        self.eps_clip = args.eps_clip
        self.critic_loss_weight = args.critic_loss_weight
        self.entropy_weight = args.entropy_weight
        self.entropy_decay = args.entropy_decay
        self.std_scale = args.std_scale_init
        self.std_scale_decay = args.std_scale_decay

        self.is_training = False

        # for tracking
        self.episodic_rewards = deque(maxlen=1000)
        self.total_steps = deque(maxlen=100)
        self.running_rewards = np.zeros(self.env.agent_n)
        
        self.losses = defaultdict(lambda: deque(maxlen=1000))
        # {
        #     "total_loss": deque(maxlen=1000),
        #     "critic_loss": deque(maxlen=1000),
        #     "actor_loss": deque(maxlen=1000),
        #     "entropy": deque(maxlen=1000)
        # }

    def _to_tensor(self, s, dtype=torch.float32):
        return torch.tensor(s, dtype=dtype, device=self.device)

    def _collect_trajectory_data(self, remain):

        state = self.env.reset()
        self.running_rewards = np.zeros((self.env.agent_n, 1))

        episode_len = 0
        trajectory = []
        is_collecting = True

        while True:
            actor = self.model.actor(torch.from_numpy(state).to(self.device), std_scale=self.std_scale)
            action, dis_action = actor["action"], actor["log_prob"]
            action = np.clip(action.detach().numpy(), -1., 1.)
            next_state, reward, done = self.env.step(action)

            # Collect rewards for tracking
            if not np.any(np.isnan(reward)):
                self.running_rewards += reward
            else:
                self.running_rewards += self.nan_penalty
                print("nan reward encountered!")

            # Collect trajectories
            if is_collecting:
                trajectory.append((state, action, reward, next_state, dis_action.detach(), done))
            # Change state
            state = next_state

            if episode_len >= remain or np.any(done):
                if is_collecting:
                    is_collecting = False

                if np.any(done) or episode_len >= self.T_EPS:
                    agents_mean_eps_reward = np.nanmean(self.running_rewards + 1e-10)
                    self.episodic_rewards.append(agents_mean_eps_reward)
                    self.total_steps.append(episode_len)
                    break

            episode_len += 1

        return trajectory

    def _calculate_advantage(self, trajectory):

        trajectory = [
            self._to_tensor(data).transpose(1, 0) if type(data[0]) is np.ndarray
            else torch.stack(data).transpose(1, 0)
            for data in list(zip(*trajectory))]

        num_agents = trajectory[0].size()[0]
        ended_idx = []
        for i in range(num_agents):
            _, _, _, _, _, done = [data[i] for data in trajectory]
            
            if done.any():
                ended_idx.append(i)
                
        if len(ended_idx) == 0:
            ended_idx = [i for i in range(num_agents)]
        
        # Calculate the advantage of each agent
        all_adavantages = []
        all_returns = []
        for i in ended_idx:
            # state size: (len_trajectory, state_size)
            # action size: (len_trajectory, action_size)
            # reward size: (len_trajectory, )
            # next_state size: (len_trajectory, state_size)
            # dis_action size: (len_trajectory, )
            # done size: (len_trajectory, )
            state, action, reward, next_state, dis_action, done = [data[i] for data in trajectory]
            
            with torch.no_grad():  
                # 1 ~ n, n + 1 trajectories          
                values = self.model.critic(torch.cat((state, next_state[-1].unsqueeze(0)), dim=0))
            
            # Calc all returns and advantages
            returns = values[-1]
            advantage = 0.0
            advantage_list = []
            return_list = []
            for i in reversed(range(len(state))):
                # calculate v_t(s)
                returns = reward[i] + self.gamma * returns * (1 - done[i])
                
                # calculate advatange
                td_error = reward[i] + self.gamma * values[i + 1] - values[i]
                advantage = advantage * self.lmbda * self.gamma * (1 - done[i]) + td_error
                
                advantage_list.append([advantage])
                return_list.append([returns])
            advantage_list.reverse()
            return_list.reverse()
            
            # Collect agent_i's adavantage and returns
            all_adavantages.append(advantage_list)
            all_returns.append(return_list)

        all_adavantages = self._to_tensor(all_adavantages).transpose(0, 1)
        all_returns = self._to_tensor(all_returns).transpose(0, 1)
        state, action, reward, _, dis_action, _ = [data[ended_idx].transpose(0, 1) for data in trajectory]

        return (state, action, reward, dis_action, all_returns, all_adavantages)

    def _train(self):
        batch = self.buffer.make_batch(self.device)

        for _ in range(self.K_epoch):
            for (state, old_action, reward, old_prob, returns, advantage) in batch:
                actor = self.model.actor(state, std_scale=self.std_scale)
                new_prob = actor['log_prob']
                entropy = actor['entropy']
                assert new_prob.requires_grad == True
                assert advantage.requires_grad == False
                assert returns.requires_grad == False
                assert old_prob.requires_grad == False

                # Actor loss
                ratio = (new_prob - old_prob).exp()
                G = ratio * advantage
                G_clip = torch.clamp(ratio, min=1.0 - self.eps_clip, max=1.0 + self.eps_clip) * advantage
                clip_loss = torch.min(G, G_clip).mean()
                actor_loss = -(clip_loss + self.entropy_weight * entropy)

                # Critic loss
                critic_loss = F.smooth_l1_loss(self.model.critic(state), returns)

                # Total loss
                total_loss = actor_loss + self.critic_loss_weight * critic_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                self.losses['total_loss'].append(total_loss.item())
                self.losses['critic_loss'].append(critic_loss.item())
                self.losses['actor_loss'].append(-clip_loss.item())
                self.losses['entropy'].append(entropy.item())
                self.losses['ratio'].append(ratio.detach().numpy().mean())
                self.losses['new_p'].append(new_prob.detach().numpy().mean())
                self.losses['old_p'].append(old_prob.detach().numpy().mean())
                self.losses['max_ratio'].append(torch.max(ratio).item())
                self.losses['min_ratio'].append(torch.min(ratio).item())

    def step(self):        
        self.model.train()
        
        remain = self.T
        while remain > 0:
            trajectory = self._collect_trajectory_data(remain)
            output = self._calculate_advantage(trajectory)
            
            trajectory_with_advantage = output
            self.buffer.add(trajectory_with_advantage)

            remain -= len(trajectory_with_advantage)

        if len(self.buffer) >= self.buffer_size * self.batch_size:
            if not self.is_training:
                print("Prefetch completed. Training starts! \r")
                print("Number of Agents: ", self.env.agent_n)
                print("Device: ", self.device)
                self.is_training = True

            # Train
            self._train()

            # std decay
            self.std_scale *= self.std_scale_decay
            # entropy weight decay
            self.entropy_weight *= self.entropy_decay

            # Reset replay buffer
            self.buffer.reset()


class ReplayBuffer:
    def __init__(self, batch_size, num_agents):
        self.memory = []
        self.batch_size = batch_size
        self.num_agents = num_agents

    def add(self, single_trajectory):
        (_s, _a, _r, _prob, _rt, _adv) = single_trajectory

        for s, a, r, prob, rt, adv in zip(_s, _a, _r, _prob, _rt, _adv):

            for i in range(len(s)):
                self.memory.append((s[i, :], a[i, :], r[i, :], prob[i, :], rt[i, :], adv[i, :]))

    def make_batch(self, device):
        (all_s, all_a, all_r, all_prob, all_rt, all_adv) = list(zip(*self.memory))
        assert (len(all_s) == len(self.memory))

        # so that we can normalized Advantage before sampling
        all_adv = tuple([adv.numpy() for adv in all_adv])
        all_adv = tuple((all_adv - np.nanmean(all_adv)) / np.std(all_adv))

        indices = np.arange(len(self.memory))
        np.random.shuffle(indices)
        indices = [indices[div * self.batch_size: (div + 1) * self.batch_size]
                   for div in range(len(indices) // self.batch_size + 1)]

        batch = []
        for batch_no, sample_ind in enumerate(indices):
            if len(sample_ind) >= self.batch_size / 2:
                s_s, s_a, s_r, s_prob, s_rt, s_adv = ([] for _ in range(6))

                i = 0
                while i < len(sample_ind):
                    s_s.append(all_s[sample_ind[i]])
                    s_a.append(all_a[sample_ind[i]])
                    s_r.append(all_r[sample_ind[i]])
                    s_prob.append(all_prob[sample_ind[i]])
                    s_rt.append(all_rt[sample_ind[i]])
                    s_adv.append(all_adv[sample_ind[i]])
                    i += 1

                # change the format to tensor and make sure dims are correct for calculation
                s_s = torch.stack(s_s).to(device)
                s_a = torch.stack(s_a).to(device)
                s_r = torch.stack(s_r).to(device)
                s_prob = torch.stack(s_prob).to(device)
                s_rt = torch.stack(s_rt).to(device)
                s_adv = torch.tensor(s_adv).to(device)

                batch.append((s_s, s_a, s_r, s_prob, s_rt, s_adv))

        return batch

    def reset(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)
