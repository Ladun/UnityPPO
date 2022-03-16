import numpy as np
from collections import deque

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
            steps = self.env.step(action)
            if steps is None:
                agents_mean_eps_reward = np.nanmean(self.running_rewards + 1e-10)
                self.episodic_rewards.append(agents_mean_eps_reward)
                self.total_steps.append(episode_len)
                break
            next_state, reward, done = steps

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

            if episode_len >= remain:
                if is_collecting:
                    is_collecting = False

                if np.all(done == 1) or episode_len >= self.T_EPS:
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

        # Calculate the advantage of each agent
        num_agents = trajectory[0].size()[0]
        all_adavantages = []
        all_returns = []
        for i in range(num_agents):
            # state size: (len_trajectory, state_size)
            # action size: (len_trajectory, action_size)
            # reward size: (len_trajectory, )
            # next_state size: (len_trajectory, state_size)
            # dis_action size: (len_trajectory, )
            # done size: (len_trajectory, )
            state, action, reward, next_state, dis_action, done = [data[i] for data in trajectory]

            with torch.no_grad():
                returns = reward + self.gamma * self.model.critic(next_state) * (1 - done)
                delta = returns - self.model.critic(state)
            all_returns.append(returns)
            delta = delta.numpy()

            advantage_list = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t
                advantage_list.append(advantage)
            advantage_list.reverse()
            all_adavantages.append(advantage_list)

        all_adavantages = self._to_tensor(all_adavantages).transpose(0, 1)
        all_returns = torch.stack(all_returns).transpose(0, 1)
        state, action, reward, _, dis_action, _ = [data.transpose(0, 1) for data in trajectory]

        return state, action, reward, dis_action, all_returns, all_adavantages

    def _train(self):
        batch = self.buffer.make_batch(self.device)

        for _ in range(self.K_epoch):
            for (state, old_action, reward, old_prob, returns, advantage) in batch:
                actor = self.model.actor(state, std_scale=self.std_scale)
                new_prob = actor['log_prob']
                entropy = actor['entropy']
                assert new_prob.requires_grad == True
                assert advantage.requires_grad == False

                # Actor loss
                ratio = (new_prob - old_prob).exp()
                G = ratio * advantage
                G_clip = torch.clamp(ratio, min=1.0 - self.eps_clip, max=1.0 + self.eps_clip) * advantage
                actor_loss = -(torch.min(G, G_clip) + self.entropy_weight * entropy.mean())

                # Critic loss
                critic_loss = F.smooth_l1_loss(self.model.critic(state), returns)

                # Total loss
                total_loss = actor_loss + self.critic_loss_weight * critic_loss

                self.optimizer.zero_grad()
                total_loss.mean().backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

    def step(self):
        remain = self.T
        while remain > 0:
            trajectory = self._collect_trajectory_data(remain)
            trajectory_with_advantage = self._calculate_advantage(trajectory)
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
            i = 0
            while i < self.num_agents:
                self.memory.append((s[i, :], a[i, :], r[i, :], prob[i, :], rt[i, :], adv[i, :]))
                i += 1

    def make_batch(self, device):
        (all_s, all_a, all_r, all_prob, all_rt, all_adv) = list(zip(*self.memory))
        assert (len(all_s) == len(self.memory))

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

                #
                s_s = torch.stack(s_s).to(device)
                s_a = torch.stack(s_a).to(device)
                s_r = torch.stack(s_r).to(device)
                s_prob = torch.stack(s_prob).to(device)
                s_rt = torch.stack(s_rt).to(device)
                s_adv = torch.stack(s_adv).to(device)

                batch.append((s_s, s_a, s_r, s_prob, s_rt, s_adv))

        return batch

    def reset(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)
