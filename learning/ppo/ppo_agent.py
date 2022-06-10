import numpy as np
import os
from collections import deque, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from environment_wrapper import WrapEnvironment
from learning.replay_buffer import ReplayBuffer
from learning.ppo.ppo_model import PPOActorCritic
from learning.agent import Agent

MODEL_NAME = "model.pt"
OPTIMIZER_NAME = "optimizer.pt"
ARG_NAME = "train_args.bin"
TRAINING_NAME = "train_values.bin"


class PPOReplayBuffer(ReplayBuffer):
    def __init__(self, batch_size):
        super().__init__(batch_size)

    def add(self, single_trajectory):
        (_s, _a, _r, _prob, _rt, _adv) = single_trajectory

        for s, a, r, prob, rt, adv in zip(_s, _a, _r, _prob, _rt, _adv):
            self.memory.append((s, a, r, prob, rt, adv))

    def make_batch(self, device):
        (all_s, all_a, all_r, all_prob, all_rt, all_adv) = list(zip(*self.memory))
        assert (len(all_s) == len(self.memory))

        # so that we can normalized Advantage before sampling
        all_adv = tuple([adv.detach().cpu().numpy() for adv in all_adv])
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


class PPOAgent(Agent):
    def __init__(self,
                 args,
                 env: WrapEnvironment,
                 model: PPOActorCritic):
        super().__init__(args)
        
        self.env = env
        self.model = model
        self.model.to(args.device)
        self.buffer = PPOReplayBuffer(args.batch_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)

        self.device = args.device

        self.loss_type = args.loss_type
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

        # for tracking
        self.episodic_rewards = deque(maxlen=1000)
        self.total_steps = deque(maxlen=100)
        self.running_rewards = np.zeros(self.env.agent_n)
        
        self.losses = defaultdict(lambda: deque(maxlen=1000))
        
    def save_checkpoint(self, args, episode_len):
        
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
       
        # save model
        torch.save(self.model.state_dict(), os.path.join(args.checkpoint_dir, MODEL_NAME))
        # save optimizer
        torch.save(self.optimizer.state_dict(), os.path.join(args.checkpoint_dir, OPTIMIZER_NAME))
        # save args
        torch.save(args, os.path.join(args.checkpoint_dir, ARG_NAME))
        # save training values
        
        losses = {k: list(self.losses[k]) for k in self.losses}
        torch.save({
            "episode_len": episode_len,
            "losses": losses
        }, os.path.join(args.checkpoint_dir, TRAINING_NAME))
        
        
    def load_checkpoint(self, args):
        
        # load model
        self.model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, MODEL_NAME)))
        # load optimizer
        self.optimizer.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, OPTIMIZER_NAME)))
        # load training values
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, TRAINING_NAME))
        losses = checkpoint['losses']
        for k in losses:
            self.losses[k] = deque(losses[k], maxlen=1000)
        return checkpoint['episode_len']
        

    def _collect_trajectory_data(self):
        
        def process_term_steps(traj_for_agent, term, id, _state, _action):
            idx = term.agent_id_to_index[id]
                
            _r = term.reward[idx]
            _n_s = term.obs[0][idx]
            
            # set reward and done
            reward[id] = _r
            done[id] = 1
            
            # Collect trajectories
            if is_collecting:
                traj_for_agent[id].append((_state[id], _action[id], _r, _n_s, dis_action[id].detach(), 1))
        
        def process_dec_steps(traj_for_agent, dec, id, _state, _action):
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
                    traj_for_agent[id].append((_state[id], _action[id], _r, _n_s, dis_action[id].detach(), 0))
                
        
        state = self.env.reset()
        self.running_rewards = np.zeros((self.env.agent_n, 1))

        episode_len = 0
        is_collecting = True
        
        trajectory_for_agents = defaultdict(list)

        while True:
            
            actor = self.model.actor(torch.from_numpy(state).to(self.device), std_scale=self.std_scale)
            action, dis_action = actor["action"], actor["log_prob"]
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

    def _calculate_advantage(self, trajectory):
        
        trajectory = [
            self._to_tensor(data) if not isinstance(data[0], torch.Tensor)
            else torch.stack(data)
            for data in list(zip(*trajectory))]
        
        # state size: (len_trajectory, state_size)
        # action size: (len_trajectory, action_size)
        # reward size: (len_trajectory, )
        # next_state size: (len_trajectory, state_size)
        # dis_action size: (len_trajectory, )
        # done size: (len_trajectory, )
        state, action, reward, next_state, dis_action, done = trajectory
        

        # Calculcate all returns and advantages
        all_adavantages = []
        all_returns = []
        
        with torch.no_grad():  
            # 1 ~ n, n + 1 trajectories          
            values = self.model.critic(torch.cat((state, next_state[-1].unsqueeze(0)), dim=0))
        
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
        
        all_adavantages = self._to_tensor(advantage_list)
        all_returns = self._to_tensor(return_list)  
            
        return (state, action, reward, dis_action, all_returns, all_adavantages)

    def _train(self):
        batch = self.buffer.make_batch(self.device)

        for _ in range(self.K_epoch):
            for (state, old_action, reward, old_prob, returns, advantage) in batch:
                actor = self.model.actor(state, old_action, std_scale=self.std_scale)
                new_prob = actor['log_prob']
                entropy = actor['entropy']
                assert new_prob.requires_grad == True
                assert advantage.requires_grad == False
                assert returns.requires_grad == False
                assert old_prob.requires_grad == False

                # Actor loss
                ratio = (new_prob - old_prob).exp()
                G = ratio * advantage
                
                if self.loss_type == "clip":
                    # if clipping
                    G_clip = torch.clamp(ratio, min=1.0 - self.eps_clip, max=1.0 + self.eps_clip) * advantage
                    actor_loss = torch.min(G, G_clip)
                elif self.loss_type == "kl":                
                    # if KL-div
                    actor_loss = G - 0.01 * torch.exp(old_prob) * (old_prob - new_prob)
                else:
                    actor_loss = G
                    
                actor_loss = actor_loss.mean()
                
                actor_entropy_loss = -(actor_loss + self.entropy_weight * entropy)

                # Critic loss
                critic_loss = F.smooth_l1_loss(self.model.critic(state), returns)

                # Total loss
                total_loss = actor_entropy_loss + self.critic_loss_weight * critic_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                self.losses['total_loss'].append(total_loss.item())
                self.losses['critic_loss'].append(critic_loss.item())
                self.losses['actor_loss'].append(actor_loss.item())
                self.losses['entropy'].append(entropy.item())
                self.losses['ratio'].append(ratio.detach().cpu().numpy().mean())
                self.losses['adv'].append(advantage.detach().cpu().numpy().mean())
                self.losses['new_p'].append(new_prob.detach().cpu().numpy().mean())
                self.losses['old_p'].append(old_prob.detach().cpu().numpy().mean())
                self.losses['max_ratio'].append(torch.max(ratio).item())
                self.losses['min_ratio'].append(torch.min(ratio).item())

    def step(self):        
        self.model.train()
        
        # Collect trajectories and calculate advantages
        trajectory_for_agents = self._collect_trajectory_data() 
        for agent_id in trajectory_for_agents:
            trajectory_with_advantage = self._calculate_advantage(trajectory_for_agents[agent_id])
            self.buffer.add(trajectory_with_advantage)
            
        # learning
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
            
    def logging(self, cur_episode_len, logger):        
            
        logger.info("e: {}  score: {:.2f} "
                    "steps: {}  \n\t\t\t\t"
                    "t_l: {:.4f}  a_l: {:.4f}  c_l: {:.4f}  en: {:.4f}  "
                    "adv: {:.4f}  oldp: {:.4f}  newp: {:.4f}  r: {:.4f} maxr: {:.4f}  minr: {:.4f}  ".format(cur_episode_len + 1, np.mean(self.episodic_rewards),
                                                                int(np.mean(self.total_steps)),
                                                                np.mean(self.losses['total_loss']),
                                                                np.mean(self.losses['actor_loss']),
                                                                np.mean(self.losses['critic_loss']),
                                                                np.mean(self.losses['entropy']),
                                                                np.mean(self.losses['adv']),
                                                                np.mean(self.losses['old_p']),
                                                                np.mean(self.losses['new_p']),
                                                                np.mean(self.losses['ratio']),
                                                                np.mean(self.losses['max_ratio']),
                                                                np.mean(self.losses['min_ratio'])
                                                                ))
            


def construct_agent(args, env):
    model = PPOActorCritic(
        state_size=env.state_size, action_size=env.action_size,
        actor_hidden_layers=args.actor_hidden_layers,
        critic_hidden_layers=args.critic_hidden_layers
    )    
    agent = PPOAgent(args, env=env, model=model)   
    
    return agent