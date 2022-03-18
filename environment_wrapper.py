
import numpy as np
import copy

from mlagents_envs.environment import BaseEnv, ActionTuple


class WrapEnvironment:
    def __init__(self, env: BaseEnv):

        self.env = env

        # Get agent num
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs)[0]

        dec, _ = self.env.get_steps(self.behavior_name)
        self.agent_n = len(dec.obs[0])

        # Get action size and state size
        spec = self.env.behavior_specs[self.behavior_name]
        self.action_size = spec.action_spec.continuous_size
        self.state_size = spec.observation_specs[0].shape[0]

    def reset(self):
        self.env.reset()

        dec, _ = self.env.get_steps(self.behavior_name)

        state = np.array([dec.obs[0][i] for i in range(len(dec.obs[0]))], dtype=np.float32)

        # np array
        # (num_agent, obs_len)
        return copy.deepcopy(state)

    def step(self, action):

        action_tuple = ActionTuple()
        action_tuple.add_continuous(action)

        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

        dec, term = self.env.get_steps(self.behavior_name)

        done = np.zeros((self.agent_n, 1), dtype=np.int32)
        if len(term.agent_id) > 0:
            done[:] = 1

        reward = np.zeros((self.agent_n, 1), dtype=np.float32)
        next_state = np.zeros((self.agent_n, self.state_size), dtype=np.float32)
        term_idx = 0
        for i in range(self.agent_n):
            if done[i]:
                reward[i] = term.reward[term_idx]
                next_state[i] = term.obs[0][term_idx]
                term_idx += 1
            else:
                if not np.any(done):
                    reward[i] = dec.reward[i]
                    next_state[i] = dec.obs[0][i]

        # np array
        # (num_agent, obs_len), (num_agent, ), (num_agent, )
        return copy.deepcopy(next_state), reward, done
