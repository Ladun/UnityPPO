
import numpy as np
import copy

from mlagents_envs.environment import BaseEnv, ActionTuple


class WrapEnvironment:
    def __init__(self, env: BaseEnv):

        self.env = env
        self.env.reset()

        self.behavior_name = list(self.env.behavior_specs)[0]

        self.agent_n = 0

    def reset(self):
        self.env.reset()

        dec, _ = self.env.get_steps(self.behavior_name)
        self.agent_n = len(dec.obs[0])

        state = np.array([dec.obs[0][i] for i in range(len(dec.obs[0]))])
        # (num_agent, obs_len)
        return copy.deepcopy(state)

    def step(self, action):
        if self.agent_n == 0:
            print("agent_n is 0")
            return

        action_tuple = ActionTuple()
        action_tuple.add_continuous(action)

        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()

        dec, term = self.env.get_steps(self.behavior_name)

        done = np.zeros(self.agent_n)
        if len(term.agent_id) > 0:
            done[term.agent_id] = 1

        reward = np.array([term.reward[i] if done[i] else dec.reward[i] for i in range(self.agent_n)])
        next_state = np.array([term.obs[0][i] if done[i] else dec.obs[0][i] for i in range(self.agent_n)])

        # np array
        # (num_agent, obs_len), (num_agent, ), (num_agent, )
        return copy.deepcopy(next_state), reward, done