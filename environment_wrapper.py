
import copy

import numpy as np
from mlagents_envs.environment import ActionTuple, BaseEnv


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
        
        return dec, term

        done = np.zeros((self.agent_n, 1), dtype=np.int32)
        if len(term.agent_id) > 0:
            done[term.agent_id] = 1

        reward = np.zeros((self.agent_n, 1), dtype=np.float32)
        next_state = np.zeros((self.agent_n, self.state_size), dtype=np.float32)
        
        for id in term.agent_id:
            idx = term.agent_id_to_index[id]
            reward[id] = term.reward[idx]
            next_state[id] = term.obs[0][idx]
        
        ended = np.any(done)
        for id in dec.agent_id:
            if id not in term.agent_id:
                idx = dec.agent_id_to_index[id]
                reward[id] = dec.reward[idx] if not ended else term.reward[0]
                next_state[id] = dec.obs[0][idx]
            
        # np array
        # (num_agent, obs_len), (num_agent, ), (num_agent, )
        return copy.deepcopy(next_state), reward, done

    def get_num_agents(self):
        return len(self.env._env_state[self.behavior_name][0])
    
    def empty_action(self, n_agents : int):
        # _continuous = np.zeros((n_agents, self.action_size), dtype=np.float32)
        #_discrete = np.zeros((n_agents, self.discrete_size), dtype=np.int32)
        # return ActionTuple(continuous=_continuous)
        _continuous = np.zeros((n_agents, self.action_size), dtype=np.float32)
        
        return _continuous
    
def get_env(args):   

    from mlagents_envs.environment import UnityEnvironment
    from mlagents_envs.registry import default_registry
    from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
    
    engine_config_channel = EngineConfigurationChannel()
    if args.env_name in default_registry.keys():
        env = default_registry[args.env_name].make(side_channels = [engine_config_channel], no_graphics=args.no_graphics)
    else:
        env = UnityEnvironment(file_name=args.env_name, side_channels = [engine_config_channel], no_graphics=args.no_graphics)
    env = WrapEnvironment(env)    
    engine_config_channel.set_configuration_parameters(time_scale=args.time_scale)
    
    return env