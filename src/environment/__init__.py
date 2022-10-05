

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.registry import default_registry
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
           
from src.environment.gym_env import OpenAIGymEnvWrap
from src.environment.unity_env import UnityEnvWrap

def get_env(args):   

    if args.env_type == 'unity':
        engine_config_channel = EngineConfigurationChannel()
        if args.env_name in default_registry.keys():
            env = default_registry[args.env_name].make(side_channels = [engine_config_channel], no_graphics=args.no_graphics)
        else:
            env = UnityEnvironment(file_name=args.env_name, side_channels = [engine_config_channel], no_graphics=args.no_graphics)
        env = UnityEnvWrap(env)    
        engine_config_channel.set_configuration_parameters(time_scale=args.time_scale)
        
        return env
    elif args.env_type == 'gym':        
        env = OpenAIGymEnvWrap(args.env_name)
        
        return env
    else:
        raise NotImplementedError(f"{args.env_type} environment is not implemented")