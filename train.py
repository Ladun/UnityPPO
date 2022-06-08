
from unittest import loader
import numpy as np
import logging
import os
import random
import argparse
import yaml

import torch

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.registry import default_registry
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from learning import AGENT_CONSTRUCT_FUNC
from environment_wrapper import WrapEnvironment
from utils.configure import CONFIG_CLASS

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, agent):
    
    
    cur_episode_len = 0    
    if args.checkpoint_dir is not None:
        if os.path.exists(args.checkpoint_dir):
            if os.path.isdir(args.checkpoint_dir):
                try:
                    cur_episode_len = agent.load_checkpoint(args)
                except Exception as e:
                    logger.info(f"wrong checkpoint path, Exception: {e}")
            else:
                logger.info("checkpoint_dir must be directory")
        else:
            logger.info("There's no checkpoints, training from scratch")
    

    logger.info("************** Start training! ****************")

    while cur_episode_len < args.n_episode:

        agent.step()

        if agent.is_training:        
            agent.logging(cur_episode_len, logger)
            
            cur_episode_len += 1
            
            if (cur_episode_len + 1) % args.save_steps == 0:
                if args.checkpoint_dir is not None:
                    agent.save_checkpoint(args, cur_episode_len)
        else:
            logger.info('\rFetching experiences... {} '.format(len(agent.buffer)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default=str, required=True)
    # parser.add_env_arguments()
    # parser.add_model_arguments()
    # parser.add_train_arguments()

    _args = parser.parse_args()
    
    # Load config
    with open(_args.config_file, mode='r', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    args = CONFIG_CLASS[cfg['algo']](**cfg)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Set seed
    set_seed(args)

    logger.info('<Parsed arguments>')
    for k, v in vars(args).items():
        logger.info('{}: {}'.format(k, v))
    logger.info('')

    # ----------------- Define environments ---------------------- #
    engine_config_channel = EngineConfigurationChannel()
    if args.env_name in default_registry.keys():
        env = default_registry[args.env_name].make(side_channels = [engine_config_channel], no_graphics=args.no_graphics)
    else:
        env = UnityEnvironment(file_name=args.env_name, side_channels = [engine_config_channel], no_graphics=args.no_graphics)
    env = WrapEnvironment(env)    
    engine_config_channel.set_configuration_parameters(time_scale=args.time_scale)
    
    # ----------------- Load Models ---------------------- #
    agent = AGENT_CONSTRUCT_FUNC[cfg['algo']](args, env)

    # ----------------- Train! ---------------------- #
    train(args, agent)

    env.env.close()


if __name__ == "__main__":
    main()