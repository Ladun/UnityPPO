
import numpy as np
import logging
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlagents_envs.environment import UnityEnvironment

from ppo_agent import PPOAgent
from parser import Argument


logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, agent):

    logger.info("************** Start training! ****************")

    for e in range(args.n_episode):

        agent.step()


def main():
    parser = Argument()
    parser.add_model_arguments()
    parser.add_train_arguments()

    args = parser.parse()

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

    # Load models
    env = UnityEnvironment(file_name=args.env_name)

    agent = PPOAgent(env=env)

    train(args, agent)


if __name__ == "__main__":
    main()