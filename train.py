
import numpy as np
import logging
import random

import torch

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.registry import default_registry

from ppo_agent import PPOAgent
from ppo_model import PPOActorCritic
from environment_wrapper import WrapEnvironment
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

    mean_rewards = []
    e = 0
    while e < args.n_episode:

        agent.step()
        episode_reward = agent.episodic_rewards

        if agent.is_training:
            mean_rewards.append(np.mean(episode_reward))

            print("e: {}  score: {:.2f}  Avg score(100e): {:.2f}  "
                  "std: {:.2f}  steps: {}".format(e + 1, np.mean(episode_reward),
                                                  np.mean(mean_rewards[-100:]),
                                                  agent.std_scale,
                                                  int(np.mean(agent.total_steps))))
            e += 1
        else:
            print('\rFetching experiences... {} '.format(len(agent.buffer)), end="")


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
    if args.env_name in default_registry.keys():
        env = default_registry[args.env_name].make()
    else:
        env = UnityEnvironment(file_name=args.env_name)
    env = WrapEnvironment(env)

    model = PPOActorCritic(
        state_size=env.state_size, action_size=env.action_size,
        actor_hidden_layers=args.actor_hidden_layers,
        critic_hidden_layers=args.critic_hidden_layers
    )
    agent = PPOAgent(args, env=env, model=model)

    train(args, agent)

    env.env.close()


if __name__ == "__main__":
    main()