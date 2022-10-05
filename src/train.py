
import logging
import os
import argparse
import yaml

from src.algorithm import AGENT_CONSTRUCT_FUNC
from src.environment import get_env
from src.utils.configure import CONFIG_CLASS
from src.utils import set_seed, load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)


def train(args, agent):  
    
    cur_episode_len = load_checkpoint(args, agent)    

    logger.info("************** Start training! ****************")

    while cur_episode_len < args.n_episode:

        agent.step()

        if agent.is_training:        
            agent.logging(cur_episode_len, logger)
            
            cur_episode_len += 1
            
            if (cur_episode_len + 1) % args.save_steps == 0:
                if args.save_dir is not None:
                    save_checkpoint(args, agent, cur_episode_len)
        else:
            logger.info('\rFetching experiences... {} '.format(len(agent.buffer)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-c", default=str, required=True)
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
    env = get_env(args)
    
    # ----------------- Load Models ---------------------- #
    agent = AGENT_CONSTRUCT_FUNC[cfg['algo']](args, env)

    # ----------------- Train! ---------------------- #
    train(args, agent)

    env.env.close()


if __name__ == "__main__":
    main()