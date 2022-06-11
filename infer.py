
import logging
import os
import random
import argparse
import yaml

from environment_wrapper import get_env
from learning import AGENT_CONSTRUCT_FUNC
from utils.configure import CONFIG_CLASS
from utils import set_seed, load_checkpoint

logger = logging.getLogger(__name__)

def infer(args, agent):
      
    # Load checkpoint
    load_checkpoint(args, agent, logger)
            
    agent.inference()
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default=str, required=True)
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
    
    # ----------------- set arguments for infer ---------- #
    args.time_scale = 1
    args.no_graphics = False
    
    # ----------------- Define environments -------------- #
    env = get_env(args)  
    
    # ----------------- Load Models ---------------------- #
    agent = AGENT_CONSTRUCT_FUNC[cfg['algo']](args, env)
    
    # Do inference
    infer(args, agent)


if __name__ == "__main__":
    main()