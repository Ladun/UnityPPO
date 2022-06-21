
import numpy as np
import random
import torch
import os
import logging

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
        
def save_checkpoint(args, agent, episode_len):
    checkpoint_dir = os.path.join(args.save_dir, f"episode-{episode_len}")    
        
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
       
    agent.save_checkpoint(args, checkpoint_dir, episode_len)
    
    if args.max_save_limits > 0:
        import shutil
        ckpt_dirs = [t for t in os.listdir(args.save_dir) if t.startswith("episode-")]

        # Remove old ckpts
        if len(ckpt_dirs) >= args.max_save_limits:
            ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split('-')[-1]))
            remove_dirs = ckpt_dirs[: len(ckpt_dirs) - args.max_save_limits]

            for rd in remove_dirs:
                logger.info(f"Remove old checkpoint {rd}")
                shutil.rmtree(os.path.join(args.save_dir, rd))
        
        


def load_checkpoint(args, agent):
    
    ckpt_path=None
    if args.load_path is not None and 'episode-' in args.load_path:
        ckpt_path = args.load_path
    elif os.path.exists(args.load_path):
        ckpt_dirs = [t for t in os.listdir(args.load_path) if t.startswith("episode-")]
        if len(ckpt_dirs) > 0:
            ckpt_path = os.path.join(args.load_path, sorted(ckpt_dirs, key=lambda x: int(x.split('-')[-1]))[-1])
        

    cur_episode_len = 0
    if ckpt_path is not None:
        if os.path.exists(ckpt_path):
            if os.path.isdir(ckpt_path):
                try:
                    cur_episode_len = agent.load_checkpoint(ckpt_path)
                    logger.info(f"Loading checkpoint success, start from episode {cur_episode_len}")
                except Exception as e:
                    logger.info(f"wrong checkpoint path [{ckpt_path}], Exception: {e}")
            else:
                logger.info(f"checkpoint_dir must be directory, {ckpt_path}")
        else:
            logger.info(f"There's no checkpoints in {ckpt_path}, training from scratch")
    else:
        logger.info(f"There's no checkpoints, training from scratch")
            
    return cur_episode_len