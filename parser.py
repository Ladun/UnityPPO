
import argparse
import torch


class Argument:

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_argument("--seed", type=int, default=2022)

        # Others
        self.add_argument("--no_cuda", action="store_true")
        # Model arguments
        self.add_argument("--env_name", type=str, required=True)

    def add_model_arguments(self):
        self.add_argument("--actor_hidden_layers", nargs='+', type=int, default=[1024, 1024, 512],
                          help="actor model hidden layers")
        self.add_argument("--critic_hidden_layers", nargs='+', type=int, default=[1024, 1024, 512],
                          help="critic model hidden layers")

    def add_train_arguments(self):
        # Train arguments
        self.add_argument("--K_epoch", type=int, default=3,
                          help=". (Recommended Range: 3 to 30)")
        self.add_argument("--n_episode", type=int, default=1000)
        self.add_argument("--batch_size", type=int, default=1024,
                          help="batch size of sampling. (Recommended Range: 4 to 4096)")
        self.add_argument("--buffer_size", type=int, default=30,
                          help="min no of batches needed in the memory before learning")
        self.add_argument("--gamma", type=float, default=0.99,
                          help="discount factor. (Recommended Range: 0.8 to 0.997)")
        self.add_argument("--lmbda", type=float, default=0.95,
                          help="value control how much agent rely on current estimate. (Recommended Range: 0.9 to 1)")
        self.add_argument("--eps_clip", type=float, default=0.1,
                          help="eps for ratio clip 1+eps, 1-eps. (Recommended Range: 0.1, 0.2, 0.3)")
        self.add_argument("--T", type=int, default=512,
                          help="max number of time step for collecting trajectory")
        self.add_argument("--T_EPS", type=int, default=int(3e4),
                          help="max number of time step for collecting trajectory")
        self.add_argument("--learning_rate", type=float, default=1e-4,
                          help="learning rate. (Recommended Range: 3e-3 to 5e-6)")
        self.add_argument("--critic_loss_weight", type=float, default=1.0,
                          help="mean square error term weight. (Recommended Range: 0.5, 1)")
        self.add_argument("--nan_penalty", type=float, default=-5.0,
                          help="penalty for actions that resulted in nan reward")
        self.add_argument("--loss_type", type=str, default="clip",
                          help="loss_type, 'clip', 'kl', 'none'")

        # Entropy
        self.add_argument("--entropy_weight", type=float, default=0.01,
                          help="weight of entropy added. (Recommended Range: 0 to 0.01)")
        self.add_argument("--entropy_decay", type=float, default=0.995,
                          help="decay of entropy per 'step'")

        # action std scale
        self.add_argument("--std_scale_init", type=float, default=1.0,
                          help="initial value of std scale for action resampling")
        self.add_argument("--std_scale_decay", type=float, default=0.995,
                          help=" scale decay of std")

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def parse(self):
        args = self.parser.parse_args()
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()

        return args