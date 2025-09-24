import sys
from pathlib import Path

# Add the root directory of the repository to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from train import train
import config

def train_def():
    train()

def train_mult_nets():
    for i in range(10):
        hp = {
            "save_dir": f"checkpoints/social_mrnn_{i}",
            "model_save_name": f"social_mrnn_{i}",
        }
        train(hp=hp)

if __name__ == "__main__":
    
    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()

    if args.experiment == "train_def":
        train_def()
    elif args.experiment == "train_mult_nets":
        train_mult_nets()
    else:
        raise NotImplementedError(f"Experiment {args.experiment} not implemented")