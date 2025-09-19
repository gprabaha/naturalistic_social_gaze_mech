import sys
from pathlib import Path

# Add the root directory of the repository to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from train import train
import config

def train_def():
    train()

if __name__ == "__main__":
    
    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()

    if args.experiment == "train_def":
        train_def()
    else:
        raise NotImplementedError(f"Experiment {args.experiment} not implemented")