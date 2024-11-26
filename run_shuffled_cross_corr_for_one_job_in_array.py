import argparse
import os
import pickle
import pandas as pd
from analyze_data import _compute_crosscorrelations_for_group
import logging

import load_data

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run shuffled cross-correlation for a single group.")
    parser.add_argument('--group_keys', required=True, type=str, help="Group keys as a tuple string.")
    parser.add_argument('--binary_timeseries_file', required=True, type=str, help="Path to binary timeseries file.")
    parser.add_argument('--cross_combinations_file', required=True, type=str, help="Path to cross-combinations file.")
    parser.add_argument('--output_dir', required=True, type=str, help="Output directory.")
    parser.add_argument('--num_cpus', required=True, type=int, help="Number of CPUs per job.")
    parser.add_argument('--shuffle_count', default=50, type=int, help="Number of shuffles.")
    args = parser.parse_args()
    # Parse group keys
    group_keys = eval(args.group_keys)
    # Load the binary behavioral DataFrame
    logger.info(f'Loading binary behavioral DataFrame from {args.binary_timeseries_file}')
    binary_timeseries_df = load_data.load_binary_timeseries_df(args.binary_timeseries_file)
    # Load cross-combinations
    logger.info(f'Loading cross-combinations from {args.cross_combinations_file}')
    with open(args.cross_combinations_file, "rb") as f:
        cross_combinations = pickle.load(f)
    # Extract the group data
    group_dict = binary_timeseries_df.groupby(
        ['session_name', 'interaction_type', 'run_number']
        ).get_group(
            group_keys
            ).to_dict(
                orient="list"
                )
    # Prepare arguments for the computation function
    arg_tuple = (
        group_keys,
        group_dict,
        cross_combinations,
        args.output_dir,
        args.shuffle_count,
        True  # Set shuffled=True
    )
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    # Perform the computation
    logger.info(f"Computing shuffled cross-correlations for group {group_keys}.")
    _compute_crosscorrelations_for_group(arg_tuple)

if __name__ == "__main__":
    main()
