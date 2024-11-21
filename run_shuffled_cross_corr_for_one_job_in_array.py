import argparse
import os
import pickle
import load_data
from analyze_data import compute_crosscorr_distribution_for_shuffled_data
import logging

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run shuffled cross-correlation for a single group.")
    parser.add_argument('--group_keys', required=True, type=str, help="Group keys as a tuple string.")
    parser.add_argument('--binary_timeseries_file', required=True, type=str, help="Path to binary timeseries file.")
    parser.add_argument('--output_dir', required=True, type=str, help="Output directory.")
    parser.add_argument('--num_cpus', required=True, type=int, help="Number of CPUs per job.")
    parser.add_argument('--shuffle_count', default=50, type=int, help="Number of shuffles.")
    args = parser.parse_args()

    group_keys = eval(args.group_keys)
    logger.info(f'Loading binary behavioral df from {binary_timeseries_file_path}')
    binary_timeseries_df = load_data.load_binary_timeseries_df(args.binary_timeseries_file)
    os.makedirs(args.output_dir, exist_ok=True)

    compute_crosscorr_distribution_for_shuffled_data(group_keys, binary_timeseries_df, args.shuffle_count, args.output_dir, args.num_cpus)

if __name__ == "__main__":
    main()
