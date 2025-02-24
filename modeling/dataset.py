import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd
import numpy as np
import os

import sys
from pathlib import Path
# Add the root directory of the repository to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import curate_data
import load_data
import itertools
import pdb

class FiringRateDataset(Dataset):
    def __init__(self, dataframe, group_by_columns=None):
        """
        Args:
            dataframe (pd.DataFrame): The dataframe containing firing rate data.
            group_by_columns (list): Columns used to group data into batches.
        """
        self.dataframe = dataframe.copy()  # Avoid modifying the original dataframe
        self.group_by_columns = group_by_columns or [
            "region", "behavior_type", "location", 
            "from_location", "to_location", "behav_duration_category"
        ]
        
        self.dataframe[self.group_by_columns] = self.dataframe[self.group_by_columns].fillna("UNKNOWN")

        # Group by specified columns (ensuring same region and behavior type are in a batch)
        self.groups = list(self.dataframe.groupby(self.group_by_columns, sort=False))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        """Fetches a batch corresponding to a group."""
        group_key, group_df = self.groups[idx]

        # Extract firing rate timelines (all units in this group)
        firing_rates = []
        for fr in group_df["firing_rate_timeline"]:
            firing_rates.append(torch.tensor(fr, dtype=torch.float32))

        # Stack firing rates along the last dimension (n_units)
        firing_rates = torch.stack(firing_rates, dim=-1).unsqueeze(0)  # Shape: (sequence_length, n_units)

        return firing_rates, group_key  # Return the key for debugging & analysis