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
        self.groups = self.dataframe.groupby(self.group_by_columns, sort=False)

    def __len__(self):
        length = 0
        for i, _ in enumerate(self.groups):
            length += 1
        return length
            

    def __getitem__(self, idx):
        """Fetches a batch corresponding to a group."""
        for i, (group_key, group_df) in enumerate(self.groups): 
            if i == idx:
                break

        # Extract firing rate timelines (all units in this group)
        firing_rates = [torch.tensor(fr, dtype=torch.float32) for fr in group_df["firing_rate_timeline"]]

        # Stack firing rates along the last dimension (n_units)
        firing_rates = torch.stack(firing_rates, dim=-1)  # Shape: (sequence_length, n_units)

        # Ensure batch consistency (transpose to batch_first format)
        firing_rates = firing_rates.permute(1, 0)  # Shape: (1, sequence_length, n_units)
        print(firing_rates.shape)

        return firing_rates, group_key  # Return the key for debugging & analysis


class DurationCategoryBatchSampler(Sampler):
    """
    A sampler that ensures each batch consists of only one sequence length.
    """

    def __init__(self, dataframe, batch_size):
        """
        Args:
            dataframe (pd.DataFrame): The dataframe with firing rate data.
            batch_size (int): Number of samples per batch.
        """
        self.batch_size = batch_size
        self.dataframe = dataframe.copy()
        self.dataframe["group_index"] = np.arange(len(self.dataframe))

        # Group indices by duration category
        self.duration_groups = {
            duration: group_df["group_index"].tolist()
            for duration, group_df in self.dataframe.groupby("behav_duration_category")
        }

        # Convert dict values (lists) to iterators for batch sampling
        self.duration_iters = {k: iter(v) for k, v in self.duration_groups.items()}

    def __iter__(self):
        """
        Yields batches of indices where all sequences have the same duration category.
        """
        batches = []
        for duration, indices in self.duration_groups.items():
            # Shuffle indices within each duration category
            np.random.shuffle(indices)

            # Generate batches
            for batch in itertools.zip_longest(*[iter(indices)] * self.batch_size, fillvalue=None):
                batch = [i for i in batch if i is not None]  # Remove None values from last batch
                batches.append(batch)

        np.random.shuffle(batches)  # Shuffle batches across different duration categories
        return iter(batches)

    def __len__(self):
        """
        Returns the number of batches.
        """
        return sum(len(indices) // self.batch_size + (1 if len(indices) % self.batch_size else 0)
                   for indices in self.duration_groups.values())

# **Updated collate function to ensure consistent sequence lengths in each batch**
def collate_fn(batch):
    """Custom collate function to ensure all sequences in a batch have the same length."""
    inputs, group_keys = zip(*batch)  # Unpack batch items

    # Check that all sequence lengths are the same
    seq_lengths = [x.shape[1] for x in inputs]
    assert len(set(seq_lengths)) == 1, f"Mismatch in sequence lengths: {set(seq_lengths)}"

    inputs = torch.cat(inputs, dim=0)  # Shape: (batch_size, sequence_length, n_units)
    
    return inputs, group_keys  # Returning group keys for debugging if needed