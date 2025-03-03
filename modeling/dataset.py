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
import random

def normalization(x):
    return (x) / (np.percentile(x, 95) - np.percentile(x, 5) + 5)

class FiringRateDataset():
    def __init__(self, dataframe, group_by_columns=None):
        """
        Args:
            dataframe (pd.DataFrame): The dataframe containing firing rate data.
            group_by_columns (list): Columns used to group data into batches.
        """
        self.dataframe = dataframe.copy()  # Avoid modifying the original dataframe
        self.group_by_columns = group_by_columns or [
            "behavior_type", "location", 
            "from_location", "to_location", "behav_duration_category"
        ]
        
        self.dataframe[self.group_by_columns] = self.dataframe[self.group_by_columns].fillna("UNKNOWN")

        # Group by specified columns (ensuring same region and behavior type are in a batch)
        self.groups = list(self.dataframe.groupby(self.group_by_columns, sort=False))
        dataframe_by_regions = self.dataframe.groupby(['region'], sort=False)

        # This will give a list of unit ids and regions
        # unit ids and region labels should be grouped together within each list
        # the region at idx i in region_labels is the corresponding region for idx i in unit_ids
        self.unit_ids = []
        self.region_labels = []
        self.units_per_region = {}
        for region_key, region_df in dataframe_by_regions:
            unit_ids_in_region = region_df["unit_uuid"].unique().tolist()
            region_label = [region_key[0]] * len(unit_ids_in_region)
            self.unit_ids.extend(unit_ids_in_region)
            self.region_labels.extend(region_label)
            self.units_per_region[region_key[0]] = len(unit_ids_in_region)

        self.num_conds = len(self.groups)
        self.total_num_units = len(self.unit_ids)

        self.input_key = {}
        for i, (key, _) in enumerate(self.groups):
            self.input_key[key] = i
    
    def sample_batch(self, batch_size, probs=None):

        if probs == None:
            probs = np.ones(self.num_conds) / self.num_conds
        cond = np.arange(self.num_conds)
        # Pick a random number based on the probabilities
        random_cond = np.random.choice(cond, p=probs)
        """Fetches a batch corresponding to a group."""
        group_key, group_df = self.groups[random_cond]

        group_df_by_id = group_df.groupby(["unit_uuid"], sort=False)
        # Extract firing rate timelines (all units in this group)

        ex_fr = group_df['firing_rate_timeline'].sample(n=1).iloc[0]
        firing_rates = np.zeros(shape=(len(ex_fr), self.total_num_units))
        for _, group_id_data in group_df_by_id:
            # They should all have the same unit ids
            unit_uuid = group_id_data['unit_uuid'].sample(n=1).iloc[0]
            unit_idx = self.unit_ids.index(unit_uuid)
            firing_rates[:, unit_idx] = np.array(group_id_data["firing_rate_timeline"].sample(n=len(group_id_data["firing_rate_timeline"])).tolist())
        
        firing_rates = normalization(firing_rates)
        firing_rates = torch.tensor(firing_rates, dtype=torch.float32)

        return firing_rates, group_key  # Return the key for debugging & analysis