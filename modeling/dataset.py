import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class DataFrameDataset(Dataset):
    def __init__(self, dataframe, input_columns, target_columns=None, device="cpu"):
        """
        Args:
            dataframe (pd.DataFrame): The input dataframe.
            input_columns (list): List of column names to use as input features.
            target_columns (list, optional): List of column names for target labels. Default is None.
            device (str): Device to store tensors ('cpu' or 'cuda').
        """
        self.dataframe = dataframe
        self.input_columns = input_columns
        self.target_columns = target_columns
        self.device = device

        self.inputs = torch.tensor(self.dataframe[input_columns].values, dtype=torch.float32).to(device)
        
        if target_columns:
            self.targets = torch.tensor(self.dataframe[target_columns].values, dtype=torch.float32).to(device)
        else:
            self.targets = None

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.inputs[idx], self.targets[idx]
        return self.inputs[idx]

class DataFrameBatcher:
    def __init__(self, dataframe, input_columns, target_columns=None, batch_size=32, shuffle=True, device="cpu"):
        """
        Wrapper class to create DataLoader from a Pandas DataFrame.

        Args:
            dataframe (pd.DataFrame): The input dataframe.
            input_columns (list): List of column names to use as input features.
            target_columns (list, optional): List of column names for target labels. Default is None.
            batch_size (int): Batch size for training. Default is 32.
            shuffle (bool): Whether to shuffle the data. Default is True.
            device (str): Device to store tensors ('cpu' or 'cuda').
        """
        self.dataset = DataFrameDataset(dataframe, input_columns, target_columns, device)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)

