import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
import train_config

import sys
import os
from pathlib import Path

# Add the root directory of the repository to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import curate_data
import load_data
import itertools
from dataset import FiringRateDataset, MeanFixationDataset
import pdb
from models import Model
from losses import l1_weight, l1_rate
from utils import initialize_params, interactivity_input, create_dir, save_hp, get_mean_fixation_data

DEF_HP = {
    "inp_dim": 2,
    "epochs": 100000,
    "lr": 1e-3,
    "dt": 10,
    "tau": 100,
    "l1_weight": 1e-4,
    "l1_rate": 1e-4,
    "constrained": False,
    "batch_first": True,
    "seed": 123456,
    "tol": 1e-3,
    "patience": 50,
    "inp_noise": 0,
    "act_noise": 0,
    "spectral_radius": 1.3,
    "save_dir": "checkpoints/social_mrnn",
    "model_save_name": "social_mrnn.pth",
    "model_specifications_path": "checkpoints/model_specifications",
    "mrnn_config_file": "modeling/configurations/mRNN.json"
}

def train(hp=None):
    
    def_hp = DEF_HP
    if hp is not None:
        def_hp.update(hp)
    hp = def_hp

    # create model path for saving model and hp
    create_dir(hp["save_dir"])
    save_hp(hp, hp["save_dir"])

    # Load processed dataframe
    dataset = get_mean_fixation_data("/Users/John/naturalistic_social_gaze_mech/social_gaze")
    
    # Training variables
    model = Model(
        hp["mrnn_config_file"], 
        100, 
        dataset.units_per_region["dmpfc"], 
        dataset.units_per_region["accg"], 
        dataset.units_per_region["ofc"], 
        dataset.units_per_region["bla"], 
        hp["dt"], 
        hp["tau"], 
        hp["inp_noise"], 
        hp["act_noise"],
        hp["constrained"],
        hp["batch_first"],
        hp["spectral_radius"]
    ).cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hp["lr"])
    cur_loss = 0
    losses = []

    # Start training
    for epoch in range(hp["epochs"]):

        batch, key, loss_mask = dataset.sample_batch()
        inp = interactivity_input(key, batch.shape[1], dataset)

        # Put to device
        batch = batch.cuda()
        inp = inp.cuda()
        loss_mask = loss_mask.cuda()

        xn = torch.zeros(size=(1, model.mrnn.total_num_units), device="cuda")
        hn = torch.zeros(size=(1, model.mrnn.total_num_units), device="cuda")
        out, hn = model(inp, xn, hn)

        out = out * loss_mask

        # Compute all losses
        mse_loss = criterion(out, batch)
        rate_loss = l1_rate(hn, hp["l1_rate"])
        weight_loss = l1_weight(model.mrnn, hp["l1_weight"])
        loss = mse_loss + rate_loss + weight_loss
        cur_loss += loss.item()

        # Training stats
        if epoch % 100 == 0 and epoch > 0:

            # Get the directory part of the save path
            directory = os.path.dirname(hp["save_dir"])
            # Check if the directory exists, and create it if it doesn't
            if not os.path.exists(directory):
                os.makedirs(directory)

            torch.save(model.state_dict(), os.path.join(hp["save_dir"], hp["model_save_name"]))

            mean_loss = cur_loss / 100
            losses.append(mean_loss)

            with open(hp["save_dir"] + "losses.txt", "w") as output:
                output.write(str(losses))            

            cur_loss = 0

            print("Mean training loss at epoch {}:{}".format(epoch, mean_loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()