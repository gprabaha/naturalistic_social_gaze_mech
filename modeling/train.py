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
from dataset import FiringRateDataset
import pdb
from models import Model

def l1_weight(rnn, scale):
    l1 = 0
    for name, param in rnn.named_parameters():
        l1 += torch.mean(torch.abs(torch.flatten(param)))
    l1 *= scale
    return l1

def l1_rate(act, scale):
    l1 = scale * torch.mean(torch.abs(torch.flatten(act)))
    return l1

def _initialize_params(
    remake_firing_rate_df=False,
    neural_data_bin_size=10,
    smooth_spike_counts=True,
    guassian_smoothing_sigma=2,
    time_window_before_event=500,
    is_cluster=False,
    path_name=None
):
    params = {
        'remake_firing_rate_df': remake_firing_rate_df,
        'neural_data_bin_size': neural_data_bin_size,  # 10 ms in seconds
        'smooth_spike_counts': smooth_spike_counts,
        'gaussian_smoothing_sigma': guassian_smoothing_sigma,
        'time_window_before_event': time_window_before_event,
        'is_cluster': is_cluster,
        'path_name': path_name
    }
    params = curate_data.add_root_data_to_params(params)
    params = curate_data.add_processed_data_to_params(params)
    return params

def const_input(key, timesteps, dataset):
    input_series = torch.zeros(size=(1, timesteps, dataset.num_conds))
    input_series[..., dataset.input_key[key]] = 1
    return input_series 

# Currently 36 different conditions
def main():

    ### PARAMETERS ###
    parser = train_config.config_parser()
    args = parser.parse_args()

    # Load processed dataframe
    params = _initialize_params(
        path_name="/Users/lazza/naturalistic_social_gaze_mech/social_gaze"
    )
    behav_firing_rate_df_file_path = os.path.join(
        params['processed_data_dir'], 'averaged_neural_firing_rate_df.pkl'
    )
    print('loading data...')
    df = load_data.get_data_df(behav_firing_rate_df_file_path)
    print('creating fr dataset...')
    dataset = FiringRateDataset(df)
    
    # Training variables
    model = Model(
        args.mrnn_config_file, 
        100, 
        dataset.units_per_region["dmpfc"], 
        dataset.units_per_region["accg"], 
        dataset.units_per_region["ofc"], 
        dataset.units_per_region["bla"], 
        args.dt, 
        args.tau, 
        args.inp_noise, 
        args.act_noise,
        args.constrained,
        args.batch_first,
        args.spectral_radius
    ).cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    cur_loss = 0
    losses = []

    # Start training
    for epoch in range(args.epochs):

        batch, key, loss_mask = dataset.sample_batch(args.batch_size)
        batch = batch.unsqueeze(0)
        loss_mask = loss_mask.unsqueeze(0)
        inp = const_input(key, batch.shape[1], dataset)

        # Put to device
        batch = batch.cuda()
        inp = inp.cuda()
        loss_mask = loss_mask.cuda()

        xn = torch.zeros(size=(1, model.mrnn.total_num_units), device="cuda")
        out, hn = model(xn, inp)

        out = out * loss_mask

        # Compute all losses
        mse_loss = criterion(out, batch)
        rate_loss = l1_rate(hn, args.l1_rate)
        weight_loss = l1_weight(model.mrnn, args.l1_weight)
        loss = mse_loss + rate_loss + weight_loss
        cur_loss += loss.item()

        # Training stats
        if epoch % 100 == 0 and epoch > 0:

            # Get the directory part of the save path
            directory = os.path.dirname(args.save_dir)
            # Check if the directory exists, and create it if it doesn't
            if not os.path.exists(directory):
                os.makedirs(directory)

            torch.save(model.state_dict(), args.save_dir + args.model_save_name)

            mean_loss = cur_loss / 100
            losses.append(mean_loss)

            with open(args.save_dir + "losses.txt", "w") as output:
                output.write(str(losses))            

            cur_loss = 0

            print("Mean training loss at epoch {}:{}".format(epoch, mean_loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    main()