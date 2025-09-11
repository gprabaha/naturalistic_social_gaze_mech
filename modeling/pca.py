import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import sys
import os
from pathlib import Path

# Add the root directory of the repository to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import train_config
import curate_data
import load_data
import itertools
from dataset import MeanFixationDataset
import pdb
from sklearn.decomposition import PCA
from models import Model
import matplotlib.pyplot as plt
from math import ceil
from train import interactivity_input

MODEL_NAME = "social_mrnn"
MODEL_PATH = "checkpoints/social_mrnn/"
PCA_PATH = "results/pca"

def create_dir(save_path):
    # Get the directory part of the save path
    directory = os.path.dirname(save_path)
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_fig(save_path):
    # Simple function to save figure while creating dir and closing
    create_dir(save_path)
    plt.savefig(save_path)
    plt.close()

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

def reduce_padding(act, keys):
    # Used to compare with padded vectors
    zero_act = torch.zeros(size=(act.shape[-1],))
    # Ensure we can still access the correct condition
    saccade_act = {}
    fixation_act = {}

    # Loop through activity and get rid of padding, add to dictionaries
    # Keys and act should be same length (batch dim)
    for key, cond in zip(keys, act):
        non_padded_act = []
        for act_t in cond:
            if torch.equal(act_t, zero_act):
                 continue
            else:
                non_padded_act.append(act_t)
        non_padded_act = torch.stack(non_padded_act)
        if key[0] == "saccade":
            saccade_act[key] = non_padded_act
        elif key[0] == "fixation":
            fixation_act[key] = non_padded_act
    return saccade_act, fixation_act

def pca(act):
    # input should be a dictionary with the keys being condition
    h_pca = PCA(n_components=3)
    act_reshaped = act.reshape((-1, act.shape[-1]))
    reduced_acts = h_pca.fit_transform(act_reshaped)
    act_reshaped = reduced_acts.reshape((3, -1, reduced_acts.shape[-1]))
    return act_reshaped

def plot_pca(act, region, data_type):
    
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    font = {'size' : 10}
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.rcParams['axes.linewidth'] = 1 # set the value globally
    plt.rc('font', **font)

    ax = plt.figure().add_subplot(projection="3d")
    # Hide grid lines
    ax.grid(False)

    label_str = ["high_interactivity_face", "low_interactivity_face", "object"]
    ax.plot(act[0, :, 0], act[0, :, 1], act[0, :, 2], linewidth=4, label=label_str[0])
    ax.plot(act[1, :, 0], act[1, :, 1], act[1, :, 2], linewidth=4, label=label_str[1])
    ax.plot(act[2, :, 0], act[2, :, 1], act[2, :, 2], linewidth=4, label=label_str[2])
    ax.legend(loc="best")

    save_path = f"{PCA_PATH}/{MODEL_NAME}/{region}_{data_type}_pca.png"
    save_fig(save_path)

def plot_all_pcs(model, x, data_type):

    pfc_act = model.mrnn.get_region_activity(x, "pfc")
    acc_act = model.mrnn.get_region_activity(x, "acc")
    ofc_act = model.mrnn.get_region_activity(x, "ofc")
    bla_act = model.mrnn.get_region_activity(x, "bla")

    pfc_fixation_reduced = pca(pfc_act)
    acc_fixation_reduced = pca(acc_act)
    ofc_fixation_reduced = pca(ofc_act)
    bla_fixation_reduced = pca(bla_act)

    plot_pca(pfc_fixation_reduced, "pfc", data_type)
    plot_pca(acc_fixation_reduced, "acc", data_type)
    plot_pca(ofc_fixation_reduced, "ofc", data_type)
    plot_pca(bla_fixation_reduced, "bla", data_type)
    

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
        params['processed_data_dir'], 'mean_fixation_response_df.pkl'
    )
    print('loading data...')
    df = load_data.get_data_df(behav_firing_rate_df_file_path)
    print('creating fr dataset...')
    dataset = MeanFixationDataset(df)
    
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
        args.spectral_radius,
    ).cuda()

    checkpoint = torch.load(MODEL_PATH + MODEL_NAME + ".pth")
    model.load_state_dict(checkpoint)

    # Start training
    batch, keys, loss_mask = dataset.sample_batch()
    inp = interactivity_input(keys, batch.shape[1], dataset)
    inp = inp.cuda()

    xn = torch.zeros(size=(batch.shape[0], model.mrnn.total_num_units), device="cuda")
    hn = torch.zeros(size=(batch.shape[0], model.mrnn.total_num_units), device="cuda")
    with torch.no_grad():
        out, hn = model(inp, xn, hn, noise=False)

    out = out.detach().cpu()
    hn = hn.detach().cpu()

    plot_all_pcs(model, batch, "data")
    plot_all_pcs(model, out, "output")
    plot_all_pcs(model, hn, "hidden_activity")

if __name__ == "__main__":
    main()
