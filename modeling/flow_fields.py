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
from mRNNTorch.utils import get_region_activity
import matplotlib.pyplot as plt
from math import ceil
from train import interactivity_input
from mRNNTorch.analysis import flow_field
from mRNNTorch.utils import get_region_activity

# Model loading and input params
MODEL_NAME = "social_mrnn"
MODEL_PATH = "checkpoints/social_mrnn/"
PCA_PATH = "results/pca"
SPECIFICATIONS_PATH = "checkpoints/model_specifications/"
CHECK_PATH = "checkpoints/"
SAVE_NAME_PATH = "results/flow_fields/"

# Flow field parameters
CONDITION = 1
NUM_POINTS = 25
TIME_SKIPS = 10
X_OFFSET = 10
Y_OFFSET = 10
REGION_LIST = [
    "pfc_exc", 
    "pfc_inhib",
    "acc_exc", 
    "acc_inhib",
    "ofc_exc", 
    "ofc_inhib",
    "bla_exc", 
    "bla_inhib"
]
CANCEL_OTHER_REGIONS = False
CANCEL_INPUT = True
LINEARIZE = False
ALPHA = 1
FOLLOW_TRAJ = False

# Supress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def create_dir(save_path):
    # Get the directory part of the save path
    directory = os.path.dirname(save_path)
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_fig(save_path, eps=False):
    # Simple function to save figure while creating dir and closing
    create_dir(save_path)
    plt.tight_layout()
    if eps:
        plt.savefig(save_path, format="eps")
    else:
        plt.savefig(save_path)
    plt.close()

def plot_energy(coords, speed, save_path):

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(coords[:, :, 0], coords[:, :, 1], speed[:, :], cmap=plt.cm.coolwarm,
                linewidth=0, antialiased=False)
    ax.grid(False)
    # Set custom z-ticks
    ax.set_zticks([0, 1])  # Set the z-ticks to -1, 0, and 1
    # Optionally, you can set labels for those ticks
    ax.set_zticklabels(['0', '1'])
    save_fig(save_path)
    

def plot_flow(coords, x_vel, y_vel, speed, save_path):

    font = {'size' : 18}
    plt.rcParams['figure.figsize'] = [4, 4]
    plt.rcParams['axes.linewidth'] = 1 # set the value globally
    plt.rc('font', **font)
    
    # Add line collection
    fig, ax = plt.subplots()
    """
    if not LINEARIZE:
        subset_lines = lines
        subset_colors = color_vals
        subset_line_collection = LineCollection(subset_lines, cmap='viridis', norm=norm, linewidths=8, zorder=5)
        subset_line_collection.set_array(subset_colors)
        ax.add_collection(subset_line_collection)
        cbar = fig.colorbar(line_collection, ax=ax)
        cbar.set_label('Time')
    """
    # Create plot
    ax.streamplot(coords[:, :, 0], 
                  coords[:, :, 1], 
                  x_vel, 
                  y_vel, 
                  color=speed, 
                  cmap="plasma", 
                  linewidth=3, 
                  arrowsize=2, 
                  zorder=0,
    )

    # Other plotting parameters
    ax.set_yticks([])
    ax.set_xticks([])
    plt.tight_layout()
    save_fig(save_path)

    
def plot_pca(reduced_act, cur_idx, save_path):
    
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    font = {'size' : 18}
    plt.rcParams['figure.figsize'] = [4, 4]
    plt.rcParams['axes.linewidth'] = 1 # set the value globally
    plt.rc('font', **font)

    ax = plt.figure().add_subplot(projection="3d")
    # Hide grid lines
    ax.grid(False)
    plot_colors = ["dodgerblue", "limegreen", "gold", "darkorange"]
    ax.plot(reduced_act[:cur_idx, 0], reduced_act[:cur_idx, 1], reduced_act[:cur_idx, 2], linewidth=4, color=plot_colors[CONDITION])
    ax.plot(reduced_act[cur_idx:, 0], reduced_act[cur_idx:, 1], reduced_act[cur_idx:, 2], linewidth=4, linestyle="dashed", color=plot_colors[CONDITION])
    ax.plot(reduced_act[:, 0], reduced_act[:, 1], np.min(reduced_act[:, 2]), linewidth=4, color="gray", alpha=0.25)
    ax.set_xticks([ceil(np.min(reduced_act[:, 0])), ceil(np.max(reduced_act[:, 0]))])
    ax.set_yticks([ceil(np.min(reduced_act[:, 1])), ceil(np.max(reduced_act[:, 1]))])
    ax.set_zticks([ceil(np.min(reduced_act[:, 2])), ceil(np.max(reduced_act[:, 2]))])
    ax.plot([reduced_act[cur_idx, 0], reduced_act[cur_idx, 0]],
            [reduced_act[cur_idx, 1], reduced_act[cur_idx, 1]], 
            [reduced_act[cur_idx, 2], np.min(reduced_act[:, 2])],
            linestyle='--', 
            color="grey")
    ax.scatter(reduced_act[cur_idx, 0], 
               reduced_act[cur_idx, 1], 
               reduced_act[cur_idx, 2], 
               s=100, 
               c="black")
    plt.tight_layout()
    save_fig(save_path)



def reduce_orig_traj(rnn, xn, inp):
    
    # PCA object for plotting original trajectory on flow field
    plotting_pca = PCA(n_components=3)

    # Get original trajectory
    with torch.no_grad():
        _, orig_act = rnn(xn, inp, noise=False)
    orig_act = orig_act[CONDITION:CONDITION+1].cpu()

    # Gather activity for specified region and cell type
    # Get reduced act for plotting
    # Fit pca on activity
    orig_act_x = get_region_activity(rnn.mrnn, orig_act, *REGION_LIST)
    orig_act_x = np.reshape(orig_act_x.numpy(), (-1, orig_act_x.shape[-1]))
    reduced_act = plotting_pca.fit_transform(orig_act_x)

    return orig_act, reduced_act

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
    with torch.no_grad():
        out, hn = model(xn, inp, noise=False)

    out = out.detach().cpu()
    hn = hn.detach().cpu()
    
    # Get original trajectory and reduce it with PCA for plotting 
    orig_act, reduced_act = reduce_orig_traj(model, xn, inp)


    # Get velocities and speeds
    print("Gathering velocities...")
    data_coords, x_vels, y_vels, speeds = flow_field(
        model.mrnn.cuda(),
        orig_act.cuda(),
        inp[CONDITION:CONDITION+1].cuda(),
        time_skips=TIME_SKIPS,
        num_points=NUM_POINTS,
        lower_bound_x=-10,
        upper_bound_x=10,
        lower_bound_y=-10,
        upper_bound_y=10,
        region_list=REGION_LIST,
        cancel_other_regions=CANCEL_OTHER_REGIONS,
    )

    # Define the save paths for flow and energy
    save_path_flow = SAVE_NAME_PATH + "/" + f"flow_cancel_{CANCEL_OTHER_REGIONS}_linear_{LINEARIZE}"
    save_path_energy = SAVE_NAME_PATH + "/" + f"energy_cancel_{CANCEL_OTHER_REGIONS}_linear_{LINEARIZE}"

    # Get full path for both
    for region in REGION_LIST:
        save_path_flow = save_path_flow + f"_{region}"
        save_path_energy = save_path_energy + f"_{region}"
    save_path_flow = save_path_flow + f"_cond{CONDITION}/"
    save_path_energy = save_path_energy + f"_cond{CONDITION}/"

    print("Plotting flow fields...")
    for i in range(0, len(x_vels)):
        
        # Plot and save energy landscape
        energy_save = save_path_energy + f"t{i*TIME_SKIPS}_energy.png"
        plot_energy(data_coords, speeds[i], energy_save)

        # Plot and save flow field
        flow_save = save_path_flow + f"t{i*TIME_SKIPS}_flow.png"
        plot_flow(data_coords, x_vels[i], y_vels[i], speeds[i], flow_save)

        pca_save = save_path_flow + f"t{i*TIME_SKIPS}_pca.png"
        plot_pca(reduced_act, round(i * TIME_SKIPS), pca_save)

if __name__ == "__main__":
    main()