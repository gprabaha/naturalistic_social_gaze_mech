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
import load_data
import itertools
from dataset import MeanFixationDataset
import pdb
from sklearn.decomposition import PCA
from models import Model
import matplotlib.pyplot as plt
from math import ceil
from train import interactivity_input
from utils import initialize_params, save_fig, get_mean_fixation_data, load_hp
import config


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
    
def plot_pca(model_path, region, data_type):

    hp = load_hp(model_path)
    exp_path = "results/pca"
    
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
        hp["spectral_radius"],
    ).cuda()

    checkpoint = torch.load(model_path + ".pth")
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
    

def main():

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()


if __name__ == "__main__":
    main()
