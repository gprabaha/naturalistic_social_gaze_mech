import os
import sys
from pathlib import Path

# Add the root directory of the repository to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import torch
import config
from plt_utils import standard_2d_ax
from utils import load_hp, get_mean_fixation_data, interactivity_input, save_fig, stim_inp
from models import Model
from mRNNTorch.analysis import linearized_eigendecomposition
from scipy.special import rel_entr
import numpy as np

# Supress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def _get_max_eigenvalues(model_path, condition, *args, stim_inp=None):

    hp = load_hp(model_path)
    
    dataset = get_mean_fixation_data("/Users/John/naturalistic_social_gaze_mech/social_gaze") 

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
        device="cpu"
    )
    
    checkpoint = torch.load(os.path.join(hp["save_dir"], hp["model_save_name"] + ".pth"))
    model.load_state_dict(checkpoint)
    model.eval()

    batch, keys, _ = dataset.sample_batch()
    inp = interactivity_input(keys, batch.shape[1])

    xn = torch.zeros(size=(batch.shape[0], model.mrnn.total_num_units), device="cpu")
    hn = torch.zeros(size=(batch.shape[0], model.mrnn.total_num_units), device="cpu")

    with torch.no_grad():
        if stim_inp is not None:
            out, hn = model(inp, xn, hn, stim_inp, noise=False)
        else:
            out, hn = model(inp, xn, hn, noise=False)

    out = out.detach().cpu()
    hn = hn.detach().cpu()

    hn_cond = hn[condition]

    max_eigs = []
    eigs = []
    for hn_t in hn_cond:
        reals, _, _ = linearized_eigendecomposition(model.mrnn, hn_t, *args)
        max_real = max(reals)
        eigs.append(reals)
        max_eigs.append(max_real)
    
    return max_eigs, eigs




def _get_ablation_stims(model_path, *args):
    
    hp = load_hp(model_path)
    
    dataset = get_mean_fixation_data("/Users/John/naturalistic_social_gaze_mech/social_gaze") 

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
        device="cpu"
    )
    
    checkpoint = torch.load(os.path.join(hp["save_dir"], hp["model_save_name"] + ".pth"))
    model.load_state_dict(checkpoint)
    model.eval()

    batch, keys, _ = dataset.sample_batch()
    inp = interactivity_input(keys, batch.shape[1])

    regions = []
    for region in model.mrnn.region_dict:
        if region not in args:
            regions.append(region)

    stim_list = []
    for s in range(len(regions)):
        stim_list.append(stim_inp(
            model.mrnn, 
            50, 
            100,
            inp.shape[1],
            0,
            -5,
            inp.shape[0], 
            5,
            10,
            regions[s]
        ))
    
    return stim_list, regions




def _plot_max_eigs(model_path, region):
    
    hp = load_hp(model_path)
    exp_path = f"results/{hp["model_save_name"]}/linear_analysis"
    
    fig, ax = standard_2d_ax()
    max_eigs_high_int, _ = _get_max_eigenvalues(model_path, 0, region)
    max_eigs_low_int, _ = _get_max_eigenvalues(model_path, 1, region)
    max_eigs_object, _ = _get_max_eigenvalues(model_path, 2, region)

    ax.plot(max_eigs_high_int, linewidth=4, color="red")
    ax.plot(max_eigs_low_int, linewidth=4, color="blue")
    ax.plot(max_eigs_object, linewidth=4, color="green")
    save_fig(os.path.join(exp_path, f"{region}_max_eigs_all_conds"))



def plot_max_eigs_pfc(model_path):
    _plot_max_eigs(model_path, "pfc")

def plot_max_eigs_acc(model_path):
    _plot_max_eigs(model_path, "acc")

def plot_max_eigs_bla(model_path):
    _plot_max_eigs(model_path, "bla")

def plot_max_eigs_ofc(model_path):
    _plot_max_eigs(model_path, "ofc")

def run_all_max_eigs(model_path):
    plot_max_eigs_pfc(model_path)
    plot_max_eigs_acc(model_path)
    plot_max_eigs_bla(model_path)
    plot_max_eigs_ofc(model_path)




# ABLATION
def _plot_max_eigs_ablation(model_path, region):
    
    hp = load_hp(model_path)
    exp_path = f"results/{hp["model_save_name"]}/linear_analysis"
    stim_list, regions = _get_ablation_stims(model_path, region)
    
    for s in range(3):
        fig, ax = standard_2d_ax()
        max_eigs_high_int, _ = _get_max_eigenvalues(model_path, 0, region, stim_inp=stim_list[s])
        max_eigs_low_int, _ = _get_max_eigenvalues(model_path, 1, region, stim_inp=stim_list[s])
        max_eigs_object, _ = _get_max_eigenvalues(model_path, 2, region, stim_inp=stim_list[s])

        ax.plot(max_eigs_high_int, linewidth=4, color="red")
        ax.plot(max_eigs_low_int, linewidth=4, color="blue")
        ax.plot(max_eigs_object, linewidth=4, color="green")
        ax.axvline(x=50, linestyle="--", color="grey", linewidth=2)
        ax.axvline(x=100, linestyle="--", color="grey", linewidth=2)
        save_fig(os.path.join(exp_path, f"{region}_max_eigs_all_conds_ablate_{regions[s]}"))

def plot_max_eigs_pfc_ablation(model_path):
    _plot_max_eigs_ablation(model_path, "pfc")

def plot_max_eigs_acc_ablation(model_path):
    _plot_max_eigs_ablation(model_path, "acc")

def plot_max_eigs_bla_ablation(model_path):
    _plot_max_eigs_ablation(model_path, "bla")

def plot_max_eigs_ofc_ablation(model_path):
    _plot_max_eigs_ablation(model_path, "ofc")

def run_all_max_eigs_ablation(model_path):
    plot_max_eigs_pfc_ablation(model_path)
    plot_max_eigs_acc_ablation(model_path)
    plot_max_eigs_bla_ablation(model_path)
    plot_max_eigs_ofc_ablation(model_path)




def plot_eigs_ablation_all_models(region):
    
    model_paths = [
        "checkpoints/social_mrnn_0",
        "checkpoints/social_mrnn_1",
        "checkpoints/social_mrnn_2",
        "checkpoints/social_mrnn_3",
        "checkpoints/social_mrnn_4"
    ]

    all_model_dists_high_int_ab = {}
    all_model_dists_low_int_ab = {}
    all_model_dists_object_ab = {}

    all_model_dists_high_int_ctrl = {}
    all_model_dists_low_int_ctrl = {}
    all_model_dists_object_ctrl = {}

    for model_path in model_paths:

        stim_list, regions = _get_ablation_stims(model_path, region)

        # intialize lists for each region
        for r in regions:
            if r not in all_model_dists_high_int_ab:
                all_model_dists_high_int_ab[r] = []
            if r not in all_model_dists_low_int_ab:
                all_model_dists_low_int_ab[r] = []
            if r not in all_model_dists_object_ab:
                all_model_dists_object_ab[r] = []
            if r not in all_model_dists_high_int_ctrl:
                all_model_dists_high_int_ctrl[r] = []
            if r not in all_model_dists_low_int_ctrl:
                all_model_dists_low_int_ctrl[r] = []
            if r not in all_model_dists_object_ctrl:
                all_model_dists_object_ctrl[r] = []
        
        for s in range(3):
            _, eigs_high_int_ab = _get_max_eigenvalues(model_path, 0, region, stim_inp=stim_list[s])
            _, eigs_low_int_ab = _get_max_eigenvalues(model_path, 1, region, stim_inp=stim_list[s])
            _, eigs_object_ab = _get_max_eigenvalues(model_path, 2, region, stim_inp=stim_list[s])

            all_model_dists_high_int_ab[regions[s]].append(eigs_high_int_ab)
            all_model_dists_low_int_ab[regions[s]].append(eigs_low_int_ab)
            all_model_dists_object_ab[regions[s]].append(eigs_object_ab)

            _, eigs_high_int_ctrl = _get_max_eigenvalues(model_path, 0, region)
            _, eigs_low_int_ctrl = _get_max_eigenvalues(model_path, 1, region)
            _, eigs_object_ctrl = _get_max_eigenvalues(model_path, 2, region)

            all_model_dists_high_int_ctrl[regions[s]].append(eigs_high_int_ctrl)
            all_model_dists_low_int_ctrl[regions[s]].append(eigs_low_int_ctrl)
            all_model_dists_object_ctrl[regions[s]].append(eigs_object_ctrl)

    for s in range(3):

        high_int_eigs_ab_arr = np.array(all_model_dists_high_int_ab[regions[s]])
        low_int_eigs_ab_arr = np.array(all_model_dists_low_int_ab[regions[s]])
        object_eigs_ab_arr = np.array(all_model_dists_object_ab[regions[s]])

        high_int_eigs_ctrl_arr = np.array(all_model_dists_high_int_ctrl[regions[s]])
        low_int_eigs_ctrl_arr = np.array(all_model_dists_low_int_ctrl[regions[s]])
        object_eigs_ctrl_arr = np.array(all_model_dists_object_ctrl[regions[s]])

    fig, ax = standard_2d_ax()
    for region in regions_ablate:
        for run in range(len(model_paths)):
            ax.plot(max_eigs_high_int[region][run], linewidth=2, color="red", alpha=0.25)
            ax.fill_between(
                x,
                mean - std,
                mean + std,
                color="C0",
                alpha=0.3,
                label="±1 std",
            )
            ax.plot(max_eigs_low_int[region][run], linewidth=2, color="blue", alpha=0.25)
            ax.plot(max_eigs_object[region][run], linewidth=2, color="green", alpha=0.25)
        ax.set_title(f"Ablate {region} to pfc")
        ax.axvline(x=50, linestyle="--", color="grey", linewidth=2)
        ax.axvline(x=100, linestyle="--", color="grey", linewidth=2)
        save_fig(os.path.join("results/linear_analysis", f"pfc_max_eigs_all_conds_ablate_{region}_all_models"), eps=True)





def main():
    
    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()

    if args.experiment == "plot_max_eigs_pfc":
        plot_max_eigs_pfc(args.model_path)
    elif args.experiment == "plot_max_eigs_acc":
        plot_max_eigs_acc(args.model_path)
    elif args.experiment == "plot_max_eigs_bla":
        plot_max_eigs_bla(args.model_path)
    elif args.experiment == "plot_max_eigs_ofc":
        plot_max_eigs_ofc(args.model_path)
    elif args.experiment == "run_all_max_eigs":
        run_all_max_eigs(args.model_path)

    elif args.experiment == "plot_max_eigs_pfc_ablation":
        plot_max_eigs_pfc_ablation(args.model_path)
    elif args.experiment == "plot_max_eigs_acc_ablation":
        plot_max_eigs_acc_ablation(args.model_path)
    elif args.experiment == "plot_max_eigs_bla_ablation":
        plot_max_eigs_bla_ablation(args.model_path)
    elif args.experiment == "plot_max_eigs_ofc_ablation":
        plot_max_eigs_ofc_ablation(args.model_path)
    elif args.experiment == "run_all_max_eigs_ablation":
        run_all_max_eigs_ablation(args.model_path)

    else:
        raise NotImplementedError(f"Experiment {args.experiment} not implemented")

if __name__ == "__main__":
    main()