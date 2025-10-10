import os
import sys
from pathlib import Path

# Add the root directory of the repository to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import torch
import config
from plt_utils import standard_2d_ax
from utils import load_hp, get_mean_fixation_data, interactivity_input, save_fig, stim_inp, get_other_regions
from models import Model
from mRNNTorch.analysis import linearized_eigendecomposition
from scipy.special import rel_entr
import numpy as np
from scipy.stats import wasserstein_distance

# Supress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def _get_eigenvalues(model_path, condition, *args, stim_inp=None):

    hp = load_hp(model_path)
    
    dataset = get_mean_fixation_data("/Users/lazza/naturalistic_social_gaze_mech/social_gaze") 

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
    eigs_r, eigs_im = [], []
    for hn_t in hn_cond:
        reals, ims, _ = linearized_eigendecomposition(model.mrnn, hn_t, *args)
        max_real = max(reals)
        eigs_r.append(reals)
        eigs_im.append(ims)
        max_eigs.append(max_real)
    
    return max_eigs, eigs_r, eigs_im




def _get_ablation_stims(model_path, region, start_silence=50, end_silence=75):
    
    hp = load_hp(model_path)
    
    dataset = get_mean_fixation_data("/Users/lazza/naturalistic_social_gaze_mech/social_gaze") 

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

    regions = get_other_regions(model, region)

    stim_list = []
    for s in range(len(regions)):
        stim_list.append(stim_inp(
            model.mrnn, 
            start_silence, 
            end_silence,
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
    max_eigs_high_int, _ = _get_eigenvalues(model_path, 0, region)
    max_eigs_low_int, _ = _get_eigenvalues(model_path, 1, region)
    max_eigs_object, _ = _get_eigenvalues(model_path, 2, region)

    ax.plot(max_eigs_high_int, linewidth=4, color="red")
    ax.plot(max_eigs_low_int, linewidth=4, color="blue")
    ax.plot(max_eigs_object, linewidth=4, color="green")
    save_fig(os.path.join(exp_path, f"{region}_max_eigs_all_conds"))




def _plot_eig_dist(model_path, region):

    def _make_scatters(data_r, data_im, cond):
        for t, (eigs_r, eigs_im) in enumerate(zip(data_r, data_im)):
            fig, ax = standard_2d_ax()
            ax.scatter(eigs_r, eigs_im, color="red", alpha=0.75, edgecolors="black", s=100)
            save_fig(os.path.join(exp_path, f"{region}", f"cond{cond}", f"eig_dist_t{t}"))

    hp = load_hp(model_path)
    exp_path = f"results/{hp["model_save_name"]}/linear_analysis"
    
    _, eigs_hi_r, eigs_hi_im = _get_eigenvalues(model_path, 0, region)
    _, eigs_li_r, eigs_li_im = _get_eigenvalues(model_path, 1, region)
    _, eigs_o_r, eigs_o_im = _get_eigenvalues(model_path, 2, region)

    _make_scatters(eigs_hi_r, eigs_hi_im, 0)
    _make_scatters(eigs_li_r, eigs_li_im, 1)
    _make_scatters(eigs_o_r, eigs_o_im, 2)
    




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




def plot_eigs_dist_pfc(model_path):
    _plot_eig_dist(model_path, "pfc")

def plot_eigs_dist_acc(model_path):
    _plot_eig_dist(model_path, "acc")

def plot_eigs_dist_bla(model_path):
    _plot_eig_dist(model_path, "bla")

def plot_eigs_dist_ofc(model_path):
    _plot_eig_dist(model_path, "ofc")

def run_all_eigs_dist(model_path):
    plot_eigs_dist_pfc(model_path)
    plot_eigs_dist_acc(model_path)
    plot_eigs_dist_bla(model_path)
    plot_eigs_dist_ofc(model_path)





# ABLATION
def _plot_max_eigs_ablation(model_path, region):
    
    hp = load_hp(model_path)
    exp_path = f"results/{hp["model_save_name"]}/linear_analysis"
    stim_list, regions = _get_ablation_stims(model_path, region)
    
    for s in range(3):
        fig, ax = standard_2d_ax()
        max_eigs_high_int, _ = _get_eigenvalues(model_path, 0, region, stim_inp=stim_list[s])
        max_eigs_low_int, _ = _get_eigenvalues(model_path, 1, region, stim_inp=stim_list[s])
        max_eigs_object, _ = _get_eigenvalues(model_path, 2, region, stim_inp=stim_list[s])

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





def _eigs_all_models(region, stim=False, start_silence=50, end_silence=75):
    
    model_paths = [
        "checkpoints/social_mrnn_0",
        "checkpoints/social_mrnn_1",
        "checkpoints/social_mrnn_2",
        "checkpoints/social_mrnn_3",
        "checkpoints/social_mrnn_4"
    ]

    all_model_dists_high_int = {}
    all_model_dists_low_int = {}
    all_model_dists_object = {}

    for model_path in model_paths:

        # This is horrible
        if stim:
            stim_list, regions = _get_ablation_stims(model_path, region, start_silence=start_silence, end_silence=end_silence)
        else:
            _, regions = _get_ablation_stims(model_path, region)
            
        # intialize lists for each region
        for r in regions:
            if r not in all_model_dists_high_int:
                all_model_dists_high_int[r] = []
            if r not in all_model_dists_low_int:
                all_model_dists_low_int[r] = []
            if r not in all_model_dists_object:
                all_model_dists_object[r] = []
        
        for s in range(len(regions)):

            stim_cur = stim_list[s] if stim else None

            _, eigs_high_int_ab, _ = _get_eigenvalues(model_path, 0, region, stim_inp=stim_cur)
            _, eigs_low_int_ab, _ = _get_eigenvalues(model_path, 1, region, stim_inp=stim_cur)
            _, eigs_object_ab, _ = _get_eigenvalues(model_path, 2, region, stim_inp=stim_cur)

            all_model_dists_high_int[regions[s]].append(eigs_high_int_ab)
            all_model_dists_low_int[regions[s]].append(eigs_low_int_ab)
            all_model_dists_object[regions[s]].append(eigs_object_ab)
    
    for r in regions:

        high_int_eigs_arr = np.array(all_model_dists_high_int[r]) 
        low_int_eigs_arr = np.array(all_model_dists_low_int[r])
        object_eigs_arr = np.array(all_model_dists_object[r])

        high_int_eigs_arr = np.reshape(high_int_eigs_arr,  (-1, high_int_eigs_arr.shape[-1])) 
        low_int_eigs_arr = np.reshape(low_int_eigs_arr,  (-1, low_int_eigs_arr.shape[-1]))
        object_eigs_arr = np.reshape(object_eigs_arr,  (-1, object_eigs_arr.shape[-1]))
        
        all_model_dists_high_int[r] = high_int_eigs_arr
        all_model_dists_low_int[r] = low_int_eigs_arr
        all_model_dists_object[r] = object_eigs_arr

    return all_model_dists_high_int, all_model_dists_low_int, all_model_dists_object, regions





def _plot_eigs_ablation_all_models(region):

    exp_path = f"results/all_social_mrnn/linear_analysis"
    
    # regions stuff is trash rn
    hi_eigs_ctrl, li_eigs_ctrl, o_eigs_ctrl, regions = _eigs_all_models(region, stim=False)
    hi_eigs_stim, li_eigs_stim, o_eigs_stim, _ = _eigs_all_models(region, stim=True)

    def _kl_div_regions(stim_data, ctrl_data):
        dists = {}
        for r in stim_data:
            dists[r] = []
            for t in range(stim_data[r].shape[0]):
                dists[r].append(wasserstein_distance(stim_data[r][t], ctrl_data[r][t]))
        return dists

    def _make_bar_plot(kl_divs, cond):
        fig, ax = standard_2d_ax()
        ys = []
        errs = []
        for r in kl_divs:
            ys.append(np.mean(kl_divs[r]))
            errs.append(np.std(kl_divs[r]))
        ax.bar(regions, ys, yerr=errs, capsize=10)
        save_fig(os.path.join(exp_path, f"{region}_kl_div_ablation_cond{cond}"))

    hi_kl_divs = _kl_div_regions(hi_eigs_stim, hi_eigs_ctrl)
    li_kl_divs = _kl_div_regions(li_eigs_stim, li_eigs_ctrl)
    o_kl_divs = _kl_div_regions(o_eigs_stim, o_eigs_ctrl)

    _make_bar_plot(hi_kl_divs, 0)
    _make_bar_plot(li_kl_divs, 1)
    _make_bar_plot(o_kl_divs, 2)




def plot_eigs_ablation_all_models_pfc():
    _plot_eigs_ablation_all_models("pfc")

def plot_eigs_ablation_all_models_acc():
    _plot_eigs_ablation_all_models("acc")

def plot_eigs_ablation_all_models_bla():
    _plot_eigs_ablation_all_models("bla")

def plot_eigs_ablation_all_models_ofc():
    _plot_eigs_ablation_all_models("ofc")





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

    elif args.experiment == "plot_eigs_dist_pfc":
        plot_eigs_dist_pfc(args.model_path)
    elif args.experiment == "plot_eigs_dist_acc":
        plot_eigs_dist_acc(args.model_path)
    elif args.experiment == "plot_eigs_dist_bla":
        plot_eigs_dist_bla(args.model_path)
    elif args.experiment == "plot_eigs_dist_ofc":
        plot_eigs_dist_ofc(args.model_path)
    elif args.experiment == "run_all_eigs_dist":
        run_all_eigs_dist(args.model_path)

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

    elif args.experiment == "plot_eigs_ablation_all_models_pfc":
        plot_eigs_ablation_all_models_pfc()
    elif args.experiment == "plot_eigs_ablation_all_models_acc":
        plot_eigs_ablation_all_models_acc()
    elif args.experiment == "plot_eigs_ablation_all_models_bla":
        plot_eigs_ablation_all_models_bla()
    elif args.experiment == "plot_eigs_ablation_all_models_ofc":
        plot_eigs_ablation_all_models_ofc()

    else:
        raise NotImplementedError(f"Experiment {args.experiment} not implemented")

if __name__ == "__main__":
    main()