import sys
from pathlib import Path

# Add the root directory of the repository to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import torch
from utils.models import Model
from utils.exp_utils import pca_batched, save_fig, load_hp
from utils.train_utils import get_mean_fixation_data, interactivity_input
import config
from utils.plt_utils import ax_3d_no_grid
import os

REGIONS = ["pfc", "acc", "ofc", "bla"]


def plot_pca(act, exp_path, region, data_type):
    fig, ax = ax_3d_no_grid()

    label_str = ["high_interactivity_face", "low_interactivity_face", "object"]
    ax.plot(
        act[0, :, 0],
        act[0, :, 1],
        act[0, :, 2],
        linewidth=4,
        label=label_str[0],
        color="blue",
    )
    ax.plot(
        act[1, :, 0],
        act[1, :, 1],
        act[1, :, 2],
        linewidth=4,
        label=label_str[1],
        color="maroon",
    )
    ax.plot(
        act[2, :, 0],
        act[2, :, 1],
        act[2, :, 2],
        linewidth=4,
        label=label_str[2],
        color="green",
    )
    # ax.legend(loc="best")

    save_path = os.path.join(exp_path, f"{region}_{data_type}_pca")
    save_fig(save_path, eps=True)


def plot_all_pcs(model, exp_path, x, data_type):
    """Plot the PCs for each region in REGIONS"""
    # TODO change this to make it work for latent (get correct output regions)
    for region in REGIONS:
        act = model.mrnn.get_region_activity(x, region)
        act_reduced = pca_batched(act)
        plot_pca(act_reduced, exp_path, region, data_type)


def _plot_pca(model_path, data_type):
    hp = load_hp(model_path)
    exp_path = f"results/{hp['model_save_name']}/pca"

    dataset = get_mean_fixation_data(
        "/Users/lazza/naturalistic_social_gaze_mech/social_gaze"
    )

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
        hp["rec_constrained"],
        hp["inp_constrained"],
        hp["batch_first"],
        hp["spectral_radius"],
        output_layer=hp["output_layer"],
        device="cpu",
    ).cpu()

    checkpoint = torch.load(
        os.path.join(hp["save_dir"], hp["model_save_name"] + ".pth")
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # Start training
    batch, _ = dataset.sample_batch()
    inp = interactivity_input(dataset.group_by_columns, batch.shape[1])
    inp = inp.cpu()

    if hp["output_layer"]:
        xn = torch.zeros(size=(batch.shape[0], model.mrnn.total_num_units))
        hn = torch.zeros(size=(batch.shape[0], model.mrnn.total_num_units))
    else:
        xn = checkpoint["xn_0"].cpu()
        hn = checkpoint["hn_0"].cpu()

    with torch.no_grad():
        out, hn = model(inp, xn, hn, noise=False)

    if data_type == "data":
        plot_all_pcs(model, exp_path, batch, "data")
    elif data_type == "output":
        plot_all_pcs(model, exp_path, out, "output")
    elif data_type == "hidden_activity":
        plot_all_pcs(model, exp_path, hn, "hidden_activity")


def data_pca(model_path):
    _plot_pca(model_path, "data")


def output_pca(model_path):
    _plot_pca(model_path, "output")


def hidden_activity_pca(model_path):
    _plot_pca(model_path, "hidden_activity")


def main():
    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()

    if args.experiment == "data_pca":
        data_pca(args.model_path)
    elif args.experiment == "output_pca":
        output_pca(args.model_path)
    elif args.experiment == "hidden_activity_pca":
        hidden_activity_pca(args.model_path)
    else:
        raise ValueError("Experiment not recognized")


if __name__ == "__main__":
    main()
