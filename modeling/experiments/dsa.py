import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import numpy as np
from sklearn.decomposition import PCA


def dsa_similarity_matrix(model_name):

    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/{model_name}/compositionality/dsa"

    options = {"batch_size": 32*4, "reach_conds": np.tile(np.arange(0, 32, 1), 4), "speed_cond": 5}

    trial_data_h = []
    trial_data_colors = []

    for env in env_dict:

        trial_data = _test(model_path, model_file, options, env=env_dict[env], noise=True)

        if env == "DlyFullReach" or env == "DlyFullCircleClk" or env == "DlyFullCircleCClk" or env == "DlyFigure8" or env == "DlyFigure8Inv":

            halfway = int((trial_data["epoch_bounds"]["movement"][0] + trial_data["epoch_bounds"]["movement"][1]) / 2)
            extend = trial_data["h"][:, trial_data["epoch_bounds"]["movement"][0]:halfway]
            retract = trial_data["h"][:, halfway:trial_data["epoch_bounds"]["movement"][1]]

            pca_extend = PCA(n_components=12)
            extend_reduced = pca_extend.fit_transform(extend.reshape((-1, extend.shape[-1])))
            extend_reduced = extend_reduced.reshape((extend.shape[0], extend.shape[1], 12))

            pca_retract = PCA(n_components=12)
            retract_reduced = pca_retract.fit_transform(retract.reshape((-1, retract.shape[-1])))
            retract_reduced = retract_reduced.reshape((retract.shape[0], retract.shape[1], 12))

            trial_data_h.append(extend_reduced)
            trial_data_colors.append("pink")

            trial_data_h.append(retract_reduced)
            trial_data_colors.append("purple")
        
        else:

            extend = trial_data["h"][:, trial_data["epoch_bounds"]["movement"][0]:trial_data["epoch_bounds"]["movement"][1]]
            pca_extend = PCA(n_components=12)
            extend_reduced = pca_extend.fit_transform(extend.reshape((-1, extend.shape[-1])))
            extend_reduced = extend_reduced.reshape((extend.shape[0], extend.shape[1], 12))

            trial_data_h.append(extend_reduced)
            trial_data_colors.append("blue")

    # TODO play around with hyperparameters
    dsa = DSA(trial_data_h, n_delays=90, rank=150, verbose=True, score_method="euclidean", device="cpu")
    similarities = dsa.fit_score()

    dsa_data = {"similarities": similarities, "colors": trial_data_colors}

    with open(os.path.join(model_path, "dsa_similarity.txt"), 'wb') as f:
        pickle.dump(dsa_data, f)
    
    dsa_scatter(model_name)


def dsa_scatter(model_name):

    model_path = f"checkpoints/{model_name}"
    exp_path = f"results/{model_name}/compositionality/dsa"

    fig, ax = standard_2d_ax()

    dsa_data = load_pickle(os.path.join(model_path, "dsa_similarity.txt"))
    similarities = dsa_data["similarities"]
    colors = dsa_data["colors"]

    reduced = PCA(n_components=2).fit_transform(similarities)
    ax.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.75, s=250)
    ax.set_xticks([])
    ax.set_yticks([])
    save_fig(os.path.join(exp_path, f"neural_dsa_scatter"), eps=True)


def dsa_heatmap(model_name):

    model_path = f"checkpoints/{model_name}"
    exp_path = f"results/{model_name}/compositionality/dsa"

    # Create figure and 3D axes
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)  # or projection='3d'

    dsa_data = load_pickle(os.path.join(model_path, "dsa_similarity.txt"))
    similarities = dsa_data["similarities"]

    # Reorder indices
    indices_extension = [0, 1, 2, 3, 4]
    indices_extension_long = [5, 7, 9, 11, 13]
    indices_retraction = [6, 8, 10, 12, 14]

    # full reordering index
    new_order = indices_extension + indices_extension_long + indices_retraction

    # reorder rows and columns at once
    re_similarity = similarities[np.ix_(new_order, new_order)]

    sns.heatmap(re_similarity, cmap="Purples")
    ax.set_xticks([])
    ax.set_yticks([])
    save_fig(os.path.join(exp_path, f"neural_dsa_similarity_vis"), eps=True)