import os 
import matplotlib.pyplot as plt
import curate_data
import torch
from dataset import MeanFixationDataset
import load_data
import json
import pickle
from matplotlib import rcParams


def initialize_params(
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

def get_mean_fixation_data(path):
    
    #"/Users/John/naturalistic_social_gaze_mech/social_gaze"
    # Load processed dataframe
    params = initialize_params(
        path_name=path
    )
    behav_firing_rate_df_file_path = os.path.join(
        params['processed_data_dir'], 'mean_fixation_response_df.pkl'
    )

    print('loading data...')
    df = load_data.get_data_df(behav_firing_rate_df_file_path)
    print('creating fr dataset...')
    dataset = MeanFixationDataset(df)

    return dataset

def interactivity_input(key, timesteps):
    input_series = []
    for cond in key:
        inp = torch.zeros(size=(1, timesteps, 2))
        if cond == "high_interactivity_face":
            inp[..., 0] = 0.7
        if cond == "low_interactivity_face":
            inp[..., 0] = 0.8
        if cond == "object":
            inp[..., 1] = 0.25
        input_series.append(inp)
    input_series = torch.cat(input_series, dim=0)
    return input_series

def save_hp(hp, model_dir):
    """Save the hyper-parameter file of model save_name"""
    hp_copy = hp.copy()
    with open(os.path.join(model_dir, 'hp.json'), 'w') as f:
        json.dump(hp_copy, f)

def create_dir(save_path):
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(save_path):
        os.makedirs(save_path)

def load_hp(model_dir):
    """Load the hyper-parameter file of model save_name"""
    fname = os.path.join(model_dir, 'hp.json')
    with open(fname, 'r') as f:
        hp = json.load(f)
    return hp

def save_fig(save_path, eps=False): 
    # Tell matplotlib to embed fonts as text, not outlines
    rcParams['pdf.fonttype'] = 42  # 42 = TrueType (editable in Illustrator)
    rcParams['ps.fonttype'] = 42
    # Simple function to save figure while creating dir and closing
    dir = os.path.dirname(save_path)
    create_dir(dir)
    plt.tight_layout()
    if eps:
        plt.savefig(save_path + ".pdf", format="pdf")
    else:
        plt.savefig(save_path + ".png")
    plt.close()

def load_pickle(file):
    try:
        with open(file, 'rb') as f:
            data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', file, ':', e)
        raise
    return data

def stim_inp(mrnn, batch_size, timesteps, *args):
    # Create stimulus input to match the one used in training
    stim_inp = []
    for region in mrnn.region_dict:
        if region in args:
            inp = -5 * torch.ones(size=(batch_size, timesteps, mrnn.get_region_size(region)))
        else:
            inp = torch.zeros(size=(batch_size, timesteps, mrnn.get_region_size(region)))
        stim_inp.append(inp)
    stim_inp = torch.cat(stim_inp, dim=-1)

    return stim_inp