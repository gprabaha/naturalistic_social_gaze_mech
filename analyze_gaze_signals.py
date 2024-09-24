#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:19:31 2024

@author: pg496
"""

import os
import re

import util
import filter_behav
import load_data
import fix_and_saccades

import logging

from data_manager import DataManager


def main():
    # Configure the root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    params = {}
    params.update({
        'is_cluster': True,
        'is_grace': False,
        'use_parallel': True,
        'submit_separate_jobs_for_sessions': True,
        'use_toy_data': False,
        'remake_toy_data': False,
        'num_cpus': None,
        'do_local_reclustering_in_parallel': False,

        'remake_gaze_data_dict': True,


        'remake_gaze_position_dict_m1': False,
        'remake_gaze_position_dict_m2': False,
        'remake_gaze_position_labels_m1': False,
        'remake_gaze_position_labels_m2': False,
        'inter_eye_dist_denom_for_eye_bbox_offset': 3,
        'offset_multiples_in_x_dir': 2.5,
        'offset_multiples_in_y_dir': 1.2,
        'bbox_expansion_factor': 1.3,
        'use_euclidean_dist_for_eyes': False,
        'make_eye_bbox_based_on_face': False,

        'pixel_threshold_for_boundary_outlier_removal': 50,
        'fixation_detection_method': 'cluster_fix',
        'do_local_reclustering_in_parallel': False,
        'detect_fixations_and_saccades_again': False,
        'remake_behavioral_dataframes': False,
        'remake_labelled_fixations_m1': False,
        'remake_labelled_fixations_m2': False,
        'remake_labelled_saccades_m1': True,
        'remake_labelled_saccades_m2': True,
        'remake_labelled_microsaccades_m1': True,
        'remake_labelled_microsaccades_m2': True,
        'remake_combined_behav_m1': False,
        'remake_combined_behav_m2': False,
        'remake_events_within_attention_frame_m1': False,
        'remake_events_within_attention_frame_m2': False,

        'remake_binary_vectors_m1': False,
        'remake_binary_vectors_m2': False,


        'remake_labelled_spiketimes': True,
        'remake_labelled_fixation_rasters': True,


        'make_plots': True,
        'plot_clashes': False,
        'recalculate_unit_ROI_responses': True,
        'replot_face/eye_vs_obj_violins': True,
        'remap_source_coord_from_inverted_to_standard_y_axis': True,
        'map_roi_coord_to_eyelink_space': False,
        'map_gaze_pos_coord_to_eyelink_space': True,
        'export_plots_to_local_folder': False,
        
        'raster_bin_size': 0.001,  # in seconds
        'raster_pre_event_time': 0.5,
        'raster_post_event_time': 0.5,
        'flush_before_reload': False,
        'use_existing_variables': False,
        'reload_existing_unit_roi_comp_stats': False
    })
    data_manager = DataManager(params)
    data_manager.run()


if __name__ == "__main__":
    main()


