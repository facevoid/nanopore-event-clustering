import argparse
from omegaconf import OmegaConf
import pyabf
from signal_processing import SignalProcessor
from utils import find_significant_dips, save_dips_as_npy
from plot_utils import plot_and_save_dips, plot_log_dwell_time_distribution_with_annotations, plot_distributions_with_subplots
import numpy as np
# Load configuration variables from a YAML file
config = OmegaConf.load('configs/event-analysis.yaml')

if __name__ == '__main__':
    version = config.version

    # Configuration variables using OmegaConf
    start_time = config.start_time
    end_time = config.end_time  # Each chunk has 5 seconds of data
    base_sigma = config.base_sigma
    sampling_rate = config.sampling_rate
    plot_dir = f"{config.plot_dir_prefix}/"
    save_dip_dir = f"{config.save_dip_dir_prefix}/"
    plot_dir_rejection = f"{config.plot_dir_prefix}_rejection/" 
    save_dip_dir_rejection = f"{config.save_dip_dir_prefix}_rejection/"
    threshold = config.threshold

    # Load data
    abf_07 = pyabf.ABF(config.data_file_path)
    
    data_chunk = abf_07.sweepY
    if not threshold:
        print('mean threshold used')
        threshold = np.mean(data_chunk) * 0.95
        print(f'threshold {threshold}')

    # Process and analyze data
    smoothed_data_full = SignalProcessor.gaussian_filter(data_chunk, sigma=base_sigma)
    significant_dips, rejected_dips = find_significant_dips(smoothed_data_full, threshold=threshold, min_length=40, min_separation=40)
    
    # plot_and_save_dips(data_chunk, SignalProcessor.gaussian_filter, significant_dips, f"{plot_dir}{start_time}s-{end_time}s", chunk_start_time=start_time, dip_counter=0)
    save_dips_as_npy(data_chunk, significant_dips, f"{save_dip_dir}{start_time}s-{end_time}s", sampling_rate=sampling_rate, chunk_start_time=start_time, dip_counter=0, context=1000)
    # plot_log_dwell_time_distribution_with_annotations(data_chunk, significant_dips, plot_dir)
    # plot_distributions_with_subplots(data_chunk, significant_dips, plot_dir)
    
    # Additional code for handling rejected dips
    # plot_and_save_dips(data_chunk, SignalProcessor.gaussian_filter, rejected_dips, f"{plot_dir_rejection}rejected_{start_time}s-{end_time}s", chunk_start_time=start_time, dip_counter=1)
    save_dips_as_npy(data_chunk, rejected_dips, f"{save_dip_dir_rejection}rejected_{start_time}s-{end_time}s", sampling_rate=sampling_rate, chunk_start_time=start_time, dip_counter=1, context=1000)

    print(f'Total significant dips {len(significant_dips)}')
