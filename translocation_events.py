
import numpy as np
from scipy.ndimage import gaussian_filter1d
import os
import glob
import pyabf 
from signal_processing import SignalProcessor
from utils import find_significant_dips, save_dips_as_npy
from plot_utils import plot_and_save_dips, plot_log_dwell_time_distribution_with_annotations, plot_distributions_with_subplots 
import sys 
import argparse

if __name__ == '__main__':
        # Create the parser
        parser = argparse.ArgumentParser()

        # Add the --version argument
        parser.add_argument('--version', type=int, help='Version number')

        # Parse the command-line arguments
        args = parser.parse_args()

        start_time = 0 
        total_significant_dip = 0
        base_sigma=35
        version = 8
        plot_dir = f'plots/dips_plots_07_0s_300s_soft_v{args.version}/'
        save_dip_dir = f'dips/dips_07_0s_300s_soft_v{args.version}/'      
        sampling_rate = 250000
        #     data_chunk_path = 'processed_data/chunk_data/chunk_0.npy'
        abf_07 = pyabf.ABF('DATA/raw_bin_data/2023_08_10_0007.abf')
        end_time = 300 #Each chunk has 5 seconds of data
        data_chunk = abf_07.sweepY
        dip_counter = 0
             
        
        smoothed_data_full = SignalProcessor.gaussian_filter(data_chunk, sigma=base_sigma)  # Initial smoothing
        significant_dips = find_significant_dips(smoothed_data_full, threshold=830, min_length=40, min_separation=40)
        
        plot_and_save_dips(data_chunk, SignalProcessor.gaussian_filter , significant_dips, f"{plot_dir}{start_time}s-{end_time}s",chunk_start_time=start_time, dip_counter=dip_counter)
        save_dips_as_npy(data_chunk, significant_dips, f"{save_dip_dir}{start_time}s-{end_time}s", sampling_rate= sampling_rate,chunk_start_time=start_time,dip_counter=dip_counter, context=1000)
        plot_log_dwell_time_distribution_with_annotations(data_chunk, significant_dips, plot_dir)
        plot_distributions_with_subplots(data_chunk, significant_dips, plot_dir)
        total_significant_dip += len(significant_dips)
        start_time = end_time
        print(f'Total significant dips {total_significant_dip}')

