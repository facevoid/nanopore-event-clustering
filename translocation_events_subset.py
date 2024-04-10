
import numpy as np

import os
import glob 
from signal_processing import SignalProcessor
from utils import find_significant_dips, save_dips_as_npy
from plot_utils import plot_and_save_dips

if __name__ == '__main__':
        start_time = 16 
        total_significant_dip = 0
        plot_dir = 'plots/dips_plots_07_16s_46s_soft/'
        save_dip_dir = 'dips/dips_07_16s_46s_soft/'
        chunk_files = glob.glob('DATA/raw_chunk_data/chunk_data_07/*.npy') #16-46s [t=30s, each chunk has 5 seconds]
        dip_counter = 0
        for chunk_id, chunk_file in enumerate(chunk_files):
             
        #     data_chunk_path = 'processed_data/chunk_data/chunk_0.npy'
            end_time = start_time + 5 #Each chunk has 5 seconds of data

            base_sigma=35
            data_chunk = np.load(chunk_file)
            smoothed_data_full = SignalProcessor.gaussian_filter(data_chunk, sigma=base_sigma)  # Initial smoothing
            significant_dips = find_significant_dips(smoothed_data_full, threshold=830, min_length=50, min_separation=50)
            # plot_and_save_dips(data_chunk, significant_dips, "plots/dips_plots_07_chunk", sigma=20)
            # plot_and_save_dips(data_chunk, smoothed_data_full, significant_dips, threshold=400, dir_name='plots/dips_plots_07_overview_chunks')
            
            dip_sigma=50
            sampling_rate = 250000
            plot_and_save_dips(data_chunk, SignalProcessor.gaussian_filter , significant_dips, f"{plot_dir}{start_time}s-{end_time}s",chunk_start_time=start_time, dip_counter=dip_counter)
            save_dips_as_npy(data_chunk, significant_dips, f"{save_dip_dir}{start_time}s-{end_time}s", sampling_rate= sampling_rate,chunk_start_time=start_time,dip_counter=dip_counter, context=1000)
            print(len(significant_dips))
            dip_counter += len(significant_dips)
            total_significant_dip += len(significant_dips)
            start_time = end_time
        print(f'Total significant dips {total_significant_dip}')

