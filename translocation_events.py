
import numpy as np
from scipy.ndimage import gaussian_filter1d
import os
import glob
import pyabf 



if __name__ == '__main__':
        start_time = 0 
        total_significant_dip = 0
        
             
        #     data_chunk_path = 'processed_data/chunk_data/chunk_0.npy'
        abf_07 = pyabf.ABF('DATA/raw_bin_data/2023_08_10_0007.abf')
        end_time = 300 #Each chunk has 5 seconds of data
        
        data_chunk = abf_07.sweepY
        smoothed_data_full = gaussian_filter1d(data_chunk, sigma=40)  # Initial smoothing
        significant_dips = find_significant_dips(smoothed_data_full, threshold=700, min_length=45, min_separation=800)
        # plot_and_save_dips(data_chunk, significant_dips, "plots/dips_plots_07_chunk", sigma=20)
        # plot_and_save_dips(data_chunk, smoothed_data_full, significant_dips, threshold=400, dir_name='plots/dips_plots_07_overview_chunks')
        base_sigma=20
        dip_sigma=300
        sampling_rate = 250000
        plot_and_save_dips(data_chunk, base_sigma, dip_sigma, significant_dips, f"plots/dips_plots_07_0s_300s/{start_time}s-{end_time}s")
        save_dips_as_npy(data_chunk, significant_dips, f"dips/dips_07_0s_300ss/{start_time}s-{end_time}s", sampling_rate= sampling_rate,context=1000)
        print(len(significant_dips))
        total_significant_dip += len(significant_dips)
        start_time = end_time
        print(f'Total significant dips {total_significant_dip}')

