import glob
import pyabf
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from bokeh.plotting import figure, output_file, save
from scipy.signal import find_peaks


def load_data(file_path):
    abf = pyabf.ABF(file_path)
    return abf.sweepY, abf.dataRate  # returning the data and the sampling rate

def smooth_data(data, sigma=40):
    return gaussian_filter1d(data, sigma=sigma)

def save_smoothed_data(data, filename):
    np.save(filename, data)

def plot_chunk(time_points, original_data, smoothed_data, peaks, dips, title, x_label, y_label, output_file_path):
    output_file(output_file_path)
    p = figure(title=title, x_axis_label=x_label, y_axis_label=y_label)
    if original_data is not None:
        p.line(time_points, original_data, legend_label="Original Data", line_width=1, line_color="blue", alpha=0.5)
    if smoothed_data is not None:
        p.line(time_points, smoothed_data, legend_label="Smoothed Data", line_width=2, line_color="red", alpha=0.8)
    if peaks is not None:
        peak_heights = smoothed_data[peaks]
        peak_times = time_points[peaks]
        p.scatter(peak_times, peak_heights, size=5, color="green", legend_label="Peaks")
    
    if dips is not None:
        dip_heights = smoothed_data[dips]
        dip_times = time_points[dips]
        p.scatter(dip_times, dip_heights, size=5, color="orange", legend_label="Dips")
    print(dips)

    save(p)


def process_data_subset(file_path, start_time_seconds, end_time_seconds, smooth_sigma, chunk_size, output_folder):
    abf_data, sampling_rate = load_data(file_path)
    start_index = int(start_time_seconds * sampling_rate)
    end_index = int(end_time_seconds * sampling_rate)
    subset_data = abf_data[start_index:end_index]
    time_points_subset = np.arange(start_index, end_index) / sampling_rate
    smoothed_subset_data = smooth_data(subset_data, sigma=smooth_sigma)
    
    for i in range(0, len(subset_data), chunk_size):
        chunk_end = i + chunk_size if i + chunk_size < len(subset_data) else len(subset_data)
        time_chunk = time_points_subset[i:chunk_end] - start_time_seconds
        data_chunk = subset_data[i:chunk_end]
        smoothed_data_chunk = smoothed_subset_data[i:chunk_end]
        output_file_path = f"{output_folder}/chunk_{i//chunk_size}.html"
        
        peaks, _ = find_peaks(-smoothed_data_chunk) #invert to find dips
        
        
        # Find dips in the smoothed data chunk
        dips = argrelextrema(smoothed_data_chunk, np.less)[0]
        dips_2 = argrelextrema(data_chunk, np.less)[0] 
        print(dips, dips_2)
        plot_chunk(time_chunk, data_chunk, smoothed_data_chunk, peaks, dips, "Data Chunk", "Time (s)", "Amplitude", output_file_path)


# Example usage
file_path = 'DATA/raw_bin_data/2023_08_10_0007.abf'  # Update with your actual file path
start_time_seconds = 16.00
end_time_seconds = 46.00
smooth_sigma = 500
chunk_size = 250000 * 5  # Adjust chunk size if necessary
output_folder = "plots/htmls_with_peaks"  # Ensure this folder exists or create it as needed

process_data_subset(file_path, start_time_seconds, end_time_seconds, smooth_sigma, chunk_size, output_folder)
