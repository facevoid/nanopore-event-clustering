from bokeh.plotting import figure, output_file, save
from bokeh.models import Span, BoxAnnotation
import numpy as np
from scipy.ndimage import gaussian_filter1d
import os

def find_significant_dips(data, threshold, min_length=200, min_separation=1000):
    under_threshold_indices = np.where(data < threshold)[0]
    if len(under_threshold_indices) == 0:
        return []

    dips = []
    current_dip = [under_threshold_indices[0]]

    for index in under_threshold_indices[1:]:
        if index - current_dip[-1] > 1:
            if len(current_dip) >= min_length:
                dips.append(current_dip)
            current_dip = [index]
        else:
            current_dip.append(index)
    if len(current_dip) >= min_length:
        dips.append(current_dip)

    merged_dips = []
    current_dip = dips[0]
    for next_dip in dips[1:]:
        if next_dip[0] - current_dip[-1] < min_separation:
            current_dip += next_dip
        else:
            merged_dips.append(current_dip)
            current_dip = next_dip
    merged_dips.append(current_dip)

    return [(dip[0], dip[-1]) for dip in merged_dips]

def plot_and_save_dips(data, significant_dips, dir_name, sigma):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    # Smooth the entire data for the overview plot
    smoothed_data = gaussian_filter1d(data, sigma=sigma)
    
    # Plot all dips overview
    output_file(os.path.join(dir_name, "all_dips.html"))
    p = figure(title="All Identified Dips", x_axis_label='Index', y_axis_label='Amplitude')
    p.line(range(len(data)), data, line_color="blue", alpha=0.5, legend_label="Original Data")
    p.line(range(len(smoothed_data)), smoothed_data, line_color="red", alpha=0.8, legend_label="Smoothed Data")
    for start, end in significant_dips:
        p.add_layout(BoxAnnotation(left=start, right=end, fill_alpha=0.3, fill_color='orange'))
    save(p)

    # Plot each significant dip with increased smoothing
    for i, (start, end) in enumerate(significant_dips):
        segment = data[max(start-1000,0):min(end+1000,len(data))]
        smoothed_segment = gaussian_filter1d(segment, sigma=sigma*10)  # Increase smoothing for individual dip plots
        output_file(os.path.join(dir_name, f"dip_{i}.html"))
        p = figure(title=f"Dip {i}", x_axis_label='Index', y_axis_label='Amplitude')
        p.line(range(len(segment)), segment, line_color="blue", alpha=0.5, legend_label="Original Data")
        p.line(range(len(smoothed_segment)), smoothed_segment, line_color="red", alpha=0.8, legend_label="Smoothed Data")
        save(p)
        
def plot_and_save_dips(data, smoothed_data, significant_areas, threshold, dir_name):
    # Ensure the output directory exists
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    # Convert data points for Bokeh compatibility
    x_range = np.arange(len(data))  # Convert range to numpy array
    
    # Plot with all dips
    output_file(f"{dir_name}/all_dips.html")
    p = figure(title="Signal with All Identified Dips", x_axis_label='Index', y_axis_label='Amplitude')
    p.line(x_range, data, legend_label="Original Data", line_color="blue", alpha=0.5)
    p.line(x_range, smoothed_data, legend_label="Smoothed Data", line_color="red", alpha=0.8)
    
    for i, segment in enumerate(significant_areas):
        start, end = segment[0], segment[-1]
        p.add_layout(BoxAnnotation(left=start, right=end, fill_alpha=0.3, fill_color='orange'))
    
    save(p)
    
    # Plot each dip separately
    for i, segment in enumerate(significant_areas):
        start, end = max(segment[0] - 1000, 0), min(segment[-1] + 1000, len(data) - 1)
        output_file(f"{dir_name}/dip_{i}.html")
        p = figure(title=f"Dip {i}", x_axis_label='Index', y_axis_label='Amplitude')
        p.line(np.arange(start, end), data[start:end], line_color="blue", alpha=0.5)
        p.line(np.arange(start, end), smoothed_data[start:end], line_color="red", alpha=0.8)
        p.add_layout(BoxAnnotation(left=segment[0], right=segment[-1], fill_alpha=0.3, fill_color='orange'))
        save(p)
# Example usage
def plot_and_save_dips(data, smoothed_data, significant_areas, threshold, dir_name):
    # Ensure the output directory exists
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    # Convert data points for Bokeh compatibility
    x_range = np.arange(len(data))  # Convert range to numpy array
    
    # Plot with all dips
    output_file(f"{dir_name}/all_dips.html")
    p = figure(title="Signal with All Identified Dips", x_axis_label='Index', y_axis_label='Amplitude')
    p.line(x_range, data, legend_label="Original Data", line_color="blue", alpha=0.5)
    p.line(x_range, smoothed_data, legend_label="Smoothed Data", line_color="red", alpha=0.8)
    
    for i, segment in enumerate(significant_areas):
        start, end = segment[0], segment[-1]
        p.add_layout(BoxAnnotation(left=start, right=end, fill_alpha=0.3, fill_color='orange'))
    
    save(p)
    
    # Plot each dip separately
    for i, segment in enumerate(significant_areas):
        start, end = max(segment[0] - 1000, 0), min(segment[-1] + 1000, len(data) - 1)
        output_file(f"{dir_name}/dip_{i}.html")
        p = figure(title=f"Dip {i}", x_axis_label='Index', y_axis_label='Amplitude')
        p.line(np.arange(start, end), data[start:end], line_color="blue", alpha=0.5)
        p.line(np.arange(start, end), smoothed_data[start:end], line_color="red", alpha=0.8)
        p.add_layout(BoxAnnotation(left=segment[0], right=segment[-1], fill_alpha=0.3, fill_color='orange'))
        save(p)
 
def plot_and_save_individual_dips(data, significant_dips, dir_name, base_sigma, dip_sigma):
    # Ensure the output directory exists
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for i, (start, end) in enumerate(significant_dips):
        # Define the segment including some context around the dip
        context = 1000  # Number of points before and after the dip to include in the plot
        segment_start = max(start - context, 0)
        segment_end = min(end + context, len(data))
        
        # Ensure x_range is explicitly a NumPy array
        x_range = np.arange(segment_start, segment_end)

        segment = data[segment_start:segment_end]
        # Apply higher smoothing specifically to this segment
        smoothed_segment = gaussian_filter1d(segment, sigma=dip_sigma)

        # Plotting
        output_file(f"{dir_name}/dip_{i}.html")
        p = figure(title=f"Dip {i}", x_axis_label='Index', y_axis_label='Amplitude')
        
        # Plot using the explicitly defined x_range
        p.line(x_range, segment, line_color="blue", alpha=0.5, legend_label="Original Data")
        p.line(x_range, smoothed_segment, line_color="red", alpha=0.8, legend_label="Smoothed Data")
        
        save(p)


data_chunk_path = 'processed_data/chunk_data/chunk_0.npy'
data_chunk = np.load(data_chunk_path)
smoothed_data_full = gaussian_filter1d(data_chunk, sigma=20)  # Initial smoothing
significant_dips = find_significant_dips(smoothed_data_full, threshold=600, min_length=200, min_separation=1000)
# plot_and_save_dips(data_chunk, significant_dips, "plots/dips_plots_07_chunk", sigma=20)
plot_and_save_dips(data_chunk, smoothed_data_full, significant_dips, threshold=400, dir_name='plots/dips_plots_07_overview_chunks')
plot_and_save_individual_dips(data_chunk, significant_dips, "plots/dips_plots_07_individual", base_sigma=20, dip_sigma=200)