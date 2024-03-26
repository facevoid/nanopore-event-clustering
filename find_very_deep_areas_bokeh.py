from bokeh.plotting import figure, output_file, save
from bokeh.models import Span, BoxAnnotation
import numpy as np
from scipy.ndimage import gaussian_filter1d
import os

def find_very_deep_areas_without_length_constraint(data, threshold=600):
    return np.where(data <= threshold)[0]

def find_significant_dips(data, threshold, min_length=200, min_separation=1000):
    under_threshold_indices = np.where(data < threshold)[0]
    if len(under_threshold_indices) == 0:
        return []

    dips = []
    current_dip = [under_threshold_indices[0]]

    for index in under_threshold_indices[1:]:
        if index - current_dip[-1] > 1:
            # Check if the current dip is long enough to be significant
            if len(current_dip) >= min_length:
                dips.append(current_dip)
            current_dip = [index]
        else:
            current_dip.append(index)

    # Check the last dip
    if len(current_dip) >= min_length:
        dips.append(current_dip)

    # Merge dips that are too close to each other
    merged_dips = []
    current_dip = dips[0]

    for next_dip in dips[1:]:
        if next_dip[0] - current_dip[-1] < min_separation:
            # Merge the current and next dips
            current_dip = current_dip + next_dip
        else:
            merged_dips.append(current_dip)
            current_dip = next_dip
    merged_dips.append(current_dip)  # Add the last dip

    # Convert index lists to start-end pairs
    significant_dips = [(dip[0], dip[-1]) for dip in merged_dips]

    return significant_dips

def group_consecutive_indices(indices):
    return np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)

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
data_chunk_path = 'processed_data/chunk_data/chunk_0.npy'
data_chunk = np.load(data_chunk_path)
smoothed_data_full = gaussian_filter1d(data_chunk, sigma=20)  # Adjust sigma as needed

very_deep_area_indices_no_length_constraint = find_very_deep_areas_without_length_constraint(data_chunk)
contiguous_deep_areas = group_consecutive_indices(very_deep_area_indices_no_length_constraint)
significant_contiguous_deep_areas = [segment for segment in contiguous_deep_areas if len(segment) > 10]
significant_dips = find_significant_dips(smoothed_data_full, threshold=600, min_length=1000, min_separation=1000)


print(len(significant_dips))
plot_and_save_dips(data_chunk, smoothed_data_full, significant_contiguous_deep_areas, 400, "plots/dips_plots")
plot_and_save_dips(data_chunk, smoothed_data_full, significant_dips, 400, "plots/dips_plots_significant")
