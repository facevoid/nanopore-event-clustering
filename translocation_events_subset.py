from bokeh.plotting import figure, output_file, save
from bokeh.io import export_png
from bokeh.models import Span, BoxAnnotation
import numpy as np
from scipy.ndimage import gaussian_filter1d
import os
import glob 

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

def save_dips_as_npy(data, significant_dips, dir_name, context=1000):
    """
    Saves segments of the data corresponding to significant dips as .npy files.

    Parameters:
    - data: The entire dataset as a NumPy array.
    - significant_dips: A list of tuples, where each tuple contains the start and end indices of a dip.
    - dir_name: Directory where the .npy files will be saved.
    - context: Number of points before and after the dip to include in the saved segment.
    """
    
    # Ensure the output directory exists
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    for i, (start, end) in enumerate(significant_dips):
        # Define segment with some context around the dip
        segment_start = max(start - context, 0)
        segment_end = min(end + context, len(data))
        segment_data = data[segment_start:segment_end]
        
        # Save the segment as a .npy file
        np.save(os.path.join(dir_name, f"dip_{i}.npy"), segment_data)

def plot_and_save_dips(data, base_sigma, dip_sigma, significant_areas, dir_name):
    # Ensure the output directory exists
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    # Convert data points for Bokeh compatibility
    x_range = np.arange(len(data))
    
    # Smooth the entire dataset for the overview plot
    smoothed_data = gaussian_filter1d(data, sigma=base_sigma)
    
    # Plot with all dips overview
    output_file(os.path.join(dir_name, "all_dips.html"))
    
    p = figure(title="Signal with All Identified Dips", x_axis_label='Index', y_axis_label='Amplitude')
    p.line(x_range, data, legend_label="Original Data", line_color="blue", alpha=0.5)
    p.line(x_range, smoothed_data, legend_label="Smoothed Data", line_color="red", alpha=0.8)
    
    for i, (start, end) in enumerate(significant_areas):
        p.add_layout(BoxAnnotation(left=start, right=end, fill_alpha=0.3, fill_color='orange'))
        
    
    save(p)
    # export_png(p, os.path.join(dir_name, "all_dips.png"))
    # export_png(p, filename=os.path.join(dir_name, "all_dips.png")) 

    
    # Plot each dip separately with higher smoothing
    for i, (start, end) in enumerate(significant_areas):
        # Define segment with some context around the dip
        segment_start = max(start - 1000, 0)
        segment_end = min(end + 1000, len(data))
        segment_data = data[segment_start:segment_end]
        
        # Apply higher smoothing to this segment
        smoothed_segment = gaussian_filter1d(segment_data, sigma=dip_sigma)
        
        output_file(os.path.join(dir_name, f"dip_{i}.html"))
        p = figure(title=f"Dip {i}", x_axis_label='Index', y_axis_label='Amplitude')
        x_range = np.arange(segment_start, segment_end) 
        # Plot the segment before and after smoothing
        p.line(x_range, segment_data, line_color="blue", alpha=0.5, legend_label="Original Data")
        p.line(x_range, smoothed_segment, line_color="red", alpha=0.8, legend_label="Smoothed Data")
        p.add_layout(BoxAnnotation(left=start, right=end, fill_alpha=0.3, fill_color='orange'))
        
        save(p)


if __name__ == '__main__':
        start_time = 16 
        total_significant_dip = 0
        chunk_files = glob.glob('processed_data/chunk_data/*.npy') #16-46s [t=30s, each chunk has 5 seconds]
        for chunk_id, chunk_file in enumerate(chunk_files):
             
        #     data_chunk_path = 'processed_data/chunk_data/chunk_0.npy'
            end_time = start_time + 5 #Each chunk has 5 seconds of data
            
            data_chunk = np.load(chunk_file)
            smoothed_data_full = gaussian_filter1d(data_chunk, sigma=20)  # Initial smoothing
            significant_dips = find_significant_dips(smoothed_data_full, threshold=895, min_length=155, min_separation=1000)
            # plot_and_save_dips(data_chunk, significant_dips, "plots/dips_plots_07_chunk", sigma=20)
            # plot_and_save_dips(data_chunk, smoothed_data_full, significant_dips, threshold=400, dir_name='plots/dips_plots_07_overview_chunks')
            base_sigma=20
            dip_sigma=300
            plot_and_save_dips(data_chunk, base_sigma, dip_sigma, significant_dips, f"plots/dips_plots_07_16s_46s/{start_time}s-{end_time}s")
            save_dips_as_npy(data_chunk, significant_dips, f"dips/dips_07_16s_46s/{start_time}s-{end_time}s", context=1000)
            print(len(significant_dips))
            total_significant_dip += len(significant_dips)
            start_time = end_time
        print(f'Total significant dips {total_significant_dip}')

