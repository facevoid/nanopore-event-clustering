import os 
import matplotlib.pyplot as plt 
import numpy as np 
from bokeh.plotting import figure, output_file, save
from bokeh.io import export_png
from bokeh.models import Span, BoxAnnotation
from scipy.cluster.hierarchy import dendrogram
from smoothness_analysis import calculate_smoothness, calculate_symmetry


            


def plot_and_save_dips(data, smooth_function, significant_areas, dir_name, chunk_start_time, dip_counter, sampling_rate=250000):
    # Ensure the output directory exists
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    # Convert data points for Bokeh compatibility
    x_range = np.arange(len(data))
    
    # Smooth the entire dataset for the overview plot
    smoothed_data = smooth_function(data)
    
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
        smoothed_segment = smooth_function(segment_data)
        # Convert indices to time (in seconds)
        start_time = start / sampling_rate + chunk_start_time
        end_time = end / sampling_rate + chunk_start_time
        filename = f"dip_{i+dip_counter}_start_{start_time:.6f}s_end_{end_time:.6f}s.html"
         
        output_file(os.path.join(dir_name, filename))
        p = figure(title=f"Dip {i+dip_counter}", x_axis_label='Index', y_axis_label='Amplitude')
        x_range = np.arange(segment_start, segment_end) 
        # Plot the segment before and after smoothing
        p.line(x_range, segment_data, line_color="blue", alpha=0.5, legend_label="Original Data")
        p.line(x_range, smoothed_segment, line_color="red", alpha=0.8, legend_label="Smoothed Data")
        p.add_layout(BoxAnnotation(left=start, right=end, fill_alpha=0.3, fill_color='orange'))
        
        save(p)


def plot_dips_by_cluster(clusters, dir_name, dip_sigma, sampling_rate):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for label, filenames in clusters.items():
        cluster_dir = os.path.join(dir_name, f"cluster_{label}")
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)

        # Prepare Matplotlib figure for aggregated cluster plot
        plt.figure(figsize=(10, 6))
        
        for i, filename in enumerate(filenames):
            # Load the dip data
            dip_data = np.load(filename)
            smoothed_dip = gaussian_filter1d(dip_data, sigma=dip_sigma)
            
            # Bokeh plot for each dip
            output_filename = filename.split('/')[-1][:-4] + '.html'
            output_file(os.path.join(cluster_dir, output_filename))
            p = figure(title=f"Cluster {label} - Dip {i}", x_axis_label='Time (s)', y_axis_label='Amplitude')
            time_axis = np.arange(len(dip_data)) / sampling_rate
            p.line(time_axis, dip_data, line_color="blue", alpha=0.5, legend_label="Original Data")
            p.line(time_axis, smoothed_dip, line_color="red", alpha=0.8, legend_label="Smoothed Data")
            save(p)
            
            # Add to Matplotlib aggregated plot
            plt.plot(time_axis, smoothed_dip, label=f"Dip {i}")
        
        plt.title(f"Cluster {label} - All Dips")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.savefig(os.path.join(cluster_dir, "aggregated_cluster_dips.png"))
        plt.close()
        
def plot_dips_by_cluster_matplotlib(clusters, dir_name, sampling_rate, smoothing_function):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Variables to hold the global min and max for amplitude across all dips
    global_min_amplitude = float('inf')
    global_max_amplitude = float('-inf')

    # First pass: find the global min and max amplitude to use consistent y-axis for all subplots
    for filenames in clusters.values():
        for filename in filenames:
            dip_data = np.load(filename)
            global_min_amplitude = min(global_min_amplitude, np.min(dip_data))
            global_max_amplitude = max(global_max_amplitude, np.max(dip_data))

    # Second pass: create plots with consistent y-axis
    for label, filenames in clusters.items():
        cluster_dir = os.path.join(dir_name, f"cluster_{label}")
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)

        # Maximum subplots per figure
        max_subplots_per_fig = 18  # Adjust this number based on your needs
        total_files = len(filenames)
        figures_needed = np.ceil(total_files / max_subplots_per_fig).astype(int)

        for fig_idx in range(figures_needed):
            remaining_files = total_files - fig_idx * max_subplots_per_fig
            subplots_in_this_fig = min(max_subplots_per_fig, remaining_files)
            
            ncols = 2  # Two images per row
            nrows = int(np.ceil(subplots_in_this_fig / ncols))  #

            fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15), squeeze=False)
            fig.suptitle(f'Cluster {label} Dips, Fig {fig_idx + 1}', fontsize=16)

            for i in range(subplots_in_this_fig):
                file_idx = fig_idx * max_subplots_per_fig + i
                filename = filenames[file_idx]
                dip_data = np.load(filename)
                if smoothing_function is not None:
                    smoothed_dip = smoothing_function(dip_data)
                else:
                    smoothed_dip = dip_data
                smoothness_score = calculate_smoothness(smoothed_dip)
                symmetry_score = calculate_symmetry(smoothed_dip)
                time_axis = np.arange(len(dip_data)) / sampling_rate

                ax = axs[i // ncols, i % ncols]
                ax.plot(time_axis, dip_data,  alpha=0.5)
                ax.plot(time_axis, smoothed_dip,  alpha=0.8)
                title = filename.split('/')[-1][:-4]
                ax.set_title(f"{title}")
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.legend()

                # Set consistent x and y axis limits
                ax.set_xlim([0, np.max(time_axis)])
                ax.set_ylim([global_min_amplitude, global_max_amplitude])

            for ax in axs.flat:
                ax.set_visible(False)  # Hide all first
            for i in range(subplots_in_this_fig):
                axs[i // ncols, i % ncols].set_visible(True)  # Only show those with data

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect so the title fits
            plt.savefig(os.path.join(cluster_dir, f"cluster_{label}_fig_{fig_idx + 1}.png"))
            plt.close()
# Function to plot hierarchical clustering dendrogram
def plot_dendrogram(Z, labels, max_d=None, plot_dir=None):
    plt.figure(figsize=(10, 7))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    dendrogram(
        Z,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=12,  # show only the last p merged clusters
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,
        labels=labels,
        color_threshold=max_d,
    )
    if max_d:
        plt.axhline(y=max_d, c='k')
    plt.savefig(f'{plot_dir}.png')

