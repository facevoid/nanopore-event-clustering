import os 
import matplotlib.pyplot as plt 
import numpy as np 
from bokeh.plotting import figure, output_file, save
from bokeh.models import Span, BoxAnnotation
from scipy.cluster.hierarchy import dendrogram
from smoothness_analysis import calculate_symmetry


            


def plot_and_save_dips(data, smooth_function, significant_areas, dir_name, chunk_start_time, dip_counter, sampling_rate=250000):
    # Ensure the output directory exists
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
    # global_min_amplitude = float('inf')
    # global_max_amplitude = float('-inf')

    # First pass: find the global min and max amplitude to use consistent y-axis for all subplots
    # Initialize these to None so that we can update them with actual data values later
    global_min_amplitude = None
    global_max_amplitude = None

    for filenames in clusters.values():
        for filename in filenames:
            dip_data = np.load(filename)
            
            # Calculate the quartiles and the IQR
            Q1 = np.percentile(dip_data, 25)
            Q3 = np.percentile(dip_data, 75)
            IQR = Q3 - Q1
            
            # Define bounds to identify outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Filter the data to exclude outliers
            filtered_data = dip_data[(dip_data >= lower_bound) & (dip_data <= upper_bound)]
            
            # It's possible all data are considered outliers, so check if filtered_data is not empty
            if filtered_data.size > 0:
                # Update global min and max with filtered data
                min_amplitude = np.min(filtered_data)
                max_amplitude = np.max(filtered_data)
                
                # Initialize global_min_amplitude and global_max_amplitude if they are None
                if global_min_amplitude is None or global_max_amplitude is None:
                    global_min_amplitude = min_amplitude
                    global_max_amplitude = max_amplitude
                else:
                    global_min_amplitude = min(global_min_amplitude, min_amplitude)
                    global_max_amplitude = max(global_max_amplitude, max_amplitude)

    # Second pass: create plots with consistent y-axis
    for label, filenames in clusters.items():
        cluster_dir = os.path.join(dir_name, f"")
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
                dip_data = dip_data[700:-700]
                if smoothing_function is not None:
                    smoothed_dip = smoothing_function(dip_data)
                else:
                    smoothed_dip = dip_data
                # smoothness_score = calculate_smoothness(smoothed_dip)
                symmetry_score = calculate_symmetry(smoothed_dip[:])
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


def plot_dwell_time_distribution(data_chunk, significant_dips, plot_dir):
    # Assuming data_chunk is a 2D array with the first column being time and the second current.
    time = [i for i in range(len(data_chunk))]
    current = data_chunk[:]
    
    # Prepare log values for current, ignoring negative values.
    log_current = [np.log(c) if c > 0 else 0 for c in current]

    # Calculate dwelling times for all significant dips.
    dwelling_times = [time[end_idx] - time[start_idx] for start_idx, end_idx in significant_dips]

    plt.figure(figsize=(10, 6))

    # Plot the histogram of dwelling times.
    plt.hist(dwelling_times, bins='auto', color='skyblue', alpha=0.7, rwidth=0.85)

    plt.xlabel('Dwelling Time (s)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Dwelling Times for Significant Dips')
    plt.grid(axis='y', alpha=0.75)

    plt.savefig(plot_dir + '/dwelling_time_distribution.png')
    
def plot_delta_current_distribution(data_chunk, plot_dir):
    # Generate a time sequence based on the length of the data_chunk.
    time = [i for i in range(len(data_chunk))]
    # Assuming data_chunk is a list of current readings.
    current = data_chunk[:]
    
    # Calculate delta current as the difference between consecutive current readings.
    # To calculate delta current, first convert the list to a numpy array.
    current_array = np.array(current)
    delta_current = np.diff(current_array)
    
    plt.figure(figsize=(10, 6))
    
    # Plot the histogram of delta current.
    plt.hist(delta_current, bins='auto', color='skyblue', alpha=0.7, rwidth=0.85)
    
    plt.xlabel('Delta Current (A)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Delta Current')
    plt.grid(axis='y', alpha=0.75)
    
    # Save the plot to the specified directory.
    plt.savefig(f"{plot_dir}/delta_current_distribution.png")
    
def plot_log_dwell_time_distribution_with_annotations(data_chunk, significant_dips, plot_dir):
    # Assuming data_chunk is a 2D array with the first column being time and the second current.
    time = [i for i in range(len(data_chunk))]
    current = data_chunk[:]
    
    # Calculate dwelling times for all significant dips.
    dwelling_times = [time[end_idx] - time[start_idx] for start_idx, end_idx in significant_dips]

    plt.figure(figsize=(10, 6))

    # Plot the histogram of dwelling times with a logarithmic y-axis and capture the bin counts.
    n, bins, patches = plt.hist(dwelling_times, bins='auto', color='skyblue', alpha=0.7, rwidth=0.85, log=True)

    # Annotate each bar with the count of items.
    for i in range(len(n)):
        plt.text(bins[i]+((bins[i+1]-bins[i])/2), n[i], str(int(n[i])), ha='center', va='bottom', fontsize=9)

    plt.xlabel('Dwelling Time (s)')
    plt.ylabel('Frequency (Log Scale)')
    plt.title('Logarithmic Distribution of Dwelling Times for Significant Dips')
    plt.grid(axis='y', alpha=0.75)

    plt.savefig(plot_dir + '/log_dwelling_time_distribution_annotated.png')
    
def plot_distributions_with_subplots(data_chunk, significant_dips, plot_dir):
    # data_chunk is a 1D list or array of current values.
    
    # Initialize lists to store delta currents and minimum currents for all significant dips.
    delta_currents = []
    min_currents = []
    for start_idx, end_idx in significant_dips:
        dip_currents = data_chunk[start_idx:end_idx+1]  # Include end_idx in the slice
        delta_currents.append(dip_currents[-1] - dip_currents[0])
        min_currents.append(min(dip_currents))

    plt.figure(figsize=(12, 6))

    # Subplot 1: Delta Currents Distribution
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, subplot 1
    n, bins, patches = plt.hist(delta_currents, bins='auto', color='skyblue', alpha=0.7, rwidth=0.85, log=True)
    plt.title('Delta Currents for Significant Dips')
    plt.xlabel('Delta Current (A)')
    plt.ylabel('Frequency (Log Scale)')
    plt.grid(axis='y', alpha=0.75)

    # Subplot 2: Minimum Currents Distribution
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, subplot 2
    plt.hist(min_currents, bins='auto', color='lightgreen', alpha=0.7, rwidth=0.85, log=True)
    plt.title('Minimum Currents for Significant Dips')
    plt.xlabel('Minimum Current (A)')
    plt.ylabel('Frequency (Log Scale)')
    plt.grid(axis='y', alpha=0.75)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure with both subplots
    plt.savefig(f"{plot_dir}/distributions_with_subplots.png")