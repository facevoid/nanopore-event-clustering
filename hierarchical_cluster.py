

import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import os
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
from bokeh.plotting import figure, output_file, save
from bokeh.models import BoxAnnotation
import os
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.fft import rfft


def load_dips_and_extract_features(directory):
    features = []
    filenames = glob.glob(os.path.join(directory, '*.npy'))

    for filename in filenames:
        dip = np.load(filename)
        depth = np.min(dip)
        width = len(dip)
        area = np.trapz(-dip)  # Assuming dip values are negative
        std_dev = np.std(dip)
        skewness = skew(dip)
        kurt = kurtosis(dip)

        # Simple slope features - difference between consecutive points
        slope_start = dip[1] - dip[0]
        slope_end = dip[-1] - dip[-2]

        # Fourier descriptors - using only the first few as features
        fft_coeffs = rfft(dip)
        fft_feature = np.abs(fft_coeffs[1])  # Just as an example, using the second coeff
        
        features.append([depth, width, area, std_dev, skewness, kurt, slope_start, slope_end, fft_feature])
    
    return np.array(features), filenames


    
    
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
        
def plot_dips_by_cluster_matplotlib(clusters, dir_name, dip_sigma, sampling_rate):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for label, filenames in clusters.items():
        cluster_dir = os.path.join(dir_name, f"cluster_{label}")
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)

        # Maximum subplots per figure
        max_subplots_per_fig = 20  # Adjust this number based on your needs

        # Calculate the total number of figures needed
        total_files = len(filenames)
        figures_needed = np.ceil(total_files / max_subplots_per_fig).astype(int)

        for fig_idx in range(figures_needed):
            # Calculate subplots for the current figure
            remaining_files = total_files - fig_idx * max_subplots_per_fig
            subplots_in_this_fig = min(max_subplots_per_fig, remaining_files)
            
            nrows = int(np.ceil(np.sqrt(subplots_in_this_fig)))
            ncols = nrows if subplots_in_this_fig > nrows * (nrows - 1) else nrows - 1

            fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15), squeeze=False)
            fig.suptitle(f'Cluster {label} Dips, Fig {fig_idx + 1}', fontsize=16)

            for i in range(subplots_in_this_fig):
                file_idx = fig_idx * max_subplots_per_fig + i
                filename = filenames[file_idx]
                # Load the dip data
                dip_data = np.load(filename)
                smoothed_dip = gaussian_filter1d(dip_data, sigma=dip_sigma)
                time_axis = np.arange(len(dip_data)) / sampling_rate

                # Find the correct subplot
                ax = axs[i // ncols, i % ncols]
                ax.plot(time_axis, dip_data, label="Original Data", alpha=0.5)
                ax.plot(time_axis, smoothed_dip, label="Smoothed Data", alpha=0.8)
                title = filename.split('/')[-1][:-4]
                ax.set_title(f"{file_idx}: {title}")
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.legend()

            # Adjust layout and save the figure
            for ax in axs.flat:
                ax.set_visible(False)  # Hide all first
            for i in range(subplots_in_this_fig):
                axs[i // ncols, i % ncols].set_visible(True)  # Only show those with data

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect so the title fits
            plt.savefig(os.path.join(cluster_dir, f"all_dips_subplot_fig_{fig_idx + 1}.png"))
            plt.close()

# Function to plot hierarchical clustering dendrogram
def plot_dendrogram(Z, labels, max_d=None):
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
    plt.savefig('plots/hr_cluster/dredogram.png')

# Main script
if __name__ == '__main__':
    directory = "dips/dips_07_16s_46s/*/"  # Update this to your correct directory path
    features, filenames = load_dips_and_extract_features(directory)

    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    # Perform hierarchical clustering
    Z = linkage(features_normalized, 'ward')

    # Plot dendrogram to help determine the number of clusters
    plot_dendrogram(Z, labels=filenames)

    # Optional: Automatically form flat clusters from the hierarchical clustering defined by the given linkage matrix
    max_distance = 4  # Adjust this threshold to cut the dendrogram and form clusters
    labels = fcluster(Z, max_distance, criterion='distance')

    # Organize filenames by cluster
    clusters = defaultdict(list)
    for label, filename in zip(labels, filenames):
        clusters[label].append(filename)

    # For debugging: Print file-cluster assignments
    for filename, label in zip(filenames, labels):
        print(f"{filename} -> Cluster {label}")
    
    # Adjust these parameters as needed
    dir_name = f"plots/clustered_dips_plots_07_16s_46s_{len(clusters)}"
    dip_sigma = 5
    sampling_rate = 250000  # Example sampling rate in Hz
    
    # Plot dips by cluster
    plot_dips_by_cluster_matplotlib(clusters, dir_name, dip_sigma, sampling_rate)

    # Here you can follow up with plotting or processing specific to clusters
    # For instance, you could plot dips by cluster as in the earlier example
    # For debugging: Print file-cluster assignments
    # for filename, label in zip(filenames, labels):
    #     print(f"{filename} -> Cluster {label}")
    # # Adjust these parameters as needed
    # dir_name = "plots/clustered_dips_plots_07_0s_16s"
    # dip_sigma = 5
    # sampling_rate = 250000  # Example sampling rate in Hz
    # plot_dips_by_cluster(clusters, dir_name, dip_sigma, sampling_rate)
    # plot_dips_by_cluster_matplotlib(clusters, dir_name, dip_sigma, sampling_rate)
