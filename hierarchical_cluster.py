

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
import numpy as np
import glob
from scipy.stats import skew, kurtosis
from scipy.fft import rfft
from scipy.signal import find_peaks, peak_widths
from utils import load_dips_and_extract_all_features, select_features
from plot_utils import plot_dendrogram, plot_dips_by_cluster_matplotlib
from signal_processing import SignalProcessor

        
# Main script
if __name__ == '__main__':
    dip_directory = "dips/dips_07_0s_300s_soft_v4/*/"  # Update this to your correct directory path
    plot_dir_base = 'clustered_dips_plots_07_00s_300s_soft_v4' 
    # features, filenames = load_dips_and_extract_features(directory)
    
     
    features, features_labels, filenames = load_dips_and_extract_all_features(dip_directory, smoothing_function=SignalProcessor.wavelet_then_savgol, remove_context=True) 
    label_dict = {label:index for index, label in enumerate(features_labels)}
    # features_labels = ['Depth', 'Width', 'Area', 'Std Dev', 'Skewness', 'Kurtosis',
    #                        'Slope Start', 'Slope End', 'Dwelling Time', 'FFT Feature', 'Inflection Count',
    #                        'Num Peaks', 'Peak Widths Mean']
    selected_labels = features_labels
    selected_features = select_features(features, label_dict, labels_to_select=selected_labels)
    print(selected_features.shape)
    print(selected_features[:,0])
    print(max(selected_features[:,0]))
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(selected_features)

    # Perform hierarchical clustering
    Z = linkage(features_normalized, 'ward')

    # Plot dendrogram to help determine the number of clusters
    plot_dendrogram(Z, labels=filenames, plot_dir=f'plots/dendrogram/{plot_dir_base}')

    # Optional: Automatically form flat clusters from the hierarchical clustering defined by the given linkage matrix
    max_distance = 20# Adjust this threshold to cut the dendrogram and form clusters
    labels = fcluster(Z, max_distance, criterion='distance')

    # Organize filenames by cluster
    clusters = defaultdict(list)
    for label, filename in zip(labels, filenames):
        clusters[label].append(filename)

    # For debugging: Print file-cluster assignments
    for filename, label in zip(filenames, labels):
        print(f"{filename} -> Cluster {label}")
    
    # Adjust these parameters as needed
    dir_name = f"plots/{plot_dir_base}_{len(clusters)}"
    
    sampling_rate = 250000  # Example sampling rate in Hz
    
    # Plot dips by cluster
    plot_dips_by_cluster_matplotlib(clusters, dir_name, sampling_rate, SignalProcessor.wavelet_then_savgol)

    