

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

if __name__ == '__main__':
    dip_directory = "dips/dips_07_0s_300s_soft_v26/*/"  # Update this to your correct directory path
    n_clusters = 5  # Define the number of clusters
    dir_name = f"plots/clustered_kMeans(k={n_clusters})"
    features, features_labels, filenames = load_dips_and_extract_all_features(dip_directory, smoothing_function=SignalProcessor.wavelet_then_savgol, remove_context=True) 

    label_dict = {label:index for index, label in enumerate(features_labels)}
    # features_labels = ['Depth', 'Width', 'Area', 'Std Dev', 'Skewness', 'Kurtosis',
    #                        'Slope Start', 'Slope End', 'Dwelling Time', 'FFT Feature', 'Inflection Count',
    #                        'Num Peaks', 'Peak Widths Mean']
    # selected_labels = ['Depth', 'Width', 'Area', 'Std Dev', 'Skewness', 'Kurtosis','Dwelling Time','FFT Feature']
    selected_labels = features_labels
    selected_features = select_features(features, label_dict, labels_to_select=selected_labels)
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    # Perform K-Means Clustering
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features_normalized)

    # Organize dips by cluster
    clusters = defaultdict(list)
    for label, filename in zip(labels, filenames):
        clusters[label].append(filename)

    # For debugging: Print file-cluster assignments
    for filename, label in zip(filenames, labels):
        print(f"{filename} -> Cluster {label}")
    # Adjust these parameters as needed
    
    dip_sigma = 5
    sampling_rate = 250000  # Example sampling rate in Hz
    plot_dips_by_cluster_matplotlib(clusters, dir_name, sampling_rate, SignalProcessor.wavelet_then_savgol)

