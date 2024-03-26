import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Load the provided .npy data file
data_chunk_path = 'processed_data/chunk_data/chunk_0.npy'
data_chunk = np.load(data_chunk_path)

# Smooth the data
chosen_sigma = 20  # Define your chosen_sigma value
smoothed_data_full = gaussian_filter1d(data_chunk, sigma=chosen_sigma)

def find_very_deep_areas_without_length_constraint(data, threshold=400):
    """
    Find indices where the data is below a certain threshold value without length constraint.
    """
    return np.where(data <= threshold)[0]

very_deep_area_indices_no_length_constraint = find_very_deep_areas_without_length_constraint(data_chunk)

def group_consecutive_indices(indices):
    """
    Group consecutive indices into individual dips.
    """
    return np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)

contiguous_deep_areas = group_consecutive_indices(very_deep_area_indices_no_length_constraint)
significant_contiguous_deep_areas = [segment for segment in contiguous_deep_areas if len(segment) > 10]

# Save plots
def save_plot_with_significant_dips(data, smoothed_data, significant_areas, threshold, file_path):
    plt.figure(figsize=(15, 5))
    plt.plot(data, label='Original Data', alpha=0.5)
    plt.plot(smoothed_data, label='Smoothed Data', alpha=0.8)
    plt.axhline(y=threshold, color='purple', linestyle='--', label='Threshold')
    for segment in significant_areas:
        plt.axvspan(segment[0], segment[-1], color='orange', alpha=0.3)
    plt.title('Original Signal with Highlighted Significant Dips')
    plt.xlabel('Data Point Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.savefig(file_path)
    plt.close()

# Example of saving a plot
save_plot_with_significant_dips(
    data_chunk,
    smoothed_data_full,
    significant_contiguous_deep_areas,
    threshold=400,
    file_path="plots/finding_dips/significant_dips_plot.png"
)

# For checking and debugging
print(f"Number of significant dips: {len(significant_contiguous_deep_areas)}")
if significant_contiguous_deep_areas:
    print(f"Indices of the first few significant dips: {significant_contiguous_deep_areas[:3]}")
