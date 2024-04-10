import os 
import numpy as np 
from scipy.ndimage import gaussian_filter1d
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks, peak_widths
from scipy.fft import rfft
import glob 

import numpy as np

def detect_dips(data, threshold):
    """
    Identifies indices where data falls below the threshold.
    
    Parameters:
    - data: Array-like, the dataset to analyze.
    - threshold: Numeric, threshold value below which a point is considered part of a dip.
    
    Returns:
    - A list of indices where data falls below the threshold.
    """
    return np.where(data < threshold)[0]

def is_dip_significant(dip, data, min_length, max_length, min_value):
    """
    Determines if a dip is significant based on various criteria, including its length,
    negativity, minimum value within the dip, its context relative to neighboring points,
    and the difference between the first context point before the start of the dip
    and the minimum current within the dip. Additionally, returns the reason if not significant.
    """
    
    dip_length = len(dip)
    if not (min_length <= dip_length < max_length):
        return False, "Invalid length"
    if any(data[index] < 0 for index in dip):  # Check for negative values within the dip
        return False, "Contains negative values"
    
    dip_data = data[dip]
    min_dip_value = np.min(dip_data)
    if min_dip_value < min_value:  # Ensure the minimum value in the dip is not below min_value
        return False, "Minimum value below threshold"
    
    if dip[0] > 0:  # Ensure there's at least one point before the dip
        first_context_point = data[dip[0] - 1]
    else:
        first_context_point = data[dip[0]]  # Use the start of the dip if it's the very beginning of the data
    
    if (first_context_point - min_dip_value) <= 20:
        return False, "Insufficient drop from context start"
    
    start_context_index = max(0, dip[0] - 5)
    end_context_index = min(len(data), dip[-1] + 5)
    context_data = np.concatenate([data[start_context_index:dip[0]], data[dip[-1]+1:end_context_index]])

    if len(context_data) == 0 or min_dip_value > np.median(context_data) * 0.90:
        # print(f'context median {np.median(context_data)} min dip value {min_dip_value}')
        return False, "Insufficient drop from surrounding context"

    return True, ""


def merge_dips(dips, data, min_separation, max_length):
    """
    Merges adjacent dips based on the minimum separation criteria.
    
    Parameters:
    - dips: List of dips, where each dip is a list of indices.
    - data: Array-like, the dataset to analyze.
    - min_separation: Minimum separation required to consider dips distinct.
    - max_length: Maximum allowed length of the dip.
    
    Returns:
    - A list of merged dips.
    """
    merged_dips = [dips[0]]
    for current_dip in dips[1:]:
        if current_dip[0] - merged_dips[-1][-1] < min_separation:
            # Check if merging is valid based on max_length
            if len(merged_dips[-1] + current_dip) < max_length:
                merged_dips[-1].extend(current_dip)
            else:
                merged_dips.append(current_dip)
        else:
            merged_dips.append(current_dip)
    return merged_dips

def find_significant_dips(data, threshold, min_length=200, min_separation=1000, min_value=0):
    max_length = 0.08 * 250000
    under_threshold_indices = detect_dips(data, threshold)
    if len(under_threshold_indices) == 0:
        return []

    dips = []
    current_dip = [under_threshold_indices[0]]
    rejection_reasons = {}  # To hold individual dip rejections
    rejection_counts = {}  # To count different reasons for rejection

    for index in under_threshold_indices[1:]:
        if index - current_dip[-1] > 1:
            is_significant, reason = is_dip_significant(current_dip, data, min_length, max_length, min_value)
            if is_significant:
                dips.append(current_dip)
            else:
                rejection_reasons[(current_dip[0], current_dip[-1])] = reason
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1  # Increment the count for this reason
            current_dip = [index]
        else:
            current_dip.append(index)
    
    is_significant, reason = is_dip_significant(current_dip, data, min_length, max_length, min_value)
    if is_significant:
        dips.append(current_dip)
    else:
        rejection_reasons[(current_dip[0], current_dip[-1])] = reason
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1  # Increment here as well

    merged_dips = merge_dips(dips, data, min_separation, max_length)

    # Optionally print or return rejection_reasons and rejection_counts for analysis
    # print("Rejection Reasons:", rejection_reasons)
    print("Rejection Counts:", rejection_counts)
    return [(dip[0], dip[-1]) for dip in merged_dips]


# def find_significant_dips(data, threshold, outlier_threshold=50, min_length=200, min_separation=1000):
#     """
#     Identifies significant dips in the provided dataset based on a threshold. A dip is considered
#     significant if it falls below the threshold, does not contain negative values, meets the
#     minimum length criteria, does not exceed the maximum length criteria, and has no more than
#     10% of its data points falling below 200. Adjacent dips separated by less than the minimum
#     separation are merged.

#     Parameters:
#     - data: Array-like, the dataset in which to find dips.
#     - threshold: Numeric, the threshold value below which a dip is considered significant.
#     - outlier_threshold: Numeric, used for an unrelated filtering criterion and can be ignored for the task of filtering out negative dips.
#     - min_length: Integer, the minimum number of consecutive data points required for a dip to be considered significant.
#     - min_separation: Integer, the minimum number of data points that must separate two dips for them to be considered distinct.

#     Returns:
#     - List of tuples, where each tuple contains the start and end indices of a significant dip in the dataset.
#     """
    
#     # Calculate maximum allowed dip length.
#     max_length = 0.08 * 250000

#     # Identify indices where data falls below threshold and is non-negative.
#     under_threshold_indices = np.where((data < threshold) & (data >= 0))[0]

#     if len(under_threshold_indices) == 0:
#         return []

#     dips = []
#     current_dip = [under_threshold_indices[0]]

#     for index in under_threshold_indices[1:]:
#         if index - current_dip[-1] > 1:
#             # Finish current dip and check for length and below 200 condition.
#             if len(current_dip) >= min_length and (current_dip[-1] - current_dip[0]) < max_length:
#                 if np.sum(data[current_dip] < 500) / len(current_dip) <= 0.40:
#                     dips.append(current_dip)
#             current_dip = [index]
#         else:
#             current_dip.append(index)
    
#     if len(current_dip) >= min_length and (current_dip[-1] - current_dip[0]) < max_length:
#         if np.sum(data[current_dip] < 200) / len(current_dip) <= 0.10:
#             dips.append(current_dip)

#     merged_dips = []
#     current_dip = dips[0]
#     for next_dip in dips[1:]:
#         if next_dip[0] - current_dip[-1] < min_separation:
#             # Attempt to merge and check length and below 200 condition before adding.
#             potential_merge = current_dip + next_dip
#             if (next_dip[-1] - current_dip[0]) < max_length and np.sum(data[potential_merge] < 200) / len(potential_merge) <= 0.10:
#                 current_dip.extend(next_dip)
#             else:
#                 merged_dips.append(current_dip)
#                 current_dip = next_dip
#         else:
#             merged_dips.append(current_dip)
#             current_dip = next_dip
#     if (current_dip[-1] - current_dip[0]) < max_length and np.sum(data[current_dip] < 200) / len(current_dip) <= 0.10:
#         merged_dips.append(current_dip)

#     return [(dip[0], dip[-1]) for dip in merged_dips]

def save_dips_as_npy(data, significant_dips, dir_name, sampling_rate, chunk_start_time, dip_counter, context=1000):
    """
    Saves segments of the data corresponding to significant dips as .npy files, including the start and end time in the filename.

    Parameters:
    - data: The entire dataset as a NumPy array.
    - significant_dips: A list of tuples, where each tuple contains the start and end indices of a dip.
    - dir_name: Directory where the .npy files will be saved.
    - sampling_rate: The sampling rate of the data (samples per second).
    - context: Number of points before and after the dip to include in the saved segment.
    """
    
    # Ensure the output directory exists
    os.makedirs(dir_name, exist_ok=False)
    
    for i, (start, end) in enumerate(significant_dips):
        # Define segment with some context around the dip
        segment_start = max(start - context, 0)
        segment_end = min(end + context, len(data))
        segment_data = data[segment_start:segment_end]
        
        # Convert indices to time (in seconds)
        start_time = start / sampling_rate + chunk_start_time
        end_time = end / sampling_rate + chunk_start_time
        
        # Save the segment as a .npy file with start and end time in the filename
        filename = f"dip_{i+dip_counter}_start_{start_time:.6f}s_end_{end_time:.6f}s.npy"
        np.save(os.path.join(dir_name, filename), segment_data)


def load_dip_and_extract_all_features(dip_file, smoothing_function, remove_context):
    
    dip = np.load(dip_file)
    if remove_context:
        dip = dip[920:-920]  # Removing context signal to focus on the dip
    if smoothing_function:
        dip = smoothing_function(dip)
    # Basic Features
    depth = np.min(dip)  # Depth of the dip
    width = len(dip)  # Width of the dip
    area = np.trapz(-dip)  # Area under the dip curve assuming negative values
    std_dev = np.std(dip)  # Standard deviation of dip values
    skewness = skew(dip)  # Skewness of the dip
    kurt = kurtosis(dip)  # Kurtosis of the dip
    slope_start = dip[1] - dip[0]  # Slope at the start
    slope_end = dip[-1] - dip[-2]  # Slope at the end
    sampling_rate = 250000  # samples per second, adjust this to your actual sampling rate
    start_index = 0  # replace with actual start index of the dip
    end_index = len(dip)  # replace with actual end index of the dip

    # Calculate dwelling time in seconds
    dwelling_time = (end_index - start_index) / sampling_rate
    fft_feature = np.abs(rfft(dip)[1])  # Magnitude of the second FFT coefficient

    # Advanced Features
    inflections = np.diff(np.sign(np.diff(dip)))  # Calculate second derivative
    inflection_count = np.sum(inflections != 0)  # Count of inflection points
    peaks, _ = find_peaks(-dip)  # Peaks assuming dip values are negative
    num_peaks = len(peaks)  # Number of peaks
    peak_widths_mean = np.mean(peak_widths(-dip, peaks)[0]) if num_peaks > 0 else 0  # Mean peak width

    # Combine all features into a list for this dip
    dip_features = [depth, width, area, std_dev, skewness, kurt,
                    slope_start, slope_end, dwelling_time, fft_feature, inflection_count,
                    num_peaks, peak_widths_mean]
    
    
    
    return dip_features



def load_dips_and_extract_all_features(directory, smoothing_function=None, remove_context=False):
    """
    Load dip signals from .npy files, apply Gaussian smoothing, and extract a comprehensive set of features.

    Parameters:
    - directory: str, path to the directory containing the .npy files.
    - smooth_sigma: int, the sigma value for Gaussian smoothing. If None, smoothing is not applied.

    Returns:
    - features: np.array, an array of extracted features for each dip.
    - filenames: list, a list of filenames corresponding to each dip.
    """
    features = []
    filenames = glob.glob(os.path.join(directory, '*.npy'))

    for filename in filenames:
        this_dip_feature = load_dip_and_extract_all_features(filename, smoothing_function=smoothing_function, remove_context=remove_context)    
        features.append(this_dip_feature)
        
    features_labels = ['Depth', 'Width', 'Area', 'Std Dev', 'Skewness', 'Kurtosis',
                           'Slope Start', 'Slope End', 'Dwelling Time', 'FFT Feature', 'Inflection Count',
                           'Num Peaks', 'Peak Widths Mean']
    return np.array(features), features_labels, filenames

def select_features(features, label_dict, labels_to_select):
    selected_features = []
    for this_file_features in features:
        this_file_selected_feature = []
        for label in labels_to_select:
            this_file_selected_feature.append(this_file_features[label_dict[label]])
        selected_features.append(this_file_selected_feature)
    return np.asarray(selected_features)