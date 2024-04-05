import os 
import numpy as np 

def find_significant_dips(data, threshold, min_length=200, min_separation=1000):
    """
    Identifies significant dips in the provided dataset based on a threshold. A dip is considered
    significant if it falls below the threshold and meets the minimum length criteria. Adjacent
    dips separated by less than the minimum separation are merged.

    Parameters:
    - data: Array-like, the dataset in which to find dips.
    - threshold: Numeric, the threshold value below which a dip is considered significant.
    - min_length: Integer, the minimum number of consecutive data points required for a dip to be considered significant.
    - min_separation: Integer, the minimum number of data points that must separate two dips for them to be considered distinct.

    Returns:
    - List of tuples, where each tuple contains the start and end indices of a significant dip in the dataset.
    """
    
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
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
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
