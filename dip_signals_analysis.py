import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
import pywt
from scipy.stats import skew, kurtosis
from scipy.integrate import simps
import glob
from scipy.ndimage import gaussian_filter1d
from signal_processing import SignalProcessor
import sys 
import argparse 

def plot_signal_with_features(original_signal, smoothed_signals, features, filename, dir_name='plots/signal_feature_analysis'):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    num_signals = len(smoothed_signals)
    num_cols = 2
    num_rows = num_signals // num_cols + (num_signals % num_cols > 0) + 1  # +1 for the original signal

    plt.figure(figsize=(15, 3 * num_rows))

    # Plot original signal
    plt.subplot(num_rows, num_cols, 1)
    plt.plot(original_signal, label='Original Signal')
    plt.title(f'Original Signal {filename}')
    plt.legend()

    # Plot each smoothed signal
    for i, (label, signal) in enumerate(smoothed_signals.items(), start=2):
        plt.subplot(num_rows, num_cols, i)
        plt.plot(signal, label=f'Smoothed: {label}')
        plt.title(f'{label}')
        plt.legend()

    # Adjust layout
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name, f'{filename}.png'))
    plt.close()


# Example usage
if __name__ == '__main__':
# Create the parser
    parser = argparse.ArgumentParser()
    # Add the --version argument
    parser.add_argument('--version', type=int, help='Version number')
    # Parse the command-line arguments
    args = parser.parse_args()

        
    files = glob.glob(f'dips/dips_07_0s_300s_soft_v{args.version}/*/*.npy')  # Update your path
    dir_name=f'plots/signal_feature_analysis_07_00s_300s_soft_v{args.version}'
    sampling_rate = 250000
    for file_index, file_path in enumerate(files):
        file_id = file_path.split('/')[-1][:-4]
        signal = np.load(file_path)
        signal_origin = signal.copy()
        signal = signal[920:-920]

        # Assume the first 1000 and last 1000 data points are baseline
        adaptively_smoothed_signal = SignalProcessor.adaptive_smoothing(signal, 0, 1000, target_noise_level=2)
        

        # Apply various smoothing and filtering techniques
        smoothed_signal = SignalProcessor.gaussian_filter(signal, sigma=50)
        gaussian_smoothed_signal_30 = SignalProcessor.gaussian_filter(signal, sigma=30)
        gaussian_smoothed_signal_20 = gaussian_filter1d(signal, sigma=20)
        median_smoothed_signal = SignalProcessor.median_filter(signal, kernel_size=37)
        savgol_smoothed_signal = SignalProcessor.savgol_filter(signal, window_length=73, polyorder=2)
        cutoff_freq_hz = 10000
        bessel_filtered_signal = SignalProcessor.apply_low_pass_bessel_filter(signal, cutoff_freq_hz=cutoff_freq_hz, sampling_rate_hz=250000)
        wavelet_denoised_signal = SignalProcessor.wavelet_denoising(signal, mode='hard')
        
        
        #Combination filters
        # Combination 1: Bessel Filter followed by Wavelet Denoising
        bessel_then_wavelet_signal = SignalProcessor.bessel_then_wavelet(signal, cutoff_freq_hz=cutoff_freq_hz, sampling_rate_hz=sampling_rate)

    
        
        
        wavelet_then_savgol_signal = SignalProcessor.wavelet_then_savgol(signal)
        # Prepare a dictionary with different smoothed versions of the signal
        smoothed_signals_dict = {
            f'Gaussian σ=30': gaussian_smoothed_signal_30,
            f'Gaussian σ=20': gaussian_smoothed_signal_20,
            'Median': median_smoothed_signal,
            'Savitzky-Golay': savgol_smoothed_signal,
            'Adaptive': adaptively_smoothed_signal,
            f'Bessel Cutoff {cutoff_freq_hz}Hz': bessel_filtered_signal,
            'Wavelet Denoised': wavelet_denoised_signal,
            'Bessel then Wavelet': bessel_then_wavelet_signal,
            'Wavelet then Savitzky-Golay': wavelet_then_savgol_signal
            
        }

        # Extract features from one of the smoothed signals or the original signal
        features = None  # Or replace 'signal' with one of the smoothed signals
        
        
        # Plot all signals and save the figure
        plot_signal_with_features(signal_origin, smoothed_signals_dict, features, f'{file_id}', dir_name=dir_name)
