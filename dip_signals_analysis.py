import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.ndimage import gaussian_filter1d
from signal_processing import SignalProcessor
from omegaconf import OmegaConf

def plot_signal_with_features(original_signal, smoothed_signals, features, filename, dir_name):
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

# Load configuration
config = OmegaConf.load('configs/event-analysis.yaml')

if __name__ == '__main__':
    version = config.version
    files = glob.glob(f'{config.save_dip_dir_prefix}/*/*.npy')
    dir_name = f'{config.signal_feature_analysis_dir}'

    for file_index, file_path in enumerate(files):
        file_id = os.path.basename(file_path)[:-4]  # Strip the extension
        signal = np.load(file_path)
        signal_origin = signal.copy()
        signal = signal[920:-920]  # Adjust based on your specific needs

        # Apply various smoothing and filtering techniques
        smoothed_signal = SignalProcessor.gaussian_filter(signal, sigma=config.gaussian_sigma[0])
        gaussian_smoothed_signal_30 = SignalProcessor.gaussian_filter(signal, sigma=config.gaussian_sigma[1])
        median_smoothed_signal = SignalProcessor.median_filter(signal, kernel_size=37)
        savgol_smoothed_signal = SignalProcessor.savgol_filter(signal, window_length=73, polyorder=2)
        cutoff_freq_hz = config.cutoff_freq_hz
        bessel_filtered_signal = SignalProcessor.apply_low_pass_bessel_filter(signal, cutoff_freq_hz=cutoff_freq_hz, sampling_rate_hz=config.sampling_rate)
        wavelet_denoised_signal = SignalProcessor.wavelet_denoising(signal, mode='hard')
        
        # Combination filters
        bessel_then_wavelet_signal = SignalProcessor.bessel_then_wavelet(signal, cutoff_freq_hz=cutoff_freq_hz, sampling_rate_hz=config.sampling_rate)
        wavelet_then_savgol_signal = SignalProcessor.wavelet_then_savgol(signal)
        
        # Prepare a dictionary with different smoothed versions of the signal
        smoothed_signals_dict = {
            'Gaussian σ={config.gaussian_sigma[0]}': smoothed_signal,
            f'Gaussian σ={config.gaussian_sigma[1]}': gaussian_smoothed_signal_30,
            'Median': median_smoothed_signal,
            'Savitzky-Golay': savgol_smoothed_signal,
            'Bessel Cutoff': bessel_filtered_signal,
            'Wavelet Denoised': wavelet_denoised_signal,
            'Bessel then Wavelet': bessel_then_wavelet_signal,
            'Wavelet then Savitzky-Golay': wavelet_then_savgol_signal,
        }
        features = None
        # Plot all signals and save the figure
        plot_signal_with_features(signal_origin, smoothed_signals_dict,  features, file_id, dir_name)
