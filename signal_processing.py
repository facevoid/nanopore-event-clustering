import numpy as np
import pywt
from scipy.signal import savgol_filter, medfilt, bessel, filtfilt, find_peaks
from scipy.stats import skew, kurtosis
from scipy.integrate import simps
from scipy.ndimage import gaussian_filter1d

class SignalProcessor:
    @staticmethod
    def calculate_derivative(signal, spacing=1):
        return np.gradient(signal, spacing)

    @staticmethod
    def gaussian_filter(signal, sigma=30):
        return gaussian_filter1d(signal, sigma=sigma)

    @staticmethod
    def median_filter(signal, kernel_size=37):
        return medfilt(signal, kernel_size=kernel_size)

    @staticmethod
    def savgol_filter(signal, window_length=73, polyorder=2):
        return savgol_filter(signal, window_length=window_length, polyorder=polyorder)

    @staticmethod
    def adaptive_smoothing(signal, baseline_start, baseline_end, target_noise_level=0.1):
        baseline_noise = np.std(signal[baseline_start:baseline_end])
        smoothing_factor = min(max(baseline_noise / target_noise_level, 2), 50)
        return gaussian_filter1d(signal, sigma=smoothing_factor)

    @staticmethod
    def wavelet_denoising(signal, wavelet_name='db4', mode='soft', level=None):
        coeffs = pywt.wavedec(signal, wavelet_name, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
        coeffs[1:] = (pywt.threshold(i, value=uthresh, mode=mode) for i in coeffs[1:])
        denoised_signal = pywt.waverec(coeffs, wavelet_name)[:len(signal)]
        return denoised_signal

    @staticmethod
    def apply_low_pass_bessel_filter(signal, cutoff_freq_hz, sampling_rate_hz, filter_order=10):
        normalized_cutoff = cutoff_freq_hz / (0.5 * sampling_rate_hz)
        b, a = bessel(filter_order, normalized_cutoff, btype='low', analog=False)
        return filtfilt(b, a, signal)

    @staticmethod
    def bessel_then_wavelet(signal, cutoff_freq_hz, sampling_rate_hz, wavelet_name='db4', mode='soft', level=None):
        bessel_filtered_signal = SignalProcessor.apply_low_pass_bessel_filter(signal, cutoff_freq_hz, sampling_rate_hz)
        return SignalProcessor.wavelet_denoising(bessel_filtered_signal, wavelet_name, mode, level)

    @staticmethod
    def wavelet_then_savgol(signal, wavelet_name='db4', mode='soft', level=None, window_length=73, polyorder=2):
        wavelet_denoised_signal = SignalProcessor.wavelet_denoising(signal, wavelet_name, mode, level)
        window_length = min(window_length, len(wavelet_denoised_signal))
        if window_length % 2 == 0:
            window_length -= 1
        return SignalProcessor.savgol_filter(wavelet_denoised_signal, window_length, polyorder)

    # Additional methods for feature extraction, plotting, etc., can be added here as static or class methods.
