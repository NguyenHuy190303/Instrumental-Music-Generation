import librosa
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from sklearn.preprocessing import MinMaxScaler

def load_and_resample_audio(audio_path, sr=22050):
    """
    Load and resample an audio file.
    
    Parameters:
        audio_path (str): Path to the audio file.
        sr (int): Sample rate (default 22050).
    
    Returns:
        y (np.ndarray): Audio time series.
        sr (int): Sample rate.
    """
    y, sr = librosa.load(audio_path, sr=sr)
    return y, sr

def audio_to_melspec(audio, sr, n_mels, n_fft=2048, hop_length=512, to_db=False):
    """
    Convert an audio signal to a Mel Spectrogram.
    
    Parameters:
        audio (np.ndarray): Input audio signal.
        sr (int): Sample rate.
        n_mels (int): Number of Mel bands.
        n_fft (int): FFT window size (default 2048).
        hop_length (int): Hop length between frames (default 512).
        to_db (bool): Convert power spectrogram to decibel scale (default False).
    
    Returns:
        spec (np.ndarray): Mel Spectrogram.
    """
    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=None,
        window="hann",
        center=True,
        pad_mode="reflect",
        power=2.0,
        n_mels=n_mels
    )
    if to_db:
        spec = librosa.power_to_db(spec, ref=np.max)
    return spec

def normalize_melspec(melspec, norm_range=(0, 1)):
    """
    Normalize a Mel Spectrogram to the specified range.
    
    Parameters:
        melspec (np.ndarray): Input Mel Spectrogram.
        norm_range (tuple): Desired range (default (0,1)).
    
    Returns:
        np.ndarray: Normalized Mel Spectrogram.
    """
    scaler = MinMaxScaler(feature_range=norm_range)
    melspec = melspec.T  # Transpose to normalize along the time axis
    melspec_normalized = scaler.fit_transform(melspec)
    return melspec_normalized.T

def denormalize_melspec(melspec_normalized, original_melspec, norm_range=(0, 1)):
    """
    Denormalize a normalized Mel Spectrogram back to its original scale.
    
    Parameters:
        melspec_normalized (np.ndarray): Normalized Mel Spectrogram.
        original_melspec (np.ndarray): Original Mel Spectrogram.
        norm_range (tuple): Normalization range (default (0,1)).
    
    Returns:
        np.ndarray: Denormalized Mel Spectrogram.
    """
    scaler = MinMaxScaler(feature_range=norm_range)
    melspec = original_melspec.T
    scaler.fit(melspec)
    melspec_denormalized = scaler.inverse_transform(melspec_normalized.T)
    return melspec_denormalized.T

def melspec_to_audio(melspec, sr, n_fft=2048, hop_length=512, n_iter=64):
    """
    Convert a Mel Spectrogram back to an audio signal using the Griffin-Lim algorithm.
    
    Parameters:
        melspec (np.ndarray): Input Mel Spectrogram.
        sr (int): Sample rate.
        n_fft (int): FFT size (default 2048).
        hop_length (int): Hop length (default 512).
        n_iter (int): Number of iterations for Griffin-Lim (default 64).
    
    Returns:
        np.ndarray: Reconstructed audio signal.
    """
    # If spectrogram is in decibels, convert back to power
    if np.any(melspec < 0):
        melspec = librosa.db_to_power(melspec)
    audio_reconstructed = librosa.feature.inverse.mel_to_audio(
        melspec,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=None,
        window="hann",
        center=True,
        pad_mode="reflect",
        power=2.0,
        n_iter=n_iter
    )
    return audio_reconstructed

def display_audio_files(reconstructed_audio, sr, title="", original_audio=None):
    """
    Display original and/or reconstructed audio.
    
    Parameters:
        reconstructed_audio (np.ndarray): Reconstructed audio signal.
        sr (int): Sample rate.
        title (str): Title to display (if no original audio provided).
        original_audio (np.ndarray, optional): Original audio signal.
    """
    if original_audio is not None:
        print("Original Audio:")
        ipd.display(ipd.Audio(original_audio, rate=sr))
        print("Reconstructed Audio (from Mel Spectrogram):")
    else:
        print(title)
    ipd.display(ipd.Audio(reconstructed_audio, rate=sr))

def show_spectrogram(spectrogram, title="Mel-Spectrogram", denormalize=False, is_numpy=False):
    """
    Display a Mel Spectrogram.
    
    Parameters:
        spectrogram (torch.Tensor or np.ndarray): Input spectrogram.
        title (str): Plot title.
        denormalize (bool): If True, the spectrogram is denormalized.
        is_numpy (bool): If True, the spectrogram is a numpy array.
    """
    if not is_numpy:
        spectrogram = spectrogram.squeeze().cpu().numpy()
    plt.figure(figsize=(10, 4))
    if denormalize:
        plt.imshow(spectrogram, aspect="auto", origin="lower", cmap="viridis")
    else:
        plt.imshow(spectrogram, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency")
    plt.colorbar()
    plt.show()
