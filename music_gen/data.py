import os
import json
from tqdm import tqdm
import numpy as np
import torch
from music_cvae.audio_utils import load_and_resample_audio, audio_to_melspec, normalize_melspec

def load_and_get_genre_data(data_path):
    """
    Load genre data from a JSON file.
    
    Parameters:
        data_path (str): Path to the JSON file.
    
    Returns:
        list: List of genres.
    """
    with open(data_path, "r") as f:
        data = json.load(f)
    return data.get("genres", [])

def get_id2label_label2id(data_dir):
    """
    Create mappings between genre IDs and labels using JSON files in a directory.
    
    Parameters:
        data_dir (str): Directory containing JSON files.
    
    Returns:
        tuple: (id2label, label2id) dictionaries.
    """
    genres = []
    for file in os.listdir(data_dir):
        if not file.endswith(".json"):
            continue
        genres_one_sample = load_and_get_genre_data(os.path.join(data_dir, file))
        genres.extend(genres_one_sample)
    genres = list(set(genres))
    id2label = {i: genre for i, genre in enumerate(genres)}
    label2id = {genre: i for i, genre in enumerate(genres)}
    return id2label, label2id

def tokenize_genres(genres, label2id):
    """
    Convert genre labels to token IDs.
    
    Parameters:
        genres (list): List of genre labels.
        label2id (dict): Mapping from genre to ID.
    
    Returns:
        list: List of token IDs.
    """
    return [label2id[genre] for genre in genres if genre in label2id]

def detokenize_genres(genre_ids, id2label):
    """
    Convert token IDs back to genre labels.
    
    Parameters:
        genre_ids (list): List of token IDs.
        id2label (dict): Mapping from ID to genre.
    
    Returns:
        list: List of genre labels.
    """
    return [id2label[i] for i in genre_ids if i in id2label]

def get_onehot_genres(genre_ids, max_classes):
    """
    Create a one-hot encoded vector for the given genre IDs.
    
    Parameters:
        genre_ids (list): List of genre IDs.
        max_classes (int): Total number of genres.
    
    Returns:
        np.ndarray: One-hot encoded genre vector.
    """
    onehot = np.zeros(max_classes, dtype=np.float32)
    onehot[genre_ids] = 1
    return onehot

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, audio_path, json_path, sample_rate, duration, max_genres,
                 n_mels=256, n_fft=2048, hop_length=512, testset_amount=32):
        """
        Custom Dataset for loading audio files and their genre labels.
        
        Parameters:
            audio_path (str): Directory containing audio (.mp3) files.
            json_path (str): Directory containing JSON files with genre data.
            sample_rate (int): Audio sample rate.
            duration (int): Duration (in seconds) of each audio segment.
            max_genres (int): Total number of genre classes.
            n_mels (int): Number of Mel bands (default 256).
            n_fft (int): FFT window size (default 2048).
            hop_length (int): Hop length (default 512).
            testset_amount (int): Number of samples reserved for testing.
        """
        self.audio_path = audio_path
        self.json_path = json_path
        # Build genre mappings
        self.id2label, self.label2id = get_id2label_label2id(json_path)
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.duration = duration
        self.fixed_length = self.sample_rate * self.duration
        self.max_genres = max_genres

        # Get lists of audio and JSON files (limit to 10 for demo purposes)
        self.audio_files = [f for f in os.listdir(audio_path) if f.endswith(".mp3")][:10]
        self.json_files = [f for f in os.listdir(json_path) if f.endswith(".json")][:10]

        self.audios = []
        for json_file, audio_file in tqdm(zip(self.json_files, self.audio_files),
                                          total=len(self.audio_files)):
            # Load genres and tokenize
            genres = load_and_get_genre_data(os.path.join(self.json_path, json_file))
            genre_ids = tokenize_genres(genres, self.label2id)
            onehot_genres = torch.tensor(get_onehot_genres(genre_ids, self.max_genres)).unsqueeze(0)
            
            # Load audio file and split into segments
            audio_file_path = os.path.join(self.audio_path, audio_file)
            audio, _ = load_and_resample_audio(audio_file_path)
            n_segments = int(len(audio) / self.fixed_length)
            for i in range(n_segments):
                start = i * self.fixed_length
                end = start + self.fixed_length
                audio_segment = audio[start:end]
                mel_spectrogram = audio_to_melspec(audio_segment, self.sample_rate,
                                                   n_mels=self.n_mels, n_fft=n_fft, hop_length=hop_length)
                mel_spectrogram_norm = torch.tensor(normalize_melspec(mel_spectrogram)).unsqueeze(0)
                mel_spectrogram = torch.tensor(mel_spectrogram).unsqueeze(0)
                self.audios.append((mel_spectrogram_norm, onehot_genres, mel_spectrogram))
        
        # Split into training and test sets
        self.audios = self.audios[:len(self.audios) - testset_amount]
        self.testset = self.audios[len(self.audios) - testset_amount:]
    
    def __len__(self):
        return len(self.audios)
    
    def __getitem__(self, idx):
        return self.audios[idx]
