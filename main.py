import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from music_cvae.data import AudioDataset, get_id2label_label2id
from music_cvae.models import CVAE, loss_function
from music_cvae.train import train_vae, plot_losses
from music_cvae.inference import inference
from music_cvae.audio_utils import load_and_resample_audio, melspec_to_audio, display_audio_files, show_spectrogram

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print version information
print("Torch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

# Parameters
sample_rate = 22050  # Samples per second
duration = 3         # Duration (seconds) for each audio segment
n_mels = 256         # Number of Mel bands
batch_size = 128
num_epochs = 100
lr = 2e-4
gamma = 0.5
step_size = num_epochs // 2
verbose_interval = num_epochs // 10
testset_amount = 1

# Paths (update these as necessary)
audio_dir = '/kaggle/input/aio-music-generation/crawled_data/audio'
json_dir = '/kaggle/input/aio-music-generation/crawled_data'

# Load genre mappings to determine the number of genres
_, label2id = get_id2label_label2id(json_dir)
max_genres = len(label2id)

# Create Dataset and DataLoader
trainset = AudioDataset(audio_dir, json_dir, sample_rate, duration, max_genres,
                        n_mels=n_mels, testset_amount=testset_amount)
testset = trainset.testset

if len(trainset) == 0:
    raise ValueError(f"No audio files found in {audio_dir}.")

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=testset_amount, shuffle=False)

# Get a batch to check data shapes
data_iter = iter(trainloader)
data_batch, genres_batch, original_audio_batch = next(data_iter)

print(f"Batch data shape: {data_batch.shape}")
print(f"Genres shape: {genres_batch.shape}")
print(f"Original audio shape: {original_audio_batch.shape}")

# Determine frame length from a sample
sample_index = 0
sample_data = data_batch[sample_index]
frame = sample_data.shape[-1]
print("Frame length:", frame)

# Initialize the CVAE model
d_model = 64
latent_dim = 128
# Note: Order of dimensions is (n_frames, n_mels) where n_frames is set to n_mels from preprocessing.
model = CVAE(d_model, latent_dim, n_frames=n_mels, n_mels=frame, n_genres=max_genres).to(device)

# Optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Print total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

# Train the model
mu, logvar, losses = train_vae(model, trainloader, optimizer, scheduler, num_epochs, device,
                               verbose_interval=verbose_interval)

# Plot training losses
plot_losses(losses)

# Perform inference and reconstruct audio
gen_mels, genres_input, ori_data = inference(model, testloader, device)
recon_audios = []
ori_audios = []
num_samples = min(5, len(gen_mels))
for i in range(num_samples):
    show_spectrogram(ori_data[i], title=f"Original Spectrogram {i+1}", denormalize=True)
    ori_reconstructed = melspec_to_audio(ori_data[i].cpu().numpy().squeeze(), sample_rate)
    ori_audios.append(ori_reconstructed)
    
    # For demonstration, we use the normalized version of the generated mel spectrogram.
    spec_denorm = ori_data[i].cpu().numpy().squeeze()  # Alternatively, apply a custom denormalization if needed
    show_spectrogram(spec_denorm, title=f"Reconstructed Spectrogram {i+1}", denormalize=True, is_numpy=True)
    audio_reconstructed = melspec_to_audio(spec_denorm, sample_rate)
    recon_audios.append(audio_reconstructed)
    
    display_audio_
