import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from music_cvae.audio_utils import show_spectrogram
from music_cvae.models import loss_function

def train_vae(model, dataloader, optimizer, scheduler, num_epochs, device, verbose_interval=50):
    """
    Train the CVAE model.
    
    Parameters:
        model (nn.Module): The CVAE model.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer.
        scheduler (LRScheduler): Learning rate scheduler.
        num_epochs (int): Number of training epochs.
        device (torch.device): Device (CPU or CUDA).
        verbose_interval (int): Interval (in epochs) to display spectrograms.
    
    Returns:
        tuple: (mu, logvar, losses) from training.
    """
    model.train()
    losses = []
    for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
        train_loss = 0
        for batch_idx, (data, genres_input, ori_data) in enumerate(dataloader):
            data = data.to(device)
            genres_input = genres_input.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data, genres_input)
            loss = loss_function(recon, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        scheduler.step()
        avg_loss = train_loss / len(dataloader.dataset)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Lr: {scheduler.get_last_lr()[0]}")
        if epoch == 0 or (epoch + 1) % verbose_interval == 0:
            data_sample = data[0].detach().cpu()
            recon_sample = recon[0].detach().cpu()
            show_spectrogram(data_sample, title="Original Spectrogram")
            show_spectrogram(recon_sample, title="Reconstructed Spectrogram")
    return mu, logvar, losses

def plot_losses(losses, title="Training Loss", xlabel="Epochs", ylabel="Loss", color='b', grid=True):
    """
    Plot training losses over epochs.
    
    Parameters:
        losses (list): List of loss values.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        color (str): Line color.
        grid (bool): Whether to show grid lines.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, color=color, linewidth=2)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if grid:
        plt.grid(True, linestyle="--", alpha=0.6)
    min_loss_idx = losses.index(min(losses))
    max_loss_idx = losses.index(max(losses))
    plt.annotate(f"Min Loss: {min(losses):.4f}", xy=(min_loss_idx, min(losses)),
                 xytext=(min_loss_idx + 1, min(losses) + 0.1),
                 arrowprops=dict(arrowstyle="->", color="green"), fontsize=12, color='green')
    plt.annotate(f"Max Loss: {max(losses):.4f}", xy=(max_loss_idx, max(losses)),
                 xytext=(max_loss_idx + 1, max(losses) + 0.1),
                 arrowprops=dict(arrowstyle="->", color="red"), fontsize=12, color="red")
    plt.tight_layout()
    plt.show()
