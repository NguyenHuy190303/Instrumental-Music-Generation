import torch
import torch.nn as nn
import numpy as np

class CVAE(nn.Module):
    def __init__(self, d_model, latent_dim, n_frames, n_mels, n_genres):
        """
        Conditional Variational Autoencoder for Mel Spectrograms.
        
        Parameters:
            d_model (int): Base model dimension.
            latent_dim (int): Dimension of the latent space.
            n_frames (int): Number of time frames in the spectrogram.
            n_mels (int): Number of Mel frequency bins.
            n_genres (int): Number of genre classes.
        """
        super(CVAE, self).__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        
        # Calculate reduced dimensions for decoder input after three conv layers
        self.n_frames = int(np.ceil(n_frames / 2 ** 3))
        self.n_mels = int(np.ceil(n_mels / 2 ** 3))
        self.n_genres = n_genres
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1 + self.n_genres, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model),
            nn.SiLU(),
            nn.Dropout2d(0.05),
            
            nn.Conv2d(d_model, d_model * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model * 2),
            nn.SiLU(),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(d_model * 2, d_model * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model * 4),
            nn.SiLU(),
            nn.Dropout2d(0.15),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(d_model * 4, latent_dim)
        self.fc_logvar = nn.Linear(d_model * 4, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim + self.n_genres, d_model * 4 * self.n_frames * self.n_mels)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d_model * 4, d_model * 2, kernel_size=3,
                               stride=2, padding=1, output_padding=(1, 0)),
            nn.BatchNorm2d(d_model * 2),
            nn.SiLU(),
            nn.Dropout2d(0.1),
            
            nn.ConvTranspose2d(d_model * 2, d_model, kernel_size=3,
                               stride=2, padding=1, output_padding=(1, 0)),
            nn.BatchNorm2d(d_model),
            nn.SiLU(),
            nn.Dropout2d(0.05),
            
            nn.ConvTranspose2d(d_model, 1, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, genres_input):
        # Prepare genre embedding and concatenate with input spectrogram
        ori_genres_embed = genres_input.view(genres_input.size(0), -1)
        genres_embed = ori_genres_embed.unsqueeze(-1).unsqueeze(-1)
        genres_embed = genres_embed.expand(-1, -1, x.size(2), x.size(3))
        x_genres = torch.cat((x, genres_embed), dim=1)
        
        # Encoder pass with saving intermediate activations for shortcuts
        h = x_genres
        shortcuts = []
        for block in self.encoder:
            h = block(h)
            if isinstance(block, nn.SiLU):
                shortcuts.append(h)
        
        # Latent space
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        z_genres = torch.cat((z, ori_genres_embed), dim=1)
        
        # Decoder pass
        h_dec = self.decoder_input(z_genres)
        h_dec = h_dec.view(-1, self.d_model * 4, self.n_frames, self.n_mels)
        for block in self.decoder:
            if isinstance(block, nn.ConvTranspose2d) and shortcuts:
                shortcut = shortcuts.pop()
                h_dec = h_dec + shortcut
            h_dec = block(h_dec)
        
        # Crop the output to match the input size
        recon = h_dec[:, :, :x.size(2), :x.size(3)]
        return recon, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    """
    Compute the CVAE loss as the sum of reconstruction loss and KL divergence.
    
    Parameters:
        recon_x (torch.Tensor): Reconstructed spectrogram.
        x (torch.Tensor): Original spectrogram.
        mu (torch.Tensor): Mean of the latent distribution.
        logvar (torch.Tensor): Log variance of the latent distribution.
    
    Returns:
        torch.Tensor: Total loss.
    """
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + KLD
