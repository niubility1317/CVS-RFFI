import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SincConv_Fast(nn.Module):
    """
    Sinc-based convolution layer specifically for Raw IQ Data.
    Learns cut-off frequencies (f1, f2) instead of weights.
    """
    def __init__(self, out_channels, kernel_size, sample_rate=1.0, in_channels=2):
        super(SincConv_Fast, self).__init__()

        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1 # Force odd kernel size

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Frequency parameters initialization
        # Initialize frequencies linearly suitable for WiFi bandwidth
        low_freq = np.linspace(0.01, 0.45, out_channels)
        band_freq = np.linspace(0.01, 0.05, out_channels)

        self.f1_raw = nn.Parameter(torch.Tensor(low_freq))
        self.band_raw = nn.Parameter(torch.Tensor(band_freq))

        # Hamming window to reduce Gibbs phenomenon
        n = torch.linspace(0, kernel_size, steps=kernel_size) - (kernel_size - 1) / 2
        self.window = 0.54 - 0.46 * torch.cos(2 * math.pi * n / kernel_size)
        self.window = nn.Parameter(self.window, requires_grad=False)
        self.n = nn.Parameter(n, requires_grad=False)

    def forward(self, x):
        # Constrain frequencies to be valid
        f1 = torch.abs(self.f1_raw) + 0.001
        band = torch.abs(self.band_raw) + 0.001
        f2 = f1 + band

        # Helper function: sinc(x) = sin(x)/x
        def _sinc(x):
            return torch.where(x == 0, torch.tensor(1.0, device=x.device), torch.sin(x) / x)

        # Generate filters in time domain
        # g[n] = 2*f2*sinc(2*pi*f2*n) - 2*f1*sinc(2*pi*f1*n)
        g = 2 * f2.unsqueeze(1) * _sinc(2 * math.pi * f2.unsqueeze(1) * self.n) - \
            2 * f1.unsqueeze(1) * _sinc(2 * math.pi * f1.unsqueeze(1) * self.n)

        g = g * self.window.to(g.device)
        filters = g.unsqueeze(1) # Shape: (out_channels, 1, kernel_size)

        # Apply same filter to I and Q channels independently
        # Input x: (Batch, 2, L) -> I=[:,0,:], Q=[:,1,:]
        I_in = x[:, 0:1, :] 
        Q_in = x[:, 1:2, :] 

        I_out = F.conv1d(I_in, filters)
        Q_out = F.conv1d(Q_in, filters)

        # Concatenate features: (Batch, 2*Out, L')
        out = torch.cat([I_out, Q_out], dim=1) 
        return out

class CVSincNet(nn.Module):
    def __init__(self, num_classes=16):
        super(CVSincNet, self).__init__()
        
        # Layer 1: SincConv
        # Input: 2 channels (I/Q), Output: 80 filters * 2 = 160 channels
        self.sinc_conv = SincConv_Fast(out_channels=80, kernel_size=251)
        self.bn1 = nn.BatchNorm1d(160)
        self.pool1 = nn.MaxPool1d(2)
        
        # Layer 2: Standard Conv
        self.layer2 = nn.Sequential(
            nn.Conv1d(160, 60, kernel_size=5),
            nn.BatchNorm1d(60),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2)
        )
        
        # Layer 3: Standard Conv
        self.layer3 = nn.Sequential(
            nn.Conv1d(60, 60, kernel_size=5),
            nn.BatchNorm1d(60),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2)
        )

        # Classification Head
        self.gap = nn.AdaptiveAvgPool1d(1) # Global Average Pooling
        
        self.fc = nn.Sequential(
            nn.Linear(60, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, 2, 1024)
        x = self.sinc_conv(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.pool1(x)
        
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.gap(x)       # (Batch, 60, 1)
        x = x.view(x.size(0), -1) # Flatten -> (Batch, 60)
        
        x = self.fc(x)
        return x