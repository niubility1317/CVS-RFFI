"""PyTorch implementation of SatIQ v1.0.2's split-IQ triplet autoencoder."""

import torch
from torch import nn
from torch.nn import functional as F


INPUT_SAMPLES = 11_000
EMBEDDING_SIZE = 512
CONVOLUTION = ((64, 2), (64, 4), (32, 8), (32, 16), (32, 32), (32, 32))


class PaperSatIQ(nn.Module):
    """Accept per-packet min-max-scaled IQ in [batch, 2, 11000] order."""

    def __init__(self):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(2, eps=1e-3, momentum=.01)
        encoder_shapes = []
        length = INPUT_SAMPLES + 4
        for _, kernel in CONVOLUTION:
            encoder_shapes.append(length)
            length = (length - kernel + 1) // 2
        self.encoded_length = length
        self.encoded_channels = CONVOLUTION[-1][0]
        self.encoders = nn.ModuleList(self._make_encoder() for _ in range(2))
        self.decoders = nn.ModuleList(self._make_decoder(encoder_shapes) for _ in range(2))
        flat_width = length * 2 * self.encoded_channels
        self.to_embedding = nn.Linear(flat_width, EMBEDDING_SIZE)
        self.from_embedding = nn.Linear(EMBEDDING_SIZE, flat_width)

        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_encoder(self):
        layers = []
        previous = 1
        for channels, kernel in CONVOLUTION:
            layers.extend((nn.Conv1d(previous, channels, kernel), nn.ReLU(), nn.MaxPool1d(2)))
            previous = channels
        return nn.Sequential(*layers)

    def _make_decoder(self, encoder_shapes):
        layers = []
        previous = self.encoded_channels
        length = self.encoded_length
        for (channels, kernel), output_length in zip(reversed(CONVOLUTION), reversed(encoder_shapes)):
            right_padding = output_length - (2 * length - kernel + 1)
            layers.extend((nn.Upsample(scale_factor=2, mode="nearest"),
                           nn.Conv1d(previous, channels, kernel), nn.ReLU(),
                           nn.ConstantPad1d((0, right_padding), 0)))
            previous, length = channels, output_length
        layers.append(nn.ConvTranspose1d(previous, 1, kernel_size=5))
        return nn.Sequential(*layers)

    def embed(self, iq):
        if iq.ndim != 3 or iq.shape[1:] != (2, INPUT_SAMPLES):
            raise ValueError("PaperSatIQ expects [batch, 2, 11000] IQ")
        padded = F.pad(self.input_norm(iq), (2, 2))
        encoded = torch.cat([encoder(padded[:, index:index + 1])
                             for index, encoder in enumerate(self.encoders)], dim=1)
        # Keep the original time-major order of the concatenated I/Q channels in the dense layers.
        dense = self.to_embedding(encoded.transpose(1, 2).flatten(1))
        return F.normalize(dense, dim=1, eps=1e-6)

    def forward(self, iq):
        embedding = self.embed(iq)
        decoded = F.relu(self.from_embedding(embedding))
        decoded = decoded.reshape(len(iq), self.encoded_length, 2 * self.encoded_channels).transpose(1, 2)
        i, q = decoded.chunk(2, dim=1)
        recon = torch.cat((self.decoders[0](i), self.decoders[1](q)), dim=1)
        return embedding, recon[:, :, 4:-4].contiguous()


def angular_triplet(embedding, labels, margin=1.0):
    """Mean semi-hard triplet loss in cosine distance over ordered positive pairs."""
    if embedding.dtype in (torch.float16, torch.bfloat16):
        embedding = embedding.float()
    unit = F.normalize(embedding, dim=1, eps=1e-6)
    distance = (1 - unit @ unit.T).clamp_min(0)
    labels = labels.reshape(-1)
    if labels.numel() != len(embedding):
        raise ValueError("SatIQ triplet loss expects one label per embedding")
    same = labels[:, None] == labels[None, :]
    different = ~same
    # For each ordered positive pair, choose the nearest negative strictly farther
    # than the positive. When none is farther, use the farthest negative.
    candidates = distance[:, None, :]
    farther = different[:, None, :] & (candidates > distance[:, :, None])
    nearest = candidates.masked_fill(~farther, torch.inf).amin(dim=2)
    farthest = distance.masked_fill(same, 0).amax(dim=1, keepdim=True)
    negative = torch.where(nearest.isfinite(), nearest, farthest)
    self_pairs = torch.eye(len(embedding), device=embedding.device, dtype=torch.bool)
    positive = same & ~self_pairs & different.any(dim=1, keepdim=True)
    terms = F.relu(margin + distance - negative).masked_fill(~positive, 0)
    return terms.sum() / positive.sum().clamp_min(1)


def paper_loss(embedding, recon, iq, labels):
    reconstruction = F.mse_loss(recon, iq)
    triplet = angular_triplet(embedding, labels)
    return reconstruction + triplet, {"reconstruction_loss": reconstruction.detach(), "triplet_loss": triplet.detach()}
