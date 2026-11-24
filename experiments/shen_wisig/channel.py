"""Source-only short-preamble approximation of Shen's exponential TDL."""
import torch


def linear_fir(signal, taps):
    """Causal linear convolution, cropped to the unchanged observation window."""
    count = signal.shape[-1]
    fft_size = 1 << (count + taps.shape[-1] - 2).bit_length()
    filtered = torch.fft.ifft(torch.fft.fft(signal, n=fft_size) *
                              torch.fft.fft(taps, n=fft_size))
    return filtered[..., :count]


def exponential_tdl(iq, generator, probability):
    """Draw new complex taps per packet, before AWGN and CIS.

    At 25 MS/s the delay grid is 40 ns. A 10 Hz Doppler component accumulates
    less than 0.04 degrees over the WiSig window, so taps are block-static.
    """
    signal = torch.complex(iq[:, 0], iq[:, 1])
    tau = torch.empty(len(signal), 1, device=iq.device).uniform_(5., 300., generator=generator)
    delays = 40 * torch.arange(39, dtype=iq.dtype, device=iq.device)
    pdp = torch.softmax(-delays / tau, dim=-1)
    taps = torch.randn(pdp.shape, dtype=signal.dtype, device=iq.device, generator=generator)
    augmented = linear_fir(signal, taps * pdp.sqrt())
    use = torch.rand(len(signal), 1, device=iq.device, generator=generator) < probability
    mixed = torch.where(use, augmented, signal)
    return torch.stack((mixed.real, mixed.imag), dim=1)
