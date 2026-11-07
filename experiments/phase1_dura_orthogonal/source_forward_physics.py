"""Source-only true-channel waveform loss; no simulator input at inference."""
import torch

WEIGHT = .1


def source_channel_error(output, clean, kernel, coefficients):
    if (output.ndim != 3 or output.shape[1:] != (2, 256) or clean.shape != output.shape
            or kernel.shape != (len(output), 2, 64) or coefficients.shape != (len(output), 4, 256)):
        raise ValueError('Matching source IQ and simulator action required')
    clean, kernel, coefficients = clean.detach(), kernel.detach(), coefficients.detach()
    y = torch.complex(clean[:, 0], clean[:, 1])
    delta = torch.complex(output[:, 0], output[:, 1]) - y
    h = torch.complex(kernel[:, 0], kernel[:, 1])
    main = torch.complex(coefficients[:, 0], coefficients[:, 1])
    image = torch.complex(coefficients[:, 2], coefficients[:, 3])
    index = torch.cat((torch.arange(63, 0, -1, device=output.device),
                       torch.arange(256, device=output.device)))
    response = torch.fft.fft(h, n=512)

    def channel(value):
        extended = value.index_select(-1, index)
        filtered = torch.fft.ifft(torch.fft.fft(extended, n=512) * response)[:, 63:319]
        return main * filtered + image * filtered.conj()

    ref = channel(y)
    err = channel(delta)
    return (err.abs().square().mean(1) / ref.abs().square().mean(1).clamp_min(1e-8)).mean()
