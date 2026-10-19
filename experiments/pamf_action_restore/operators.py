"""Real-linear IQ operators with causal reflect filtering and its true adjoint."""
import torch


class Action:
    def __init__(self, kernel, main, image):
        if kernel.ndim != 2 or main.ndim != 2 or main.shape != image.shape or len(kernel) != len(main):
            raise ValueError('Batched complex FIR/main/image required')
        if not all(t.is_complex() and torch.isfinite(t).all() for t in (kernel, main, image)):
            raise ValueError('Finite complex action required')
        self.kernel, self.main, self.image = kernel, main, image
        self.count, self.length = main.shape[-1], kernel.shape[-1]
        if not 1 <= self.length <= self.count:
            raise ValueError('Reflect FIR support must fit within the current window')
        self.index = torch.cat((torch.arange(self.length-1, 0, -1, device=main.device),
                                torch.arange(self.count, device=main.device)))
        self.nfft = 1 << (self.count+2*self.length-3).bit_length()
        self.response = torch.fft.fft(kernel, n=self.nfft)

    def filter(self, value):
        if value.shape != self.main.shape:
            raise ValueError('Only matching current-window IQ is accepted')
        extended = value.index_select(-1, self.index)
        full = torch.fft.ifft(torch.fft.fft(extended, n=self.nfft)*self.response)
        return full[:, self.length-1:self.length-1+self.count]

    def filter_adjoint(self, value):
        zeros = torch.zeros((len(value), self.nfft), device=value.device, dtype=value.dtype)
        zeros[:, self.length-1:self.length-1+self.count] = value
        extended = torch.fft.ifft(torch.fft.fft(zeros)*self.response.conj())[:, :len(self.index)]
        return torch.zeros_like(value).scatter_add(1, self.index.expand(len(value), -1), extended)

    def __call__(self, value):
        filtered = self.filter(value)
        return self.main*filtered+self.image*filtered.conj()

    def adjoint(self, value):
        # Adjoint for Re <u,v>; conjugating only the FIR is insufficient.
        return self.filter_adjoint(self.main.conj()*value+self.image*value.conj())


def real_dot(left, right):
    return (left.conj()*right).real.sum(-1)


def soft_map(action, observed, prior, real_noise_variance, prior_variance, steps=6):
    if (real_noise_variance.shape != (len(observed),) or
            not torch.isfinite(real_noise_variance).all() or (real_noise_variance <= 0).any() or
            prior_variance <= 0 or steps < 1):
        raise ValueError('Positive matched noise/prior variances and iteration budget required')
    regularization = real_noise_variance[:, None]/prior_variance
    rhs = action.adjoint(observed-action(prior))
    delta = torch.zeros_like(rhs)
    residual, direction = rhs, rhs
    squared = real_dot(residual, residual)
    for _ in range(steps):
        normal = action.adjoint(action(direction))+regularization*direction
        alpha = squared/real_dot(direction, normal).clamp_min(1e-20)
        delta = delta+alpha[:, None]*direction
        residual = residual-alpha[:, None]*normal
        following = real_dot(residual, residual)
        direction = residual+(following/squared.clamp_min(1e-20))[:, None]*direction
        squared = following
    return prior+delta
