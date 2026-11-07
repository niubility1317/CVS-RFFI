"""Build source-only same-record DURA mean teachers in view-major order."""
import torch


def build_teacher(anchor, clean, fit_count):
    if (fit_count < 1 or anchor.shape != clean.shape or anchor.shape != (3*fit_count, 2, 256)):
        raise ValueError('Three matching full source IQ views required')
    if (not torch.equal(clean[:fit_count], clean[fit_count:2*fit_count]) or
            not torch.equal(clean[:fit_count], clean[2*fit_count:])):
        raise ValueError('Source view groups must correspond to the same clean physical IQ')
    mean=(anchor[:fit_count]+anchor[fit_count:2*fit_count]+anchor[2*fit_count:])/3
    return torch.cat((mean,mean,mean),0)


def build_leave_one_view_teacher(anchor, clean, fit_count):
    """Use the other two aligned source views as each view's frozen teacher."""
    if (fit_count < 1 or anchor.shape != clean.shape or anchor.shape != (3*fit_count, 2, 256)):
        raise ValueError('Three matching full source IQ views required')
    if (not torch.equal(clean[:fit_count], clean[fit_count:2*fit_count]) or
            not torch.equal(clean[:fit_count], clean[2*fit_count:])):
        raise ValueError('Source view groups must correspond to the same clean physical IQ')
    a, b, c = anchor.split(fit_count)
    return torch.cat(((b+c)/2,(a+c)/2,(a+b)/2),0)
