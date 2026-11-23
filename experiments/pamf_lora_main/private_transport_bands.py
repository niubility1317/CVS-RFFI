"""Fixed private-spectrum full/half combinations on three disjoint LoRa bands."""

import torch


SUBSETS = ('000', '100', '010', '001', '110', '101', '011', '111')


def named_band_masks(reference):
    frequencies = (reference.shape[-1] * torch.fft.fftfreq(reference.shape[-1],
                    device=reference.device)).round().abs()
    bounds = (0, 8, 32, 64)
    return torch.stack(tuple((frequencies > lo) & (frequencies <= hi)
                             for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)))


def band_deltas(half, full, masks):
    # Exact unchanged endpoints permit a packet-by-packet lineage check.
    difference = torch.fft.fft(full-half, norm='ortho')
    rows = {'000': half}
    for label in SUBSETS[1:-1]:
        chosen = [i for i,value in enumerate(label) if value=='1']
        selected = masks[chosen].any(0)
        rows[label] = half + torch.fft.ifft(difference * selected, norm='ortho')
    rows['111'] = full
    return rows
