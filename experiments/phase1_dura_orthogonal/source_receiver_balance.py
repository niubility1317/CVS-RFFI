"""Source-only inverse-frequency receiver weighting for waveform recovery."""
from collections import Counter

SOURCE_RECEIVERS = (1, 3, 4, 6, 8)
EXPECTED_COUNTS = {'1': 8262, '3': 8182, '4': 5458, '6': 5341, '8': 5525}


def receiver_weights(receivers):
    counts = Counter(int(r) for r in receivers)
    if set(counts) != set(SOURCE_RECEIVERS):
        raise ValueError('All and only the five source receivers are required')
    total = len(receivers)
    weights = [total / (len(SOURCE_RECEIVERS) * counts[int(r)]) for r in receivers]
    return weights, {str(r): counts[r] for r in SOURCE_RECEIVERS}


def receiver_weighted_guard(output, anchor, clean, weights):
    if (output.ndim != 3 or output.shape[1:] != (2, 256) or
            anchor.shape != output.shape or clean.shape != output.shape or weights.shape != (len(output),)):
        raise ValueError('Matching source IQ and receiver weights required')
    energy = clean.square().mean((1, 2)).clamp_min(1e-8)
    error = (output - clean).square().mean((1, 2)) / energy
    change = (output - anchor).square().mean((1, 2)) / energy
    baseline = (anchor.detach() - clean).square().mean((1, 2)) / energy
    excess = (error - baseline).clamp_min(0)
    return ((error + .02 * change + .002 * excess / (baseline + 1e-4)) * weights).mean()
