"""Count fixed-epoch U exposure when a shuffled loader cycles through short batches."""


def fixed_epoch_exposure(unlabeled_count, *, steps=222, batch_size=256, physical_limit=128):
    """Return seen/selected samples for one fresh FixedEpochLoader iterator."""
    if any(type(v) is not int or v <= 0 for v in (unlabeled_count, steps, batch_size, physical_limit)):
        raise ValueError('Positive integer loader sizes required')
    full, tail = divmod(unlabeled_count, batch_size)
    batches = full + bool(tail)
    cycles, remaining = divmod(steps, batches)
    seen = cycles * unlabeled_count + remaining * batch_size
    selected = cycles * (full * min(batch_size, physical_limit) + min(tail, physical_limit))
    selected += remaining * min(batch_size, physical_limit)
    return dict(steps=steps, cycles=cycles, remaining_batches=remaining, seen=seen, selected=selected)
