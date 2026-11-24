"""Map one WiSig window into the LoRa packet interface for transfer diagnostics.

The six entries are repeated representations of one observation; they do not
provide six independent WiSig or LoRa measurements.
"""


def pack(iq):
    if iq.ndim != 3 or iq.shape[1:] != (2, 256):
        raise ValueError('Expected WiSig IQ with shape (batch, 2, 256)')
    return iq.unsqueeze(1).expand(-1, 6, -1, -1).contiguous()


def unpack(views):
    if views.ndim != 4 or views.shape[1:] != (6, 2, 256):
        raise ValueError('Expected six aligned (2, 256) representations')
    return views.mean(dim=1)


def restore_expanded(iq, recover, tracker, models, threshold):
    outputs, active, _ = recover(pack(iq), tracker, models, threshold)
    return unpack(outputs[3]), active


def restore_guard(iq, recover, tracker, model, gate, threshold):
    outputs, active = recover(pack(iq), tracker, model, gate, threshold)
    return unpack(outputs[2]), active
