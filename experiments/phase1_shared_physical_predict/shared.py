"""Evaluate fixed physical-fusion models while reusing same-record inference."""
import torch


@torch.no_grad()
def predict_shared(x, models, gate, dura, physical, reader):
    """Return the same tuples as separate physical-fusion predict_batch calls.

    All models share the same input batch, frozen anchor, posterior, gate and
    reader. Each model and each restored reader call remains separate to keep
    the existing batch shape and operation order unchanged.
    """
    if not models or any(model.training for model in models):
        raise ValueError('One or more evaluation-mode fusion models are required')
    scores = gate(x).reshape(-1)
    anchor = dura(x)
    if (scores.shape != (len(x),) or anchor.shape != x.shape
            or not all(torch.isfinite(v).all() for v in (x, scores, anchor))):
        raise FloatingPointError('Finite gate scores and DURA waveforms are required')
    active = scores.sigmoid() > .8
    outputs = [anchor.clone() for _ in models]
    if active.any():
        posterior, logvar = physical(x[active])
        packed = torch.cat((anchor[active], posterior), dim=1)
        for model, output in zip(models, outputs):
            output[active] = model(x[active], packed, logvar)

    def classify(wave):
        logits = reader.classify_features(reader.features(wave))
        if logits.shape != (len(x), 6) or not all(torch.isfinite(v).all() for v in (wave, logits)):
            raise FloatingPointError('Finite IQ and frozen reader logits are required')
        return logits.argmax(1)

    raw_class = classify(x)
    anchor_class = classify(anchor)
    return [(anchor, output, active, [raw_class, anchor_class, classify(output)])
            for output in outputs]
