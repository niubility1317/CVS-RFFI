"""Fixed source-only masked-IQ teacher constraint for recovery training."""
import numpy as np
import torch

SOURCE = '20261107-phase1-masked-iq-prior-source-s2026092701-r01'
SOURCE_COMMIT = 'acdfd17ae205fa5e04b32a5ecdd5eea5d7bb21d4'
MASK_SEED = 2026110711
WEIGHT = .05


def load_prior(root, fit, device, *, smoke=False):
    from experiments.pamf_phase1_flow import experiment as core
    from experiments.phase1_dura_orthogonal.source_masked_prior import MaskedIQPrior
    folder = root/'runs'/SOURCE
    summary = core.read(folder/'result.json')
    state = torch.load(folder/'final.pt', map_location='cpu', weights_only=False)
    expected = dict(status='SOURCE_MASKED_PRIOR_COMPLETE', commit=SOURCE_COMMIT,
                    fit_ids=32768, epochs=5, TX_labels_read=0, F_weights_read=0,
                    F_outputs_read=0, V_reads=0, target_reads=0)
    if any(summary.get(k) != v for k, v in expected.items()) or state.get('commit') != SOURCE_COMMIT:
        raise ValueError('Fixed source-only masked prior provenance required')
    teacher_fit = np.asarray(state.get('fit_ids', []))
    if len(teacher_fit) != 32768 or not np.array_equal(teacher_fit[:len(fit)], fit) or (not smoke and len(fit) != 32768):
        raise ValueError('Exact source fit IDs required for masked teacher')
    model = MaskedIQPrior().to(device)
    model.load_state_dict(state['model'], strict=True)
    return model.eval().requires_grad_(False)


def masked_prior_loss(prior, output, clean, starts):
    from experiments.phase1_dura_orthogonal.source_masked_prior import SPAN
    if output.shape != clean.shape or output.ndim != 3 or output.shape[1:] != (2, 256) or starts.shape != (len(output),):
        raise ValueError('Matching source output and clean IQ required')
    if bool(((starts < 1) | (starts + SPAN >= 256)).any()):
        raise ValueError('Mask requires both adjacent samples')
    positions = torch.arange(256, device=output.device)[None, :]
    mask = (positions >= starts[:, None]) & (positions < starts[:, None] + SPAN)
    def visible(value):
        observed = value.masked_fill(mask[:, None], 0.)
        scale = (observed.square().sum((1, 2), keepdim=True) / (2*(256-SPAN))).clamp_min(1e-12).sqrt()
        return observed, scale
    observed, scale = visible(output)
    clean_observed, clean_scale = visible(clean)
    with torch.no_grad():
        target = prior(clean_observed, mask, clean_scale)
    prediction = prior(observed, mask, scale)
    energy = clean.square().mean((1, 2)).clamp_min(1e-8)
    return (((prediction-target).square()*mask[:, None]).sum((1, 2))/(2*SPAN*energy)).mean()
