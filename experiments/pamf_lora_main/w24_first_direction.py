"""Train the first W24 MAP edit direction from matched source IQ."""

import torch


DIRECTION_WEIGHT = 0.10


def direction_risk(correction, target):
    """Measure signed complex alignment per chirp in the receive frame."""
    target_power = target.abs().square().mean(-1)
    edit_power = correction.abs().square().mean(-1)
    agreement = (correction * target.conj()).real.mean(-1)
    cosine = agreement * (edit_power + 1e-4).rsqrt() * target_power.clamp_min(1e-8).rsqrt()
    usable = target_power > 1e-8
    return ((1 - cosine) * usable).sum() / usable.sum().clamp_min(1)


def objective_for(archive, *, output_bias=False):
    """Retain the frozen recipe's draws and losses and add one matched term."""
    if output_bias:
        from experiments.pamf_lora_main.w24_source_bias import BIAS_WEIGHT, coherent_error

    window = archive.WIDE
    risk = archive.waveform_risk

    def objective(model, tracker, raw, rng, *, with_identity, recipe, step):
        raw = raw / raw.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        clean = tracker.aperture(raw)
        center = clean[..., :window]
        framed, phase, _, rms = archive.coordinates(tracker, archive.real_iq(center))
        clean_frame = phase * rms
        delta = archive.weak_difference(framed)
        change = delta * clean_frame
        a, b = (tracker.aperture(archive.condition_waveforms(raw.unflatten(-1, (8, 1024)), rng).flatten(-2))
                for _ in range(2))
        if with_identity:
            a[-2:] = b[-2:] = clean[-2:]
        data = torch.stack((a, b, torch.cat((a[..., :window] + change, a[..., window:]), dim=-1)), dim=1)
        observed = data.flatten(0, 1)
        y, phase, quality, rms = archive.coordinates(tracker, archive.real_iq(observed[..., :window]))
        frame = phase * rms
        ground = torch.stack((center, center, center + change), dim=1).flatten(0, 1) / frame
        (coarse, first, final), action, variance, _, _ = model.propose(y, quality, rms, observed[..., window:])
        half = y + .5 * (final - y)
        prediction = (half * frame).unflatten(0, data.shape[:2])
        corruption = (y - action(ground)).abs().square()
        signal = archive.split((prediction[:, 2] - prediction[:, 0]) / clean_frame)
        expected = archive.split(delta)
        preserve = archive.transfer_risk(signal[:, :1], expected[:, :1]) + archive.transfer_risk(
            signal[:, 1:], expected[:, 1:])
        consistency = ((prediction[:, 0] - prediction[:, 1]) / clean_frame).abs().square().mean()
        identity = ((prediction - data[..., :window])[-2:, :2].abs().square().mean()
                    if with_identity else y.real.new_zeros(()))
        values = torch.stack((risk(coarse, ground), risk(first, ground), risk(final, ground), risk(half, ground),
                              preserve, consistency, (corruption / (.01 + variance.detach())).log1p().mean(),
                              (corruption.detach() / variance + variance.log()).mean(), identity))
        a_weight, b_weight, c_weight = recipe.loss_weights(step)
        weights = values.new_tensor((.6 * a_weight, .4 * a_weight, .35 * b_weight, .65 * b_weight,
                                     .30, c_weight, .07, .015, .35))
        if output_bias:
            bias = coherent_error(prediction, center, clean_frame, with_identity=with_identity)
            return values @ weights + BIAS_WEIGHT * bias, torch.cat((values, bias[None]))
        # The source clean/received edit is only a loss target. Exclude the
        # clean exposures explicitly and leave the artificial probe to its
        # original transfer loss. Deployment calls the unchanged model.
        edit = (first - y).unflatten(0, data.shape[:2])[:, :2]
        desired = (ground - y).unflatten(0, data.shape[:2])[:, :2]
        if with_identity:
            edit, desired = edit[:-2], desired[:-2]
        alignment = direction_risk(edit, desired)
        return values @ weights + DIRECTION_WEIGHT * alignment, torch.cat((values, alignment[None]))

    return objective
