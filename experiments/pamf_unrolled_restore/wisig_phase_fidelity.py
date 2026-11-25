"""Train the native WiSig FIR to retain source-IQ phase at both MAP outputs."""

import torch


PHASE_WEIGHT = .20


def phase_chord(wave, reference):
    """Compare regularized unit phasors, weighted by reference phase reliability."""
    target_power = reference.abs().square()
    level = target_power.mean(dim=-1, keepdim=True).clamp_min(1e-12)
    floor = .0025 * level
    confidence = target_power / (target_power + .04*level)
    estimate = wave / (wave.abs().square() + floor).sqrt()
    target = reference / (target_power + floor).sqrt()
    numerator = (.5 * confidence * (estimate-target).abs().square()).sum(dim=-1)
    return (numerator / confidence.sum(dim=-1).clamp_min(1e-8)).mean()


def objective_components(model, source, rng):
    """Replicate the archived data draws and costs, and expose the same MAP outputs to phase loss."""
    from experiments.pamf_unrolled_restore.wisig_native_fir import natural_delta, observe
    from experiments.pamf_unrolled_restore.wisig_uncertain_prior import normalized_error

    detail = natural_delta(source)
    targets = torch.stack((source, source, source+detail), 1)
    shared = observe(targets[:, (0, 2)], rng, model)
    fresh = observe(targets[:, 1:2], rng, model)
    views = torch.stack((shared[:, 0], fresh[:, 0], shared[:, 1]), 1)
    views[-3:] = targets[-3:]
    clean = targets.flatten(0, 1)
    analysis = model.analyze(views.flatten(0, 1))
    stages, action = model.stages(analysis)
    measured, level, _, _, _, _, variance = analysis
    residual = (action(clean/level)-measured).abs().square()
    forward = residual.log1p().mean()
    likelihood = (.5*residual.detach()/variance+variance.log()).mean()
    result = stages[-1].unflatten(0, (len(source), 3))
    consistency = normalized_error(result[:, 1], result[:, 0])
    preserved = (result[:, 2]-result[:, 0]-detail).abs().square().mean(-1)
    transfer = (preserved/(.002+detail.abs().square().mean(-1))).log1p().mean()
    error = torch.fft.fft(stages[-1]-clean, norm='ortho').abs().square()
    weak = (error/(.025+torch.fft.fft(clean, norm='ortho').abs().square())).log1p().mean()
    costs = torch.stack((*(normalized_error(stage, clean) for stage in stages),
                         forward, likelihood, consistency, transfer, weak))
    weights = costs.new_tensor((.30, .13, .35, .12, 1., .75, .035, .10, .20, .12))
    phases = torch.stack(tuple(phase_chord(stages[index], clean) for index in (2, 4)))
    return (weights*costs).sum(), costs, phases
