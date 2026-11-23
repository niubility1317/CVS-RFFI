"""Release matched learned polar and received repeat-drift WiSig restorers."""

from functools import partial

import torch

from . import wisig_anchored_window_source as runner
from .wisig_repeat_drift import RepeatDrift, generic_drift, objective


RUN = '20261124-phase1-wisig-repeat-drift-source-m4-r01'
ARMS = ('polar','drift')
BASE_READOUTS = ('timed','coarse','joint','gated')
DRIFT_READOUTS = ('timed','engineering','coarse','joint','gated','phase_only','no_observed')
MODES = ('direct','r69',*(f'polar_{x}' for x in BASE_READOUTS),*(f'drift_{x}' for x in DRIFT_READOUTS))
COMPARISONS = (*( (f'drift_{kind}',f'polar_{kind}') for kind in BASE_READOUTS),
               ('drift_gated','direct'),('drift_gated','drift_no_observed'),
               ('drift_gated','drift_phase_only'),('drift_coarse','drift_engineering'),
               ('drift_engineering','drift_timed'),('drift_joint','drift_coarse'),
               *((mode,'r69') for mode in MODES[2:]))


def build(arm):
    return RepeatDrift(arm=='drift')


def proposals(model, gate, y):
    analysis = model.analyze(y)
    (timed,estimate,coarse,fine),shift,margin,noise = model.stages(analysis)
    active = (margin>gate[0]) | (noise>gate[1])

    def guard(value):
        return torch.where(active[:,None],value,y)

    if not model.apply_observed:
        return (timed,coarse,fine,guard(fine)),active,shift
    phase_only = torch.polar(timed.abs(),fine.angle())
    unassisted = model.stages(analysis,use_observed=False)[0][-1]
    return (timed,estimate,coarse,fine,guard(fine),guard(phase_only),guard(unassisted)),active,shift


def measurements(model,y):
    raw = y / y.abs().square().mean(-1,keepdim=True).clamp_min(1e-8).sqrt()
    amplitude, phase, confidence = model.estimate_drift(raw)
    return dict(absolute_log_amp=amplitude.abs().amax(-1),phase_rad=phase.abs().amax(-1),
                repeat_confidence=confidence)


if __name__ == '__main__':
    load = partial(runner.load_models,arms=ARMS,build=build)
    runner.main(run=RUN,runner=__spec__.name,arms=ARMS,modes=MODES,comparisons=COMPARISONS,
                fit_arm=partial(runner.fit,build=build,calculate=objective),
                run_smoke=partial(runner.smoke,models_from=load,variants=proposals,disturb=generic_drift),
                run_evaluation=partial(runner.evaluate,arms=ARMS,models_from=load,variants=proposals,
                                       auxiliary=measurements))
