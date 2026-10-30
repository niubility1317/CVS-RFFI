"""Single-forward scratch training on exactly rotated source IQ/action pairs."""
import time
import numpy as np
import torch

from . import no_variance
from experiments.pamf_action_restore.operators import Action
from experiments.pamf_dura_diagnostics.runner import tensor
from experiments.pamf_restoration.models import complex_iq, pair

RUN = '20261030-phase1-DURA-source-phasePair-s2026092701-r01'
MODULE = 'experiments.pamf_dura_direct_train.phase_pair'
SEED = 2026103019


@torch.no_grad()
def rotate_pair(received, clean, action, angle):
    unit = torch.polar(torch.ones_like(angle), angle)[:, None]
    # A(z)=a H(z)+b conj(H(z)). Conjugating A by a common rotation R
    # keeps H/a and maps b to R^2 b. Rotating IQ alone is inconsistent.
    transformed = Action(action.kernel, action.main, action.image * unit.square())
    return pair(unit * complex_iq(received)), pair(unit * complex_iq(clean)), transformed


class PhaseObjective:
    def __init__(self, objective, *, audit=False):
        self.objective = objective
        self.rng = np.random.default_rng(SEED)
        self.seen = self.calls = 0
        self.angle_bins = np.zeros(8, np.int64)
        self.angle_sum = self.angle_square_sum = 0.
        self.audit = audit
        self.audit_metrics = None

    def __call__(self, model, received, clean, action, variance, probes):
        angles = self.rng.uniform(-np.pi, np.pi, len(clean)).astype(np.float32)
        angle = tensor(angles, clean.device)
        observed, reference, transformed = rotate_pair(received, clean, action, angle)
        self.seen += len(clean); self.calls += 1
        self.angle_bins += np.histogram(angles, bins=8, range=(-np.pi, np.pi))[0]
        self.angle_sum += float(angles.astype(np.float64).sum())
        self.angle_square_sum += float(np.square(angles.astype(np.float64)).sum())
        if self.audit:
            with torch.no_grad():
                unit = torch.polar(torch.ones_like(angle), angle)[:, None]
                noise = complex_iq(received) - action(complex_iq(clean))
                replay = complex_iq(observed) - transformed(complex_iq(reference))
                self.audit_metrics = dict(
                    residual_rotation_max_error=float((replay-unit*noise).abs().max()),
                    relative_sample_power_max_error=float(((reference.square().sum(1)-clean.square().sum(1)).abs()
                        / clean.square().sum(1).clamp_min(1e-8)).max()))
        return self.objective(model, observed, reference, transformed, variance, probes)

    def summary(self):
        return dict(seen=self.seen, calls=self.calls, probability=1., seed=SEED,
                    angle_range_radians=[-float(np.pi), float(np.pi)], angle_bins=self.angle_bins.tolist(),
                    angle_sum=self.angle_sum, angle_square_sum=self.angle_square_sum,
                    numpy_version=np.__version__, G_forwards_per_visit=1, TX_labels_read=0,
                    F_training_inputs=0, learned_teacher_inputs=0)


def configure():
    no_variance.configure(); base = no_variance.base
    previous_configuration = base.configuration; previous_command = base.command
    previous_smoke = base.smoke; previous_train = base.train
    source_objective = base.direct_objective

    def configuration():
        value = previous_configuration()
        value.update(run=RUN, recipe='single_stage_scratch_no_variance_aux_exact_source_phase_pair',
            source_phase_seed=SEED, source_phase_probability=1., source_phase_range_radians=[-float(np.pi), float(np.pi)],
            source_pairing='y2=R*y,x2=R*x,A2=R*A*R^-1;kernel/main unchanged,image2=R^2*image;noise rotates,variance unchanged',
            G_forwards_per_training_visit=1, learned_teachers=0,
            source_clean_identity='Existing native cache retains its original15percent clean visits;no added/replaced visits',
            max_total_visible_GPU_processes=3, allowed_GPUs=[4, 6, 7],
            resource_scope='One own worker on GPU4/6/7,>=12GB free,at most2 existing compute PIDs before adding;other owners unchanged',
            host_numpy_conversion='DLPack_owned_FP32',
            comparison='Fixed source-defined noVar E20(cf40),not a checkpoint initializer;no source/target selection')
        return value

    def command(args, mode, arm=None, role=None):
        return [MODULE if item == no_variance.MODULE else item
                for item in previous_command(args, mode, arm, role)]

    def smoke(args, p):
        objective = PhaseObjective(source_objective, audit=True)
        base.direct_objective = objective
        previous_smoke(args, p)
        metrics = objective.audit_metrics
        if objective.seen != 2 or objective.calls != 1 or metrics is None:
            raise ValueError('Real source2Ls rotation/objective smoke required')
        if not (metrics['residual_rotation_max_error'] <= 2e-6
                and metrics['relative_sample_power_max_error'] <= 2e-6):
            raise ValueError('Real source phase pairing changed residual/power')
        base.write_json(p['output']/'phase_smoke.json', dict(status='PASS', **objective.summary(), **metrics))

    def train(args, p):
        started = time.perf_counter()
        objective = PhaseObjective(source_objective)
        base.direct_objective = objective
        previous_train(args, p)
        if (objective.seen, objective.calls) != (1260000, 4940):
            raise ValueError('Complete fixed source rotation exposure required')
        base.write_json(p['output']/args.arm/'source_pairing_summary.json',
            dict(status='SOURCE_PAIRING_COMPLETE', total_G_training_pipeline_seconds=time.perf_counter()-started,
                 **objective.summary()))

    base.RUN = RUN; base.configuration = configuration; base.command = command
    base.smoke = smoke; base.train = train


if __name__ == '__main__':
    configure(); no_variance.base.main()
