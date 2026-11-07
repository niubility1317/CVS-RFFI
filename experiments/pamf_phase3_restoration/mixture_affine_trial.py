"""Matched scratch30 comparison of centered affine mixture fusion."""
from . import bounded_projected_resource_trial as resource
from . import covariance_detail_trial as training
from . import mixture_gate_trial as parent
from . import multiview_trial as trial
from .mixture_affine_model import scratch_affine_mixture
from .source_pair import read
from .variance_trial import recovery_gpu_choice

RUN = '20261108-phase1-Restore-MixtureAffine-E30-s2026092701-r01'
MODULE = 'experiments.pamf_phase3_restoration.mixture_affine_trial'
ARM = 'TF_MixtureAffine'


def configuration(arm):
    if arm != ARM:
        raise ValueError('Only fixed centered-affine mixture candidate allowed')
    cfg = parent.configuration(parent.ARM)
    cfg.update(run_id=RUN, arm=ARM,
        architecture=('Matched source mixture fusion model, replacing posterior-logit softmax adjustment '
                      'with centered, unconstrained affine IQ combination; same multiscale residual.'),
        scientific_change=('Only fusion mapping changes; initial tensors, unadjusted mixture IQ, source data, '
                           'loss, seeds and single-stage schedule matched. Coefficients can be signed, not probabilities.'),
        matched_control=parent.RUN+'/'+parent.ARM,
        fusion='base posterior IQ + sum_k (adjustment_k - mean(adjustment))*component_k',
        source_evidence=('On two original L_s/U_s nonclean records, initial IQ-only gate gradient '
                         '4.21e-25 for posterior-softmax adjustment versus 0.185 for centered affine; '
                         'no F0, labels, target or optimizer updates.'),
        resource_policy=('GPU3 only after the original mixture version has completed its GPU stages or '
                         'its own worker has terminated in failure; admit at most one existing compute PID, '
                         'total cap3, free >=12288MiB.'))
    return cfg


def factory(arm):
    configuration(arm)
    return scratch_affine_mixture(2026092701)


def source_readout(model, reader, received, clean, features, scale, objective, *, source_indices, epoch):
    restored, aux = model(received, return_aux=True)
    loss, metrics = training.detail_objective(reader, restored, clean, features, scale, objective)
    energy = clean.detach().square().mean((1, 2)).clamp_min(1e-4)
    centered = aux['mixture_adjustment'] - aux['mixture_adjustment'].mean(1, keepdim=True)
    metrics.update(gate_adjustment_rms=centered.square().mean(1).sqrt(),
        gate_iq_correction_nmse=(aux['mixture_affine_iq']-aux['base_iq']).square().mean((1, 2))/energy)
    return restored, aux, loss, metrics


def train(args, out):
    training.train(args, out, build=factory, config_for=configuration,
        bind_prior=parent.bind_prior, training_readout=source_readout,
        smoke_gradient_check=parent.gradient_smoke,
        additional_prior_keys=('mixture_within','mixture_offsets','mixture_weights'))


def previous_gpu_stages_finished(root):
    folder = root/'runs'/parent.RUN
    for stage in ('train','cross_rx','cross_both'):
        path = folder/f'terminal-{parent.ARM}-{stage}.json'
        if path.is_file() and (read(path)['exit_code'] != 0 or stage == 'cross_both'):
            return True
    return False


def admitted_gpus():
    if not previous_gpu_stages_finished(_root):
        return []
    return [g for g in recovery_gpu_choice(allowed_gpus=(3,), maximum_processes=2)
            if g['free_memory_MB'] >= 12288]


def controller(args):
    global _root
    _root = args.root
    resource.controller(args)


def install():
    trial.RUN, trial.MODULE, trial.ARMS = RUN, MODULE, (ARM,)
    trial.configuration, trial.factory, trial.train = configuration, factory, train
    resource.RUN, resource.ARM, resource.admitted_gpus = RUN, ARM, admitted_gpus
    trial.controller = controller


if __name__ == '__main__':
    install()
    trial.main()
