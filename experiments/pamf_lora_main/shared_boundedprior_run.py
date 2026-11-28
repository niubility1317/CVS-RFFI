"""Source-teacher training of a common real-gain and last R17 prior fusion."""
import argparse
import copy
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from time import monotonic

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_lora_main.dura_real_gain_fit import (DEVICE, LORA_ACTION, LORA_GAIN,
    WISIG_CACHE, WISIG_PACKET, WISIG_PARENT, as_complex)
from experiments.pamf_lora_main.shared_dura_realgain import SharedBoundedGain
from experiments.pamf_lora_main.shared_dura_realgain_run import REFERENCE, dispatch, score
from experiments.pamf_lora_main import shared_twoprior_run as shared
from experiments.pamf_lora_main.shared_practical_urban_run import practical_pool, source_teacher, teacher_kl
from experiments.pamf_restoration.models import complex_iq, pair

RUN = '20261128-phase1-shared-boundedprior-gain-s2026112957-r01'
PARENT = '20261128-phase1-dura-realgain-crossdata-s2026112845-r02'
SEED, STEPS = 2026112957, 1000
BATCH = {'lora': 2, 'wisig': 16}
ANCHOR_REAL = {'lora': True, 'wisig': False}


def load_trainable_r17(project, dataset):
    if dataset == 'lora':
        from experiments.pamf_lora_main import w24_joint_pamf as joint
        from experiments.pamf_lora_main import w24_native_training as native
        old = read(project / 'runs' / LORA_GAIN / 'frequency_inphase_native' / 'fit-inputs.json')
        contract = joint.source_contract(Path(old['source_root']))
        model = native.dura_model(project / 'runs' / LORA_ACTION, contract, DEVICE).eval()
        prior = model.priors[-1]

        def forward(y):
            with torch.no_grad():
                _, action, _ = model.estimate(pair(y))
            return model(y), action
    else:
        from experiments.pamf_unrolled_restore.runtime import load_checkpoint
        path = project / 'runs' / WISIG_PARENT / 'unroll_unshared' / 'native_map_prior' / 'final.pt'
        model, proof = load_checkpoint(path, DEVICE)
        if (proof['candidate'], proof['epoch'], proof['updates']) != ('unroll_unshared', 20, 4940):
            raise ValueError('Expected original same-contract R17 source checkpoint')
        prior = model.prior[-1]

        def forward(y):
            p, _, action, _ = model.training_outputs(pair(y))
            return complex_iq(p), action
    original = copy.deepcopy(model).eval().requires_grad_(False)
    model.requires_grad_(False)
    prior.requires_grad_(True)
    raw_forward = forward

    def forward(y):
        physical, action = raw_forward(y)
        with torch.no_grad():
            reference = original(y) if dataset == 'lora' else complex_iq(original.training_outputs(pair(y))[0])
        return bound_prior(reference, physical), action

    return model, prior, forward


def bound_prior(reference, physical):
    delta = physical-reference
    rms = delta.abs().square().mean(-1, keepdim=True).clamp_min(1e-20).sqrt()
    limit = .01*reference.abs().square().mean(-1, keepdim=True).sqrt()
    return reference + (limit/rms).clamp(max=1.)*delta


def restore(received, r17, gain, model):
    scale = received.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    y = received / scale
    with torch.no_grad():
        real = gain(y)
    physical, action = r17(y)
    return model(y, real, physical, action) * scale, physical * scale, real * scale


def make_eval_gain(dataset, r17):
    class EvalGain(SharedBoundedGain):
        def forward(self, received, real_gain, physical, action):
            if dataset == 'wisig':
                physical, action = r17(received)
            return super().forward(received, real_gain, physical, action)
    return lambda width: EvalGain(width, anchor_real=ANCHOR_REAL[dataset], correction_cap=.05)


def fit(rt, dataset):
    source, ids, frozen_r17, gain = shared.experts(rt, dataset)
    teacher = source_teacher(rt, dataset)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    r17_model, last, r17 = load_trainable_r17(Path(rt['project']), dataset)
    model = SharedBoundedGain(shared.WIDTH[dataset], anchor_real=ANCHOR_REAL[dataset], correction_cap=.05).to(DEVICE).train()
    params = list(model.parameters()) + list(last.parameters())
    opt = torch.optim.AdamW([{'params': model.parameters(), 'lr': 3e-4},
                             {'params': last.parameters(), 'lr': 1e-5}], weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: .1+.9*(1+math.cos(math.pi*step/STEPS))/2)
    started = monotonic()

    def objective(clean, observed, check=False):
        input_scale = observed.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        clean, observed = clean/input_scale, observed/input_scale
        output, physical, real = restore(observed, r17, gain, model)
        preserved, physical_clean, real_clean = restore(clean, r17, gain, model)
        clean_scale = clean.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
        with torch.no_grad():
            reference, _ = frozen_r17(clean/clean_scale)
            reference = reference*clean_scale
            clean_logits = teacher(clean)
        fixed_anchor = real_clean if ANCHOR_REAL[dataset] else reference
        wave = ((output-clean)/clean_scale).abs().square().mean()
        keep = ((preserved-fixed_anchor)/clean_scale).abs().square().mean()
        prior_keep = ((physical_clean-reference)/clean_scale).abs().square().mean()
        loss = (teacher_kl(clean_logits, teacher(output)) + .2*teacher_kl(clean_logits, teacher(physical))
                + .2*wave + 1.5*keep + .5*prior_keep)
        if check:
            anchor = real if ANCHOR_REAL[dataset] else physical
            torch.testing.assert_close(output, anchor, rtol=0, atol=0)
            grads = torch.autograd.grad(loss, (model.net[-1].weight, last.out.weight), retain_graph=True)
            if any(not torch.isfinite(g).all() or not bool(g.norm() > 0) for g in grads):
                raise ValueError('Both gain and last prior need finite nonzero gradients')
        return loss

    def smoke(clean, observed):
        n = min(BATCH[dataset], len(clean))
        objective(as_complex(clean[:n]), as_complex(observed[:n]), check=True)
        print('SHARED_BOUNDEDPRIOR_REAL_CHECKPOINT_SMOKE', dataset, n, 'PASS', flush=True)

    clean_pool, observed_pool, counts = practical_pool(source, ids, dataset, smoke)
    for step in range(1, STEPS+1):
        chosen = rng.choice(len(clean_pool), BATCH[dataset], replace=False)
        loss = objective(as_complex(clean_pool[chosen]), as_complex(observed_pool[chosen]), check=step == 1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite source-only objective')
        opt.step()
        schedule.step()
        if step == 1 or step % 50 == 0:
            print('SHARED_BOUNDEDPRIOR_FIT', dataset, step, float(loss.detach()), float(norm), flush=True)
    torch.cuda.synchronize()
    proof = dict(status='SOURCE_FROZEN', dataset=dataset, seed=SEED, updates=STEPS,
        batch=BATCH[dataset], U_pool=len(ids), simulation_pool=4096, simulation_scenes=counts,
        simulation_seed=SEED+61, U_exposures=STEPS*BATCH[dataset],
        F_reads=3*(STEPS+1)*BATCH[dataset], TX_reads=0, V_reads=0, target_reads=0,
        teacher='LoRa W24 E104' if dataset == 'lora' else 'WiSig Tri-SIG-TIME SSL F0 default head',
        lora_backward='exact scipy forward; WideTracker surrogate gradient' if dataset == 'lora' else 'not applicable',
        anchor='real_gain' if ANCHOR_REAL[dataset] else 'r17', correction_rms_fraction_cap=.05,
        anchor_selection='previously exposed source V low-urban strongest of real-gain and R17',
        prior_parent=LORA_ACTION if dataset == 'lora' else WISIG_PARENT, prior_rms_fraction_cap=.01,
        gain_parameters=sum(p.numel() for p in model.parameters()),
        last_prior_parameters=sum(p.numel() for p in last.parameters()),
        total_r17_parameters=sum(p.numel() for p in r17_model.parameters()),
        seconds=monotonic()-started, peak_cuda_bytes=torch.cuda.max_memory_allocated())
    folder = Path(rt['run_root']) / dataset
    with (folder / 'gain.pt').open('xb') as stream:
        torch.save(dict(weights=model.state_dict(), proof=proof), stream)
    with (folder / 'prior_last.pt').open('xb') as stream:
        torch.save(last.state_dict(), stream)
    write(folder / 'fit.json', proof)


def predict(rt, dataset):
    source, ids, _, gain = shared.experts(rt, dataset)
    r17_model, last, r17 = load_trainable_r17(Path(rt['project']), dataset)
    last.load_state_dict(torch.load(Path(rt['run_root']) / dataset / 'prior_last.pt', map_location=DEVICE, weights_only=True), strict=True)
    r17_model.eval().requires_grad_(False)
    old_experts, old_gain = shared.experts, shared.SharedTwoPriorGain
    shared.experts = lambda _rt, _dataset: (source, ids, r17, gain)
    shared.SharedTwoPriorGain = make_eval_gain(dataset, r17)
    shared.SEED, shared.STEPS = SEED, STEPS
    try:
        shared.predict(rt, dataset, allow_source_f_guidance=True)
    finally:
        shared.experts, shared.SharedTwoPriorGain = old_experts, old_gain


def worker(runtime, dataset):
    rt = read(runtime)
    fit(rt, dataset)
    predict(rt, dataset)
    result = subprocess.run([sys.executable, '-u', '-m', __spec__.name, 'score', str(runtime), dataset],
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdin=subprocess.DEVNULL, check=False)
    if result.returncode:
        raise RuntimeError('Independent source scoring failed')


def publish(repo, output, context, ssh_config):
    from experiments.pamf_lora_main.publish import release, release_commit
    source = read(context)
    rt = {k: source[k] for k in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    runs, releases = project / 'runs', project / 'releases'
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
        release_root=str(releases / f'{RUN}-release'), run_root=str(runs / RUN),
        lora_frozen_root=str(runs / PARENT), allowed_gpus=[4, 6],
        external_releases=[str(runs / n) for n in (LORA_ACTION, LORA_GAIN, WISIG_CACHE,
            WISIG_PACKET, WISIG_PARENT, PARENT, '20261128-diagnostic-lora-r69-full-w24-s2026112895-r01')]
            + [str(releases / REFERENCE)])
    release(repo, output, rt, ssh_config, archive_extras=('tools/r17_activation_factorization.py',))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for phase in ('dispatch', 'worker', 'score'):
        child = commands.add_parser(phase)
        child.add_argument('runtime', type=Path)
        if phase != 'dispatch':
            child.add_argument('dataset', choices=('lora', 'wisig'))
    pub = commands.add_parser('publish')
    for field in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--' + field, type=Path, required=True)
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action == 'dispatch':
        dispatch(args['runtime'], worker_module=__spec__.name)
    elif action == 'score':
        score(read(args['runtime']), args['dataset'])
    elif action == 'worker':
        worker(**args)
    else:
        publish(**args)
