"""Run the source-only, two-route pilot of the uncompressed spectral DURA."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from experiments.cric_dura_comparison.g_lineage import STAGES, load
from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_unrolled_restore.spectral import SpectralUnroll, paired_loss

from .data import SOURCE, VIEWS
from .dura_repair import no_tone
from .full_adaptive import SET_METHOD, SET_RUN, case, complete_f
from .full_dura_lineage import OBSERVED_SHAPE_RUN, TRACE_RUN, frozen_g, load_utility
from .full_dura_lineage_data import LoRaView
from .full_dura_observed_shape import ObservedShape
from .full_packet_publish import DATA_RUN
from .full_packet_views import source_role
from .full_supervised import construct as recognizer
from .packet_physics import aligned_view, chirp_windows
from .packet_runner import schedule
from .physical import ChirpResponse
from .publish import release, release_commit
from .refine_runner import start
from .residual_dura import RUN as RESIDUAL_RUN, SEED, ResidualView
from .residual_dura_bandfit import correction
from .residual_dura_calibration import source_model, source_state
from .residual_dura_donormix import sixth_panel
from .residual_dura_gate import load as load_gate
from .residual_dura_gate_run import RUN as GATE_RUN
from .runner import activate


RUN = '20261123-phase1-lora-dura-spectral-unroll-source-m4-r01'
ROUTES = ('full', 'residual')
ARMS = ('temporal', 'spectral')
STEPS = 600
BATCH = 64
EVAL_PER_RX = 100
MODES = ('direct', 'r69', 'r17', *ARMS, 'spectral_equal_r69')


def bind(rt, route, device, *, pairs=False):
    parent, residual_pairs, contract, residual_contract = source_state(rt)
    if route == 'residual':
        r17 = source_model(parent, 'r17', residual_contract, device)
        return r17, residual_contract, ResidualView(residual_pairs) if pairs else None
    if route != 'full':
        raise ValueError('An independently registered practical route is required')
    full = parent.parent / OBSERVED_SHAPE_RUN
    lineage = full / 'G' / 'lineage'
    proof = read(lineage / 'complete.json')
    if (proof['status'] != 'SOURCE_FROZEN' or proof['updates'] != 39520 or proof['stages'] != list(STAGES)
            or any(proof[k] for k in ('F_reads', 'TX_labels_read', 'target_reads'))):
        raise ValueError('The full practical parent must be independently source-frozen')
    r17, saved = load(lineage / 'r17' / 'final.pt', contract, 'LU', device)
    if (saved['model_seed'] != SEED or saved['stage'] != 'r17'
            or saved['ancestors'] != [str(lineage / s / 'final.pt') for s in STAGES[:2]]):
        raise ValueError('Full practical R17 has the wrong ancestry')
    view = LoRaView(dict(rt, run_root=str(full), parent_root=str(parent.parent / DATA_RUN)),
                    observed_phase=True, full_action=True, full_coarse=True) if pairs else None
    return r17, contract, view


def build(parent, arm, reference, device):
    if arm == 'spectral':
        return SpectralUnroll(parent, reference).to(device)
    if arm == 'temporal':
        return deepcopy(parent)
    raise ValueError('A registered matched continuation arm is required')


def fit(rt, route, arm, *, smoke=False):
    device = 'cuda:0'
    torch.manual_seed(SEED)
    r17, contract, stream = bind(rt, route, device, pairs=True)
    if not stream.ready():
        raise ValueError('Incomplete route-matched source pairs')
    reference = ChirpResponse().chirp.to(device)
    model = build(r17, arm, reference, device).train()
    model.prior.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.prior.parameters(), lr=6e-5, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    steps = 1 if smoke else STEPS
    for step, indices in enumerate(rng.permutation(len(stream))[:steps * BATCH].reshape(steps, BATCH), 1):
        batch = stream.batch(indices, 1 + (step - 1) // (STEPS // 2))
        received, clean = (tensor(batch[key], device) for key in ('received', 'clean'))
        optimizer.zero_grad(set_to_none=True)
        outputs = model.stage_outputs(received)[0]
        value = paired_loss(outputs, clean, reference).mean()
        if not torch.isfinite(value):
            raise FloatingPointError('Nonfinite paired reconstruction')
        value.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.prior.parameters(), 1., error_if_nonfinite=True)
        if step == 1 and (norm <= 0 or any(param.grad is not None for param in model.frozen.parameters())):
            raise ValueError('Prior update or frozen action/source isolation failed')
        if step == 1 and arm == 'spectral' and any(
                stage.spectral.out.weight.grad.norm() <= 0 for stage in model.prior):
            raise ValueError('All three spectral stages must receive source gradients')
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print('SPECTRAL_UNROLL_STEP', route, arm, step, float(value.detach()), float(norm), flush=True)
    result = dict(status='PASS' if smoke else 'SOURCE_FROZEN', route=route, arm=arm, seed=SEED,
                  steps=steps, parameters=sum(p.numel() for p in model.parameters()),
                  TX_reads=0, F_reads=0, V_reads=0, target_reads=0)
    root = Path(rt['run_root'])
    if smoke:
        write(root / f'smoke-{route}.json', result)
    else:
        output = root / f'{route}-{arm}'
        output.mkdir(exist_ok=False)
        torch.save(dict(proof=result, contract=contract, prior=model.prior.state_dict()), output / 'prior.pt')
        write(output / 'complete.json', result)


def read_model(root, route, arm, contract, parent, reference, device):
    folder = root / f'{route}-{arm}'
    saved = torch.load(folder / 'prior.pt', map_location=device, weights_only=True)
    proof = read(folder / 'complete.json')
    if (saved['contract'] != contract or saved['proof'] != proof or proof['route'] != route
            or proof['arm'] != arm or proof['status'] != 'SOURCE_FROZEN' or proof['steps'] != STEPS
            or proof['seed'] != SEED or any(proof[k] for k in ('TX_reads', 'F_reads', 'V_reads', 'target_reads'))):
        raise ValueError('Wrong route, arm or ancestry in the frozen source candidate')
    model = build(parent, arm, reference, device)
    model.prior.load_state_dict(saved['prior'], strict=True)
    return model.eval().requires_grad_(False)


def observed(scene, route, row):
    iq, rx, pid = row
    return aligned_view(iq, scene, rx, pid, 'source-V', processing_route=route)


def proposals(iq, models, projector, gate):
    active = gate(iq)
    waveforms = [iq, *(iq.clone() for _ in MODES[1:])]
    if not active.any():
        return torch.stack(waveforms), 0, 0.
    y = iq[active].flatten(0, 1)
    coordinates = projector.coordinates(y)
    shape, rot, scale = coordinates
    raw = no_tone(shape)
    delta = [correction(projector, model(y), coordinates, raw) for model in models]
    reference_rms = delta[0].abs().square().mean(-1, keepdim=True).sqrt()
    candidate_rms = delta[-1].abs().square().mean(-1, keepdim=True).sqrt()
    same_rms = torch.where(candidate_rms > 1e-7,
                           delta[-1] * (reference_rms / candidate_rms.clamp_min(1e-7)), delta[0])
    drift = (same_rms.abs().square().mean(-1, keepdim=True).sqrt() - reference_rms).abs().max()
    for out, step in zip(waveforms[1:], (*delta, same_rms), strict=True):
        step = step * rot * scale
        out[active] = (y + torch.stack((step.real, step.imag), dim=1)).reshape_as(iq[active])
    return torch.stack(waveforms), int(active.sum()), float(drift)


@torch.no_grad()
def score(rt, route):
    device = 'cuda:0'
    parent, _, contract, paired = source_state(rt)
    r17, train_contract, _ = bind(rt, route, device)
    root = Path(rt['run_root'])
    ref = ChirpResponse().chirp.to(device)
    if route == 'residual':
        r69 = source_model(parent, 'r69', paired, device)
        gate = load_gate(parent.parent / GATE_RUN / 'gate.json', contract, device)
    else:
        r69 = frozen_g(parent.parent / OBSERVED_SHAPE_RUN, contract, device)
        full_gate = load_utility(parent.parent / TRACE_RUN, contract, SEED, device)
        gate = lambda iq: full_gate.impairment(iq).gt(0)
    models = [r69, r17, *(read_model(root, route, arm, train_contract, r17, ref, device) for arm in ARMS)]
    projector = ObservedShape().to(device).eval()
    f = recognizer(device).eval().requires_grad_(False)
    ckpt = complete_f(parent.parent / SET_RUN, contract, case(SEED, SET_METHOD)) / 'model.pt'
    f.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=True)
    source = Path(rt['source_root'])
    iq, rx, ids, _ = source_role(source, contract, 'V')
    panel = sixth_panel(rx).reshape(len(SOURCE), -1)
    subset = np.random.default_rng(SEED + 2).choice(panel.shape[1], EVAL_PER_RX, replace=False)
    chosen = panel[:, np.sort(subset)].flatten()
    rows = list(zip(iq[chosen], rx[chosen], ids[chosen], strict=True))
    predictions, metrics = {}, {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for scene in VIEWS:
            inputs = (np.stack([chirp_windows(row[0]) for row in rows]) if scene == 'clean' else
                      np.stack(list(pool.map(partial(observed, scene, route), rows, chunksize=15))))
            count, parts = 0, []
            for y in tensor(inputs, device).split(16):
                reads, enabled, drift = proposals(y, models, projector, gate)
                logits = f(reads.flatten(0, 1))[0]
                if drift > 1e-6 or not torch.isfinite(logits).all():
                    raise ValueError('Invalid frozen readout or unmatched dose')
                parts.append(logits.argmax(1).reshape(len(MODES), len(y)).cpu().tolist())
                count += enabled
            predictions[scene] = np.concatenate(parts, axis=1)
            metrics[scene] = dict(active=count)
            print('SPECTRAL_SOURCE_PREDICTED', route, scene, count, flush=True)
    write(root / f'{route}-source-predictions.json', {name: values.tolist() for name, values in predictions.items()})
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as metadata:
        truth = metadata['y'][chosen]
    hits = {name: value == truth for name, value in predictions.items()}
    for name, hit in hits.items():
        metrics[name].update(correct=hit.sum(1).tolist(),
                             by_rx=hit.reshape(len(MODES), len(SOURCE), EVAL_PER_RX).sum(-1).tolist())
    six = np.stack([hits[name] for name in VIEWS[1:]]).sum(0)
    low = hits[VIEWS[-1]].astype(int)
    rng = np.random.default_rng(SEED + 3)
    boot = np.concatenate([rng.integers(i * EVAL_PER_RX, (i + 1) * EVAL_PER_RX, (1600, EVAL_PER_RX))
                           for i in range(len(SOURCE))], axis=1)
    intervals = {name: [np.quantile(delta[boot].sum(1), (.025, .975)).tolist() for delta in v - v[1]]
                 for name, v in (('six', six), ('low_urban', low))}
    write(root / f'{route}-source-scores.json', dict(status='SOURCE_SCORED', route=route, n=len(chosen),
          modes=list(MODES), metrics=metrics, six=six.sum(1).tolist(), paired_ci_vs_r69=intervals,
          target_reads=0, truth_after_predictions=True))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root / 'state.json'
    launch = partial(start, module=rt['runner'])
    try:
        for action, jobs in (('smoke', [f'{route}-spectral' for route in ROUTES]),
                             ('fit', [f'{route}-{arm}' for route in ROUTES for arm in ARMS]),
                             ('score', list(ROUTES))):
            write(state, dict(status=action.upper()), exclusive=False)
            schedule(rt, [(action, job) for job in jobs], launch)
        write(state, dict(status='SOURCE_SCORED'), exclusive=False)
    except Exception as error:
        write(state, dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config, parent_submit):
    cfg = read(context)
    rt = {key: cfg[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    parent = project / 'runs' / RESIDUAL_RUN
    receipt = read(parent_submit)
    if receipt['archive_transfer'] != 'VERIFIED' or PurePosixPath(receipt['run_root']) != parent:
        raise ValueError('Matching residual parent receipt required')
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              run_root=str(project / 'runs' / RUN), release_root=str(project / 'releases' / f'{RUN}-release'),
              source_root=str(project / 'releases' / f'{DATA_RUN}-release' / 'data' / 'source'),
              parent_root=str(parent), allowed_gpus=[3, 6, 2, 0, 4, 1, 5], max_workers=4,
              max_compute_processes=4)
    release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='kind', required=True)
    pub = commands.add_parser('publish')
    for option in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--' + option, type=Path, required=True)
    for kind in ('dispatch', 'smoke', 'fit', 'score'):
        command = commands.add_parser(kind)
        command.add_argument('runtime', type=Path)
        command.add_argument('job', nargs='?')
    options = vars(parser.parse_args())
    kind = options.pop('kind')
    if kind == 'publish':
        publish(**options)
    else:
        runtime = read(options['runtime'])
        if kind == 'dispatch':
            dispatch(runtime)
        else:
            activate(runtime)
            if kind == 'score':
                score(runtime, options['job'])
            else:
                fit(runtime, *options['job'].split('-'), smoke=kind == 'smoke')
