"""Equal-dose, two-route LoRa test of source-estimated repeatable details."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write

from . import bypass_utility_source as previous
from .bypass_utility import MAX_DOSE, Utility
from .data import SOURCE, VIEWS
from .dura_repair import PacketRestorer, complex_iq
from .full_adaptive import SET_METHOD, SET_RUN, case, complete_f
from .full_dura_observed_shape import ObservedShape
from .full_packet_publish import DATA_RUN
from .full_packet_views import source_role
from .full_restore_draw import RUN as DRAW_RUN, load_draw
from .full_supervised import construct as recognizer, handles
from .packet_physics import chirp_windows
from .packet_runner import schedule
from .publish import release, release_commit
from .r17_anchored_source import validation_indices as anchor_panel
from .refine_runner import start
from .repeatable_detail import basis_spec, estimate_basis, matched_shape, pack_spectrum, project_corrections
from .residual_dura import L_COUNT, RUN as RESIDUAL_RUN, SEED, ResidualView
from .residual_dura_calibration import source_model, source_state
from .residual_dura_gate import load as load_gate
from .residual_dura_gate_run import RUN as GATE_RUN
from .runner import activate
from .spectral_unroll import ROUTES, bind, observed


RUN = '20261123-phase1-lora-dura-repeatable-detail-source-m2-r01'
RANK = 96
FIT_PER_RX = 900
EVAL_PER_RX = 200
KEY = SEED + 980
MODES = ('direct', 'reference', 'extra_r17', 'stable_equal_dose', 'complement_equal_dose', 'stable_shrink')


def fit_arrays(rt, route):
    parent, pairs, contract, residual = source_state(rt)
    if route == 'residual':
        view = ResidualView(pairs)
        if not view.ready():
            raise ValueError('Completed residual source paired arrays required')
        return residual, (view.clean[L_COUNT:], view.received[0, L_COUNT:], view.received[1, L_COUNT:])
    _, cache, bound = handles(dict(rt, parent_root=str(parent.parent / DATA_RUN)))
    if bound != contract:
        raise ValueError('Full practical source contract differs')
    return contract, (np.load(cache / 'U_s.npy', mmap_mode='r', allow_pickle=False)[0],
                      np.load(cache / 'U_s_practical.npy', mmap_mode='r', allow_pickle=False),
                      load_draw(parent.parent / DRAW_RUN, 'U_s'))


def binding(rt, route):
    _, _, base, residual = source_state(rt)
    proof = dict(status='SOURCE_FROZEN', route=route, fit_role='U_s', rank=RANK,
                 fit_per_rx=FIT_PER_RX, seed=KEY, **previous.ISOLATION)
    return (base if route == 'full' else residual), proof


@torch.no_grad()
def fit(rt, route, smoke=False):
    device = 'cuda:0'
    contract, arrays = fit_arrays(rt, route)
    expected, proof = binding(rt, route)
    if contract != expected:
        raise ValueError('Cannot fit the subspace across unmatched contracts')
    _, rx, _, _ = source_role(Path(rt['source_root']), contract, 'U_s')
    if any(a.shape != (len(rx), 6, 2, 256) or a.dtype != np.float32 for a in arrays):
        raise ValueError('Invalid source paired observation arrays')
    count = 24 if smoke else FIT_PER_RX
    rng = np.random.default_rng(KEY)
    groups = [rng.choice(np.flatnonzero(rx == name), count, replace=False) for name in SOURCE]
    tracker = PacketRestorer('consensus').to(device).eval()
    _, mask = basis_spec(device)
    data = []
    for number, array in enumerate(arrays):
        rows = []
        for group in groups:
            batches = [pack_spectrum(matched_shape(tensor(array[ids], device), tracker), mask).cpu()
                       for ids in np.array_split(group, 1 + len(group) // 96)]
            batch = torch.cat(batches)
            # Remove source receiver means from the clean covariance only.
            # The paired draw difference estimates nuisance with the same RX.
            rows.append(batch - batch.mean(0) if number == 0 else batch)
        data.append(torch.cat(rows))
        print('REPEATABLE_FEATURES', route, number, len(data[-1]), flush=True)
    basis, values = estimate_basis(*data, rank=RANK)
    if not torch.isfinite(basis).all() or not torch.isfinite(values).all():
        raise ValueError('Source generalized noise decomposition is nonfinite')
    if not torch.allclose(basis.T @ basis, torch.eye(RANK), atol=5e-5, rtol=5e-5):
        raise ValueError('Source serialization requires an orthonormal physical basis')
    root = Path(rt['run_root'])
    record = dict(proof, eigenvalues_at_ranks=[float(values[i]) for i in (0, 15, 31, 63, 95, 127, 255, 511)])
    if smoke:
        propose, tracking = route_predictor(rt, route, device)
        observations = tensor(arrays[1][groups[0][:12]], device)
        baseline, extra, active = propose(observations)
        variants, drift = project_corrections(baseline, extra, tracking, basis.to(device))
        if not all(torch.isfinite(wave).all() for wave in (baseline, extra, *variants)) or drift > 2e-6:
            raise ValueError('Real frozen checkpoint or equal-dose path failed')
        if not all(torch.equal(wave[~active], baseline[~active]) for wave in (extra, *variants)):
            raise ValueError('No-query smoke did not preserve inactive packets')
        write(root / f'smoke-{route}.json', dict(record, status='PASS', fit_per_rx=count,
              checkpoint_packets=len(active), active_packets=int(active.sum()), dose_drift=drift))
    else:
        torch.save(dict(proof=proof, contract=contract, basis=basis), root / f'{route}.pt')
        write(root / f'{route}-complete.json', record)


def load_basis(rt, route, device):
    root = Path(rt['run_root'])
    contract, proof = binding(rt, route)
    item = torch.load(root / f'{route}.pt', map_location=device, weights_only=True)
    if item['contract'] != contract or item['proof'] != proof:
        raise ValueError('Frozen source-only matched basis required')
    return item['basis']


def validate_panel(rx):
    excluded = np.union1d(anchor_panel(rx), previous.validation_indices(rx))
    rng = np.random.default_rng(KEY + 1)
    return np.concatenate([np.sort(rng.choice(np.setdiff1d(np.flatnonzero(rx == name), excluded),
                                             EVAL_PER_RX, replace=False)) for name in SOURCE])


def residual_readouts(y, gate, r17, r69, projector):
    reference, active = gate.restore(y, r69, projector)
    proposal = reference.clone()
    if active.any():
        selected = y[active]
        raw = selected.flatten(0, 1)
        projected = projector.restore(raw, r17(raw)).reshape_as(selected)
        r17_half = selected + .5 * (projected - selected)
        # Keep the original source-gated R69 readout and extend it through an
        # R17 proposal computed from the original received coordinates.
        diff = r17_half - reference[active]
        rms = complex_iq(reference[active]).abs().square().mean((1, 2)).clamp_min(1e-8).sqrt()
        size = complex_iq(diff).abs().square().mean((1, 2)).clamp_min(1e-8).sqrt()
        proposal[active] = reference[active] + diff * (MAX_DOSE * rms / size).clamp_max(1.)[:, None, None, None]
    return reference, proposal, active


def full_reader(rt, contract, device):
    _, _, _, models = previous.contract_and_models(rt, device)
    source = Path(rt['run_root']).parent / previous.RUN
    stored = torch.load(source / 'model.pt', map_location=device, weights_only=True)
    proof = dict(status='SOURCE_FROZEN', seed=SEED, epochs=previous.EPOCHS, train_packets=previous.TRAIN,
                 calibration_packets=contract['counts']['U_s'] - previous.TRAIN, **previous.ISOLATION)
    if stored['contract'] != contract or stored['receipt'] != proof or read(source / 'complete.json') != proof:
        raise ValueError('Require the frozen full practical disturbance detector')
    detector = Utility().to(device).eval().requires_grad_(False)
    detector.load_state_dict(stored['utility'], strict=True)

    def proposals(y):
        values, eligible, features, _ = models.candidates(y)
        active = eligible[:, 1] & (detector.detection(features) > detector.boundary)
        return values[0], torch.where(active[:, None, None, None], values[2], values[0]), active

    return proposals, models.projector.tracker


def route_predictor(rt, route, device):
    parent, _, contract, paired = source_state(rt)
    if route == 'full':
        return full_reader(rt, contract, device)
    r17, _, _ = bind(rt, 'residual', device)
    r69 = source_model(parent, 'r69', paired, device)
    projector = ObservedShape().to(device).eval()
    gate = load_gate(parent.parent / GATE_RUN / 'gate.json', contract, device)
    return partial(residual_readouts, gate=gate, r17=r17, r69=r69, projector=projector), projector.tracker


@torch.no_grad()
def score(rt, route):
    device = 'cuda:0'
    parent, _, contract, _ = source_state(rt)
    root = Path(rt['run_root'])
    basis = load_basis(rt, route, device)
    proposals, tracker = route_predictor(rt, route, device)
    if route == 'full':
        _, cache, _ = handles(dict(rt, parent_root=str(parent.parent / DATA_RUN)))
        archive = np.load(cache / 'V.npy', mmap_mode='r', allow_pickle=False)
    else:
        archive = None
    source = Path(rt['source_root'])
    iq, rx, ids, _ = source_role(source, contract, 'V')
    if archive is not None and (archive.shape != (len(VIEWS), len(ids), 6, 2, 256) or archive.dtype != np.float32):
        raise ValueError('Unexpected full practical source V cache')
    chosen = validate_panel(rx)
    rows = list(zip(iq[chosen], rx[chosen], ids[chosen], strict=True))
    f = recognizer(device).eval().requires_grad_(False)
    weights = complete_f(parent.parent / SET_RUN, contract, case(SEED, SET_METHOD)) / 'model.pt'
    f.load_state_dict(torch.load(weights, map_location=device, weights_only=True), strict=True)
    predictions, metrics = {}, {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for i, scene in enumerate(VIEWS):
            if archive is not None:
                obs = archive[i, chosen]
            elif scene == 'clean':
                obs = np.stack([chirp_windows(row[0]) for row in rows])
            else:
                obs = np.stack(list(pool.map(partial(observed, scene, route), rows, chunksize=16)))
            chunks, coverage, max_drift = [], 0, 0.
            for y in tensor(obs, device).split(24):
                baseline, extra, active = proposals(y)
                filtered = [baseline.clone() for _ in range(3)]
                if active.any():
                    variants, drift = project_corrections(baseline[active], extra[active], tracker, basis)
                    for output, candidate in zip(filtered, variants, strict=True):
                        output[active] = candidate
                    max_drift = max(max_drift, drift)
                wave = torch.stack((y, baseline, extra, *filtered))
                logits = f(wave.flatten(0, 1))[0]
                if max_drift > 2e-6 or not torch.isfinite(logits).all():
                    raise ValueError('Invalid frozen scores or mismatched frequency-band doses')
                coverage += int(active.sum())
                chunks.append(logits.argmax(1).reshape(len(MODES), len(y)).cpu().tolist())
            predictions[scene] = np.concatenate(chunks, axis=1)
            metrics[scene] = dict(active=coverage, maximum_norm_difference=max_drift)
            print('REPEATABLE_SOURCE_PREDICTED', route, scene, coverage, flush=True)
    write(root / f'{route}-source-predictions.json', dict(ids=ids[chosen].tolist(),
          views={v: y.tolist() for v, y in predictions.items()}))
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as m:
        truth = m['y'][chosen]
    hits = {v: y == truth for v, y in predictions.items()}
    for v, hit in hits.items():
        metrics[v].update(correct=hit.sum(-1).tolist(),
                          by_rx=hit.reshape(len(MODES), len(SOURCE), EVAL_PER_RX).sum(-1).tolist())
    six = np.stack([hits[v] for v in VIEWS[1:]]).sum(0)
    low = hits['practical_low_urban'].astype(int)
    rng = np.random.default_rng(KEY + 2)
    draws = np.concatenate([rng.integers(i * EVAL_PER_RX, (i + 1) * EVAL_PER_RX, (1600, EVAL_PER_RX))
                            for i in range(len(SOURCE))], axis=1)
    intervals = {name: [np.quantile(delta[draws].sum(1), (.025, .975)).tolist() for delta in v - v[1]]
                 for name, v in (('six', six), ('low_urban', low))}
    write(root / f'{route}-source-scores.json', dict(status='SOURCE_SCORED', route=route, n=len(chosen),
          modes=list(MODES), metrics=metrics, six=six.sum(1).tolist(), paired_ci_vs_reference=intervals,
          target_reads=0, truth_after_predictions=True))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    launch = partial(start, module=rt['runner'])
    try:
        for action in ('smoke', 'fit', 'score'):
            write(root / 'state.json', dict(status=action.upper()), exclusive=False)
            schedule(rt, [(action, route) for route in ROUTES], launch)
        write(root / 'state.json', dict(status='SOURCE_SCORED'), exclusive=False)
    except Exception as error:
        write(root / 'state.json', dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config, parent_submit):
    cfg = read(context)
    rt = {k: cfg[k] for k in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    parent = project / 'runs' / RESIDUAL_RUN
    receipt = read(parent_submit)
    if receipt['archive_transfer'] != 'VERIFIED' or PurePosixPath(receipt['run_root']) != parent:
        raise ValueError('Registered LoRa parent is required')
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              run_root=str(project / 'runs' / RUN), release_root=str(project / 'releases' / f'{RUN}-release'),
              source_root=str(project / 'releases' / f'{DATA_RUN}-release' / 'data' / 'source'),
              parent_root=str(parent), allowed_gpus=[3, 6, 2, 0, 4, 1, 5], max_workers=2, max_compute_processes=4)
    release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='kind', required=True)
    pub = commands.add_parser('publish')
    for option in ('repo', 'output', 'context', 'ssh-config', 'parent-submit'):
        pub.add_argument('--' + option, type=Path, required=True)
    dispatch_parser = commands.add_parser('dispatch')
    dispatch_parser.add_argument('runtime', type=Path)
    for kind in ('smoke', 'fit', 'score'):
        command = commands.add_parser(kind)
        command.add_argument('runtime', type=Path)
        command.add_argument('route', choices=ROUTES)
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
                score(runtime, options['route'])
            else:
                fit(runtime, options['route'], smoke=kind == 'smoke')
