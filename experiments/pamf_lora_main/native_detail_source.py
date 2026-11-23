"""Full practical source trial of native adjacent-band LoRa restoration."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write

from . import bypass_utility_source as previous
from . import direct_detail_source as previous_run
from .bypass_utility import Utility
from .data import SOURCE, VIEWS
from .full_adaptive import SET_METHOD, SET_RUN, case, complete_f
from .full_dura_detail import extra_correction, half_baseline, scale_detail
from .full_packet_publish import DATA_RUN
from .full_packet_views import source_role
from .full_restore_draw import DRAW, RUN as DRAW_RUN, load_draw
from .full_supervised import construct as recognizer
from .native_detail import NativeDetail, feature_pack, native_pair, restore, targets, total_risk
from .packet_runner import schedule
from .publish import release, release_commit
from .refine_runner import start
from .residual_dura import RUN as RESIDUAL_RUN, SEED
from .runner import activate


RUN = '20261123-phase1-lora-dura-native-band-source-m1-r01'
KEY = SEED + 1191
FIT_PER_RX = 200
HELD_PER_RX = 40
EVAL_PER_RX = 150
STEPS = 800
BATCH = 8
ARMS = ('baseband', 'native')
MODES = ('direct', 'reference', 'baseband_half', 'native_half', 'native_full',
         'paired_noise', 'paired_shape', 'paired_joint')


def bind(rt, device):
    source, cache, contract, models = previous.contract_and_models(rt, device)
    old = Path(rt['run_root']).parent / previous.RUN
    record = dict(status='SOURCE_FROZEN', seed=SEED, epochs=previous.EPOCHS, train_packets=previous.TRAIN,
                  calibration_packets=contract['counts']['U_s'] - previous.TRAIN, **previous.ISOLATION)
    saved = torch.load(old / 'model.pt', map_location=device, weights_only=True)
    if saved['receipt'] != record or saved['contract'] != contract or read(old / 'complete.json') != record:
        raise ValueError('Require the frozen source-only full practical detector')
    detector = Utility().to(device).eval().requires_grad_(False)
    detector.load_state_dict(saved['utility'], strict=True)

    @torch.no_grad()
    def baseline(y):
        ref, _ = half_baseline(models.base, models.prior, y)
        selected, restored = extra_correction(models.response, ref, torch.zeros(len(y), dtype=torch.bool, device=device))
        ref[selected] = scale_detail(ref[selected], restored, .5)
        evidence, _ = models.features(y)
        return ref, detector.detection(evidence) > detector.boundary

    return source, cache, contract, baseline, models.projector.tracker


def source_record(contract):
    return dict(status='SOURCE_FROZEN', contract=contract, seed=KEY, steps=STEPS,
                fit_per_rx=FIT_PER_RX, held_per_rx=HELD_PER_RX, arms=list(ARMS), **previous.ISOLATION)


def assemble(pool, jobs, expected):
    rows = zip(pool.map(native_pair, jobs, chunksize=6), expected, strict=True)
    side, noise = [], []
    for (observed, adjacent, truth), before in rows:
        if not np.array_equal(observed, before) or not (np.isfinite(adjacent).all() and np.isfinite(truth).all()):
            raise ValueError('1 MHz replay differs from the registered 256 kHz observation')
        side.append(adjacent)
        noise.append(truth)
    return np.stack(side), np.stack(noise)


@torch.no_grad()
def fit_data(rt, devices, count, held):
    source, cache, contract, baseline, tracker = devices
    u, rx, ids, slots = source_role(source, contract, 'U_s')
    rng = np.random.default_rng(KEY)
    groups = [rng.permutation(np.flatnonzero(rx == name))[:count + held] for name in SOURCE]
    clean = np.load(cache / 'U_s.npy', mmap_mode='r', allow_pickle=False)[0]
    sources = (np.load(cache / 'U_s_practical.npy', mmap_mode='r', allow_pickle=False),
               load_draw(Path(rt['parent_root']).parent / DRAW_RUN, 'U_s'))
    result = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for row, selection in enumerate((np.concatenate([g[:count] for g in groups]), np.concatenate([g[count:] for g in groups]))):
            features, components = [], ([], [])
            for realized, observed in zip((None, DRAW), sources, strict=True):
                for idx in np.array_split(selection, 1 + len(selection) // 48):
                    jobs = [(u[i], rx[i], ids[i], VIEWS[1 + int(slots[i]) % 6], 'source-U', realized) for i in idx]
                    y0 = observed[idx]
                    adj, noise = assemble(pool, jobs, y0)
                    y = tensor(y0, 'cuda:0')
                    ref, active = baseline(y)
                    if active.any():
                        x, info = feature_pack(y[active], ref[active], tensor(adj, 'cuda:0')[active], tracker)
                        target = targets(tensor(clean[idx], 'cuda:0')[active], tensor(noise, 'cuda:0')[active], info, tracker)
                        features.append(x)
                        for aggregate, part in zip(components, target, strict=True):
                            aggregate.append(part)
                print('NATIVE_DETAIL_PAIRS', row, realized, 0 if not features else sum(len(x) for x in features), flush=True)
            if not features:
                raise ValueError('No eligible source IQ for this fixed cohort')
            result.append((torch.cat(features), tuple(torch.cat(part) for part in components)))
    return result


def inputs(features, arm):
    if arm == 'native':
        return features
    return torch.cat((features[:, :, :4], torch.zeros_like(features[:, :, 4:])), dim=2)


def fit(rt, smoke=False):
    torch.manual_seed(KEY)
    center = bind(rt, 'cuda:0')
    count, held = (5, 3) if smoke else (FIT_PER_RX, HELD_PER_RX)
    (x, expected), (xh, expected_h) = fit_data(rt, center, count, held)
    heads = {k: NativeDetail().cuda() for k in ARMS}
    heads['native'].load_state_dict(heads['baseband'].state_dict())
    optimizer = torch.optim.AdamW([v for model in heads.values() for v in model.parameters()],
                                  lr=2e-4, weight_decay=1e-4)
    rng = torch.Generator(device='cuda:0').manual_seed(KEY)
    steps = 1 if smoke else STEPS
    for step in range(1, steps + 1):
        sample = torch.randint(len(x), (BATCH,), generator=rng, device='cuda:0')
        losses = torch.stack([total_risk(model(inputs(x[sample], k)), tuple(y[sample] for y in expected))
                              for k, model in heads.items()])
        if not torch.isfinite(losses).all():
            raise FloatingPointError('Nonfinite supervised source physical residual')
        optimizer.zero_grad(set_to_none=True)
        losses.sum().backward()
        for model in heads.values():
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if step == 1 and norm <= 0:
                raise ValueError('Both physical-component heads need source gradient')
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print('NATIVE_DETAIL_FIT', step, *losses.detach().tolist(), flush=True)
    with torch.no_grad():
        errors = {}
        for k, model in heads.items():
            model.eval()
            rows = zip(xh.split(24), *(y.split(24) for y in expected_h), strict=True)
            errors[k] = sum(len(a)*float(total_risk(model(inputs(a, k)), (b, c))) for a, b, c in rows) / len(xh)
    root = Path(rt['run_root'])
    record = source_record(center[2])
    report = dict(record, fit_exposures=len(x), held_exposures=len(xh), held_error=errors)
    if smoke:
        write(root / 'smoke.json', dict(report, status='PASS', steps=steps, fit_per_rx=count, held_per_rx=held))
    else:
        torch.save(dict(proof=record, heads={k: model.state_dict() for k, model in heads.items()}), root / 'model.pt')
        write(root / 'complete.json', report)


def evaluation_panel(rx):
    prior = previous_run.prior
    excluded = np.unique(np.concatenate((previous_run.panel(rx), prior.validate_panel(rx),
                                         prior.anchor_panel(rx), previous.validation_indices(rx))))
    rng = np.random.default_rng(KEY + 11)
    return np.concatenate([np.sort(rng.choice(np.setdiff1d(np.flatnonzero(rx == name), excluded),
                                             EVAL_PER_RX, replace=False)) for name in SOURCE])


def same_dose(component, total):
    power = lambda x: x.abs().square().mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    return component * (power(total) / power(component))


@torch.no_grad()
def score(rt):
    device = 'cuda:0'
    root = Path(rt['run_root'])
    source, cache, contract, baseline, tracker = bind(rt, device)
    stored = torch.load(root / 'model.pt', map_location=device, weights_only=True)
    if stored['proof'] != source_record(contract) or set(stored['heads']) != set(ARMS):
        raise ValueError('Invalid frozen physical-component heads')
    heads = {k: NativeDetail().cuda().eval().requires_grad_(False) for k in ARMS}
    for k, model in heads.items():
        model.load_state_dict(stored['heads'][k], strict=True)
    f = recognizer(device).eval().requires_grad_(False)
    weights = complete_f(root.parent / SET_RUN, contract, case(SEED, SET_METHOD)) / 'model.pt'
    f.load_state_dict(torch.load(weights, map_location=device, weights_only=True), strict=True)
    raw, rx, ids, _ = source_role(source, contract, 'V')
    archive = np.load(cache / 'V.npy', mmap_mode='r', allow_pickle=False)
    idx = evaluation_panel(rx)
    preds, metrics = {}, {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for vi, scene in enumerate(VIEWS):
            count, parts = 0, []
            for group in np.array_split(idx, 1 + len(idx) // 32):
                y0 = archive[vi, group]
                jobs = [(raw[i], rx[i], ids[i], scene, 'source-V', None) for i in group]
                adjacent, noise = assemble(pool, jobs, y0)
                y = tensor(y0, device)
                ref, active = baseline(y)
                count += int(active.sum())
                variants = [ref.clone() for _ in MODES[2:]]
                if active.any():
                    coords, info = feature_pack(y[active], ref[active], tensor(adjacent, device)[active], tracker)
                    raw_a, raw_b = (sum(heads[k](inputs(coords, k))) for k in ARMS)
                    n, s = targets(tensor(archive[0, group], device)[active], tensor(noise, device)[active], info, tracker)
                    for dest, delta in zip(variants, (raw_a*.5, raw_b*.5, raw_b,
                            same_dose(n, n+s), same_dose(s, n+s), n+s), strict=True):
                        dest[active] = restore(ref[active], delta, info)
                logits = f(torch.stack((y, ref, *variants)).flatten(0, 1))[0]
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite fixed source evaluation')
                parts.append(logits.argmax(1).reshape(len(MODES), len(y)).cpu().tolist())
            preds[scene] = np.concatenate(parts, axis=1)
            metrics[scene] = dict(active=count)
            print('NATIVE_DETAIL_PREDICTED', scene, count, flush=True)
    write(root / 'predictions.json', dict(ids=ids[idx].tolist(), rows={k: v.tolist() for k, v in preds.items()}))
    with np.load(source / 'V_metadata.npz', allow_pickle=False) as m:
        truth = m['y'][idx]
    hit = {k: v == truth for k, v in preds.items()}
    for k, value in hit.items():
        metrics[k].update(correct=value.sum(-1).tolist(), by_rx=value.reshape(len(MODES), len(SOURCE), EVAL_PER_RX).sum(-1).tolist())
    six = np.stack([hit[v] for v in VIEWS[1:]]).sum(0)
    low = hit['practical_low_urban'].astype(int)
    rng = np.random.default_rng(KEY + 12)
    draws = np.concatenate([rng.integers(i*EVAL_PER_RX, (i+1)*EVAL_PER_RX, (1600, EVAL_PER_RX))
                            for i in range(len(SOURCE))], axis=1)
    ci = {k: [np.quantile(delta[draws].sum(1), (.025, .975)).tolist() for delta in v-v[1]]
          for k, v in (('six', six), ('low', low))}
    write(root / 'source-scores.json', dict(status='SOURCE_SCORED', n=len(idx), modes=list(MODES),
          metrics=metrics, six=six.sum(1).tolist(), paired_ci_vs_reference=ci, target_reads=0,
          truth_after_predictions=True))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    try:
        for action in ('smoke', 'fit', 'score'):
            write(root / 'state.json', dict(status=action.upper()), exclusive=False)
            schedule(rt, [(action, 'full')], partial(start, module=rt['runner']))
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
        raise ValueError('Match the registered source release before fitting')
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              run_root=str(project / 'runs' / RUN), release_root=str(project / 'releases' / f'{RUN}-release'),
              source_root=str(project / 'releases' / f'{DATA_RUN}-release' / 'data' / 'source'),
              parent_root=str(parent), allowed_gpus=[3, 6, 2, 0, 4, 1, 5], max_workers=1, max_compute_processes=4)
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
        command.add_argument('name', nargs='?')
    options = vars(parser.parse_args())
    kind = options.pop('kind')
    if kind == 'publish':
        publish(**options)
    else:
        rt = read(options['runtime'])
        if kind == 'dispatch':
            dispatch(rt)
        else:
            activate(rt)
            if kind == 'score':
                score(rt)
            else:
                fit(rt, smoke=kind == 'smoke')
