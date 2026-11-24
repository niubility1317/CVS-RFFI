"""Evaluate the source-frozen LoRa fixed midpoint on both blind practical routes."""
import argparse
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from .channel_blind_source import setup
from .complex_vector_source import load_reader
from .data import TARGET, VIEWS
from .dura_repair import PacketRestorer, complex_iq, real_iq
from .fission_risk_source import MODES as SOURCE_MODES, RUN as SOURCE_RUN, bind, readouts
from .full_dura_detail import COMPLEX_RESPONSE_RUN
from .full_packet_data import LIMIT, PROTOCOL
from .full_packet_publish import DATA_RUN
from .full_packet_views import source_contract
from .packet_runner import schedule
from .publish import release, release_commit
from .refine_runner import start
from .residual_dura import RUN as RESIDUAL_RUN, SEED
from .residual_dura_gate_run import RUN as RESIDUAL_CONTROL
from .rf_noise_transport import colored_observation
from .runner import activate
from .spectral_unroll import ROUTES


RUN = '20261124-phase1-lora-fixed-mid-target-m2-r01'
MODE = 'fixed_mid'
CONTROL_MODES = {'full': ('direct', 'coherent_half', 'complex_response_half'),
                 'residual': ('direct', 'dura_gated_half')}
BATCH = 18


def freeze_metadata(route, modes=(MODE,)):
    readouts = {'mode': MODE} if modes == (MODE,) else {'modes': list(modes)}
    return dict(status='PREDICTIONS_FROZEN', route=route, seed=SEED, **readouts,
                receivers=list(TARGET), per_receiver=LIMIT, views=list(VIEWS), target_truth_read=False)


def control_metadata(path, route):
    doc = read(path / 'complete.json')
    expected = freeze_metadata(route)
    keys = ('status', 'seed', 'receivers', 'per_receiver', 'views', 'target_truth_read')
    if (any(doc.get(k) != expected[k] for k in keys)
            or not set(CONTROL_MODES[route]).issubset(doc.get('modes', []))
            or (route == 'residual' and doc.get('practical', {}).get('processing_route') != 'residual')
            or (route == 'full' and 'practical' in doc)):
        raise ValueError(f'Wrong frozen Tri-set historical controls for {route}')


def output_path(path, rx, view, mode):
    return path / f'{rx}__{view}__{mode}.npy'


def frozen_prediction(path):
    values = np.load(path, allow_pickle=False)
    if values.shape != (LIMIT,) or values.dtype.kind not in 'iu' or (values < 0).any() or (values >= 10).any():
        raise ValueError(f'Invalid frozen LoRa predictions at {path.name}')
    return values


def blind_views(rt, route):
    target = Path(rt['target_views'][route])
    expected = dict(status='TARGET_READY', protocol=PROTOCOL, target_truth_read=False,
                    receivers=list(TARGET), per_receiver=LIMIT, views=list(VIEWS))
    if route == 'residual':
        expected['processing_route'] = 'residual'
    if read(target/'complete.json') != expected:
        raise ValueError(f'Expected registered blind {route} views')
    return target


def frozen_runtime(rt, contract, device):
    root = Path(rt['run_root'])
    if read(root.parent/SOURCE_RUN/'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('The nominated source readout must already be frozen and scored')
    models, threshold = bind(root, contract, device)
    tracker = PacketRestorer('consensus').to(device).eval()
    reader = load_reader(root, contract, device)
    return tracker, models, threshold, reader


def fixed_readout(iq, tracker, models, threshold):
    waves, active = readouts(iq, tracker, models, threshold)
    return waves[SOURCE_MODES.index(MODE) - 2], active


def midpoint_readouts(iq, tracker, models, threshold):
    wave, active = fixed_readout(iq, tracker, models, threshold)
    return (wave,), active


def update_predictions(destination, changed, scores):
    mask = np.asarray(changed.cpu().tolist(), dtype=bool)
    destination[mask] = scores.argmax(1).cpu().tolist()


@torch.no_grad()
def smoke(rt):
    root, device = Path(rt['run_root']), 'cuda:0'
    (_, cal, _), contract = setup(rt, device, smoke=True)
    tracker, models, threshold, reader = frozen_runtime(rt, contract, device)
    rng = torch.Generator(device=device).manual_seed(2026113627)
    corrupt = real_iq(colored_observation(complex_iq(cal), rng))
    iq = torch.cat((cal[:2], corrupt))
    wave, active = fixed_readout(iq, tracker, models, threshold)
    scores = reader(wave)[0]
    if (not active or not torch.equal(wave[:2], iq[:2]) or torch.equal(wave[2:], iq[2:])
            or not wave.isfinite().all() or not scores.isfinite().all() or scores.shape != (len(iq),10)):
        raise ValueError('Frozen midpoint failed real held source smoke')
    write(root/'smoke.json', dict(status='PASS', source_run=SOURCE_RUN, mode=MODE,
                                  active=active, target_reads=0))
    print('LORA_FIXED_MID_SOURCE_SMOKE_PASS', flush=True)


def source_smoke_metadata(source_run, modes):
    return dict(status='PASS', source_run=source_run, target_reads=0,
                **({'mode': MODE} if modes == (MODE,) else {'modes': list(modes)}))


@torch.no_grad()
def predict(rt, route, *, modes=(MODE,), source_run=SOURCE_RUN,
            prepare=frozen_runtime, evaluate=midpoint_readouts):
    root, device = Path(rt['run_root']), 'cuda:0'
    check = read(root/'smoke.json')
    if any(check.get(k) != v for k,v in source_smoke_metadata(source_run, modes).items()):
        raise ValueError('Actual source-only smoke required before target access')
    contract = source_contract(Path(rt['source_root']))
    tracker, models, threshold, reader = prepare(rt, contract, device)
    target = blind_views(rt, route)
    control = Path(rt['controls'][route])
    control_metadata(control, route)
    dest = root/'predictions'/route
    dest.mkdir(parents=True)
    activity = {}
    for rx in TARGET:
        iq = np.load(target/f'{rx}.npy', allow_pickle=False, mmap_mode='r')
        if iq.shape != (len(VIEWS), LIMIT, 6, 2, 256) or iq.dtype != np.float32:
            raise ValueError(f'Invalid blind LoRa IQ for {route}/{rx}')
        activity[rx] = {}
        for j, view in enumerate(VIEWS):
            direct = frozen_prediction(output_path(control, rx, view, 'direct')).astype(np.int16)
            rows = [direct.copy() for _ in modes]
            active_count = 0
            for i in range(0, LIMIT, BATCH):
                original = tensor(iq[j, i:i+BATCH], device)
                waves, active = evaluate(original, tracker, models, threshold)
                active_count += active
                if active:
                    for mode, pred, wave in zip(modes, rows, waves, strict=True):
                        changed = (wave != original).flatten(1).any(-1)
                        if changed.any():
                            scores = reader(wave[changed])[0]
                            if not scores.isfinite().all():
                                raise ValueError(f'Nonfinite logits for {mode}/{route}/{rx}/{view}')
                            update_predictions(pred[i:i+len(wave)], changed, scores)
            activity[rx][view] = active_count
            for mode, pred in zip(modes, rows, strict=True):
                path = output_path(dest, rx, view, mode)
                with path.open('xb') as stream:
                    np.save(stream, pred)
        print('LORA_TARGET_PREDICTED', route, rx, flush=True)
    write(dest/'activity.json', activity)
    write(dest/'complete.json', freeze_metadata(route, modes))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir()
    state = root/'state.json'
    execute = partial(start, module=rt['runner'], extra_keys=('controls',))
    try:
        write(state, dict(status='SOURCE_SMOKE'))
        schedule(rt, [('smoke','source')], execute)
        write(state, dict(status='PREDICTING'), exclusive=False)
        schedule(rt, [('predict',route) for route in ROUTES], execute)
        write(state, dict(status='PREDICTIONS_FROZEN', target_truth_read=False), exclusive=False)
    except Exception as error:
        write(state, dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config, source_submit, *, run=RUN, source_run=SOURCE_RUN, runner=None):
    cfg = read(context)
    runtime = {key: cfg[key] for key in ('project','python','frozen_sources')}
    project = PurePosixPath(runtime['project'])
    runs, releases = project/'runs', project/'releases'
    source = read(source_submit)
    if (source['archive_transfer'] != 'VERIFIED' or source['remote_compile'] != 'PASS'
            or PurePosixPath(source['run_root']) != runs/source_run):
        raise ValueError('Expected the exact completed source experiment')
    runtime.update(run_id=run, runner=runner or __spec__.name, code_commit=release_commit(repo),
        run_root=str(runs/run), release_root=str(releases/f'{run}-release'),
        source_root=str(releases/f'{DATA_RUN}-release'/'data'/'source'), parent_root=str(runs/DATA_RUN),
        target_views={'full':str(runs/DATA_RUN/'target_views'), 'residual':str(runs/RESIDUAL_RUN/'target_views')},
        controls={'full':str(runs/COMPLEX_RESPONSE_RUN/'predictions'/str(SEED)),
                  'residual':str(runs/RESIDUAL_CONTROL/'predictions')},
        external_releases=[str(runs/source_run/'state.json')],
        allowed_gpus=[3,6,2,0,4,1,5], max_workers=2, max_compute_processes=4)
    release(repo, output, runtime, ssh_config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    launch = actions.add_parser('publish')
    for key in ('repo','output','context','ssh-config','source-submit'):
        launch.add_argument('--'+key, type=Path, required=True)
    for action in ('dispatch','smoke','predict'):
        worker = actions.add_parser(action)
        worker.add_argument('runtime', type=Path)
        if action != 'dispatch':
            worker.add_argument('route', choices=ROUTES if action == 'predict' else ('source',))
    opts = vars(parser.parse_args())
    action = opts.pop('action')
    if action == 'publish':
        return publish(**opts)
    rt = read(opts['runtime'])
    if action == 'dispatch':
        return dispatch(rt)
    activate(rt)
    if action == 'smoke':
        return smoke(rt)
    return predict(rt, opts['route'])


if __name__ == '__main__':
    main()
