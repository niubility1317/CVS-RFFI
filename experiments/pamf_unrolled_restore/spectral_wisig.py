"""Check one source-trained DURA model on both registered WiSig practical routes."""

import argparse
from functools import partial
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_fcrossrx_readout.frozen import load_reader
from experiments.pamf_fcrossrx_readout.source_diagnostic import VIEWS, source_reference
from experiments.pamf_lora_main.packet_runner import schedule
from experiments.pamf_lora_main.publish import release, release_commit
from experiments.pamf_lora_main.refine_runner import start
from experiments.pamf_lora_main.runner import activate
from experiments.pamf_r65_four_route.runtime import load_checkpoint as load_r69

from .design import KIND, RUN as R17_RUN
from .runtime import load_checkpoint
from .spectral import SpectralUnroll, paired_loss


RUN = '20261123-phase1-wisig-dura-spectral-unroll-source-m2-r01'
R02 = '20261019-phase1-reception-restoration-selfsup-manysig-m6-r02'
R07 = '20261019-phase1-restoration-learned-action-source-s2026092701-r07'
R69 = '20261021-phase1-restoration-R65-four-route-s2026092701-m4-r69'
ROUTES = ('full', 'residual')
ARMS = ('temporal', 'spectral')
MODES = ('direct', 'r69', 'r17', *ARMS)
SEED = 2026092701
STEPS = 600
BATCH = 64


def parent_model(rt):
    root = Path(rt['parent_root'])
    if root.name != R17_RUN:
        raise ValueError('Original same-contract WiSig R17 parent required')
    checkpoint = root / 'unroll_unshared' / KIND / 'final.pt'
    model, proof = load_checkpoint(checkpoint, 'cuda:0')
    if proof['candidate'] != 'unroll_unshared' or proof['epoch'] != 20 or proof['updates'] != 4940:
        raise ValueError('Fixed final WiSig R17 required')
    return model


def source_pairs(rt):
    cache = Path(rt['source_root'])
    receipt = read(cache / 'complete.json')
    if (cache.parent.name != R02 or receipt['paired_full_residual'] is not True
            or receipt['TX_labels_read'] or receipt['target_reads']):
        raise ValueError('Expected existing TX-blind dual-route WiSig source cache')
    ids = np.load(cache / 'train/physical_ids.npy', allow_pickle=False)
    r07 = cache.parent.parent / R07 / 'cache/train/physical_ids.npy'
    if ids.shape != (63000,) or not np.array_equal(ids, np.load(r07, allow_pickle=False)):
        raise ValueError('Training physical IDs do not match the R17 parent contract')
    roles = np.load(cache / 'train/roles.npy', allow_pickle=False)
    if ((roles == 'L_s').sum(), (roles == 'U_s').sum()) != (6300, 56700):
        raise ValueError('Only original 6300 L / 56700 U source records permitted')
    clean = np.load(cache / 'train/clean.npy', mmap_mode='r', allow_pickle=False)
    epochs = []
    for index in (1, 2):
        folder = cache / 'epochs' / f'{index:02d}'
        paired = [np.load(folder / f'{route}.npy', mmap_mode='r', allow_pickle=False) for route in ROUTES]
        quantile = np.load(folder / 'quantile.npy', mmap_mode='r', allow_pickle=False)
        if any(a.shape != clean.shape or a.dtype != np.float32 for a in paired):
            raise ValueError('Unexpected WiSig route shapes or dtype')
        if quantile.shape != (len(clean),):
            raise ValueError('Expected a clean identity assignment for each source row')
        epochs.append((*paired, quantile))
    return clean, epochs


def fit(rt, arm, smoke=False):
    if arm not in ARMS:
        raise ValueError('A registered WiSig continuation arm is required')
    torch.manual_seed(SEED)
    r17 = parent_model(rt)
    model = SpectralUnroll(r17).to('cuda:0') if arm == 'spectral' else r17
    model.train()
    clean, epochs = source_pairs(rt)
    rng = np.random.default_rng(SEED)
    count = 1 if smoke else STEPS
    optimizer = torch.optim.AdamW(model.prior.parameters(), lr=6e-5, weight_decay=1e-4)
    for step, index in enumerate(rng.permutation(len(clean))[:count * BATCH].reshape(count, BATCH), 1):
        full, residual, quantile = epochs[(step - 1) // (STEPS // 2)]
        # Alternate route assignment so both routes see the same sampling rule.
        first, second = (full, residual) if step % 2 else (residual, full)
        observed = np.concatenate((first[index[:BATCH // 2]], second[index[BATCH // 2:]]))
        target = clean[index]
        identity = quantile[index] < .15
        # The legacy cache leaves identity rows as zero placeholders.
        observed[identity] = target[identity]
        inputs, target = tensor(observed, 'cuda:0'), tensor(target, 'cuda:0')
        optimizer.zero_grad(set_to_none=True)
        stage = model.stage_outputs(inputs)[0]
        value = paired_loss(stage, target).mean()
        if not torch.isfinite(value):
            raise FloatingPointError('Nonfinite WiSig source loss')
        value.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.prior.parameters(), 1., error_if_nonfinite=True)
        if step == 1:
            if norm <= 0 or any(p.grad is not None for p in model.frozen.parameters()):
                raise ValueError('Wrong frozen-versus-trainable R17 update')
            if arm == 'spectral' and any(p.spectral.out.weight.grad.norm() <= 0 for p in model.prior):
                raise ValueError('All spectral stages must receive source gradient')
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print('WISIG_SPECTRAL_STEP', arm, step, float(value.detach()), float(norm), flush=True)
    root = Path(rt['run_root'])
    result = dict(status='PASS' if smoke else 'SOURCE_FROZEN', arm=arm, seed=SEED, steps=count,
                  source_cache=R02, parent=R17_RUN, routes=list(ROUTES),
                  TX_reads=0, F_reads=0, V_reads=0, target_reads=0)
    if not smoke:
        torch.save(dict(proof=result, prior=model.prior.state_dict()), root / f'{arm}.pt')
    write(root / ('smoke.json' if smoke else f'{arm}-complete.json'), result)


@torch.no_grad()
def score(rt):
    root, cache = Path(rt['run_root']), Path(rt['source_root'])
    parent = Path(rt['parent_root'])
    args = read(parent / 'launch.json')['argv']
    source_cache, reader_path = [Path(args[args.index(key) + 1]) for key in ('--source-cache', '--reader-folder')]
    v_root = {route: cache / ('validation_full' if route == 'full' else 'validation') for route in ROUTES}
    ids, source, pos = source_reference(v_root['residual'], source_cache)
    full_ids, _, _ = source_reference(v_root['full'], source_cache)
    if not np.array_equal(ids, full_ids):
        raise ValueError('Mismatched WiSig full and residual validation physical rows')
    f = load_reader(reader_path, source_cache, 'cuda:0').eval().requires_grad_(False)
    r69, proof = load_r69(parent.parent / R69 / 'stage2' / KIND / 'final.pt', 'cuda:0')
    if (proof['arm'], proof['epoch'], proof['updates'], proof['code_commit']) != (
            'stage2', 20, 4940, 'b2a2ce937a64654e17e84c2aa817b9da4257a48f'):
        raise ValueError('The original frozen DURA v1.0 readout is required')
    r17 = parent_model(rt)
    models = [r69, r17]
    for name in ARMS:
        saved = torch.load(root / f'{name}.pt', map_location='cuda:0', weights_only=True)
        proof = read(root / f'{name}-complete.json')
        if (saved['proof'] != proof or proof['status'] != 'SOURCE_FROZEN' or proof['arm'] != name
                or proof['steps'] != STEPS or proof['source_cache'] != R02 or proof['parent'] != R17_RUN
                or any(proof[k] for k in ('TX_reads', 'F_reads', 'V_reads', 'target_reads'))):
            raise ValueError('Source-only fixed-final WiSig candidates required')
        model = parent_model(rt)
        if name == 'spectral':
            model = SpectralUnroll(model).to('cuda:0')
        model.prior.load_state_dict(saved['prior'], strict=True)
        models.append(model.eval().requires_grad_(False))
    predictions = {}
    for route, folder in v_root.items():
        predictions[route] = {}
        for scene in VIEWS:
            batches = []
            view = np.load(folder / f'{scene}.npy', mmap_mode='r', allow_pickle=False)
            for start_at in range(0, len(ids), 32):
                y = tensor(view[start_at:start_at + 32], 'cuda:0')
                reads = [f(x) for x in (y, *(model(y) for model in models))]
                logits = torch.stack(reads)
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite frozen WiSig readout')
                batches.append(logits.argmax(-1).cpu().tolist())
            predictions[route][scene] = np.concatenate(batches, axis=1).tolist()
            print('WISIG_SPECTRAL_FIXED', route, scene, len(ids), flush=True)
    write(root / 'predictions.json', dict(ids=ids.tolist(), routes=predictions))
    labels = np.load(source_cache / 'V' / 'labels.npy', allow_pickle=False)[pos]
    scores = {}
    for route, views in predictions.items():
        scores[route] = {}
        for scene, outputs in views.items():
            hit = np.asarray(outputs) == labels
            scores[route][scene] = dict(correct=hit.sum(-1).tolist(), by_rx={
                str(rx): hit[:, source.rx[pos] == rx].sum(-1).tolist() for rx in np.unique(source.rx[pos])})
    write(root / 'source-scores.json', dict(status='SOURCE_SCORED', n=len(ids), modes=list(MODES),
          records=scores, frozen_before_source_truth=True, target_reads=0))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    launch = partial(start, module=rt['runner'])
    try:
        for action, names in (('smoke', ('spectral',)), ('fit', ARMS), ('score', ('dual',))):
            write(root / 'state.json', dict(status=action.upper()), exclusive=False)
            schedule(rt, [(action, name) for name in names], launch)
        write(root / 'state.json', dict(status='SOURCE_SCORED'), exclusive=False)
    except Exception as error:
        write(root / 'state.json', dict(status='FAILED', error=str(error)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    config = read(context)
    rt = {key: config[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              run_root=str(project / 'runs' / RUN), release_root=str(project / 'releases' / f'{RUN}-release'),
              source_root=str(project / 'runs' / R02 / 'cache'), parent_root=str(project / 'runs' / R17_RUN),
              allowed_gpus=[0, 1, 4, 5, 2, 6, 3], max_workers=2, max_compute_processes=4)
    release(repo, output, rt, ssh_config, archive_extras=('tools/r17_activation_factorization.py',))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='kind', required=True)
    pub = commands.add_parser('publish')
    for option in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--' + option, type=Path, required=True)
    for kind in ('dispatch', 'smoke', 'fit', 'score'):
        worker = commands.add_parser(kind)
        worker.add_argument('runtime', type=Path)
        worker.add_argument('name', nargs='?')
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
                fit(rt, options['name'], smoke=kind == 'smoke')
