"""Audited Phase1 Full+PL methods on the immutable LoRa main split."""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, write
from experiments.practical_full_2x2.plan import SCENES, SCENE_PROBABILITIES
from .comparison import predict
from .data import CLASSES, SOURCE, SPLIT_SEED, VIEWS
from .physical import rms_normalize
from .recognition import SEEDS
from .runner import available


RUN = '20261122-phase1-lora-full-external-m12-r02'
METHODS = ('riei', 'drift', 'ibrff', 'sdg')
BUDGETS = {'riei': (100, 64, .0001, 1), 'drift': (100, 256, .0001, 1),
           'ibrff': (200, 128, .001, 2), 'sdg': (200, 32, .001, 1)}


def configuration(rt, family, seed):
    epochs, batch, lr, stages = BUDGETS[family]
    row = f'{family}-s{seed}'
    return dict(run_id=RUN, row_id=row, code_commit=rt['code_commit'], protocol_id='lora_rx5_main_v1',
        family=family, variant='full', cell='D', initialization='scratch', training_regime='pl',
        model_seed=seed, split_seed=SPLIT_SEED, epochs=epochs, batch_size=batch, lr=lr,
        stages=stages, budget_kind='native', weight_decay=0., num_classes=len(CLASSES),
        classes=list(map(str, CLASSES)), num_domains=len(SOURCE), U_s_use=True,
        satellite_train_augmentation=True, lambda_sat_cls=.68, pl_threshold=.95, pl_weight=1.,
        accepted_pl_only=True, log_native_terms=True, drift_normalized_separation=family == 'drift',
        ibrff_plateau_warmup=25 if family == 'ibrff' else 0,
        riei_adam=family == 'riei',
        default_head_joint_training=True, separate_head_training=False,
        source_cache=rt['source_root'], source_contract=str(Path(rt['source_root'])/'contract.json'),
        source_practical=rt['source_practical'], output_root=str(Path(rt['run_root'])/'F'/row),
        ibrff_model_root=rt['ibrff_model_root'], sdg_model_root=rt['sdg_model_root'])


def check_training(c):
    if (c['run_id'] != RUN or c['family'] not in METHODS or c['model_seed'] not in SEEDS
            or any(k in c for k in ('target', 'target_root', 'target_practical', 'target_inputs',
                                    'views_root', 'tests', 'truth', 'target_truth', 'dataset',
                                    'resume', 'teacher_checkpoint', 'checkpoint', 'initial_checkpoint'))):
        raise ValueError('Supplement training accepts only the registered LoRa source rows')
    contract = read(c['source_contract'])
    if (contract['protocol'] != 'lora_rx5_main_v1' or contract['receiver_names'] != list(SOURCE)
            or contract['classes'] != c['classes'] or contract['split_seed'] != SPLIT_SEED
            or contract['counts'] != dict(L_s=2800, U_s=25200, V=12000)):
        raise ValueError('Same main-LoRa source roles required')
    with np.load(Path(c['source_cache'])/'U_s_metadata.npz', allow_pickle=False) as m:
        if ('y' in m or m['ids'].tolist() != contract['role_ids']['U_s']
                or set(m.files) != {'ids','rx_i','day_i','domain','eq_i','sig_i','base_index'}):
            raise ValueError('U labels or physical IDs differ')
    return dict(counts=contract['counts'], source_contract=contract)


def array_loader(c, role, batch_size=None):
    from experiments.practical_full_2x2.compat import tensor
    root = Path(c['source_cache'])
    x = rms_normalize(tensor(np.load(root/f'{role}_x.npy', allow_pickle=False)))
    with np.load(root/f'{role}_metadata.npz', allow_pickle=False) as m:
        rx, ids = m['rx_i'], m['ids']
        contract = read(c['source_contract'])
        if len(x) != len(rx) or ids.tolist() != contract['role_ids'][role]:
            raise ValueError('LoRa loader roles are not aligned')
        cols = (x, tensor(rx)) if role == 'U_s' else (x, tensor(m['y']), tensor(rx))
    if role == 'L_s':
        cols += (torch.arange(len(x)),)
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*cols), shuffle=role != 'V',
        batch_size=batch_size or c['batch_size'], drop_last=role == 'U_s', num_workers=0,
        generator=torch.Generator().manual_seed(c['model_seed'] + (991 if role == 'U_s' else 0)))


class CachedPractical:
    """Retain Phase1 concat CE dose and stage schedule on the main-LoRa IQ cache."""
    def __init__(self, c):
        self.reader = None
        self.indices = None
        self.epoch = None
        cache = np.load(c['source_practical'], mmap_mode='r', allow_pickle=False)
        self.views = cache[[VIEWS.index(scene)-1 for scene in SCENES]]
        self.seed = 2026102301

    def __call__(self, x, progress, batch_index, stage=1):
        from experiments.practical_full_2x2.compat import tensor
        if len(self.indices) != len(x):
            raise ValueError('Physical source L required for practical augmentation')
        p = .3 if progress < .2 else .6 if progress < .45 else .8
        rng = np.random.default_rng([self.seed, stage, batch_index])
        keep = rng.random(len(x)) < p
        scenes = rng.choice(len(SCENES), size=len(x), p=SCENE_PROBABILITIES)
        result = x.clone()
        if keep.any():
            mask = torch.tensor(keep.tolist(), dtype=torch.bool, device=x.device)
            indices = np.asarray(self.indices)[keep]
            result[mask] = tensor(self.views[scenes[keep], indices], x.device)
        counts = {i: int(((scenes==i) & keep).sum()) for i in range(len(SCENES))}
        counts[-1] = len(x)-int(keep.sum())
        return result, dict(applied_count=int(keep.sum()), scenario='three_practical',
                            probability=p, scene_counts=counts)


@contextmanager
def native_source():
    from experiments.practical_full_2x2 import source
    old = source.check_training, source.array_loader, source.BatchAugment
    source.check_training, source.array_loader, source.BatchAugment = check_training, array_loader, CachedPractical
    try:
        yield source
    finally:
        source.check_training, source.array_loader, source.BatchAugment = old


def smoke(rt):
    from experiments.practical_full_2x2.objectives import step
    out = Path(rt['run_root'])/'smoke'
    out.mkdir()
    with native_source() as source:
        for family in METHODS:
            c = configuration(rt, family, SEEDS[0])
            meta = check_training(c)
            source.native.seed(c['model_seed'])
            model = source.build(c)
            L = array_loader(c, 'L_s', 16)
            author = source.native.author_sdg(c) if family == 'sdg' else None
            if author:
                ds = source.SDGDataset(L, author)
                batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=16)))
                x,y,rx,views,indices = batch
            else:
                x,y,rx,indices = next(iter(L))
                views = None
            u,u_rx = next(iter(array_loader(c, 'U_s', 16)))
            fed = None
            if family == 'riei':
                opt, fed = source.riei_optimizers(model, c)
            else:
                opt = torch.optim.Adam(model.parameters(), lr=c['lr'])
            aug = CachedPractical(c)
            aug.indices = indices.tolist()
            result = step(model, c, x, y, rx, opt, aug, .5, 0, stage=c['stages'],
                          fed_optimizer=fed, unlabeled=u, native_views=views, author=author, unlabeled_receivers=u_rx)
            if result['pl_seen'] != len(u) or result['satellite_applied_samples'] < 1:
                raise ValueError('LoRa Full+PL smoke did not exercise intended branches')
            checkpoint = out/f'{family}.pt'
            source.save_checkpoint(checkpoint, model, c, meta['source_contract'], 1, 1, 0., 16, 16, 16, result['pl_accepted'])
            payload = source.native.load(checkpoint, c, meta['source_contract'], 'cpu')
            reader = source.build(c).eval()
            reader.load_state_dict(payload['state_dict'], strict=True)
            with torch.no_grad():
                logits = source.forward(reader, c, x)
            if logits.shape != (len(x), len(CLASSES)) or not torch.isfinite(logits).all():
                raise ValueError('Native LoRa checkpoint did not roundtrip')
            print('LORA_EXTERNAL_SOURCE_SMOKE_PASS', family, flush=True)
    write(out/'complete.json', dict(status='PASS', tested=list(METHODS), target_reads=0, U_truth_reads=0))


def worker(rt, kind):
    # The old author packages are resolved only inside worker processes.
    from experiments.practical_full_2x2.runner import activate as native
    native(rt)
    from experiments.practical_full_2x2 import source as native_source_module
    # Use the class-portable default head for validation and frozen readout.
    native_source_module.native.forward = native_source_module.forward
    if kind == 'smoke':
        return smoke(rt)
    c = configuration(rt, rt['family'], rt['seed'])
    with native_source() as source:
        if kind == 'train':
            return source.train(c)
        folder = Path(c['output_root'])
        contract_path = Path(c['source_contract'])
        model, frozen = source.frozen_model(folder, read(contract_path), 'cuda:0')
        if c != frozen:
            raise ValueError('Wrong frozen LoRa external run row')
        loader = lambda x: source.forward(model, c, x)
        return predict(folder, Path(rt['target_root']), Path(rt['target_practical']),
                       Path(rt['run_root'])/'predictions'/c['row_id'], contract_path,
                       Path(rt['main_run'])/'G'/f"dura-s{c['model_seed']}",
                       reader=loader, seed=c['model_seed'])


def launch(rt, kind, gpu, **fields):
    root = Path(rt['run_root'])
    row = f"{fields['family']}-s{fields['seed']}" if fields else 'all'
    keys = ('run_root','code_commit','frozen_sources','external_releases','ibrff_model_root','sdg_model_root',
            'source_root','source_practical')
    if kind == 'predict':
        keys += ('target_root','target_practical','main_run')
    config = root/'configs'/f'{kind}-{row}.json'
    write(config, {**{k:rt[k] for k in keys}, **fields})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    log = root/'logs'/f'{kind}-{row}.log'
    log.parent.mkdir(exist_ok=True)
    python = rt['python'] if kind == 'predict' else rt['external_python']
    with log.open('x') as stream:
        child = subprocess.Popen([python, '-u','-m',__spec__.name,kind,str(config)],
                                 env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
    print('LORA_PHASE1_LAUNCHED', kind, row, child.pid, gpu, flush=True)
    return child


def schedule(rt, rows):
    waiting, active = [('train', row) for row in rows], {}
    trained = predicted = 0
    state = Path(rt['run_root'])/'state.json'
    while waiting or active:
        for gpu, (child, kind, row) in list(active.items()):
            code = child.poll()
            if code is None:
                continue
            if code:
                raise RuntimeError(f'{kind} {row} failed with status {code}')
            print('LORA_PHASE1_COMPLETED', kind, row, flush=True)
            del active[gpu]
            if kind == 'train':
                trained += 1
                waiting.insert(0, ('predict', row))
            else:
                predicted += 1
        if waiting:
            for gpu in available(rt['allowed_gpus']):
                if not waiting:
                    break
                if gpu in active:
                    continue
                kind, row = waiting.pop(0)
                family, seed = row
                active[gpu] = launch(rt, kind, gpu, family=family, seed=seed), kind, row
        write(state, dict(status='RUNNING', source_frozen=trained, predictions_frozen=predicted,
            workers={str(gpu): dict(pid=child.pid, kind=kind, row_id=f'{row[0]}-s{row[1]}')
                     for gpu, (child, kind, row) in active.items()}), exclusive=False)
        if waiting or active:
            time.sleep(20)


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    state = root/'state.json'
    try:
        write(state, dict(status='SOURCE_SMOKE'))
        if launch(rt, 'smoke', None).wait():
            raise RuntimeError('Native Full baseline smoke failed')
        if (read(Path(rt['main_run'])/'state.json')['status'] != 'PREDICTIONS_FROZEN'
                or not (Path(rt['target_practical'])/'complete.json').is_file()):
            raise ValueError('The original LoRa practical views and predictions must be frozen')
        rows = [(f,s) for s in SEEDS for f in METHODS]
        schedule(rt, rows)
        write(state, dict(status='PREDICTIONS_FROZEN', target_truth_read=False), exclusive=False)
    except Exception as error:
        write(state, dict(status='FAILED', error=str(error)), exclusive=False)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['dispatch','smoke','train','predict'])
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    (dispatch if args.kind == 'dispatch' else lambda c: worker(c,args.kind))(read(args.config))
