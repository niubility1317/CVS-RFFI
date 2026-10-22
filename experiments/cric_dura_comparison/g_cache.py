"""Reuse source-only R02/R07 observations; generate only absent physical IDs."""
from pathlib import Path
import multiprocessing
import numpy as np
from .io import read, write

_SOURCE = None


def initialize(root, frozen_sources):
    from .bootstrap import activate
    activate(dict(frozen_sources=frozen_sources))
    global _SOURCE
    root = Path(root)
    _SOURCE = dict(clean=np.load(root / 'clean.npy', mmap_mode='r'),
        ids=np.load(root / 'physical_ids.npy'), rx=np.load(root / 'rx.npy'), day=np.load(root / 'day.npy'))


def generate(item):
    i, epoch = item
    from leo_practical.channel import Config, receiver_for_session, stable_seed
    from experiments.pamf_reception_restore.data import shared_pair
    from experiments.pamf_action_restore.data import teacher, clean_teacher
    from experiments.pamf_restoration.data import SCENES
    pid = str(_SOURCE['ids'][i])
    clean = _SOURCE['clean'][i]
    session = 'rx%d:day%d:eq1' % (_SOURCE['rx'][i], _SOURCE['day'][i])
    # Physical-ID keyed draws keep an observation identical when a source pool
    # is a subset or the same L/U union is repartitioned at another label rate.
    rng = np.random.default_rng(stable_seed(2026101902, 'cric-dura-source', epoch, pid))
    q = float(rng.random())
    scene = SCENES[int(rng.integers(len(SCENES)))]
    if q < .15:
        coarse = clean.copy()
    else:
        residual, full, _ = shared_pair(clean, scene, pid, session, 2026101901, epoch,
            2026101902, namespace='reception-restoration/train/epoch%02d' % epoch)
        coarse = residual if q < .75 else full
    rng = np.random.default_rng(stable_seed(2026101912, 'cric-dura-action', epoch, pid))
    scene = SCENES[int(rng.integers(len(SCENES)))]
    if float(rng.random()) < .15:
        action, _ = clean_teacher(clean)
    else:
        cfg = Config(fs_hz=25e6, fc_hz=2462e6, scenario=scene, processing_route='residual',
            mode='post_sync', equalization_enabled=False, reference_snr_db=30., boundary='reflect')
        receiver = receiver_for_session(cfg, 2026101911, session)
        seed = stable_seed(2026101912, 'learned-action/train/epoch%02d' % epoch, scene, pid)
        action, _ = teacher(clean, cfg, seed, receiver)
    return i, np.asarray(coarse, np.float32), action


def prepare_index(rt, handles, protocols):
    root = Path(rt['run_root']) / 'g_data'
    root.mkdir(parents=True, exist_ok=True)
    old_ids = np.load(Path(rt['original_G_action_cache']) / 'train/physical_ids.npy', allow_pickle=False)
    old_lookup = {str(s): i for i, s in enumerate(old_ids)}
    new_lookup = {}
    for dataset in ('ManySig', 'ManyRx'):
        records = {}
        for pid, p in protocols.items():
            if p['dataset'] != dataset:
                continue
            folder = Path(handles[pid]['source_cache'])
            for role in ('L_s', 'U_s'):
                with np.load(folder / (role + '_metadata.npz')) as meta:
                    for i, sid in enumerate(meta['ids']):
                        if str(sid) not in old_lookup and str(sid) not in records:
                            records[str(sid)] = (str(folder / (role + '_x.npy')), i, int(meta['rx_i'][i]), int(meta['day_i'][i]))
        group = root / dataset
        group.mkdir(exist_ok=False)
        ids = sorted(records)
        x = np.lib.format.open_memmap(group / 'clean.npy', mode='w+', dtype=np.float32, shape=(len(ids), 2, 256))
        cache = {}
        for j, sid in enumerate(ids):
            path, i, _, _ = records[sid]
            if path not in cache:
                from .source_iq import aligned_source_iq
                role=Path(path).name[:-6]
                with np.load(Path(path).parent/(role+'_metadata.npz'),allow_pickle=False) as metadata:
                    cache[path],_=aligned_source_iq(Path(path).parent,role,metadata['ids'])
            x[j] = cache[path][i]
            new_lookup[sid] = (dataset, j)
        x.flush()
        np.save(group / 'physical_ids.npy', np.asarray(ids))
        np.save(group / 'rx.npy', np.asarray([records[s][2] for s in ids], np.int64))
        np.save(group / 'day.npy', np.asarray([records[s][3] for s in ids], np.int64))
    selectors = {}
    for pid, handle in handles.items():
        contract = read(handle['source_contract'])
        for roles in ('L', 'LU'):
            allowed = set(contract['role_ids']['L_s'])
            if roles == 'LU':
                allowed.update(contract['role_ids']['U_s'])
            old = sorted((old_lookup[s], s) for s in allowed if s in old_lookup)
            new = sorted((new_lookup[s][1], s) for s in allowed if s not in old_lookup)
            ids = [s for _, s in old] + [s for _, s in new]
            if set(ids) != allowed or len(ids) != len(allowed):
                raise ValueError('G role selector differs')
            path = root / (pid + '-' + roles + '.npz')
            labeled_ids = set(contract['role_ids']['L_s'])
            np.savez(path, ids=np.asarray(ids), kind=np.asarray([0] * len(old) + [1] * len(new), np.int64),
                position=np.asarray([i for i, _ in old] + [i for i, _ in new], np.int64),
                is_L=np.asarray([s in labeled_ids for s in ids], bool))
            selectors[pid + '-' + roles] = dict(selector=str(path), new_cache=str(root / protocols[pid]['dataset']),
                old_action=rt['original_G_action_cache'], old_coarse=rt['original_G_coarse_cache'],
                source_contract=handle['source_contract'], roles=roles, reused_observations=len(old), new_observations=len(new))
    write(root / 'selectors.json', selectors)
    write(root / 'index_complete.json', dict(status='SOURCE_ROLE_BINDINGS_COMPLETE', old_source_observations=len(old_ids),
        missing_counts={d: len(np.load(root / d / 'physical_ids.npy')) for d in ('ManySig', 'ManyRx')},
        TX_labels_read=0, target_IQ_read=0))
    return selectors


def prepare_observations(root, frozen_sources, workers=4):
    root = Path(root)
    n = len(np.load(root / 'physical_ids.npy'))
    if (root / 'complete.json').exists():
        return
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(workers, initializer=initialize, initargs=(str(root), frozen_sources)) as pool:
        for epoch in range(1, 21):
            folder = root / 'epochs' / ('%02d' % epoch)
            if (folder / 'complete.json').exists():
                continue
            folder.mkdir(parents=True, exist_ok=False)
            shapes = dict(coarse=(n, 2, 256), received=(n, 2, 256), kernel=(n, 2, 64), coefficients=(n, 4, 256), variance=(n,))
            arrays = {k: np.lib.format.open_memmap(folder / (k + '.npy'), mode='w+', dtype=np.float32, shape=v) for k, v in shapes.items()}
            for i, coarse, action in pool.imap(generate, ((i, epoch) for i in range(n)), chunksize=32):
                arrays['coarse'][i] = coarse
                for key, value in action.items():
                    arrays[key][i] = value
            for a in arrays.values():
                a.flush()
            write(folder / 'complete.json', dict(epoch=epoch, count=n, native_source_action=True, TX_labels_read=0, F_reads=0, target_reads=0))
            print('G_SOURCE_EPOCH_READY', root.name, epoch, n, flush=True)
    write(root / 'complete.json', dict(status='ALL_SOURCE_OBSERVATIONS_READY', epochs=20, count=n))


class CacheView:
    def __init__(self, cfg):
        if cfg['roles'] not in ('L','LU'):
            raise ValueError('Only registered source L/LU roles permitted')
        self.cfg = cfg
        with np.load(cfg['selector'], allow_pickle=False) as selector:
            self.ids, self.kind, self.position, self.is_L = (selector[k].copy() for k in ('ids', 'kind', 'position', 'is_L'))
        contract = read(cfg['source_contract'])
        allowed = set(contract['role_ids']['L_s'])
        if cfg['roles'] == 'LU':
            allowed.update(contract['role_ids']['U_s'])
        labeled=set(contract['role_ids']['L_s'])
        expected_L=np.asarray([str(i) in labeled for i in self.ids],bool)
        if len(self.ids)!=len(allowed) or set(self.ids.tolist()) != allowed or not np.array_equal(self.is_L,expected_L) or (cfg['roles'] == 'L' and not self.is_L.all()):
            raise ValueError('Forbidden G source role')
        if self.kind.shape!=self.ids.shape or self.position.shape!=self.ids.shape or not np.isin(self.kind,[0,1]).all():
            raise ValueError('G source selector shape/kind differs')
        self.old = Path(cfg['old_action'])
        self.new = Path(cfg['new_cache'])
        self.clean = [np.load(self.old / 'train/clean.npy', mmap_mode='r'), np.load(self.new / 'clean.npy', mmap_mode='r')]
        for kind,folder in ((0,self.old/'train'),(1,self.new)):
            take=self.kind==kind
            if not take.any():continue
            native_ids=np.load(folder/'physical_ids.npy',allow_pickle=False)
            pos=self.position[take]
            if pos.min()<0 or pos.max()>=len(native_ids) or not np.array_equal(native_ids[pos],self.ids[take]):
                raise ValueError('G source selector does not address declared physical IQ')
        self.arrays = {}

    def __len__(self):
        return len(self.ids)

    def ready(self):
        return not (self.kind == 1).any() or (self.new / 'complete.json').is_file()

    def batch(self, indices, epoch, coarse=False, action=False):
        if not self.ready():
            raise ValueError('G source observations are not complete')
        folder = '%02d' % epoch
        result = dict(clean=np.empty((len(indices), 2, 256), np.float32))
        names = ('coarse',) if coarse else ('received', 'kernel', 'coefficients', 'variance') if action else ('received',)
        for name in names:
            shape = (len(indices),) if name == 'variance' else (len(indices), 2, 64) if name == 'kernel' else (len(indices), 4, 256) if name == 'coefficients' else (len(indices), 2, 256)
            result[name] = np.empty(shape, np.float32)
        for kind in (0, 1):
            mask = self.kind[indices] == kind
            if not mask.any():
                continue
            pos = self.position[indices[mask]]
            result['clean'][mask] = self.clean[kind][pos]
            for name in names:
                key = (kind, epoch, name)
                if key not in self.arrays:
                    if kind == 0 and name == 'coarse':
                        base = Path(self.cfg['old_coarse']) / 'epochs' / folder
                        q = np.load(base / 'quantile.npy')
                        self.arrays[key] = (q, np.load(base / 'residual.npy', mmap_mode='r'), np.load(base / 'full.npy', mmap_mode='r'))
                    else:
                        base = self.old if kind == 0 else self.new
                        self.arrays[key] = np.load(base / 'epochs' / folder / (name + '.npy'), mmap_mode='r')
                if kind == 0 and name == 'coarse':
                    q, residual, full = self.arrays[key]
                    value = self.clean[0][pos].copy()
                    for selected, array in ((q[pos] >= .15, residual), (q[pos] >= .75, full)):
                        value[selected] = array[pos[selected]]
                    result[name][mask] = value
                else:
                    result[name][mask] = self.arrays[key][pos]
        return result
