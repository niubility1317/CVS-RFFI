"""Generate paired source IQ for the two fixed WiSig environment distributions."""

import multiprocessing as mp
from pathlib import Path

import numpy as np

from experiments.cric_dura_comparison.io import write


COUNT = 63_000
EPOCHS = 30
AUG_SEED = 2026112601
SCENES = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
          'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')
HARD_REMAP = np.asarray([0, 3, 5, 3, 6, 5, 6], dtype=np.uint8)
ACTION_SHAPES = {'kernel': (2, 64), 'coefficients': (4, 256), 'variance': ()}
IQ_SHAPE = (2, 256)


def assignments(epoch):
    order = np.random.default_rng(np.random.SeedSequence([AUG_SEED, epoch])).permutation(COUNT)
    balanced = np.zeros(COUNT, dtype=np.uint8)
    balanced[order[9450:]] = np.repeat(np.arange(1, 7, dtype=np.uint8), 8925)
    # Keep the original coarse mixture of 15% clean, 60% residual and 25% full.
    full = np.zeros(COUNT, dtype=bool)
    full[order[9450:][np.random.default_rng(np.random.SeedSequence(
        [AUG_SEED, epoch, 1])).permutation(53550)[:15750]]] = True
    return balanced, HARD_REMAP[balanced], full


def load(path):
    return np.load(path, mmap_mode='r', allow_pickle=False)


def initialize(source):
    global CLEAN, IDS, RX, DAY
    CLEAN, IDS, RX, DAY = (load(source / f'{key}.npy') for key in ('clean', 'physical_ids', 'rx', 'day'))


def generate(job):
    from leo_practical.channel import Config, receiver_for_session, stable_seed
    from leo_practical.residual import ResidualChannel
    from experiments.pamf_action_restore.data import clean_teacher, teacher
    from experiments.pamf_reception_restore.data import shared_pair

    i, epoch, balanced_code, hard_code, coarse_full = job
    clean = CLEAN[i]
    pid = str(IDS[i])
    session = f'rx{RX[i]}:day{DAY[i]}:eq1'
    namespace = f'wisig-dura-hardmix/epoch{epoch:02d}'
    early = epoch <= 20

    def received(code, action=False):
        scene = SCENES[code]
        if code == 0:
            return clean_teacher(clean)[0] if action else clean
        cfg = Config(fs_hz=25e6, fc_hz=2462e6, scenario=scene, processing_route='residual',
                     mode='post_sync', equalization_enabled=False, reference_snr_db=30., boundary='reflect')
        rx = receiver_for_session(cfg, 2026101911, session)
        seed = stable_seed(AUG_SEED, namespace, scene, pid)
        if action:
            return teacher(clean, cfg, seed, rx)[0]
        signal = clean[0].astype(float) + 1j * clean[1].astype(float)
        output, _ = ResidualChannel(cfg, seed, rx).process(signal)
        return np.stack((output.real, output.imag)).astype(np.float32)

    action = received(hard_code, action=True) if early else {}
    hard = action['received'] if early else received(hard_code)
    balanced = hard if balanced_code == hard_code else received(balanced_code)
    result = dict(action, hard=hard, balanced=balanced)
    result.pop('received', None)
    if early:
        if coarse_full:
            _, full, _ = shared_pair(clean, SCENES[hard_code], pid, session, AUG_SEED, epoch,
                                     2026101902, namespace=f'wisig-dura-hardmix/coarse/{epoch:02d}')
            result['coarse'] = full
        else:
            result['coarse'] = hard
    return i, result


def prepare(rt):
    root = Path(rt['run_root']) / 'source'
    source = Path(rt['action_iq']) / 'train'
    root.mkdir()
    shapes = dict(balanced=IQ_SHAPE, hard=IQ_SHAPE)
    coarse_shapes = dict(coarse=IQ_SHAPE, **ACTION_SHAPES)
    with mp.get_context('spawn').Pool(12, initializer=initialize, initargs=(source,)) as pool:
        for epoch in range(1, EPOCHS + 1):
            folder = root / f'{epoch:02d}'
            folder.mkdir()
            balanced, hard, full = assignments(epoch)
            layout = dict(shapes, **coarse_shapes) if epoch <= 20 else shapes
            arrays = {key: np.lib.format.open_memmap(folder / f'{key}.npy', mode='w+',
                      dtype=np.float32, shape=(COUNT, *shape)) for key, shape in layout.items()}
            jobs = ((i, epoch, int(balanced[i]), int(hard[i]), full[i]) for i in range(COUNT))
            for done, (i, values) in enumerate(pool.imap(generate, jobs, chunksize=64), 1):
                for key, array in arrays.items():
                    array[i] = values[key]
                if done % 10000 == 0:
                    print('HARDMIX_SOURCE_PROGRESS', epoch, done, flush=True)
            for array in arrays.values():
                array.flush()
            np.savez(folder / 'scenes.npz', balanced=balanced, hard=hard, coarse_full=full)
            write(folder / 'complete.json', dict(status='SOURCE_EPOCH_COMPLETE', epoch=epoch, count=COUNT,
                  scene_order=SCENES, balanced=np.bincount(balanced, minlength=7).tolist(),
                  hard=np.bincount(hard, minlength=7).tolist(), TX_labels_read=0, target_reads=0))
            print('HARDMIX_SOURCE_EPOCH_COMPLETE', epoch, flush=True)


class SourceView:
    def __init__(self, rt, profile):
        source = Path(rt['action_iq']) / 'train'
        self.clean = load(source / 'clean.npy')
        self.is_L = load(source / 'roles.npy') == 'L_s'
        self.root = Path(rt['run_root']) / 'source'
        self.profile = profile
        self.current = None

    def ready(self):
        return (self.clean.shape == (COUNT, *IQ_SHAPE) and self.clean.dtype == np.float32
                and self.is_L.shape == (COUNT,) and self.is_L.sum() == 6300)

    def __len__(self):
        return COUNT

    def batch(self, selected, epoch, coarse=False, action=False):
        if (epoch, coarse, action) != self.current:
            folder = self.root / f'{epoch:02d}'
            if not (folder / 'complete.json').is_file():
                raise ValueError('The fixed source epoch is incomplete')
            selected_views = {'coarse': 'coarse'} if coarse else {'received': self.profile}
            if action:
                selected_views.update({key: key for key in ACTION_SHAPES})
            self.arrays = {key: load(folder / f'{field}.npy') for key, field in selected_views.items()}
            self.current = epoch, coarse, action
        return dict(clean=self.clean[selected], **{key: array[selected] for key, array in self.arrays.items()})
