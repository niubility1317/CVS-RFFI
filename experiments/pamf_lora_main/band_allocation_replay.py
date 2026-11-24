"""Score the original LoRa allocation checkpoint after a smoke compatibility fix."""

import argparse
from pathlib import Path
import shutil

from experiments.cric_dura_comparison.io import read, write

from . import band_allocation_source as trained
from .joint_spectrum_source import evaluate, smoke
from .local_wiener_source import ROUTES, score_frozen
from .refine_runner import select_gpu, start
from .runner import activate
from .spectral_block_source import paired_tasks, publish


RUN = '20261124-phase1-lora-band-allocation-source-m3-r02'


def reuse_frozen_fit(root):
    original = root.parent / trained.RUN
    if read(original/'state.json') != dict(status='FAILED', error='LoRa real checkpoint no-query smoke failed'):
        raise ValueError('Expected the terminal original run with the documented smoke failure')
    metadata = read(original/'band-complete.json')
    expected = dict(status='SOURCE_FROZEN', steps=40000, F_reads=0, TX_reads=0, V_reads=0, target_reads=0,
                    practical_reads=0, trace_reads=0, n_fit=20000, n_calibration=2500, n_held=500)
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('The original run must have a legal frozen 40,000-step calibration')
    for name in ('band-model.pt', 'band-complete.json'):
        shutil.copyfile(original/name, root/name)
    write(root/'fit-reuse.json', dict(origin=trained.RUN, optimizer_updates_here=0,
                                     carried_steps=metadata['steps'], carried_threshold=metadata['threshold']))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    status = root/'state.json'
    try:
        reuse_frozen_fit(root)
        gpu = select_gpu()
        write(status, dict(status='SMOKE', gpu=gpu), exclusive=False)
        if start(rt,'smoke','source',gpu,module=rt['runner']).wait():
            raise RuntimeError('Frozen LoRa allocation smoke failed')
        paired_tasks(rt, 'evaluate', ROUTES)
        write(status, dict(status='SCORE'), exclusive=False)
        if start(rt,'score','source',module=rt['runner']).wait():
            raise RuntimeError('Independent LoRa allocation scoring failed')
        write(status, dict(status='SOURCE_SCORED'), exclusive=False)
    except Exception as exc:
        write(status, dict(status='FAILED', error=str(exc)), exclusive=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.add_subparsers(dest='action', required=True)
    publication = args.add_parser('publish')
    for name in ('repo','output','context','ssh-config'):
        publication.add_argument('--'+name, type=Path, required=True)
    for name in ('dispatch','smoke','evaluate','score'):
        stage = args.add_parser(name)
        stage.add_argument('runtime', type=Path)
        if name != 'dispatch':
            stage.add_argument('name', choices=ROUTES if name=='evaluate' else ('source',))
    values = vars(parser.parse_args())
    action = values.pop('action')
    if action == 'publish':
        publish(**values, run=RUN, runner=__spec__.name)
        return
    rt = read(values['runtime'])
    if action=='dispatch':
        dispatch(rt)
    elif action=='score':
        score_frozen(rt, trained.MODES, comparisons=trained.COMPARISONS)
    else:
        activate(rt)
        if action=='smoke':
            smoke(rt, trained.RECIPES, trained.MODES, render_fn=trained.render, recover_fn=trained.recover)
        else:
            evaluate(rt, values['name'], trained.RECIPES, trained.MODES, fields=trained.FIELDS,
                     render_fn=trained.render)


if __name__=='__main__':
    main()
