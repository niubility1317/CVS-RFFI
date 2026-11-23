"""Replay the fixed WiSig weak-detail ablation with the Main split reader."""

import argparse
from pathlib import Path, PurePosixPath

import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.publish import release, release_commit
from experiments.pamf_lora_main.runner import activate

from . import wisig_joint_preamble_source as tri
from . import wisig_uncertain_source as core


RUN = '20261123-phase1-wisig-tri-learned-replay-s2026092701-r01'
TRAIN_COMMIT = '21737d2666ae4d005da2bff5404dcc6d143abe12'
BASELINE_COMMIT = '1291fa44f9dea8659021cec24fc8c581e12f74ab'


def parent(rt, run, commit):
    root = Path(rt['run_root']).parent
    config = read(root.parent / 'releases' / f'{run}-release' / 'runtime.json')
    path = root / run
    if (config['run_id'], config['code_commit'], config['run_root'], config['source_root']) != (
            run, commit, str(path), rt['source_root']) or read(path / 'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Expected immutable same-source ancestor')
    return path


@torch.no_grad()
def smoke(rt, model_root):
    source, _, indices = core.source_iq(Path(rt['source_root']))
    x = core.as_complex(tensor(source[indices[:4]], 'cuda:0'))
    reader = tri.reader_factory(rt).eval().requires_grad_(False)
    for model, threshold in core.load_models(model_root, core.RECIPE).values():
        coarse, joint, level = model(x)
        gated = torch.where((level >= threshold)[:, None], joint, x)
        for y in (coarse, joint, gated):
            logits = reader.classify_features(reader.features(core.as_iq(y)))
            if logits.shape != (len(x), 6) or not torch.isfinite(logits).all():
                raise ValueError('Real frozen candidate or reader no-query smoke failed')
    write(Path(rt['run_root']) / 'smoke.json', dict(status='PASS', count=len(x), V_reads=0, TX_reads=0,
                                                 target_reads=0, optimizer_updates=0))


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    status = root / 'state.json'
    try:
        for action in ('smoke', 'evaluate', 'score'):
            gpu = None if action == 'score' else core.baseline.select_gpu()
            write(status, dict(status=action.upper(), gpu=gpu), exclusive=False)
            worker = core.baseline.start(rt, action, 'tri-replay', gpu, module=rt['runner'],
                                         extra_keys=('recognizer_run', 'baseline_root'))
            if worker.wait():
                raise RuntimeError(f'WiSig frozen ablation replay {action} failed')
        write(status, dict(status='SOURCE_SCORED'), exclusive=False)
    except Exception as exc:
        write(status, dict(status='FAILED', error=str(exc)), exclusive=False)
        raise


def publish(repo, output, context, ssh_config):
    c = read(context)
    rt = {key: c[key] for key in ('project', 'python', 'frozen_sources')}
    project = PurePosixPath(rt['project'])
    runs = project / 'runs'
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              release_root=str(project / 'releases' / f'{RUN}-release'), run_root=str(runs / RUN),
              source_root=str(runs / core.baseline.CACHE_RUN / 'cache'), parent_root=str(runs / core.baseline.R17_RUN),
              recognizer_run=str(runs / tri.tri_plan.RUN), baseline_root=str(runs / tri.RUN))
    release(repo, output, rt, ssh_config, archive_extras=('tools/r17_activation_factorization.py',))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    pub = actions.add_parser('publish')
    for name in ('repo', 'output', 'context', 'ssh-config'):
        pub.add_argument('--' + name, type=Path, required=True)
    for action in ('dispatch', 'smoke', 'evaluate', 'score'):
        worker = actions.add_parser(action)
        worker.add_argument('runtime', type=Path)
        worker.add_argument('name', nargs='?')
    cmd = parser.parse_args()
    if cmd.action == 'publish':
        publish(cmd.repo, cmd.output, cmd.context, cmd.ssh_config)
        return
    rt = read(cmd.runtime)
    if cmd.action == 'dispatch':
        dispatch(rt)
    elif cmd.action == 'score':
        core.baseline.score(rt, core.modes_for(core.RECIPE))
    else:
        activate(rt)
        checkpoint_root = parent(rt, core.RUN, TRAIN_COMMIT)
        if cmd.action == 'smoke':
            smoke(rt, checkpoint_root)
        else:
            old = parent(rt, tri.RUN, BASELINE_COMMIT)
            if old != Path(rt['baseline_root']) or read(old / 'predictions/complete.json')['modes'] != list(tri.MODES):
                raise ValueError('Expected the fixed same-reader predictions')
            core.evaluate(rt, model_root=checkpoint_root, reader_factory=tri.reader_factory)


if __name__ == '__main__':
    main()
