"""Fetch pinned author sources and create local, minimally repaired copies.

Upstream snapshots and weights remain local. No downloaded checkpoint is loaded.
This prepares candidates; it does not submit or start an experiment.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path


def git(*args, cwd=None):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True, encoding='utf-8').strip()


def replace_once(path, old, new):
    text = path.read_text(encoding='utf-8')
    if text.count(old) != 1:
        raise ValueError(f'Pinned patch no longer matches: {path}')
    path.write_text(text.replace(old, new), encoding='utf-8', newline='\n')


def repair(repo, method):
    if method in ('asknet', 'wavemlp'):
        loader = repo / ('WiSig/utils/load_data.py' if method == 'asknet' else 'utils/load_data.py')
        replace_once(loader, '        x[i] = x[i] / np.sqrt(power)',
                     '        if not np.isfinite(power) or power <= 0:\n'
                     '            raise ValueError(f"Non-finite or zero-power IQ sample: {i}")\n'
                     '        x[i] = x[i] / np.sqrt(power)')
    if method == 'wavemlp':
        replace_once(repo / 'main.py', '        verbose=True,\n', '')
        replace_once(repo / 'backbones/WaveletOperator.py', 'import torch.nn as nn',
                     'import torch.nn as nn\nimport torch.nn.functional as F')
    # DIFL intentionally remains unchanged: repairing its empty feature branch
    # requires an architectural choice that has not been confirmed from the paper.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream-root', type=Path, required=True)
    p.add_argument('--working-root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=Path(__file__).with_name('methods.json'))
    args = p.parse_args()
    config = json.loads(args.manifest.read_text(encoding='utf-8'))
    for m in config['methods']:
        upstream = args.upstream_root / m['repository_name']
        if not upstream.exists():
            upstream.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'clone', '--no-checkout', m['repository_url'], str(upstream)], check=True)
            subprocess.run(['git', 'checkout', '--detach', m['upstream_commit']], cwd=upstream, check=True)
        if git('rev-parse', 'HEAD', cwd=upstream) != m['upstream_commit']:
            raise ValueError(f'Unexpected upstream commit: {upstream}; existing files preserved')
        if git('status', '--porcelain', '--untracked-files=no', cwd=upstream):
            raise ValueError(f'Upstream tracked files modified: {upstream}')
        destination = args.working_root / m['id']
        if destination.exists():
            raise FileExistsError(f'Will not overwrite existing working copy: {destination}')
        # Keep all original data, weights, logs, archives and .git in upstream.
        # Working copies contain only source and author documentation.
        destination.mkdir(parents=True)
        for relative in git('ls-files', cwd=upstream).splitlines():
            source = upstream / relative
            if source.suffix.lower() not in ('.py', '.md', '.sh', '.txt') or '/logs/' in '/' + relative:
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        repair(destination, m['id'])
        (destination / 'LOCAL_SOURCE.json').write_text(json.dumps({
            'method': m['id'], 'repository_url': m['repository_url'],
            'upstream_commit': m['upstream_commit'], 'repair_version': 1,
            'formal_experiment_started': False, 'pretrained_weights_loaded': False,
        }, indent=2) + '\n', encoding='utf-8')
        print(f'PREPARED {m["id"]}: {destination}')


if __name__ == '__main__':
    main()
