"""Copy the pinned author model/loss only; never copy or load author weights."""
import argparse
import ast
import shutil
import subprocess
from pathlib import Path

PIN = '5a03a04da86c4bbf5cf90f41a55c268bf2152203'

def prepare(upstream, destination):
    upstream, destination = Path(upstream), Path(destination)
    oid = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if oid != PIN or dirty:
        raise ValueError('Pinned clean author checkout required')
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    paths = list((upstream/'backbones').rglob('*.py')) + [upstream/'utils/hsic.py']
    for source in paths:
        target = destination/source.relative_to(upstream)
        target.parent.mkdir(parents=True, exist_ok=True)
        code = source.read_text(encoding='utf-8-sig')
        # The author only imports timm for this identical inverse-CDF initializer.
        # Paired seeded initialization is checked before release.
        code = code.replace('from timm.layers import trunc_normal_', 'from torch.nn.init import trunc_normal_')
        ast.parse(code)
        target.write_text(code, encoding='utf-8', newline='\n')
    shutil.copyfile(upstream/'README.md', destination/'AUTHOR_README.md')
    return len(paths)

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    args=p.parse_args()
    print('PREPARED_MODEL_FILES', prepare(args.upstream,args.destination))
