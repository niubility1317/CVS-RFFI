"""Use the immutable CRIC/DURA source releases, with the new adapter separate."""
import importlib
import json
from pathlib import Path
import sys


def activate(runtime):
    roots = {k: Path(v) for k, v in runtime['frozen_sources'].items()}
    required = {'F', 'AF', 'BASE', 'FOLD', 'G'}
    if set(roots) != required or any(not p.is_dir() for p in roots.values()):
        raise ValueError('The five fixed historical source releases are required')
    package = importlib.import_module('experiments')
    current = Path(__file__).resolve().parents[2]
    package.__path__ = [str(roots[k] / 'experiments') for k in ('F', 'AF', 'FOLD', 'BASE', 'G')] + [str(current / 'experiments')]
    # G's fixed channel implementation is used for source simulation. The
    # native F loader explicitly inserts its own immutable SSDG directory.
    sys.path[:0] = [str(roots['G'] / 'code'), str(roots['BASE']), str(roots['BASE'] / 'code')]
    return roots


def numerical_policy():
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision('highest')


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    rt = dict(frozen_sources={k: str(root / 'local_artifacts/cric_dura_comparison_20261021/frozen_sources' / k) for k in ('F', 'AF', 'BASE', 'FOLD', 'G')})
    activate(rt)
    from experiments.cvs_f_supcon50 import design
    from experiments.cvs_rdrop_sinc_units import model
    numerical_policy()
    c = design.config(dict(row_id='CROSS_RX-s2026092701', arm='CROSS_RX', model_seed=2026092701))
    a = design.make_args(c, 'cpu')
    print(json.dumps(dict(args=vars(a), config=c), default=str))
