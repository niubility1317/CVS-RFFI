"""Bootstrap the existing immutable sceneRisk dependency tree; add this package."""
import importlib.util
import json
from pathlib import Path
import sys
import types


def bootstrap():
    here = Path(__file__).resolve().parents[2]
    rt = json.loads((here/'runtime.json').read_text())
    ref = Path(rt['reference_release'])
    if rt.get('reference_commit') and (ref/'release_commit.txt').read_text().strip() != rt['reference_commit']:
        raise ValueError('Pinned pure-F runtime differs')
    spec = importlib.util.spec_from_file_location('_epoch_replay_runtime', ref/'experiments/cvs_F_rawAs20/runner.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.bootstrap()
    package = types.ModuleType('experiments.cvs_F_epochReplay20')
    package.__path__ = [str(here/'experiments/cvs_F_epochReplay20')]
    sys.modules[package.__name__] = package
    return here


if __name__ == '__main__':
    bootstrap()
    if sys.argv[1:] == ['--checks']:
        from experiments.cvs_F_epochReplay20.checks import run
        print(json.dumps(run()))
    else:
        from experiments.cvs_F_epochReplay20.dispatch import main
        main()
