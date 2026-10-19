"""Load the unchanged P43 runtime and seed-aware P49 factory."""
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def bootstrap():
    location = ROOT / 'experiments/cvs_As_replicate49/runner.py'
    spec = importlib.util.spec_from_file_location('CE_HALF_seed_runtime', location)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.bootstrap()
    package = types.ModuleType('experiments.cvs_ce_half_replicate')
    package.__path__ = [str(ROOT / 'experiments/cvs_ce_half_replicate')]
    sys.modules[package.__name__] = package


if __name__ == '__main__':
    bootstrap()
    if sys.argv[1:] == ['--checks']:
        from experiments.cvs_ce_half_replicate.checks import run
        print(run())
    else:
        from experiments.cvs_ce_half_replicate.dispatch import main
        main()
