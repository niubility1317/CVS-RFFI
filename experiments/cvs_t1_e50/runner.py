"""Bootstrap the unchanged original T1 runtime, then this release's package."""
import importlib.util
import sys
import types
from pathlib import Path


def bootstrap():
    root = Path(__file__).resolve().parents[2]
    ref = Path('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_AsTemperature_E30_20261019_r02')
    spec = importlib.util.spec_from_file_location('t1_e50_runtime', ref / 'experiments/cvs_as_temperature84/runner.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.bootstrap()
    package = types.ModuleType('experiments.cvs_t1_e50')
    package.__path__ = [str(root / 'experiments/cvs_t1_e50')]
    sys.modules[package.__name__] = package
    return root


if __name__ == '__main__':
    bootstrap()
    from experiments.cvs_t1_e50.dispatch import main
    main()
