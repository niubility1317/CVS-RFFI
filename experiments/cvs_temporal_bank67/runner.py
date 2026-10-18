"""Pin P43 runtime and retain package ownership of two final student models."""
import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path('/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_impulse_factorial_E30_20261018_r01')
REFERENCE_COMMIT = '19aedc3604fed9c193237dd92366164178af0983'

def bootstrap():
    if (REFERENCE / 'release_commit.txt').read_text().strip() != REFERENCE_COMMIT:
        raise ValueError('Pinned P43 runtime differs')
    spec = importlib.util.spec_from_file_location('TEMPORALBANK67_original_runtime', REFERENCE / 'experiments/cvs_noise_shape_factorial43/runner.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.bootstrap()
    package = types.ModuleType('experiments.cvs_temporal_bank67')
    package.__path__ = [str(ROOT / 'experiments/cvs_temporal_bank67')]
    sys.modules[package.__name__] = package

if __name__ == '__main__':
    bootstrap()
    if sys.argv[1:] == ['--checks']:
        from experiments.cvs_temporal_bank67.checks import run
        print(json.dumps(run()))
    else:
        from experiments.cvs_temporal_bank67.dispatch import main
        main()
