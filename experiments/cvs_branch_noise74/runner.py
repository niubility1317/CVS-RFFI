"""Pinned original P43 runtime; source-only observations, no train entry point."""
import importlib.util
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT.parent / 'cvs_FcrossRX_impulse_factorial_E30_20261018_r01'

def bootstrap():
    if (REFERENCE/'release_commit.txt').read_text().strip() != '19aedc3604fed9c193237dd92366164178af0983':
        raise ValueError('Original P43 runtime identity differs')
    spec = importlib.util.spec_from_file_location('branch74_runtime', REFERENCE/'experiments/cvs_noise_shape_factorial43/runner.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.bootstrap()
    package = types.ModuleType('experiments.cvs_branch_noise74')
    package.__path__ = [str(ROOT/'experiments/cvs_branch_noise74')]
    sys.modules[package.__name__] = package

if __name__ == '__main__':
    bootstrap()
    from experiments.cvs_branch_noise74.probe import main
    main()
