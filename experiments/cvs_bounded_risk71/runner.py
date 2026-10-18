"""Pin existing P43 runtime; only finite-pool quality estimation changes."""
import importlib.util
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT.parent / 'cvs_FcrossRX_impulse_factorial_E30_20261018_r01'
REFERENCE_COMMIT = '19aedc3604fed9c193237dd92366164178af0983'


def bootstrap():
    if (REFERENCE / 'release_commit.txt').read_text().strip() != REFERENCE_COMMIT:
        raise ValueError('Pinned P43 runtime differs')
    spec = importlib.util.spec_from_file_location('BOUNDEDRISK71_original_runtime',
        REFERENCE / 'experiments/cvs_noise_shape_factorial43/runner.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.bootstrap()
    from experiments.cvs_phase1_overlay.model import native_modules
    native = native_modules()
    expected = REFERENCE.parent / 'cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/SSDG/train_ssdg.py'
    if Path(native.__file__).resolve() != expected.resolve():
        raise ValueError('Actual SSDG module is not the original pinned runtime')
    nominal_root=REFERENCE.parent / 'cvs_FcrossRX_quality_strat_E30_20261018_r01'
    if (nominal_root/'release_commit.txt').read_text().strip()!='000a34f52dc4240c8d078254ada17e7761a27a84':
        raise ValueError('Pinned P70 nominal estimator code identity differs')
    nominal_pkg=types.ModuleType('experiments.cvs_quality_risk70')
    nominal_pkg.__path__=[str(nominal_root/'experiments/cvs_quality_risk70')]
    sys.modules[nominal_pkg.__name__]=nominal_pkg
    package = types.ModuleType('experiments.cvs_bounded_risk71')
    package.__path__ = [str(ROOT / 'experiments/cvs_bounded_risk71')]
    sys.modules[package.__name__] = package


if __name__ == '__main__':
    bootstrap()
    if sys.argv[1:] == ['--checks']:
        from experiments.cvs_bounded_risk71.checks import run
        print(json.dumps(run()), flush=True)
    else:
        from experiments.cvs_bounded_risk71.dispatch import main
        main()
