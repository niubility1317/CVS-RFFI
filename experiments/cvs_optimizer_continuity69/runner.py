"""Pin existing P43 runtime; only optimizer-state inheritance changes."""
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
    spec = importlib.util.spec_from_file_location('OPTIMIZERKEEP69_original_runtime',
        REFERENCE / 'experiments/cvs_noise_shape_factorial43/runner.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.bootstrap()
    from experiments.cvs_phase1_overlay.model import native_modules
    native = native_modules()
    expected = REFERENCE.parent / 'cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code/SSDG/train_ssdg.py'
    if Path(native.__file__).resolve() != expected.resolve():
        raise ValueError('Actual SSDG module is not the original pinned runtime')
    package = types.ModuleType('experiments.cvs_optimizer_continuity69')
    package.__path__ = [str(ROOT / 'experiments/cvs_optimizer_continuity69')]
    sys.modules[package.__name__] = package


if __name__ == '__main__':
    bootstrap()
    if sys.argv[1:] == ['--checks']:
        from experiments.cvs_optimizer_continuity69.checks import run
        print(json.dumps(run()), flush=True)
    else:
        from experiments.cvs_optimizer_continuity69.dispatch import main
        main()
