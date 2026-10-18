"""Pin P68/SAT18 dependencies; never alter a running release."""
import importlib.util
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[2]
REFERENCE=ROOT.parent/'cvs_FcrossRX_full_As_pair6_E30_20261018_r01'
REFERENCE_COMMIT='f520ae98791a80775b6b5923183fcffba7e91c73'

def bootstrap():
    if (REFERENCE/'release_commit.txt').read_text().strip()!=REFERENCE_COMMIT:
        raise ValueError('Pinned P68 runtime differs')
    spec=importlib.util.spec_from_file_location('NOISERESAMPLE75_runtime',REFERENCE/'experiments/cvs_full_channel_pair68/runner.py')
    runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime);runtime.bootstrap()
    package=types.ModuleType('experiments.cvs_noise_resample75')
    package.__path__=[str(ROOT/'experiments/cvs_noise_resample75')]
    sys.modules[package.__name__]=package

if __name__=='__main__':
    bootstrap()
    from experiments.cvs_noise_resample75.dispatch import main
    main()
