"""Run one fixed AS_CE20 source-only diagnostic through the existing P68 runtime."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT.parent / 'cvs_FcrossRX_full_As_pair6_E30_20261018_r01'
REFERENCE_COMMIT = 'f520ae98791a80775b6b5923183fcffba7e91c73'


def bootstrap():
    if (REFERENCE / 'release_commit.txt').read_text().strip() != REFERENCE_COMMIT:
        raise ValueError('Fixed already-running P68 runtime commit differs')
    entry = REFERENCE / 'experiments/cvs_full_channel_pair68/runner.py'
    spec = importlib.util.spec_from_file_location('ASCE_PARENT68_PINNED_RUNTIME', entry)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.bootstrap()
    package = types.ModuleType('experiments.cvs_asce_parent68')
    package.__path__ = [str(ROOT / 'experiments/cvs_asce_parent68')]
    sys.modules[package.__name__] = package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checks', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    args = parser.parse_args()
    if args.checks:
        sys.path.insert(0, str(ROOT))
        from experiments.cvs_asce_parent68.checks import run
        print(json.dumps(run()), flush=True)
        return
    bootstrap()
    from experiments.cvs_asce_parent68.readout import execute
    result = execute(args.device)
    print(json.dumps({k: result[k] for k in ('status', 'run_id', 'checkpoint_key',
        'parent_clean_correct_count', 'NN', 'source_NN', 'smoke_NN',
        'training_updates', 'target_access', 'wall_seconds')}), flush=True)


if __name__ == '__main__':
    main()
