"""Bootstrap the pinned runtime, then perform one frozen source-only readout."""
import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path


def bootstrap():
    root = Path(__file__).resolve().parents[2]
    entry = root / 'experiments/cvs_u_quality66/runner.py'
    spec = importlib.util.spec_from_file_location('PARENT_COHORT66_PINNED_RUNTIME', entry)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.bootstrap()
    package = types.ModuleType('experiments.cvs_parent_cohort66')
    package.__path__ = [str(root / 'experiments/cvs_parent_cohort66')]
    sys.modules[package.__name__] = package
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checks', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    args = parser.parse_args()
    if args.checks:
        # Pure toy checks must not bootstrap a remote runtime or import Torch.
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root))
        from experiments.cvs_parent_cohort66.checks import run
        print(json.dumps(run()), flush=True)
        return
    bootstrap()
    from experiments.cvs_parent_cohort66.readout import execute
    result = execute(args.device)
    print(json.dumps({k: result[k] for k in (
        'status', 'run_id', 'checkpoint_key', 'NN', 'source_NN', 'smoke_NN',
        'training_updates', 'U_truth_read', 'target_access', 'wall_seconds')}), flush=True)


if __name__ == '__main__':
    main()
