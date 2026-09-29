"""Publish the fixed fixed-kernel local margin support diagnostic through the existing CPU transport."""
import argparse
import importlib.util
from pathlib import Path
from run_d92_branch_local_margin_probe import validate_spec

# Load a private instance: other diagnostic publishers also specialize this
# transport and must not mutate one another when imported in one test process.
spec = importlib.util.spec_from_file_location('_branch_local_margin_transport',
    Path(__file__).with_name('publish_d92_branch_support_probe.py'))
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
transport.validate_spec = validate_spec
transport.PATHS = ['code', 'tools/cvs_native_artifacts.py',
    'tools/export_d92_branch_support_features.py', 'tools/evaluate_d92_branch_support_probe.py',
    'tools/run_d92_branch_support_probe.py', 'tools/run_d92_branch_local_margin_probe.py',
    'tools/evaluate_d92_branch_local_margin_probe.py', 'tools/summarize_d92_branch_local_margin_probe.py',
    'tools/summarize_d92_branch_local_ridge_probe.py', 'tools/evaluate_d92_branch_local_ridge_probe.py',
    'tools/summarize_d92_branch_support_probe.py', 'tools/summarize_d92_branch_interaction_probe.py',
    'tools/evaluate_d92_branch_interaction_probe.py',
    'configs/d92_branch_local_margin_frozen_20260929.json',
    'configs/d92_branch_local_margin_support_rx3_20260929.json',
    'configs/d92_branch_local_margin_support_rx1_20260929.json']
GPU_CHECK = "used=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())\nif used>1024:raise RuntimeError('GPU0 occupied; leave existing process untouched')\n"
if transport.REMOTE.count(GPU_CHECK) != 1:
    raise RuntimeError('Publisher transport changed; reconcile the CPU-only route')
transport.REMOTE = transport.REMOTE.replace(GPU_CHECK, '').replace(
    'tools/run_d92_branch_support_probe.py', 'tools/run_d92_branch_local_margin_probe.py')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--resume-staged-commit')
    parser.parse_args()
    transport.main()
