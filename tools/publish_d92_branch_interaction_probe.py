"""Reuse the verified publisher transport for a cache-only CPU diagnostic."""
import publish_d92_branch_support_probe as transport
from run_d92_branch_interaction_probe import validate_spec

transport.validate_spec = validate_spec
transport.PATHS = ['code','tools/cvs_native_artifacts.py','tools/export_d92_branch_support_features.py',
    'tools/evaluate_d92_branch_support_probe.py','tools/run_d92_branch_support_probe.py',
    'tools/run_d92_branch_interaction_probe.py','tools/evaluate_d92_branch_interaction_probe.py',
    'tools/summarize_d92_branch_support_probe.py','tools/summarize_d92_branch_interaction_probe.py',
    'configs/d92_branch_interaction_frozen_20260929.json',
    'configs/d92_branch_interaction_support_rx3_20260929.json','configs/d92_branch_interaction_support_rx1_20260929.json']
GPU_CHECK = "used=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())\nif used>1024:raise RuntimeError('GPU0 occupied; leave existing process untouched')\n"
if transport.REMOTE.count(GPU_CHECK) != 1:
    raise RuntimeError('Publisher transport changed; reconcile CPU-only route')
transport.REMOTE = transport.REMOTE.replace(GPU_CHECK, '').replace('tools/run_d92_branch_support_probe.py','tools/run_d92_branch_interaction_probe.py')

if __name__ == '__main__':
    transport.main()
