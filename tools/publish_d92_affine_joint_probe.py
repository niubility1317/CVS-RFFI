"""Publish the committed AffineJointLocalRidge joint diagnostic through CPU transport."""
import argparse
import importlib.util
from pathlib import Path
from run_d92_affine_joint_probe import validate_spec

spec=importlib.util.spec_from_file_location('_affine_joint_transport',Path(__file__).with_name('publish_d92_branch_support_probe.py'))
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
transport.validate_spec=validate_spec
PATHS=[
    'code/cvsrffi/d92_affine_joint_local_ridge.py',
    'code/cvsrffi/d92_function_coordinate_residual8_local_ridge.py',
    'code/cvsrffi/d92_margin_constrained_residual8_local_ridge.py',
    'tools/evaluate_d92_prototype_transport_probe.py',
    'tools/summarize_d92_prototype_transport_probe.py',
    'configs/d92_prototype_transport_frozen_20260930.json',
    'code/cvsrffi/__init__.py','code/cvsrffi/d92_branch_support_probe.py',
    'code/cvsrffi/d92_branch_ridge.py','code/cvsrffi/d92_branch_interaction.py',
    'code/cvsrffi/d92_branch_local_ridge.py',
    'code/cvsrffi/d92_joint_channel_local_ridge.py',
    'code/cvsrffi/d92_prototype_transport_local_ridge.py',
    'tools/cvs_native_artifacts.py','tools/export_d92_branch_support_features.py',
    'tools/evaluate_d92_branch_support_probe.py','tools/evaluate_d92_branch_local_ridge_probe.py',
    'tools/run_d92_branch_support_probe.py','tools/run_d92_registration_diagnostic.py',
    'tools/run_d92_affine_joint_probe.py','tools/d92_registration_score_diagnostics.py',
    'tools/evaluate_d92_registration_diagnostic.py','tools/evaluate_d92_affine_joint_probe.py',
    'tools/summarize_d92_registration_diagnostic.py','tools/summarize_d92_branch_support_probe.py',
    'tools/summarize_d92_affine_joint_probe.py',
    'configs/d92_branch_local_ridge_frozen_20260929.json',
    'configs/d92_affine_joint_frozen_20261001.json',
    'configs/d92_affine_joint_support_rx3_20261001.json',
    'configs/d92_affine_joint_support_rx1_20261001.json']
transport.PATHS=PATHS
GPU_CHECK="used=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())\nif used>1024:raise RuntimeError('GPU0 occupied; leave existing process untouched')\n"
if transport.REMOTE.count(GPU_CHECK)!=1:raise RuntimeError('Publisher transport changed; reconcile CPU-only route')
transport.REMOTE=transport.REMOTE.replace(GPU_CHECK,'').replace('tools/run_d92_branch_support_probe.py','tools/run_d92_affine_joint_probe.py')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--resume-staged-commit');p.parse_args()
    transport.main()
