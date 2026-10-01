"""Margin entry bundle; no dispatch until the independent analysis chain exists."""
import argparse
import importlib.util
import json
from pathlib import Path

from run_d92_margin_joint_probe import ROOT, validate_spec

SUMMARY='tools/summarize_d92_margin_joint_probe.py'
ANALYSIS_PATHS=[SUMMARY,'tools/analyze_d92_margin_joint_probe.py',
    'tools/collect_d92_margin_joint_training_diagnostics.py','tools/report_d92_margin_joint_support.py']
PATHS=[
    'code/cvsrffi/__init__.py','code/cvsrffi/d92_margin_qp_head.py',
    'code/cvsrffi/d92_margin_joint_local_ridge.py','code/cvsrffi/d92_affine_joint_local_ridge.py',
    # The frozen margin core imports only project_coordinates from the joint
    # helper; its module-level kernel import must also exist inside the bundle.
    'code/cvsrffi/d92_conditional_joint_local_ridge.py','code/cvsrffi/d92_conditional_affine_kernel.py',
    'code/cvsrffi/d92_function_coordinate_residual8_local_ridge.py',
    'code/cvsrffi/d92_margin_constrained_residual8_local_ridge.py',
    'code/cvsrffi/d92_joint_channel_local_ridge.py','code/cvsrffi/d92_prototype_transport_local_ridge.py',
    'code/cvsrffi/d92_branch_support_probe.py','code/cvsrffi/d92_branch_ridge.py',
    'code/cvsrffi/d92_branch_interaction.py','code/cvsrffi/d92_branch_local_ridge.py',
    'tools/cvs_native_artifacts.py','tools/export_d92_branch_support_features.py',
    'tools/evaluate_d92_branch_support_probe.py','tools/evaluate_d92_branch_local_ridge_probe.py',
    'tools/evaluate_d92_registration_diagnostic.py','tools/run_d92_registration_diagnostic.py',
    'tools/d92_registration_score_diagnostics.py','tools/run_d92_branch_support_probe.py',
    'tools/evaluate_d92_prototype_transport_probe.py','tools/summarize_d92_prototype_transport_probe.py',
    'tools/summarize_d92_registration_diagnostic.py','tools/summarize_d92_branch_support_probe.py',
    'configs/d92_branch_local_ridge_frozen_20260929.json','configs/d92_prototype_transport_frozen_20260930.json',
    'tools/prepare_d92_margin_joint_probe.py','tools/run_d92_margin_joint_probe.py',
    'tools/evaluate_d92_margin_joint_probe.py','tools/preflight_d92_margin_joint_probe.py',
    'tools/publish_d92_margin_joint_probe.py','tools/publish_d92_branch_support_probe.py',
    'tools/summarize_d92_affine_joint_probe.py','tools/evaluate_d92_affine_joint_probe.py',
    'tools/run_d92_affine_joint_probe.py','tools/d92_affine_analysis_math.py',
    'tools/collect_d92_affine_joint_training_diagnostics.py',
    'docs/D92_MARGIN_QP_HEAD_IMPLEMENTATION_20261001.md','docs/D92_MARGIN_JOINT_ENTRY_20261001.md']


def release_paths(spec):
    validate_spec(spec)
    return list(dict.fromkeys(PATHS+ANALYSIS_PATHS+[spec['spec_path']]+[co['config_path'] for co in spec['probe']['cohorts'].values()]))


def readiness(spec,root=ROOT):
    paths=release_paths(spec);missing=[p for p in paths if not (Path(root)/p).is_file()]
    missing_analysis=[path for path in ANALYSIS_PATHS if path in missing]
    return dict(status='SOURCE_BUNDLE_AVAILABLE_NOT_LAUNCHED' if not missing else 'INCOMPLETE_RELEASE_CAPABILITY',
        missing=missing,missing_analysis=missing_analysis,paths=paths,
        entry_bundle_available=not any(path in missing for path in PATHS),
        independent_summary_available=SUMMARY not in missing,analysis_chain_available=not missing_analysis,
        readiness_scope='File presence only; isolated import is a separate synthetic dependency check',launched=False)


def dispatch(spec_path,root=ROOT):
    spec=json.loads((Path(root)/spec_path).read_text(encoding='utf-8'))
    if str(spec_path)!=spec['spec_path']:raise ValueError('Publication path differs from declared spec_path')
    available=readiness(spec,root)
    if available['missing']:
        raise RuntimeError('Margin source bundle is incomplete; no publication/launch: '+', '.join(available['missing']))
    transport_spec=importlib.util.spec_from_file_location('_margin_support_transport',Path(root)/'tools/publish_d92_branch_support_probe.py')
    transport=importlib.util.module_from_spec(transport_spec);transport_spec.loader.exec_module(transport)
    transport.ROOT=Path(root)
    transport.validate_spec=validate_spec;transport.PATHS=[p for p in release_paths(spec) if p!=spec['spec_path']]
    gpu="used=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())\nif used>1024:raise RuntimeError('GPU0 occupied; leave existing process untouched')\n"
    if transport.REMOTE.count(gpu)!=1:raise RuntimeError('CPU transport changed; reconcile before mutation')
    transport.REMOTE=transport.REMOTE.replace(gpu,'').replace('tools/run_d92_branch_support_probe.py','tools/run_d92_margin_joint_probe.py')
    # Existing committed/pushed release transport retains its collision checks,
    # one archive transfer/readback and sole supervisor semantics.
    transport.main()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--resume-staged-commit')
    args=p.parse_args();dispatch(args.spec)
