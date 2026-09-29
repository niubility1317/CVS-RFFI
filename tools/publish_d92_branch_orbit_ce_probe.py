"""Publish one frozen orbit support export and CE diagnostic release."""
import argparse
import importlib.util
from pathlib import Path
from run_d92_branch_orbit_ce_probe import validate_spec

definition=importlib.util.spec_from_file_location('_branch_orbit_transport',
    Path(__file__).with_name('publish_d92_branch_support_probe.py'))
transport=importlib.util.module_from_spec(definition)
definition.loader.exec_module(transport)
transport.validate_spec=validate_spec
transport.PATHS=['code','tools/cvs_native_artifacts.py','tools/export_d92_branch_support_features.py',
    'tools/evaluate_d92_branch_support_probe.py','tools/export_d92_branch_orbit_support_features.py',
    'tools/evaluate_d92_branch_orbit_ce_probe.py','tools/run_d92_branch_orbit_ce_probe.py',
    'tools/summarize_d92_branch_orbit_ce_probe.py','tools/summarize_d92_branch_support_probe.py',
    'configs/d92_branch_orbit_ce_frozen_20260929.json',
    'configs/d92_branch_orbit_ce_support_rx3_20260929.json','configs/d92_branch_orbit_ce_support_rx1_20260929.json']
transport.REMOTE=transport.REMOTE.replace('tools/run_d92_branch_support_probe.py','tools/run_d92_branch_orbit_ce_probe.py')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--spec',required=True)
    parser.add_argument('--resume-staged-commit')
    parser.parse_args()
    transport.main()
