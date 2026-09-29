"""Complete orbit support diagnostic analysis, preserving exclusive outputs."""
import argparse
import importlib.util
from pathlib import Path

definition=importlib.util.spec_from_file_location('_branch_orbit_analysis',
    Path(__file__).with_name('analyze_d92_branch_support_probe.py'))
transport=importlib.util.module_from_spec(definition)
definition.loader.exec_module(transport)
transport.PATHS=['code','tools/cvs_native_artifacts.py','tools/export_d92_branch_support_features.py',
    'tools/evaluate_d92_branch_support_probe.py','tools/export_d92_branch_orbit_support_features.py',
    'tools/evaluate_d92_branch_orbit_ce_probe.py','tools/summarize_d92_branch_support_probe.py',
    'tools/summarize_d92_branch_orbit_ce_probe.py']
transport.REMOTE=transport.REMOTE.replace('tools/summarize_d92_branch_support_probe.py',
    'tools/summarize_d92_branch_orbit_ce_probe.py')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--spec',required=True)
    parser.add_argument('--analysis-release',required=True)
    parser.parse_args()
    transport.main()
