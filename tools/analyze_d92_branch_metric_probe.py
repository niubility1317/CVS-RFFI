"""Analyze the complete metric support run through the existing artifact route."""
import argparse
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('_branch_metric_analysis_transport',
    Path(__file__).with_name('analyze_d92_branch_support_probe.py'))
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
transport.PATHS = ['code', 'tools/summarize_d92_branch_metric_probe.py',
    'tools/summarize_d92_branch_interaction_probe.py', 'tools/summarize_d92_branch_support_probe.py',
    'tools/evaluate_d92_branch_metric_probe.py', 'tools/evaluate_d92_branch_support_probe.py',
    'tools/export_d92_branch_support_features.py', 'tools/cvs_native_artifacts.py']
transport.REMOTE = transport.REMOTE.replace('tools/summarize_d92_branch_support_probe.py',
    'tools/summarize_d92_branch_metric_probe.py')

if __name__ == '__main__':
    # Do not inherit the old transport's defaults for a different run.
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--analysis-release', required=True)
    parser.parse_args()
    transport.main()
