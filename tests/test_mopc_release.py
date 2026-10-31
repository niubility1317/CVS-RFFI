import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
from experiments.pamf_phase2_practical.mopc_release import paths


def test_compact_release_contains_source_augmentation_imports(tmp_path):
    repo=Path(__file__).resolve().parents[1];root=tmp_path/'release'
    # Reproduce the original archive boundary in a fresh interpreter, without cwd imports.
    (root/'experiments/pamf_phase2_practical').mkdir(parents=True)
    command=[sys.executable,'-I','-c',"import sys;sys.path[:0]=[sys.argv[1],sys.argv[1]+'/code'];from experiments.practical_full_2x2.compat import numerical_policy;from experiments.practical_full_2x2.cache import Reader,service,configuration;from leo_practical.channel import Config;numerical_policy()",str(root)]
    missing=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True)
    assert missing.returncode and 'experiments.practical_full_2x2' in missing.stderr
    for rel in paths('test'):
        if rel.startswith(('configs/','automation_reports/','experiments/pamf_phase2_practical')):continue
        source=repo/rel;target=root/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if source.is_dir():shutil.copytree(source,target)
        else:shutil.copy2(source,target)
    fixed=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True)
    assert fixed.returncode==0,fixed.stderr
