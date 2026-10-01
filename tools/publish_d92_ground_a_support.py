"""Publish the independent CPU Ground A support supplement, without native model code."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from run_d92_ground_a_support import ROOT,validate_spec

REMOTE_CODE_PATHS=(
    'code/cvsrffi/__init__.py','code/cvsrffi/d92_ground_classifier_a.py',
    'tools/export_d92_ground_classifier_a_packet.py','tools/score_d92_ground_a_support.py',
    'tools/export_d92_branch_support_features.py','tools/cvs_native_artifacts.py','tools/run_d92_ground_a_support.py')
LOCAL_DEPENDENCIES=(
    'tools/publish_d92_ground_a_support.py','tools/publish_d92_branch_support_probe.py','tools/run_d92_branch_support_probe.py')
PATHS=list(REMOTE_CODE_PATHS+LOCAL_DEPENDENCIES)


def release_paths(spec):
    validate_spec(spec)
    return PATHS+[spec['spec_path']]


def readiness(spec,root=ROOT):
    paths=release_paths(spec);missing=[path for path in paths if not (Path(root)/path).is_file()]
    return dict(status='SOURCE_BUNDLE_AVAILABLE_NOT_LAUNCHED' if not missing else 'INCOMPLETE_SOURCE_BUNDLE',
        paths=paths,missing=missing,launched=False,native_training_release_required=False)


def verify_bundle_imports(root=ROOT):
    """Import exact seven-file runtime bundle; no source packet/cache is opened."""
    with tempfile.TemporaryDirectory(prefix='d92-ground-a-support-import-') as directory:
        bundle=Path(directory)
        for relative in REMOTE_CODE_PATHS:
            target=bundle/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(Path(root)/relative,target)
        program='''import json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
sys.path[:0]=[str(root/'tools'),str(root/'code')]
import run_d92_ground_a_support as runner
import score_d92_ground_a_support as scorer
import export_d92_ground_classifier_a_packet as exporter
import export_d92_branch_support_features as cache
import cvs_native_artifacts as artifacts
import cvsrffi
import cvsrffi.d92_ground_classifier_a as head
modules=[runner,scorer,exporter,cache,artifacts,cvsrffi,head]
assert all(Path(module.__file__).resolve().is_relative_to(root) for module in modules)
assert 'baseline_origin_sat_view' not in sys.modules and 'cvsrffi.muse_ssdg' not in sys.modules
assert 'cvsrffi.checkpoint_loading' not in sys.modules and 'cvsrffi.identity_only_forward' not in sys.modules
print(json.dumps(dict(status='VERIFIED',module_files=[Path(module.__file__).relative_to(root).as_posix() for module in modules])))
'''
        result=subprocess.run([sys.executable,'-I','-s','-c',program,str(bundle)],cwd=bundle,
            text=True,encoding='utf-8',capture_output=True,timeout=120)
        if result.returncode!=0:raise RuntimeError('Isolated support supplement import failed: '+result.stderr[-4000:])
        value=json.loads(result.stdout)
        if value.get('status')!='VERIFIED' or set(value['module_files'])!=set(REMOTE_CODE_PATHS):
            raise ValueError('Isolated import did not resolve exact remote source whitelist')
        return dict(value,scope='LOCAL_INTERPRETER_ISOLATED_IMPORT_NOT_REMOTE_RUNTIME_OR_SCORING_VERIFICATION')


def configure_transport(transport,spec,root):
    transport.ROOT=Path(root);transport.validate_spec=validate_spec;transport.PATHS=PATHS.copy()
    gpu="used=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())\nif used>1024:raise RuntimeError('GPU0 occupied; leave existing process untouched')\n"
    if transport.REMOTE.count(gpu)!=1:raise RuntimeError('Existing CPU transport needs explicit reconciliation before mutation')
    if transport.REMOTE.count('tools/run_d92_branch_support_probe.py')!=1:raise RuntimeError('Existing supervisor dispatch changed')
    transport.REMOTE=transport.REMOTE.replace(gpu,'').replace('tools/run_d92_branch_support_probe.py','tools/run_d92_ground_a_support.py')
    compile(transport.REMOTE.replace('CONFIG',repr(dict(release='/synthetic/release',archive='/synthetic/archive',root='/synthetic/run',
        python='/synthetic/python',spec=spec['spec_path'],commit='a'*40,owner='root',sha256='b'*64))),'ground-a-support-publication','exec')


def dispatch(spec_path,root=ROOT,*,import_check=verify_bundle_imports):
    spec=json.loads((Path(root)/spec_path).read_text(encoding='utf-8'))
    if str(spec_path)!=spec['spec_path']:raise ValueError('Publication path differs from declared spec path')
    available=readiness(spec,root)
    if available['missing']:raise RuntimeError('Incomplete support source bundle: '+', '.join(available['missing']))
    verified=import_check(root)
    if verified.get('status')!='VERIFIED':raise ValueError('Exact source whitelist import was not verified')
    location=importlib.util.spec_from_file_location('_ground_a_support_transport',Path(root)/'tools/publish_d92_branch_support_probe.py')
    transport=importlib.util.module_from_spec(location);location.loader.exec_module(transport)
    configure_transport(transport,spec,root)
    transport.main()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',required=True)
    parser.add_argument('--resume-staged-commit');args=parser.parse_args();dispatch(args.spec)
