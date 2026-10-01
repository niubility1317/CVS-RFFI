"""Publish one isolated source-classifier packet export; no healthy release edits."""
import argparse
import importlib.util
import json
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import sys
import tempfile

from run_d92_ground_a_packet_export import ROOT,validate_spec

REMOTE_CODE_PATHS=(
    'code/cvsrffi/__init__.py','code/cvsrffi/d92_ground_classifier_a.py',
    'tools/export_d92_ground_classifier_a_packet.py','tools/run_d92_ground_a_packet_export.py')
LOCAL_DEPENDENCIES=(
    'tools/publish_d92_ground_a_packet_export.py','tools/publish_d92_branch_support_probe.py',
    'tools/run_d92_branch_support_probe.py')
PATHS=list(REMOTE_CODE_PATHS+LOCAL_DEPENDENCIES)


def release_paths(spec):
    validate_spec(spec)
    return PATHS+[spec['spec_path']]


def readiness(spec,root=ROOT):
    paths=release_paths(spec);missing=[p for p in paths if not (Path(root)/p).is_file()]
    return dict(status='SOURCE_BUNDLE_AVAILABLE_NOT_LAUNCHED' if not missing else 'INCOMPLETE_SOURCE_BUNDLE',
        paths=paths,missing=missing,launched=False,
        external_native_dependency=dict(code_ref=str(PurePosixPath(spec['code']['native_training_release'])/'code'),
            copied_into_bundle=False,scope='EXISTING_READ_ONLY_ORIGINAL_TRAINING_RELEASE',
            runtime_module_origins_verified=False))


def verify_bundle_imports(root=ROOT):
    """Import exactly four remote code files in an isolated local Python process."""
    with tempfile.TemporaryDirectory(prefix='d92-ground-a-packet-import-') as directory:
        bundle=Path(directory)
        for relative in REMOTE_CODE_PATHS:
            target=bundle/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path(root)/relative,target)
        program='''import json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
sys.path[:0]=[str(root/'tools'),str(root/'code')]
import run_d92_ground_a_packet_export as runner
import export_d92_ground_classifier_a_packet as exporter
import cvsrffi
import cvsrffi.d92_ground_classifier_a as head
assert 'baseline_origin_sat_view' not in sys.modules and 'cvsrffi.muse_ssdg' not in sys.modules
modules=[runner,exporter,cvsrffi,head]
assert all(Path(module.__file__).resolve().is_relative_to(root) for module in modules)
print(json.dumps(dict(status='VERIFIED',module_files=[Path(module.__file__).relative_to(root).as_posix() for module in modules])))
'''
        result=subprocess.run([sys.executable,'-I','-s','-c',program,str(bundle)],cwd=bundle,
            text=True,encoding='utf-8',capture_output=True,timeout=120)
        if result.returncode!=0:raise RuntimeError('Isolated packet source import failed: '+result.stderr[-4000:])
        value=json.loads(result.stdout)
        if value.get('status')!='VERIFIED' or set(value['module_files'])!=set(REMOTE_CODE_PATHS):
            raise ValueError('Isolated import did not resolve the exact remote source whitelist')
        return dict(value,scope='LOCAL_INTERPRETER_ISOLATED_IMPORT_NOT_REMOTE_RUNTIME_OR_EXPORT_VERIFICATION')


def configure_transport(transport,spec,root):
    transport.ROOT=Path(root);transport.validate_spec=validate_spec;transport.PATHS=PATHS.copy()
    gpu="used=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())\nif used>1024:raise RuntimeError('GPU0 occupied; leave existing process untouched')\n"
    if transport.REMOTE.count(gpu)!=1:raise RuntimeError('Existing CPU transport needs explicit reconciliation before mutation')
    if transport.REMOTE.count('tools/run_d92_branch_support_probe.py')!=1:raise RuntimeError('Existing supervisor dispatch changed')
    transport.REMOTE=transport.REMOTE.replace(gpu,'').replace('tools/run_d92_branch_support_probe.py','tools/run_d92_ground_a_packet_export.py')
    compile(transport.REMOTE.replace('CONFIG',repr(dict(release='/synthetic/release',archive='/synthetic/archive',root='/synthetic/run',
        python='/synthetic/python',spec=spec['spec_path'],commit='a'*40,owner='root',sha256='b'*64))),'packet-publication-transport','exec')


def dispatch(spec_path,root=ROOT,*,import_check=verify_bundle_imports):
    spec=json.loads((Path(root)/spec_path).read_text(encoding='utf-8'))
    if str(spec_path)!=spec['spec_path']:raise ValueError('Publication path differs from declared spec path')
    available=readiness(spec,root)
    if available['missing']:raise RuntimeError('Incomplete packet source bundle: '+', '.join(available['missing']))
    verified=import_check(root)
    if verified.get('status')!='VERIFIED':raise ValueError('Exact source whitelist import was not verified')
    location=importlib.util.spec_from_file_location('_ground_a_packet_transport',Path(root)/'tools/publish_d92_branch_support_probe.py')
    transport=importlib.util.module_from_spec(location);location.loader.exec_module(transport)
    configure_transport(transport,spec,root)
    # Existing publisher checks committed/pushed paths, archive/SCP/readback,
    # exclusive release/run destinations and launches only its one supervisor.
    transport.main()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',required=True)
    parser.add_argument('--resume-staged-commit');args=parser.parse_args();dispatch(args.spec)
