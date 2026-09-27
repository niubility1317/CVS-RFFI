"""Bounded synthetic CPU/CUDA development acceptance; never opens a dataset/checkpoint."""
import argparse
import difflib
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import torch
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA acceptance requested but CUDA unavailable')
    args.output.mkdir(parents=True,exist_ok=True)
    modules={}
    for name in ('cvsrffi.xuc_fusion.ir_types','cvsrffi.xuc_fusion.ir_head','cvsrffi.xuc_fusion.ir_metric',
                 'cvsrffi.xuc_fusion.ir_cg','cvsrffi.xuc_fusion.ir_solver','cvsrffi.xuc_fusion.ir_objective',
                 'cvsrffi.game_tracking.solvers','cvsrffi.xuc_fusion.runtime','SSDG.train_ssdg','model_dual_cvsincnet'):
        mod=importlib.import_module(name); path=Path(mod.__file__).resolve()
        if not path.is_relative_to((ROOT/'code').resolve()):
            raise RuntimeError(f'import source mismatch: {name}: {path}')
        modules[name]=dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    def git(*argv):
        p=subprocess.run(['git',*argv],cwd=ROOT,text=True,encoding='utf-8',capture_output=True,check=True)
        return p.stdout.strip()
    environment=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,
                     cudnn=torch.backends.cudnn.version(),device=args.device,
                     cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
                     deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                     device_name=torch.cuda.get_device_name(0) if args.device=='cuda' else platform.processor(),
                     git_head=git('rev-parse','HEAD'),base='acf0a6407c4a2cd114851e9f60d0cd346e4b2c77',
                     modules=modules,dataset_access=False,checkpoint_access=False,target_access=False,
                     synthetic_seed_policy='explicit per-test fixture seeds; no scientific result')
    (args.output/'environment.json').write_text(json.dumps(environment,indent=2)+'\n',encoding='utf-8')
    baseline=json.loads((ROOT/'acceptance/baseline_manifest.json').read_text(encoding='utf-8'))
    original={item['path']:item for item in baseline['files']}
    differences=[];execution_files={}
    for path in sorted((ROOT/'code').rglob('*.py')):
        rel=path.relative_to(ROOT).as_posix();body=path.read_text(encoding='utf-8')
        execution_files[rel]=hashlib.sha256(path.read_bytes()).hexdigest()
        old=(ROOT.parents[1]/original[rel]['source']).read_text(encoding='utf-8') if rel in original else ''
        if old!=body:
            differences.extend(difflib.unified_diff(old.splitlines(True),body.splitlines(True),fromfile='frozen/'+rel,tofile=rel))
    (args.output/'workspace.diff').write_text(''.join(differences),encoding='utf-8')
    (args.output/'execution_files.json').write_text(json.dumps(execution_files,indent=2)+'\n',encoding='utf-8')
    tests=sorted(str(p.relative_to(ROOT)) for p in (ROOT/'tests').glob('test_ir_*.py'))
    junit=(args.output/'pytest.xml').resolve()
    command=[sys.executable,'-X','utf8','-m','pytest',*tests,'-q','-o','addopts=',f'--junitxml={junit}']
    env=dict(os.environ,IR_ACCEPTANCE_DEVICE=args.device,PYTHONUTF8='1')
    start=time.perf_counter()
    run=subprocess.run(command,cwd=ROOT,env=env,text=True,encoding='utf-8',errors='strict',capture_output=True)
    (args.output/'pytest.log').write_text(run.stdout+run.stderr,encoding='utf-8')
    suite=ET.parse(junit).getroot()
    counts={k:sum(int(s.attrib.get(k,0)) for s in suite.findall('testsuite')) for k in ('tests','failures','errors','skipped')}
    result=dict(status='VERIFIED' if run.returncode==0 else 'FAILED',command=command,exit_code=run.returncode,
                elapsed_seconds=time.perf_counter()-start,device=args.device,counts=counts,
                performance_claim=False,training_launch=False)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    return run.returncode


if __name__=='__main__':
    raise SystemExit(main())
