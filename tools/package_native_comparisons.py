"""Package only committed runtime code and these two comparison registrations."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

RUNS=('20260930-phase1-native-baselines-practical-manysig-m5-r01','20260930-phase12-native-baselines-practical-m5-r01')

def package(root,output):
    root=Path(root);output=Path(output)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    names=subprocess.check_output(['git','ls-files'],cwd=root,text=True).splitlines()
    selected=[]
    for name in names:
        include=(name.startswith('comparison_suite/') and name.endswith('.py')
            or name.startswith('baselines/common/') and name.endswith('.py')
            or name.startswith('paper_reproduction/') and name.endswith('.py')
            or name.startswith('code/leo_practical/') and Path(name).suffix in {'.py','.json'}
            or name=='code/dataset_wisig.py'
            or any(name.startswith('automation_reports/CV-SincNet/'+run+'/') for run in RUNS)
            or name in {'baselines/__init__.py','docs/CVS_NATIVE_COMPARISON_DESIGN_20260930.md'})
        if include:selected.append(name)
    if not any(name.endswith('/launch_spec.json') for name in selected): raise ValueError('Preregistration not committed')
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): raise FileExistsError(output)
    with tarfile.open(output,'w:gz') as archive:
        for name in selected:
            # Git blob, not an uncommitted working-file overlay.
            blob=subprocess.check_output(['git','show',commit+':'+name],cwd=root)
            info=tarfile.TarInfo(name);info.size=len(blob);info.mode=0o644
            archive.addfile(info,io.BytesIO(blob))
        blob=(commit+'\n').encode();info=tarfile.TarInfo('release_commit.txt');info.size=len(blob)
        archive.addfile(info,io.BytesIO(blob))
    metadata=dict(commit=commit,archive=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),files=len(selected))
    output.with_suffix(output.suffix+'.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(package(a.root,a.output)))
