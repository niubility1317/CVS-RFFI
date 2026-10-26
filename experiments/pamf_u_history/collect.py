"""Read back this pilot and existing source-only predictions, without remote writes."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from experiments.cric_dura_comparison.publish import CONNECTION, ssh
from .source import RUN, PREFIX_RUN
from .vat_source import RUN as VAT_RUN


def collect(runtime_file, output):
    rt=json.loads(Path(runtime_file).read_text(encoding='utf-8'))
    if rt['run_id'] not in (RUN,VAT_RUN):raise ValueError('Only these registered source pilots can be collected')
    remote=r'''
import json,time
from pathlib import Path
rt=RUNTIME
run=Path(rt['run_root']);project=Path(rt['project']);files=[]
def add(path,key,live=False):
 if path.is_file():files.append(dict(remote=str(path),local=key,bytes=path.stat().st_size,live=live))
for p in run.rglob('*'):
 if not p.is_file():continue
 rel=p.relative_to(run)
 if p.suffix in ('.json','.jsonl','.csv','.log','.gz'):
  add(p,'pilot/'+rel.as_posix(),live=not (p.parent/'complete.json').exists())
 elif p.name=='source_V_predictions.npz' and (p.parent/'complete.json').exists():add(p,'pilot/'+rel.as_posix())
prefix=project/'runs'/PREFIX/'prefixes/F0-A1B1-SIG_TIME-F21-s2026092701/segment20'
for p in prefix.iterdir():
 if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.log','.gz'):add(p,'prefix20/'+p.name)
old=project/'runs/20261025-diagnostic-PAMF-RestoredPL-sourceShift-m9-r01'
for name in ('complete.json','source_V_predictions.npz'):add(old/name,'existing_PL_source/'+name)
print(json.dumps(dict(observed_at=time.time(),files=files)))
'''.replace('RUNTIME',repr(rt)).replace('PREFIX',repr(PREFIX_RUN))
    manifest=ssh(remote);root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True)
    for row in manifest['files']:
        path=(root/row['local']).resolve()
        if root not in path.parents:raise ValueError('Readback path escapes output folder')
        path.parent.mkdir(parents=True,exist_ok=True)
        # Reuse already complete immutable evidence; live files are always refreshed.
        if path.exists() and not row['live'] and path.stat().st_size==row['bytes']:
            row['transport']='REUSED_EXISTING'
            continue
        subprocess.run(['scp',*CONNECTION,'N607:'+row['remote'],str(path)],check=True,timeout=60,
                       stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        row.update(transport='COPIED',local_bytes=path.stat().st_size)
    manifest['readback_finished_at']=time.time()
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(files=len(manifest['files']),copied=sum(r['transport']=='COPIED' for r in manifest['files']),
                          existing_control_retraining=False,output=str(root))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();collect(a.runtime,a.output)
