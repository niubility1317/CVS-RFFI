"""Fetch fixed completed source records,never IQ or weights;stdlib only."""
import argparse,os,subprocess,tarfile
from pathlib import Path
from . import design as d

def harvest(output):
 out=Path(output);archive=out.with_name(out.name+'.fetch.partial.tar.gz')
 if out.exists() or archive.exists():raise FileExistsError('Reconcile possibly completed fetch before retry')
 parent=out.parent;parent.mkdir(parents=True,exist_ok=True)
 B=d.PROJECT+'/runs/'+d.RUN;R=d.PROJECT+'/releases/'+d.RELEASE;row=d.rows()[0]['row_id']
 roots=['execution.json','progress.json','completion.json','checkpoint_noquery_smoke.json'];names=['resolved_config.json','source_contract.json','source_labels.json','data_roles.json','actual_modules.json','receivers.json','predictions.jsonl.gz','physics.jsonl.gz','summary.json'];release=['preflight.json','submit.json','dispatcher.log']
 entries=[(B+'/'+n,n) for n in roots]+[(B+'/'+row+'/'+n,row+'/'+n) for n in names]+[(R+'/'+n,n) for n in release]
 source="import sys,tarfile\nfrom pathlib import Path\nB=Path("+repr(B)+")\nif not (B/'completion.json').exists() or (B/'failure.json').exists():raise ValueError('Full source completion first')\nentries="+repr(entries)+"\nfor p,n in entries:\n if not Path(p).is_file():raise FileNotFoundError(p)\nwith tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:\n for p,n in entries:tar.add(p,arcname=n,recursive=False)\n"
 env=dict(os.environ,SystemRoot='C:/Windows',windir='C:/Windows',PATHEXT='.COM;.EXE;.BAT;.CMD')
 cmd=['C:/Program Files/Git/usr/bin/ssh.exe','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','-T','N607','python3 -']
 r=subprocess.run(cmd,input=source.encode('utf8'),capture_output=True,env=env,timeout=50)
 if r.returncode:raise RuntimeError(r.stderr.decode('utf8',errors='replace'))
 archive.write_bytes(r.stdout);allowed={n for p,n in entries}
 with tarfile.open(archive) as tar:
  members=tar.getmembers()
  if {m.name for m in members}!=allowed or len(members)!=len(allowed) or any(not m.isfile() or not (out/m.name).resolve().is_relative_to(out.resolve()) for m in members):raise ValueError('Only fixed source metadata,not IQ/weights')
  out.mkdir();tar.extractall(out)
 print('FETCHED '+str(len(entries))+' fixed source files;0IQ/weights')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();harvest(a.output)
