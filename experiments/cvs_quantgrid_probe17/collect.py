"""Read fixed completed statistics only;never IQ/cache/model files."""
import argparse,json,os,subprocess,tarfile
from pathlib import Path
from . import design as d
FILES=('resolved_config.json','execution.json','source_contract.json','data_roles.json','synthetic_noquery_smoke.json','progress.json','packets.jsonl.gz','summary.json','completion.json')
def fixed_entries():
 return [((d.BASE/n).as_posix(),n) for n in FILES]+[(d.BASE.as_posix()+'-launch/'+n,n) for n in ('launch.json','worker.stdout.log','worker.stderr.log')]
def collect(out):
 out=Path(out);archive=out.with_name(out.name+'.fetch.partial.tar.gz')
 if out.exists() or archive.exists():raise FileExistsError('Reconcile existing fetch,no overwrites')
 entries=fixed_entries()
 code="import json,sys,tarfile\nfrom pathlib import Path\nB=Path("+repr(str(d.BASE).replace(chr(92),'/'))+")\nif json.loads((B/'completion.json').read_text())['status']!='ARTIFACTS_COMPLETE':raise ValueError('Wait exact original source process;no relaunch')\nentries="+repr(entries)+"\nfor p,n in entries:\n if not Path(p).is_file():raise FileNotFoundError(p)\nwith tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:\n for p,n in entries:t.add(p,arcname=n,recursive=False)\n"
 env=dict(os.environ,SystemRoot='C:/Windows',windir='C:/Windows',PATHEXT='.COM;.EXE;.BAT;.CMD');cmd=['C:/Program Files/Git/usr/bin/ssh.exe','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','-T','N607','python3 -'];r=subprocess.run(cmd,input=code.encode(),capture_output=True,env=env,timeout=50)
 if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
 archive.write_bytes(r.stdout);allowed={n for p,n in entries}
 with tarfile.open(archive) as t:
  members=t.getmembers()
  if len(members)!=len(allowed) or {m.name for m in members}!=allowed or any(not m.isfile() or not (out/m.name).resolve().is_relative_to(out.resolve()) for m in members):raise ValueError('Only fixed metadata/source statistics,0IQ/cache/CP')
  out.mkdir();t.extractall(out)
 print(json.dumps(dict(status='FETCHED',files=len(entries),IQ=0,CP=0,NN=0)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();collect(a.output)
