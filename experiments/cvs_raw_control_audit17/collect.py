"""Read only fixed historical source artifacts,including untrained init states."""
import argparse,os,subprocess,tarfile
from pathlib import Path
from . import design as d

def collect(out):
 out=Path(out);archive=out.with_name(out.name+'.fetch.partial.tar.gz')
 if out.exists() or archive.exists():raise FileExistsError('Reconcile existing fetch before retry,no overwrite')
 entries=[]
 for r in d.sources():
  for n in d.FILES:entries.append((r['source']+'/'+n,r['key']+'/'+n))
  for v in d.VIEWS:entries.append((r['source']+'/source_readouts/'+v+'.json',r['key']+'/source_readouts/'+v+'.json'))
  entries.append((d.PROJECT+'/runs/'+r['run_id']+'/logs/'+r['arm']+'-RX1-s2026092701.stdout.log',r['key']+'/stdout.log'))
 code="import sys,tarfile\nfrom pathlib import Path\nentries="+repr(entries)+"\nfor p,n in entries:\n if not Path(p).is_file():raise FileNotFoundError(p)\nwith tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:\n for p,n in entries:tar.add(p,arcname=n,recursive=False)\n"
 compile(code,'fixed_readonly_collection','exec');env=dict(os.environ,SystemRoot='C:/Windows',windir='C:/Windows',PATHEXT='.COM;.EXE;.BAT;.CMD')
 cmd=['C:/Program Files/Git/usr/bin/ssh.exe','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','-T','N607','python3 -'];r=subprocess.run(cmd,input=code.encode('utf8'),capture_output=True,env=env,timeout=50)
 if r.returncode:raise RuntimeError(r.stderr.decode('utf8',errors='replace'))
 archive.write_bytes(r.stdout);allowed={n for p,n in entries}
 with tarfile.open(archive) as tar:
  m=tar.getmembers()
  if len(m)!=len(allowed) or {x.name for x in m}!=allowed or any(not x.isfile() or not (out/x.name).resolve().is_relative_to(out.resolve()) for x in m):raise ValueError('Only fixed source records and initial_smoke.pt;noIQ/trainedCP')
  out.mkdir();tar.extractall(out)
 print('FETCHED '+str(len(entries))+' source artifacts;5untrainedinitialstates;0IQ/trainedCP')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();collect(a.output)
