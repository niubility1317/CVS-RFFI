"""Recover complete existing structured diagnostics; never retrain or read target scores."""
import argparse,base64,io,json,tarfile
from pathlib import Path
from experiments.cvs_phase1_stack.publish import ssh
from . import design as d

def recover(output):
 output=Path(output);output.mkdir(parents=True,exist_ok=False)
 script=r'''
from pathlib import Path
import io,tarfile,base64,json
root=Path('/home/szu2070436088/2510044040/CV-SincNet/runs/20261009-phase1-multi-state-action-manysig-m36-r02')
paths=[]
for folder in sorted(root.glob('audit-s*')):
 paths.extend(p for p in folder.iterdir() if p.suffix in ('.json','.jsonl','.csv'))
for folder in sorted(root.glob('*/source')):
 for name in ('epoch_metrics.jsonl','epoch_metrics.csv','auxiliary_cost.json','completion.json'):
  if (folder/name).is_file():paths.append(folder/name)
manifest=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size) for p in paths]
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as archive:
 for p in paths:archive.add(p,arcname=str(p.relative_to(root)),recursive=False)
print(json.dumps(dict(manifest=manifest,archive=base64.b64encode(buf.getvalue()).decode(),source=str(root))))
'''
 result=json.loads(ssh(script));blob=base64.b64decode(result.pop('archive'))
 archive=output/'source_diagnostics.tar.gz';archive.write_bytes(blob)
 with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as t:
  for m in t.getmembers():
   if not m.isfile() or not (output/m.name).resolve().is_relative_to(output.resolve()):raise ValueError('Unsafe recovered path')
  t.extractall(output)
 for item in result['manifest']:
  p=output/item['path']
  if p.stat().st_size!=item['bytes']:raise ValueError('Incomplete recovery')
  if p.suffix=='.json':json.loads(p.read_text(encoding='utf-8'))
  if p.suffix=='.jsonl':
   for line in p.read_text(encoding='utf-8').splitlines():json.loads(line)
 result.update(status='VERIFIED',target_scores_read=False,remote_modified=False,
  historical_gradient_caveat='Epoch sparse gradients wrongly divided by all auxiliary calls; do not interpret them as actual cosines. Original raw records retained.',
  local_archive=str(archive),files=len(result['manifest']))
 d.write(output/'recovery.json',result);print(json.dumps({k:v for k,v in result.items() if k!='manifest'}))
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);recover(p.parse_args().output)
