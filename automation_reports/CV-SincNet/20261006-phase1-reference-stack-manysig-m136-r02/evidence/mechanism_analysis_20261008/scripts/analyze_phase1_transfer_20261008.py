from pathlib import Path
import sys,json,subprocess
WT=Path('E:/type10-7/code/snapshots/daot_practical_three_20260918_wt')
sys.path.insert(0,str(WT))
from experiments.cvs_phase1_stack.publish import ssh
OUT=Path('E:/type10-7/local_artifacts/phase1_transfer_analysis_20261008')
OUT.mkdir(parents=True,exist_ok=True)
SCRIPT=r'''
from pathlib import Path
import json,time,re
P=Path('/home/szu2070436088/2510044040/CV-SincNet')
runs=['20261004-phase1-reference-overlay-r1-manysig-m16-r01','20261006-phase1-reference-stack-manysig-m136-r02']
def load(p):return json.loads(p.read_text()) if p.is_file() else None
result={'read_at':time.time(),'runs':{},'rows':[]}
for run in runs:
 root=P/'runs'/run
 meta={p.name:load(p) for p in root.glob('*.json') if any(s in p.name for s in ['selection','freeze','frozen','queue_state','failure','completion','resolved_matrix'])}
 result['runs'][run]=meta
 configs=sorted((root/'configs').glob('source-*.json'))
 for cp in configs:
  c=load(cp);s=Path(c['output_root']);rid=c['row_id'];e=s/'epoch_metrics.jsonl'
  if not e.is_file():continue
  item={'run_id':run,'row_id':rid,'config':c,'source_path':str(s),'epochs':[json.loads(x) for x in e.read_text().splitlines() if x.strip()],
        'completion':load(s/'completion.json'),'resolved':load(s/'resolved_config.json'),'args':load(s/'resolved_native_args.json'),
        'files':{p.name:p.stat().st_size for p in s.iterdir() if p.is_file()}}
  log=P/'logs'/c['run_id']/('source-'+rid+'.log')
  if not log.is_file():log=P/'logs'/run/('source-'+rid+'.log')
  if log.is_file():
   lines=log.read_text(errors='replace').splitlines()
   pats={'error':r'Traceback \(most recent|out of memory|CUDA error|FloatingPointError|NONFINITE|Killed','warning':r'Warning|warning','resume':r'early.stop|resum|restart'}
   item['stdout']={'path':str(log),'lines':len(lines),'matches':{k:[{'line':i+1,'text':v[:600]} for i,v in enumerate(lines) if re.search(p,v)] for k,p in pats.items()}}
  result['rows'].append(item)
print(json.dumps(result))
'''
if __name__=='__main__':
 data=json.loads(ssh(SCRIPT))
 (OUT/'metadata_epochs.json').write_text(json.dumps(data,ensure_ascii=False)+'\n',encoding='utf-8')
 print(json.dumps({'rows':len(data['rows']),'complete':sum(bool(x['completion']) for x in data['rows']),'stages':{s:sum(x['row_id'].startswith(s) and bool(x['completion']) for x in data['rows']) for s in ['r2','r3','r4','r5','r6']},'first':{'id':data['rows'][0]['row_id'],'files':data['rows'][0]['files'],'epoch_keys':list(data['rows'][0]['epochs'][0])},'run_keys':{k:list(v) for k,v in data['runs'].items()}}))
