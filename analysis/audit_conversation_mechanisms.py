"""Read-only activation audit across the 13 A1 run roots in this conversation."""
import gzip,json,subprocess
from pathlib import Path
REMOTE=r'''
import csv,datetime,io,json,math,os,re,statistics
from pathlib import Path
p=Path('/home/szu2070436088/2510044040/CV-SincNet')
names=['a1_ecrs_cross_rx_s392005_20260909_r1','a1_extended_s392005_20260910_r1',
 'a1_fast_matched_core90_s392005_20260908_r1','a1_fast_matched_core90_s392005_20260908_r2','a1_fast_matched_core90_s392005_20260908_r3',
 'a1_fast_selected_adv3b02_s392005_20260908_r1','a1_fast_v2_s392005_20260909_r1',
 'a1_mechanism_periodic_s392005_20260910_r1','a1_mechanism_screen_s392005_20260910_r2',
 'a1_r3_budget_s392005_20260909_r1','a1_r3_budget_s392005_20260909_r2',
 'a1_r3_clean_cross_rx_s392005_20260909_r1','a1_r3_scratch_s392005_20260908_r1']
out={'captured_at':datetime.datetime.now().isoformat(),'runs':{}}
pattern=re.compile(r'ecrs|r3|recon|swap|response|fisher|nmfdu|physical_gate|daot|orbit|tangent|ema|coverage|balanced|concat_sat|group_ce|identity_coupling|identity_domain|loss_cons|lambda_cons|nonfinite|loss_sat|val_tx_acc|source_val_sat|^lr')
for run in names:
 root=p/'runs'/run
 state=json.loads((root/'pipeline_state.json').read_text()) if (root/'pipeline_state.json').exists() else {}
 matrix=json.loads((root/'effective_matrix.json').read_text()) if (root/'effective_matrix.json').exists() else {}
 rows={}
 rownames=set(state.get('rows',{}))|{x['id'] for x in matrix.get('rows',[])}|{x.parent.name for x in root.glob('*/metrics_epoch.jsonl')}
 for name in sorted(rownames):
  folder=root/name; mf=folder/'metrics_epoch.jsonl'; cf=folder/'metrics_epoch.csv'
  records=[];parse_errors=[]
  for i,line in enumerate(mf.read_text().splitlines() if mf.exists() else []):
   if not line.strip():continue
   try:records.append(json.loads(line))
   except json.JSONDecodeError as e:parse_errors.append({'line':i+1,'error':str(e)})
  log=p/'logs'/run/(name+'.train.log'); txt=log.read_text(encoding='utf-8') if log.exists() else '';lines=txt.splitlines()
  csvrows=list(csv.DictReader(io.StringIO(cf.read_text()))) if cf.exists() else []
  proc=json.loads((folder/'process.json').read_text()) if (folder/'process.json').exists() else {}
  launch=json.loads((folder/'launch_config.json').read_text()) if (folder/'launch_config.json').exists() else {}
  itemstate=state.get('rows',{}).get(name,{})
  live=Path('/proc')/str(itemstate.get('pid',0)); liveargv=[];livecwd=None
  if live.exists():
   try:
    liveargv=live.joinpath('cmdline').read_bytes().decode().split('\0')[:-1];livecwd=os.readlink(live/'cwd')
   except FileNotFoundError:pass
  keys=sorted(set().union(*(x.keys() for x in records))) if records else []
  stats={}
  for key in keys:
   if not pattern.search(key):continue
   vals=[(int(x['epoch']),float(x[key])) for x in records if isinstance(x.get(key),(float,int)) and not isinstance(x.get(key),bool)]
   finite=[(e,v) for e,v in vals if math.isfinite(v)];nz=[(e,v) for e,v in finite if v!=0]
   if finite:stats[key]={'count':len(vals),'finite':len(finite),'nonzero_epochs':len(nz),'first_nonzero':nz[0][0] if nz else None,
    'min':min(v for e,v in finite),'max':max(v for e,v in finite),'mean':statistics.fmean(v for e,v in finite),'last':finite[-1][1]}
  scores=[]
  for sp in sorted(list(folder.glob('target_prediction/score.json'))+list(folder.glob('target_epochs/E*/score.json'))):
   scores.append({'path':str(sp),'score':json.loads(sp.read_text()),'scope':json.loads(sp.with_name('evaluation_scope.json').read_text()) if sp.with_name('evaluation_scope.json').exists() else None})
  matching=[x for x in matrix.get('rows',[]) if x['id']==name]
  rows[name]={'state':itemstate,'matrix':matching,'base_options':matrix.get('core90_options',{}),'process':proc,'launch':launch,
   'live_argv':liveargv,'live_cwd':livecwd,'jsonl_records':len(records),'csv_records':len(csvrows),'csv_bad_widths':sum(None in x for x in csvrows),
   'epochs':[x['epoch'] for x in records],'parse_errors':parse_errors,'stdout_lines':len(lines),
   'config_lines':sorted(set(x for x in lines if x.startswith('[CONFIG-') or 'init=scratch' in x or '[BATCH-GEOM]' in x)),
   'errors':[x for x in lines if re.search(r'Traceback|RuntimeError:|CUDA out of memory|RC4_SYSTEMIC_NONFINITE',x)],
   'tail':lines[-4:],'fields':keys,'stats':stats,'scores':scores,
   'last':{k:v for k,v in records[-1].items() if pattern.search(k)} if records else {},
   'final_checkpoint_exists':(folder/'final_ssdg.pth').exists()}
  del records,csvrows
 out['runs'][run]={'state':state,'rows':rows,'stop_files':{s.name:json.loads(s.read_text()) for s in root.glob('*stop*result*.json')}}
print(json.dumps(out,separators=(',',':')))
'''
compile(REMOTE,'<read-only-mechanism-audit>','exec')
r=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],input=REMOTE,text=True,encoding='utf-8',capture_output=True,timeout=60)
if r.returncode:raise RuntimeError(r.stderr)
d=json.loads(r.stdout)
out=Path(__file__).parent/'mechanism_activation_audit_20260910';out.mkdir(exist_ok=True)
with gzip.open(out/'evidence.json.gz','wt',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,separators=(',',':'))
print(d['captured_at'])
for run,data in d['runs'].items():
 print(run,[(n,x['state'].get('status'),x['jsonl_records'],bool(x['live_argv']),len(x['scores'])) for n,x in data['rows'].items()])
print('epochs',sum(x['jsonl_records'] for r in d['runs'].values() for x in r['rows'].values()))
