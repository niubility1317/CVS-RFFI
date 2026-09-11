import json,math,re,statistics,csv,argparse
from pathlib import Path
from collections import Counter,defaultdict
parser=argparse.ArgumentParser(description='Parse complete C2/S4 and control logs'); parser.add_argument('--input',required=True); parser.add_argument('--target-scores',required=True); args=parser.parse_args(); root=Path(args.input)
out=root/'analysis';out.mkdir(exist_ok=True)
rows=['C2','S4','B0','B1','C1','S3','B8']
target=json.loads(Path(args.target_scores).read_text(encoding='utf-8'))['scores']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def stats(xs):
 return dict(min=min(xs),mean=statistics.mean(xs),median=statistics.median(xs),max=max(xs)) if xs else None
result={}; audits_csv=[];epochs_csv=[]
for row in rows:
 run=root/row;files={};jsonl={}
 for p in sorted(run.rglob('*')):
  if p.suffix=='.jsonl':
   data=[json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
   jsonl[p.name]=data;files[str(p.relative_to(run))]=len(data)
  elif p.suffix=='.json':read(p);files[str(p.relative_to(run))]=1
 epochs=jsonl['logs.jsonl'];actions=jsonl['game_actions.jsonl'];audits=jsonl.get('game_audit.jsonl',[]);curr=jsonl.get('curriculum_events.jsonl',[])
 assert [e['epoch'] for e in epochs]==list(range(1,201))
 assert [x['step'] for x in actions]==list(range(9800))
 assert all(x['accepted'] for x in actions)
 stdout=(run/'stdout.log').read_text(encoding='utf-8')
 errors=[line for line in stdout.splitlines() if re.search(r'Traceback|CUDA out of memory|\bKilled\b|\bRuntimeError\b|\bnan\b|\binf\b',line,re.I)]
 warnings=[line for line in stdout.splitlines() if re.search(r'warning|resume',line,re.I)]
 st={}
 for name,lo,hi in [('E1_40',1,40),('E41_79',41,79),('E80_90',80,90),('E91_130',91,130),('E131_200',131,200)]:
  es=[e for e in epochs if lo<=e['epoch']<=hi];ac=[a for a in actions if lo<=a['epoch']<=hi]
  st[name]=dict(epochs=len(es),mean_loss=statistics.mean(e['mean_loss'] for e in es),
    terms={k:statistics.mean(e['terms'].get(k,0) for e in es) for k in sorted(set().union(*(e['terms'] for e in es)))},
    actions=dict(Counter(x['action'] for x in ac)),corrections=sum(x['field_evaluations']>1 for x in ac),
    head_steps=sum(x['committed_head_steps'] for x in ac),satellite_scenario_steps=dict(Counter(x['satellite_scenario'] for x in ac)),
    sample_count=sum(x['sample_count'] for x in ac),satellite_count=sum(x['satellite_count'] for x in ac),
    pseudo_selected=sum(x['pseudo_selected'] for x in ac),unlabeled_samples=sum(x['unlabeled_samples'] for x in ac),
    epoch_seconds=statistics.mean(e['epoch_seconds'] for e in es))
 for e in epochs:
  epochs_csv.append(dict(row=row,epoch=e['epoch'],mean_loss=e['mean_loss'],epoch_seconds=e['epoch_seconds'],**e['terms']))
 for a in audits:
  g=a['gradient']['online_recovered']
  audits_csv.append(dict(row=row,step=a['step'],epoch=a['epoch'],G_lag=a['G_lag'],S_domain=a.get('S_domain'),ce_gap_raw=a['ce_gap_raw'],
    online_ce=a['online']['ce'],recovered_ce=a['recovered']['ce'],online_acc=a['online']['accuracy'],recovered_acc=a['recovered']['accuracy'],
    direction_imbalance=a['direction_imbalance'],gradient_cosine=g['cosine'],gradient_norm_ratio=g['norm_ratio'],
    identity=a['identity'],margin=a['margin'],consistency=a['consistency'],
    rx_linear_acc=a.get('independent',{}).get('rx',{}).get('linear',{}).get('accuracy'),rx_mlp_acc=a.get('independent',{}).get('rx',{}).get('mlp',{}).get('accuracy')))
 correction_epochs=Counter(x['epoch'] for x in actions if x['field_evaluations']>1)
 summary=dict(parsed_files=files,stdout_lines=len(stdout.splitlines()),errors=errors,warnings=warnings,
   epochs=200,steps=len(actions),actions=dict(Counter(x['action'] for x in actions)),reasons=dict(Counter(x['reason'] for x in actions)),
   field_counts=dict(Counter(x['field_evaluations'] for x in actions)),head_steps=sum(x['committed_head_steps'] for x in actions),
   first_correction=next((dict(step=x['step'],epoch=x['epoch']) for x in actions if x['field_evaluations']>1),None),
   correction_epochs=dict(correction_epochs),curriculum_events=curr,stages=st,
   config=read(run/'resolved_config.json'),calibration=read(run/'source_calibration.json') if (run/'source_calibration.json').exists() else None,
   resources={k:v for k,v in read(run/'resource_summary.json').items() if k!='budget'},
   audit_count=len(audits),audit_valid=sum(a.get('valid',False) for a in audits),
   audit_stats={k:stats([a[k] for a in audits]) for k in ['G_lag','S_domain','ce_gap_raw','direction_imbalance','identity','margin','consistency']},
   recovery_improved=sum(a['ce_gap_raw']>0 for a in audits),last_epoch=epochs[-1],
   peak_training_cuda_bytes=max(e['peak_memory_bytes'] for e in epochs),
   source_scores=read(run/'source_final_eval/source_scores.json'),target_scores=target[row])
 result[row]=summary
for name,data in [('epochs',epochs_csv),('audits',audits_csv)]:
 keys=list(dict.fromkeys(k for r in data for k in r))
 with (out/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
(out/'full_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(rows=list(result),epochs=len(epochs_csv),audits=len(audits_csv),output=str(out))))
