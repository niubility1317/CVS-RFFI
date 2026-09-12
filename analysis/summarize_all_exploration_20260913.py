import csv,gzip,json,math,re
from pathlib import Path
OUT=Path(__file__).parent/'all_exploration_20260913'
d=json.load(gzip.open(OUT/'evidence.json.gz','rt',encoding='utf-8'))
scenes=['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak']
inventory=[];scores=[];classes=[];curves=[];finals=[];roots=[]
def family(run):
 if 'selected_adv3b02' in run:return '继承A1'
 if 'fast_matched' in run:return '早期CORE90'
 if 'fast_v2_' in run:return 'V2'
 if 'ecrs_cross_rx_' in run:return 'X/ECRS'
 if 'mechanism_periodic_' in run:return '机制'
 if 'mechanism_screen_' in run:return '已替代机制'
 if 'r3_scratch_' in run:return 'R3'
 if 'r3_clean_cross_rx_' in run:return 'R3+clean'
 if 'r3_budget_' in run:return 'R3预算'
 if 'e600_repair_' in run:return 'E600修复'
 return '长训练'
def parsed_argv(a):
 out={}
 for i,k in enumerate(a):
  if k.startswith('--'):out[k]=a[i+1] if i+1<len(a) and not a[i+1].startswith('--') else True
 return out
for run,v in d['runs'].items():
 roots.append({'run':run,'rows':len(v['rows']),'stored_status':v['state'].get('status'),'family':family(run)})
 for row,x in v['rows'].items():
  argv=x['process'].get('argv') or x['launch'].get('argv') or []
  opts={**x['base_options'],**(x['matrix'][0].get('options',{}) if x['matrix'] else {}),**parsed_argv(argv)}
  total=int(opts.get('--epochs',200));live=bool(x['live_argv']) and any(run in str(a) for a in x['live_argv'])
  assert not x['parse_errors'] and not x['csv_bad_widths']
  assert x['epochs']==list(range(1,x['jsonl_records']+1))
  assert x['jsonl_records']==x['csv_records']
  if 'mechanism_screen' in run or 'fast_matched' in run:status='SUPERSEDED'
  elif live:status='RUNNING'
  elif x['final_checkpoint_exists'] and x['jsonl_records']==total:status='COMPLETE_SCORED' if x['scores'] else 'SOURCE_ONLY_COMPLETE'
  else:status='FAILED_OR_NOT_STARTED'
  inherited='selected_adv3b02' in run
  info={'run':run,'family':family(run),'row':row,'status':status,'stored_status':x['state'].get('status'),
   'epoch':x['jsonl_records'],'total':total,'target_tests':len(x['scores']),'inherited_target_contact':inherited,
   'from_scratch_argv':opts.get('--from_scratch'),'scratch_only_argv':opts.get('--a1_scratch_only'),
   'checkpoint_exists':x['final_checkpoint_exists'],**x['resources']}
  inventory.append(info)
  curves.extend({'run':run,'row':row,**c} for c in x['curve'])
  row_scores=[]
  for entry in x['scores']:
   s=entry['score'];m=s['metrics'];match=re.search(r'/E(\d+)/score.json$',entry['path'])
   epoch=int(match[1]) if match else total
   assert entry['prediction_bytes']>0 and s['record_count']==672000 and set(m)==set(scenes)
   if entry['scope']:
    assert entry['scope']['feeds_training'] is False
    if 'record_count' in entry['scope']:assert entry['scope']['record_count']==672000
   z={'run':run,'family':family(run),'row':row,'epoch':epoch,'kind':'periodic' if match else 'final_only','path':entry['path']}
   for scene in scenes:
    t=m[scene];pc=t['per_class_accuracy']
    assert t['total']==168000 and math.isclose(t['accuracy'],t['correct']/t['total'],abs_tol=1e-12)
    assert set(pc)==set(map(str,range(6))) and math.isclose(sum(pc.values())/6,t['accuracy'],abs_tol=1e-12)
    z[scene]=100*t['accuracy'];z[scene+'_correct']=t['correct']
    classes.extend({'run':run,'row':row,'epoch':epoch,'scene':scene,'class':k,'accuracy_pct':100*a} for k,a in pc.items())
   z['leo_mean']=sum(z[s] for s in scenes[1:])/3
   z['scene_floor']=min(z[s] for s in scenes[1:])
   z['class_leo']=[100*sum(m[s]['per_class_accuracy'][str(k)] for s in scenes[1:])/3 for k in range(6)]
   z['weak_class_leo']=min(z['class_leo']);z['weak_class']=z['class_leo'].index(z['weak_class_leo'])
   scores.append(z);row_scores.append(z)
  if row_scores:
   latest=max(row_scores,key=lambda z:z['epoch']);peak=max(row_scores,key=lambda z:z['leo_mean'])
   info.update(latest_test_epoch=latest['epoch'],latest_test_leo=latest['leo_mean'],peak_epoch=peak['epoch'],peak_leo=peak['leo_mean'])
   if status=='COMPLETE_SCORED':
    assert latest['epoch']==total
    finals.append({**info,**latest,'epoch_hours':x['resources']['epoch_hours']})
  if status=='COMPLETE_SCORED' and 'mechanism_periodic' in run:assert sorted(s['epoch'] for s in row_scores)==list(range(80,201,10))
valid=[x for x in finals if not x['inherited_target_contact']]
pareto=[x for x in valid if not any(y['clean']>=x['clean'] and y['leo_mean']>=x['leo_mean'] and (y['clean']>x['clean'] or y['leo_mean']>x['leo_mean']) for y in valid)]
winner=max(valid,key=lambda x:x['leo_mean']);e200=max((x for x in valid if x['total']==200),key=lambda x:x['leo_mean'])
weak=max(valid,key=lambda x:x['weak_class_leo'])
counts={s:sum(x['status']==s for x in inventory) for s in sorted({x['status'] for x in inventory})}
lora=[]
v5=json.loads((OUT/'lora_v5_live.json').read_text(encoding='utf-8'))['result']
v7=next(iter(next(iter(d['side_results'].values())).values()))
for name,result in [('LoRa_v5',v5),('LoRa_v7',v7)]:
 assert len(result['training']['epoch_rows'])==100
 for section in ['single_domain_calibration_4x4','multiple_configuration_calibration']:
  for r in result[section]:
   assert r['decisions']==39060 and math.isclose(r['accuracy'],r['correct_decisions']/r['decisions'],abs_tol=1e-7)
   lora.append({'run':name,'calibration':r.get('calibration_configuration','multiple'),'test':r['test_configuration'],'accuracy_pct':r['accuracy']*100,'correct':r['correct_decisions'],'total':r['decisions']})
assert len(inventory)==60 and len(finals)==33 and len(valid)==31 and len(lora)==40
def save_csv(name,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
for name,rows in [('all_60_rows.csv',inventory),('all_14_roots.csv',roots),('all_target_tests.csv',scores),('all_class_tests.csv',classes),('all_training_curves.csv',curves),('all_final_results.csv',finals),('lora_all_40_tests.csv',lora)]:save_csv(name,rows)
summary={'at':d['captured_at'],'counts':counts,'rows':len(inventory),'roots':len(roots),'epochs':len(curves),'scores':len(scores),'class_scores':len(classes),
 'stdout_lines':sum(x['stdout_lines'] for v in d['runs'].values() for x in v['rows'].values()),'valid_final_rows':len(valid),
 'winner':winner,'e200_winner':e200,'weak_class_winner':weak,'pareto_clean_leo':[x['row'] for x in pareto],
 'finals':finals,'inventory':inventory,'lora':lora}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print('COUNTS',counts,'scores',len(scores),'classes',len(classes),'PARETO',summary['pareto_clean_leo'])
for r in sorted(valid,key=lambda r:r['leo_mean'],reverse=True):print(r['family'],r['row'],round(r['clean'],4),round(r['leo_mean'],4),'weak',round(r['weak_class_leo'],4),'h',round(r['epoch_hours'],2),'peak',r['peak_epoch'],round(r['peak_leo'],4))
print('G1',[(r['epoch'],r['clean'],r['leo_mean']) for r in scores if r['row']=='G1_FISHER_GATE'])
print('LORA',[(r['run'],r['calibration'],r['test'],round(r['accuracy_pct'],3)) for r in lora if r['calibration']==r['test'] or r['calibration']=='multiple'])
