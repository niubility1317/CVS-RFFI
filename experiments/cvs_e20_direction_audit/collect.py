import argparse,json,math
from pathlib import Path
from . import design as d
from experiments.cvs_phase1_stack.publish import ssh

SCRIPT=r'''
import json
from pathlib import Path
c=CONFIG;base=Path(c['base'])
def read(p):return json.loads(p.read_text()) if p.is_file() else None
rows=[]
for row in c['rows']:
 p=base/row['row_id'];models={}
 for name in ('P0','CE20'):
  q=p/name;models[name]=dict(summary=read(q/'summary.json'),audit=read(q/'audit.json'),warm=read(q/'head_warmup.json'),calibration=read(q/'calibration.json'),warm_steps=[json.loads(l) for l in (q/'head_warmup.jsonl').read_text().splitlines()] if (q/'head_warmup.jsonl').is_file() else [])
 log=base/'logs'/(row['row_id']+'.log')
 rows.append(dict(row=row,completion=read(p/'completion.json'),resolved=read(p/'resolved_config.json'),models=models,log=log.read_text() if log.is_file() else None))
print(json.dumps(dict(completion=read(base/'completion.json'),failure=read(base/'failure.json'),rows=rows)))
'''

def validate(v):
 if v['failure']:raise ValueError('Diagnostic failure; retain artifacts')
 expected={r['row_id'] for r in d.rows()};done=v['completion']
 assert done and done['status']=='SOURCE_DIAGNOSTIC_COMPLETE' and set(done['rows'])==expected
 assert {r['row']['row_id'] for r in v['rows']}==expected and len(v['rows'])==3
 variants={'+'.join(level)+'@'+str(float(s)) for level in d.config(d.rows()[0])['level_sets'] for s in (.3,1.)}
 def finite(x):
  if isinstance(x,float):assert math.isfinite(x)
  elif isinstance(x,dict):
   for a in x.values():finite(a)
  elif isinstance(x,list):
   for a in x:finite(a)
 for row in v['rows']:
  c=d.config(row['row']);z=row['completion'];assert z['config']==c and z['identity_updates']==0 and z['head_updates']==400 and not z['target_access']
  assert z['checkpoint_sources']==c['checkpoint_sources'] and z['U_forward']==z['V_forward']==0
  assert all(row['resolved'][k]==value for k,value in c.items())
  assert row['log'] and 'Traceback' not in row['log'] and 'SOURCE_DIAGNOSTIC_COMPLETE' in row['log']
  assert set(row['models'])=={'P0','CE20'}
  for name,m in row['models'].items():
   a=m['audit'];assert a['status']=='COMPLETE' and a['source_role']=='L_s' and a['optimizer_steps']==0
   assert all(a[k] for k in ('state_unchanged','training_flags_restored','RNG_preserved','parameter_grad_unchanged'))
   assert not any(a[k] for k in ('target_access','U_access','V_access'))
   assert a['rounds']==len(a['records'])==36 and a['variants_per_round']==6
   assert {str(k):v for k,v in a['day_counts'].items()}=={'1':12,'2':12,'3':12}
   assert a['resources']==dict(model_forwards=369,source_packet_readouts=39960)
   assert a['fixed_coefficients']==m['calibration']['fixed_coefficients']
   assert m['summary']['identity_state_unchanged'] and m['summary']['identity_updates']==0
   assert m['summary']['checkpoint']==c['checkpoint_sources'][c['models'].index(name)]
   assert m['warm']['optimizer_steps']==200 and m['warm']['source_L_cache_packets']==6300 and m['warm']['base_state_unchanged'] and len(m['warm_steps'])==200
   assert m['calibration']['pairs']==12 and m['calibration']['extra_source_readouts']==1440 and m['calibration']['state_unchanged']
   assert {r['round'] for r in a['records']}==set(range(36))
   for r in a['records']:
    assert len(r['styles'])==6 and {s['variant'] for s in r['styles']}==variants and r['split_combined_clean_close']
    assert r['pair']['pair_A_distinct_PID']==r['pair']['pair_B_distinct_PID']==60
    assert 0<=r['pair']['cross_batch_PID_overlap']<=60 and r['pair']['independent_AB']
    for s in r['styles']:
     assert s['style']['donor_rules_verified']
     q=s['comparison'];assert sum(q[k] for k in ('repaired','broken','persistent_wrong','persistent_correct'))==120
     assert abs((q['styled']['accuracy']-r['clean']['accuracy'])*120-(q['repaired']-q['broken']))<2e-5
     assert set(s['directions'])=={'full_identity','time_down','t2.pool'}
   assert len(a['dropout_controls'])==3 and {c['day'] for c in a['dropout_controls']}=={1,2,3}
   for c0 in a['dropout_controls']:
    assert c0.get('same_mask_replay_close',c0['same_mask_replay_exact']) and c0['persistent_statistics_eval'] and c0['dropout_modules']>0
    assert len(c0['variants'])==6 and {s['variant'] for s in c0['variants']}==variants
   finite(a)
 return dict(v,status='VERIFIED',models=6,gradient_pairs=216,style_trials=1296,dropout_controls=18,identity_updates=0,head_updates=1200,diagnostic_model_forwards=2214,diagnostic_L_readouts=239760,warmup_L_readouts=37800,calibration_L_readouts=8640,total_L_readouts_excluding_preflight=286200,target_access=False)

def collect(output):
 v=validate(json.loads(ssh(SCRIPT.replace('CONFIG',repr(dict(base=d.BASE.as_posix(),rows=d.rows()))))));d.write(output,v);return v

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=collect(a.output)
 print(json.dumps({k:v[k] for k in ('status','models','gradient_pairs','style_trials','dropout_controls','identity_updates','total_L_readouts_excluding_preflight','target_access')}))
