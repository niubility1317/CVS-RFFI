import argparse,json
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
 for model in ('P0','CE20'):
  q=p/model;models[model]=dict(summary=read(q/'summary.json'),levels={level:read(q/(level.replace('.','_')+'.json')) for level in ('time_down','t2.pool')},
   warm=read(q/'head_warmup.json'),warm_steps=[json.loads(l) for l in (q/'head_warmup.jsonl').read_text().splitlines()] if (q/'head_warmup.jsonl').is_file() else [])
 log=base/'logs'/(row['row_id']+'.log')
 rows.append(dict(row=row,completion=read(p/'completion.json'),resolved=read(p/'resolved_config.json'),models=models,log=log.read_text() if log.is_file() else None))
print(json.dumps(dict(completion=read(base/'completion.json'),failure=read(base/'failure.json'),rows=rows)))
'''
def validate(v):
 if v['failure']:raise ValueError('Diagnostic failure retained')
 done=v['completion']
 if not done or done['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or set(done['rows'])!={r['row_id'] for r in d.rows()}:raise ValueError('Incomplete matrix')
 for row in v['rows']:
  c=d.config(row['row']);z=row['completion'];assert z['config']==c and z['identity_updates']==0 and not z['target_access'] and z['head_updates']==400
  assert z['checkpoint_sources']==c['checkpoint_sources'] and all(row['resolved'][k]==a for k,a in c.items())
  assert row['log'] and 'Traceback' not in row['log'] and 'SOURCE_DIAGNOSTIC_COMPLETE' in row['log']
  assert z['resources']['L_model_packet_readouts']==25200 and z['resources']['U_forward']==z['resources']['V_forward']==0
  for name,model in row['models'].items():
   assert model['summary']['identity_state_unchanged'] and model['summary']['identity_updates']==0
   assert model['warm']['optimizer_steps']==200 and model['warm']['base_state_unchanged'] and len(model['warm_steps'])==200
   for level,a in model['levels'].items():
    f,s=a['fishr'],a['style'];assert f['identity_updates']==f['head_updates']==0 and f['real_head_and_inputs_unchanged']
    assert not any(f[k] or s[k] for k in ('target_access','V_access','U_access'))
    assert len(f['rounds'])==216 and len(f['references'])==6 and len(f['summary'])==54
    assert {(r['control'],r['day'],r['m'],r['round']) for r in f['rounds']}=={(ctrl,day,m,i) for ctrl in ('original_RX','within_TX_day_shuffled_RX') for day in (1,2,3) for m in (2,8,16) for i in range(12)}
    assert s['summary']['count']==s['summary']['unique_pid']==6300 and len(s['rounds'])==36 and len(s['conditional_cells'])==18
    assert all(t['donor_rules_verified'] and t['count']==60 for t in s['rounds'])
 return dict(v,status='VERIFIED',identity_updates=0,head_updates=1200,L_model_packet_readouts=75600,target_access=False)
def collect(output):
 v=validate(json.loads(ssh(SCRIPT.replace('CONFIG',repr(dict(base=d.BASE.as_posix(),rows=d.rows()))))));d.write(output,v);return v
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=collect(a.output)
 print(json.dumps({k:v[k] for k in ('status','identity_updates','head_updates','L_model_packet_readouts','target_access')}))
