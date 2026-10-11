"""Read complete source diagnostic JSON only; never target values or predictions."""
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
 p=base/row['row_id']
 rows.append(dict(row=row,completion=read(p/'completion.json'),resolved=read(p/'resolved_config.json'),paired=read(p/'paired_readouts.json'),gradient=read(p/'gradient_audit.json'),log_tail=(base/'logs'/(row['row_id']+'.log')).read_text()[-1500:] if (base/'logs'/(row['row_id']+'.log')).is_file() else None))
print(json.dumps(dict(completion=read(base/'completion.json'),failure=read(base/'failure.json'),queue=read(base/'queue_state.json'),rows=rows)))
'''


def collect(output):
 value=json.loads(ssh(SCRIPT.replace('CONFIG',repr(dict(base=d.BASE.as_posix(),rows=d.rows())))))
 if value['failure']:raise ValueError('Source diagnostic failed; inspect existing artifacts')
 done=value['completion']
 if not done or done['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or set(done['rows'])!={r['row_id'] for r in d.rows()}:raise ValueError('All three source diagnostic rows required')
 for row in value['rows']:
  c=d.config(row['row']);finished=row['completion'];g=row['gradient'];p=row['paired']
  assert finished['config']==c and finished['identity_updates']==0 and not finished['target_access'] and not finished['V_backward'] and not finished['U_truth']
  assert all(row['resolved'][k]==v for k,v in c.items())
  assert len(p['rows'])==16 and {(v['arm'],v['role']) for v in p['rows']}=={(a,r) for a in c['endpoint_arms'] for r in ('L','V')}
  assert len(p['versus_CE'])==14 and {(v['arm'],v['role']) for v in p['versus_CE']}=={(a,r) for a in c['endpoint_arms'] if a!='CE' for r in ('L','V')}
  for item in p['rows']+p['versus_CE']:
   res=item['result'];s=res['overall'];assert s['count']==c['roles'][item['role']]
   assert s['final_correct']-s['initial_correct']==s['fixed']-s['broken']
   assert sum(t['count'] for t in res['initial_groups'].values())==s['count']
  assert len(g['rows'])==192 and {(v['round'],v['arm']) for v in g['rows']}=={(i,a) for i in range(24) for a in c['endpoint_arms']}
  assert g['state_unchanged'] and g['RNG_restored'] and not g['V_gradient_access'] and not g['U_truth_access']
  assert all(v['state_unchanged'] for v in g['rows'])
  assert finished['resources']['source_readout_packets']==299700 and finished['resources']['head_updates']==200
 value.update(status='VERIFIED',target_access=False,source_readout_packets=899100,gradient_records=576,identity_updates=0)
 d.write(output,value);return value


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();v=collect(a.output)
 print(json.dumps({k:v[k] for k in ('status','source_readout_packets','gradient_records','identity_updates','target_access')}))
