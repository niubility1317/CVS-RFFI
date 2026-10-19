"""Independent complete source-logit/CM/paired recount; no model/target read."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.cvs_sig_boundary_source import design as d
from experiments.cvs_sig_boundary_source.publish import remote
SCRIPT=r'''
import json
from pathlib import Path
import numpy as np
c=CONFIG;run=Path(c['project'])/'runs'/c['run_id']
def read(p):return json.loads(p.read_text())
r=read(run/'results.json');done=read(run/'complete.json');cfg=read(run/'config.json');smoke=read(run/'smoke.json')
if r['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or done['NN']!=21604 or r['NN']!=21604 or r['additional_default_head_decisions']!=86400 or r['total_source_head_decisions']!=108000 or smoke['NN']!=4 or smoke['source_IQ_read'] or smoke['query_read'] or r['fit_updates'] or r['U_used'] or r['V_used'] or r['target_access'] or r['formal_scores_read'] or r['new_physical_simulations']:raise ValueError('Fixed source-only scope/budget differs')
recounts={};pairs={};cm_count=0;pair_count=0;all_ids=None
for name in ('B0_FINAL20','FULL_As_JOINT_FINAL30'):
 with np.load(run/(name+'_source_logits.npz'),allow_pickle=False) as f:
  ids=f['ids'];y=f['y'];rx=f['rx'];day=f['day'];scene=f['training_scene']
  if len(ids)!=1800 or len(set(ids))!=1800 or (all_ids is not None and not np.array_equal(ids,all_ids)):raise ValueError('Exact same source physical IDs required')
  all_ids=ids.copy();views={};model_pairs={}
  for view in cfg['views']:
   predictions={mode:f[view+'_'+mode].argmax(1) for mode in cfg['modes']};source=r['rows'][name]['views'][view];conditions={}
   for mode,records in source['conditions'].items():
    values=[]
    for z in records:
     dim,v=z['dimension'],z['stratum'];take=np.ones(1800,bool) if dim=='overall' else rx==int(v) if dim=='receiver' else day==int(v) if dim=='day' else y==int(v) if dim=='transmitter' else scene==v if dim=='training_scene' else None
     if take is None or not take.any():raise ValueError('Unknown source stratum')
     a=predictions[mode][take];t=y[take];cm=np.bincount(t*6+a,minlength=36).reshape(6,6);den=cm.sum(0)+cm.sum(1);acc=float(cm.trace()/cm.sum());f1=float(np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0).mean())
     if z['count']!=int(take.sum()) or z['confusion']!=cm.tolist() or abs(z['accuracy']-acc)>1e-12 or abs(z['macro_f1']-f1)>1e-12:raise ValueError('Independent source CM/accuracy/F1 differs')
     values.append(dict(z));cm_count+=1
    conditions[mode]=values
   views[view]=dict(conditions=conditions)
   ps={}
   for a,b in [('E','NATIVE'),('I','NATIVE'),('EI','NATIVE'),('E','I')]:
    records=[]
    for z in source['conditions']['NATIVE']:
     dim,v=z['dimension'],z['stratum'];take=np.ones(1800,bool) if dim=='overall' else rx==int(v) if dim=='receiver' else day==int(v) if dim=='day' else y==int(v) if dim=='transmitter' else scene==v
     p,q,t=predictions[a][take],predictions[b][take],y[take];repair=int(((p==t)&(q!=t)).sum());damage=int(((p!=t)&(q==t)).sum());records.append(dict(dimension=dim,stratum=v,count=int(take.sum()),repair=repair,damage=damage,net_pp=100*(repair-damage)/int(take.sum())));pair_count+=1
    ps[a+'_minus_'+b]=records
   ps['EI_interaction_overall_pp']=100*((predictions['EI']==y).mean()-(predictions['E']==y).mean()-(predictions['I']==y).mean()+(predictions['NATIVE']==y).mean());model_pairs[view]=ps
  if not r['rows'][name]['parameters_unchanged'] or not r['rows'][name]['capture_closed'] or not r['rows'][name]['native_base_pool_joint_head_bit_equal'] or max(r['rows'][name]['displacement_max_error'],r['rows'][name]['time_area_max_error'])>2e-5:raise ValueError('Native source replay/state/endpoint identities not verified')
  recounts[name]=dict(views=views);pairs[name]=model_pairs
v=recounts['FULL_As_JOINT_FINAL30']['views'];acc=lambda view,mode:v[view]['conditions'][mode][0]['accuracy'];fresh=lambda mode:sum(acc(x,mode) for x in ('mid_draw0','low_draw0','mid_draw1','low_draw1'))/4;a=100*(fresh('E')-fresh('NATIVE'));b=100*(fresh('E')-fresh('I'));keep=100*(acc('raw','E')-acc('raw','NATIVE'));q=dict(E_minus_NATIVE_fresh_midlow_pp=a,E_minus_I_fresh_midlow_pp=b,E_minus_NATIVE_raw_L_pp=keep,supports_boundary_hypothesis=a>=.5 and b>=.5 and keep>=-.2)
for key,value in q.items():
 if isinstance(value,bool):
  if value!=r['decision'][key]:raise ValueError('Fixed source condition differs')
 elif abs(value-r['decision'][key])>1e-10:raise ValueError('Fixed source delta differs')
print(json.dumps(dict(status='SOURCE_ANALYZED',run_id=c['run_id'],results=r,independent_recount=recounts,paired=pairs,CM_recounted=cm_count,paired_strata_recounted=pair_count,decision=q,extra_NN=0,CP_loads=0,IQ_read=False,target_access=False,formal_scores_read=False,goal_complete=False)))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--output',required=True);a=p.parse_args();rt=d.read(a.runtime)
 if rt['run_id']!=d.RUN:raise ValueError('Wrong source diagnostic runtime')
 out=Path(a.output)
 if out.exists():raise FileExistsError('Existing analysis;reconcile no overwrite')
 c=dict(project=rt['project'],run_id=d.RUN);compile(SCRIPT.replace('CONFIG','{}'),'<sig-boundary-source-analysis>','exec');r=remote(SCRIPT.replace('CONFIG',repr(c)),rt);out.mkdir(parents=True);d.write(out/'source_analysis.json',r);print(json.dumps({k:r[k] for k in ('status','CM_recounted','paired_strata_recounted','decision','extra_NN','target_access','goal_complete')}))
if __name__=='__main__':main()
