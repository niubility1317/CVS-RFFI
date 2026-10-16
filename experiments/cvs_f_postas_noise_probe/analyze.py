"""Independent offline default-head recount;no IQ,checkpoint or fitted parameters."""
import argparse,gzip,json,math
from collections import Counter,defaultdict
from pathlib import Path
from . import design as d
FEATURES=('time_projection','frequency_projection','reference_projection','SIG_TIME','base_identity','PA_physical','joint160')
def close(a,b,tol=3e-5):
 if not math.isfinite(a+b) or abs(a-b)>tol*max(1.,abs(a),abs(b)):raise ValueError('Independent scalar recount differs')
def score(records,labels):
 C=[[0]*6 for _ in range(6)]
 for p in records:C[labels[p['physical_id']]['TX']][p['prediction']]+=1
 n=sum(map(sum,C));correct=sum(C[k][k] for k in range(6));F=[]
 for k in range(6):
  den=sum(C[k])+sum(v[k] for v in C);F.append(2*C[k][k]/den if den else 0.)
 return dict(count=n,correct=correct,accuracy=correct/n,macro_F1=sum(F)/6,confusion=C)
def interval(x,edges):
 for v in edges:
  if x<v:return '<'+str(v)
 return '>='+str(edges[-1])
def group(p):
 out={k:str(p[k]) for k in ('state','lock','RX','day','agc_limited','phase_tracking_applied','iq_compensation_applied','boundary','noise_boundary','numerical_filter_latency_samples','residual_kernel_length')}
 for name,key,edges in [('quality','quality_snr_db',(0,5,10,20,30)),('added_noise_SNR','signal_to_added_noise_db',(0,5,10,20,30)),('abs_CFO','output_frequency_hz',(10,100,1000)),('kernel_energy','residual_kernel_energy',(.25,.5,1,2,4))]:out[name]=interval(abs(p[key]) if name=='abs_CFO' else p[key],edges)
 return out
def mechanism(records):
 n=len(records)
 mean=lambda vals:sum(vals)/len(vals)
 gradients={}
 for k in (*FEATURES,'frequency_gate_logits','PA_gate_logits'):
  v=[p['feature_gradient_L2'][k] for p in records];finite=[x for x in v if x is not None]
  gradients[k]=dict(mean=mean(finite) if finite else None,observed=len(finite),unused=n-len(finite),zero=sum(x==0 for x in finite))
 return dict(mean_margin_over30=mean([p['margin_over30'] for p in records]),mean_CE=mean([p['CE'] for p in records]),feature_norm={k:mean([p['feature_norms'][k] for p in records]) for k in FEATURES},feature_cosine_to_clean={k:mean([p['feature_cosine_to_clean'][k] for p in records]) for k in FEATURES},feature_gradient_L2=gradients,gate={k:{q:mean([p['gates'][k][q] for p in records]) for q in ('mean','std','scale_mean','scale_min','scale_max','alpha')} for k in ('frequency','PA')})
def analyze(root):
 root=Path(root);done=d.read(root/'completion.json')
 if done['status']!='ARTIFACTS_COMPLETE' or (root/'failure.json').exists():raise ValueError('Complete fixed prediction required')
 results={};strata=[];mechanisms=[];shared=None;shared_order=None;raw_correct=None
 for row in d.rows():
  base=root/row['row_id'];config=d.validate(d.read(base/'resolved_config.json'));actual=d.read(base/'summary.json');labels={p['physical_id']:p for p in d.read(base/'source_labels.json')};contract=d.read(base/'input_contract.json');smoke=d.read(base/'checkpoint_noquery_smoke.json');audit=d.read(base/'observation_audit.json');model=d.read(base/'model_audit.json')
  if len(labels)!=5040 or set(labels)!=set(contract['role_ids']['L_s']) or contract['source_receivers']!=list(d.FIT) or any(p['RX'] not in d.FIT or p['day'] not in (1,2,3) or p['TX'] not in range(6) for p in labels.values()):raise ValueError('Exact fitL source role differs')
  cells=Counter((p['TX'],p['RX'],p['day']) for p in labels.values())
  if len(cells)!=72 or set(cells.values())!={70}:raise ValueError('Full balanced72sourcecells required')
  if actual['NN']!=15122 or actual['decisions']!=15120 or actual['autograd_calls']!=120 or actual['autograd_packet_equivalents']!=15120 or not actual['parameters_and_buffers_unchanged'] or any(actual[k] for k in ('optimizer_updates','head_fit_updates','EMA_updates','model_epochs','target_access')) or smoke['NN']!=2 or smoke['source_IQ_before_smoke'] or smoke['status']!='PASS':raise ValueError('Zero-update diagnostic budget differs')
  if audit['full_physical_draws']!=5040 or audit['session_count']!=12 or audit['control_receiver_and_AGC_recomputed'] or model['scale']!=30 or model['parameters']!=900722:raise ValueError('Actual channel/model differs')
  with gzip.open(base/'physics.jsonl.gz','rt',encoding='utf8') as f:physics=[json.loads(line) for line in f]
  phys={p['physical_id']:p for p in physics}
  if len(physics)!=5040 or set(phys)!=set(labels):raise ValueError('Full matched source observations required')
  for p in physics:
   if any(p[k]!=labels[p['physical_id']][k] for k in ('RX','day')) or p['EQ'] or p['G']!=1 or p['control_new_noise_power']!=0 or not p['original_full_IQ_reconstructed_exact'] or not p['same_receiver_compensation_AGC']:raise ValueError('Physical intervention differs')
  if shared is None:shared=physics
  elif physics!=shared:raise ValueError('Two checkpoints must see identical full physical/noise realizations')
  with gzip.open(base/'predictions.jsonl.gz','rt',encoding='utf8') as f:records=[json.loads(line) for line in f]
  table={(p['view'],p['physical_id']):p for p in records};views={v:[p for p in records if p['view']==v] for v in d.VIEWS}
  if len(records)!=15120 or len(table)!=15120 or any(set(p['physical_id'] for p in views[v])!=set(labels) for v in d.VIEWS):raise ValueError('Complete unique3views required')
  order=[p['physical_id'] for p in views['clean']]
  if any([p['physical_id'] for p in views[v]]!=order for v in d.VIEWS):raise ValueError('View pairing differs')
  if shared_order is None:shared_order=order
  elif order!=shared_order:raise ValueError('Model physical order differs')
  for p in records:
   if p['batch'] not in range(40) or any(p[k]!=labels[p['physical_id']][k] for k in ('RX','day')) or len(p['logits'])!=6 or not all(math.isfinite(x) for x in p['logits']) or p['prediction']!=max(range(6),key=lambda k:p['logits'][k]) or p['raw_reference_CFO_coordinate']!=0:raise ValueError('Full original defaulthead prediction differs')
   y=labels[p['physical_id']]['TX'];z=p['logits'];mx=max(z);ce=mx+math.log(sum(math.exp(a-mx) for a in z))-z[y];margin=(z[y]-max(a for i,a in enumerate(z) if i!=y))/30
   close(p['CE'],ce);close(p['margin_over30'],margin)
   if set(p['feature_norms'])!=set(FEATURES) or set(p['feature_cosine_to_clean'])!=set(FEATURES) or set(p['feature_gradient_L2'])!=set(FEATURES)|{'frequency_gate_logits','PA_gate_logits'}:raise ValueError('Existing feature observation differs')
   for k in FEATURES:
    if not math.isfinite(p['feature_norms'][k]) or p['feature_norms'][k]<0 or not -1.000001<=p['feature_cosine_to_clean'][k]<=1.000001:raise ValueError('Nonfinite feature observation')
    if p['view']=='clean' and p['feature_norms'][k]>1e-12:close(p['feature_cosine_to_clean'][k],1.,1e-6)
   for x in p['feature_gradient_L2'].values():
    if x is not None and (not math.isfinite(x) or x<0):raise ValueError('Invalid gradient observation')
   for g in p['gates'].values():
    if not all(math.isfinite(x) for x in g.values()) or not 0<=g['mean']<=1 or g['std']<0 or g['scale_min']>g['scale_max']:raise ValueError('Invalid existing gate observation')
  scores={v:score(views[v],labels) for v in d.VIEWS};cells_out={}
  for dim,value in (('state','BLOCKED'),('lock','unlocked')):
   ids={p['physical_id'] for p in physics if p[dim]==value};ss={v:score([p for p in views[v] if p['physical_id'] in ids],labels) for v in d.VIEWS};cs={v:ss[v]['correct'] for v in d.VIEWS};gap=cs['clean']-cs['full'];recovered=cs[d.VIEWS[-1]]-cs['full'];passed=bool(ids) and gap>0 and 2*recovered>=gap
   cells_out[value]=dict(count=len(ids),correct=cs,full_to_clean_correct_gap=gap,recovered_correct=recovered,recovery_fraction=recovered/gap if gap>0 else None,supported=passed,scores=ss)
   expected=actual['recoverability']['strata'][value]
   if any(cells_out[value][k]!=expected[k] for k in ('count','correct','full_to_clean_correct_gap','recovered_correct','recovery_fraction','supported')):raise ValueError('Independent fixed criterion differs')
  supported=all(p['supported'] for p in cells_out.values())
  if supported!=actual['recoverability']['supported']:raise ValueError('Independent two-stratum AND differs')
  if raw_correct is None:raw_correct={p['physical_id'] for p in views['clean'] if p['prediction']==labels[p['physical_id']]['TX']}
  retention={v:score([p for p in views[v] if p['physical_id'] in raw_correct],labels) for v in d.VIEWS}
  transitions={}
  for v in ('full',d.VIEWS[-1]):
   transitions[v]=dict(repairs=sum(table[v,p]['prediction']==labels[p]['TX'] and table['clean',p]['prediction']!=labels[p]['TX'] for p in order),breaks=sum(table[v,p]['prediction']!=labels[p]['TX'] and table['clean',p]['prediction']==labels[p]['TX'] for p in order))
  bygroup=defaultdict(list)
  for p in physics:
   for dim,value in dict(group(p),TX=str(labels[p['physical_id']]['TX'])).items():bygroup[dim,value].append(p['physical_id'])
  for (dim,value),ids in sorted(bygroup.items()):
   for view in d.VIEWS:
    rec=[table[view,p] for p in ids];strata.append(dict(arm=row['arm'],view=view,dimension=dim,value=value,**score(rec,labels)));mechanisms.append(dict(arm=row['arm'],view=view,dimension=dim,value=value,count=len(rec),**mechanism(rec)))
  for view in d.VIEWS:
   mechanisms.append(dict(arm=row['arm'],view=view,dimension='all',value='all',count=5040,**mechanism(views[view])))
  results[row['arm']]=dict(scores=scores,recoverability=dict(strata=cells_out,supported=supported),retention_on_fixed_RAW10_clean_correct=retention,transitions_relative_to_own_clean=transitions,wall_seconds=actual['wall_seconds'],parent_epochs=actual['parent_epochs'],parent_updates=actual['parent_updates'],parent_NN=actual['parent_NN'],NN=actual['NN'],decisions=actual['decisions'],autograd_calls=actual['autograd_calls'],parameters_and_buffers_unchanged=True)
 primary=results['ANTI']['recoverability']['supported']
 if primary!=done['supported_noise_masking_rationale']:raise ValueError('Independent global primary decision differs')
 summary=dict(status='ANALYZED',actual_execution='VERIFIED',run_id=d.RUN,source_only=True,model_seeds=[d.SEED],source_L_unique=5040,source_cells=72,rows=results,primary='ANTI',supported_noise_masking_rationale=primary,same_physics_and_order_across_models=True,new_NN=30244,source_decisions=30240,autograd_calls=240,optimizer_updates=0,head_fit_updates=0,EMA_updates=0,model_epochs=0,target_access=False,automatic_training=False,performance_improvement_proved=False,Ag_efficacy_proved=False,joint_goal_complete=False,limits=['SourcefitL seen during both original trainings;not heldRX or target validation','Artificial fixed-receiver noise removal is diagnostic,not deployable denoising or an information-theoretic upper bound','BLOCKED/unlocked overlap;not independent strata','One channel draw per physicalsourceL,one modelseed,two own-head checkpoints','Remaining propagation,boundary,receiver and original ground RX/TX effects retained','Branch/gate gradients are intermediate CE derivatives,not training updates or a substitute for recognition'])
 d.write(root/'analysis_summary.json',summary);d.write(root/'source_strata.json',strata);d.write(root/'mechanism_strata.json',mechanisms);return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args();print(json.dumps(analyze(a.input)))
