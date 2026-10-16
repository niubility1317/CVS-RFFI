"""Independent arithmetic recount. Supply a private source artifact directory and output JSON path; no IQ, model forward or fitting. Required private files: result.json, completion.json, batch_metrics.jsonl, source_gradients.pt, initial_smoke.pt, updates_e{1,17,33,50}.pt."""
import json,math,statistics,sys,torch,numpy as np
from pathlib import Path
out=Path(sys.argv[1])
v=json.loads((out/'result.json').read_text());done=json.loads((out/'completion.json').read_text())
if done['status']!='ANALYZED' or done['rows']!=4 or done['batches']!=200:raise ValueError('Incomplete source diagnosis')
records=[json.loads(s) for s in (out/'batch_metrics.jsonl').read_text().splitlines()];raw=torch.load(out/'source_gradients.pt',map_location='cpu',weights_only=False)
if len(records)!=200 or len(raw)!=200:raise ValueError('Missing complete source vectors')
lookup={(r['arm'],r['stage'],r['batch']):r for r in records};stats={};max_error=0.
for r in raw:
 key=(r['arm'],r['stage'],r['batch']);record=lookup[key];a=r['CE_physical_Hz_gradient'].numpy();b=r['KL_physical_Hz_gradient'].numpy()
 if a.shape!=(48,) or b.shape!=(48,) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite source gradient')
 na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b));dot=float(np.dot(a,b));cos=dot/(na*nb) if na*nb else None;ratio=nb/na if na else None;row=record['groups']['Sinc']
 for k,value in dict(CE_norm=na,KL_norm=nb,dot=dot,cosine=cos,KL_to_CE_norm=ratio,total_to_CE_dot=float(np.dot(a+b,a))).items():
  if value is None:
   if row[k] is not None:raise ValueError('Undefined source ratio differs')
  elif not math.isclose(row[k],value,rel_tol=1e-10,abs_tol=1e-22):raise ValueError('Source gradient scalar mismatch '+k)
  elif k=='cosine':max_error=max(max_error,abs(row[k]-value))
 group=stats.setdefault((r['arm'],r['stage']),[]);group.append(dict(conflict=dot<0,cosine=cos,ratio=ratio,total_opposes=float(np.dot(a+b,a))<0))
summary=[]
for (arm,stage),rows in stats.items():
 original=next(r for r in v['summary'] if r['arm']==arm and r['stage']==stage)['groups']['Sinc'];count=sum(r['conflict'] for r in rows)
 if len(rows)!=50 or count!=original['conflict_count'] or sum(x['packets'] for x in records if x['arm']==arm and x['stage']==stage)!=6300:raise ValueError('Incomplete full source L row')
 summary.append(dict(arm=arm,stage=stage,batches=50,source_L_packets=6300,conflict_count=count,conflict_fraction=count/50,cosine_mean=statistics.mean(r['cosine'] for r in rows),KL_to_CE_norm_median=statistics.median(r['ratio'] for r in rows),total_gradient_opposes_CE_count=sum(r['total_opposes'] for r in rows)))
# Existing actual E step witnesses: no new model forward, fit or update.
p=out;prefix='id_backbone.encoder.core.id_backbone.sinc.'
initial=torch.load(p/'initial_smoke.pt',map_location='cpu',weights_only=False)['model']
def edges(low,band):
 low=np.clip(50+np.abs(low),50,12.5e6-51);high=np.minimum(low+50+np.abs(band),12.5e6-1);return low,high
l0,h0=edges(initial[prefix+'low_origin_hz'].numpy().ravel(),initial[prefix+'band_origin_hz'].numpy().ravel());width=h0-l0;updates=[]
for epoch in (1,17,33,50):
 rows=torch.load(p/f'updates_e{epoch}.pt',map_location='cpu',weights_only=False);ratios=[];initial_ratios=[];folds=0;norms=[]
 if len(rows)!=13:raise ValueError('Missing old actual update observations')
 for r in rows:
  values=r['parameters'];a=values[prefix+'low_coordinate'];b=values[prefix+'band_coordinate'];lb=a['physical_before'].numpy().ravel().astype(float);la=a['physical_after'].numpy().ravel().astype(float);bb=b['physical_before'].numpy().ravel().astype(float);ba=b['physical_after'].numpy().ravel().astype(float)
  before=edges(lb,bb);after=edges(la,ba);bw=before[1]-before[0];ratios.extend((np.maximum(np.abs(after[0]-before[0]),np.abs(after[1]-before[1]))/bw).tolist());initial_ratios.extend((np.maximum(np.abs(after[0]-before[0]),np.abs(after[1]-before[1]))/width).tolist());folds+=int(np.sum(lb*la<0)+np.sum(bb*ba<0));norms.append(float(np.linalg.norm(np.concatenate([la-lb,ba-bb]))))
 updates.append(dict(epoch=epoch,observed_optimizer_steps=13,channel_step_observations=len(ratios),edge_shift_over_current_bandwidth_median=statistics.median(ratios),edge_shift_over_current_bandwidth_max=max(ratios),edge_shift_over_initial_bandwidth_max=max(initial_ratios),current_bandwidth_shift_over_1pct=sum(x>.01 for x in ratios),current_bandwidth_shift_over_10pct=sum(x>.1 for x in ratios),current_bandwidth_shift_over_100pct=sum(x>1 for x in ratios),signed_leaf_zero_crossings=folds,physical_update_L2_Hz_mean=statistics.mean(norms)))
final_E=next(r for r in summary if r['arm']=='E' and r['stage']=='final50')
result=dict(status='VERIFIED',run_id=v['run_id'],source_gradient_rows=4,source_gradient_batches=200,source_gradient_components=9600,original_runtime_math_error_max_cosine=max_error,summary=summary,preregistered_conflict_candidate_eligible=final_E['conflict_count']>=10,conflict_route_closed=final_E['conflict_count']<10,source_update_observations=dict(source_run='20261015-phase1-SIGTIME-RDrop-AF-e50-clean-manysig-m18-r02',arm='E',model_seed=2026092701,epochs=updates,initial_bandwidth_Hz_min=float(width.min()),initial_bandwidth_Hz_max=float(width.max()),initial_bandwidth_Hz_median=statistics.median(width.tolist()),limitation='13 observed steps perdeclared epoch, not a complete trajectory or CE/KL split during training.'),extra_source_IQ_reads=0,extra_model_forwards=0,extra_optimizer_updates=0,head_fit_updates=0,target_access=False)
Path(sys.argv[2]).write_text(json.dumps(dict(result=v,independent_recount=result),indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps(result,allow_nan=False))
