"""Complete source-only finite-response recount, without models or target files."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.cvs_sameiq_experts.publish import remote
RUN='20261019-diagnostic-F-cleanLookahead-sourceL-s2026092701-m2-r01'
SCRIPT=r'''
import json,math,re
from pathlib import Path
import numpy as np
c=CONFIG;root=Path(c['project'])/'runs'/c['run_id'];read=lambda p:json.loads(p.read_text())
done=read(root/'complete.json');decision=read(root/'source_decision.json');roles=read(root/'source_roles.json');records=read(root/'responses.json')['records'];cfg=done['config'];views=cfg['views'];modes=['BASE','RAW_STEP','JOINT_STEP','FIRST_ORDER_LOOKAHEAD']
if done['status']!='SOURCE_DIAGNOSED' or done['NN']!=6188 or done['network_backward_packets']!=1820 or done['virtual_updates']!=42 or done['original_model_updates'] or done['head_only_fit_updates'] or done['target_access'] or done['U_truth_read'] or done['V_used'] or not done['original_states_and_grads_unchanged'] or not done['global_RNG_unchanged'] or len(records)!=12:raise ValueError('Source-only completion/budget differs')
if len(roles['step_ids'])!=180 or len(roles['response_ids'])!=180 or len(set(roles['step_ids']+roles['response_ids']))!=360 or roles['source_role']!='L_s' or not roles['both_historically_trained_L'] or roles['target_access'] or roles['U_truth_read'] or roles['V_used']:raise ValueError('Source operator roles differ')
report=dict(status='SOURCE_ANALYZED',run_id=c['run_id'],completion=done,decision=decision,per_model={},all_logits_independently_recounted=True,target_access=False,model_or_IQ_loads=0,max_CE_error=0.)
all_steps={};all_response={}
for arm in ('B0','B1'):
 rows=sorted([r for r in records if r['arm']==arm],key=lambda r:r['batch'])
 if [r['batch'] for r in rows]!=list(range(6)):raise ValueError('Complete six batches per model required')
 rowdone=read(root/arm/'complete.json')
 if rowdone['NN']!=3094 or rowdone['virtual_updates']!=21 or rowdone['original_model_updates'] or rowdone['target_access']:raise ValueError('Per-model source budget differs')
 aggregate={mode:{v:dict(CE=0.,correct=0,n=0) for v in views} for mode in modes};urban_d=[];raw_d=[];geometry=[]
 all_steps[arm]=[];all_response[arm]=[]
 for r in rows:
  y=np.asarray(r['TX'],int);rx=r['RX'];day=r['day']
  if y.shape!=(30,) or set(zip(y.tolist(),rx))!={(t,x) for t in range(6) for x in (1,3,4,6,8)} or len(set(day))!=1 or set(day)-{1,2,3}:raise ValueError('Each response batch must cover all TX/RX in one source day')
  start=r['batch']*30
  if r['step_ids']!=roles['step_ids'][start:start+30] or r['response_ids']!=roles['response_ids'][start:start+30]:raise ValueError('Actual batch physical role order differs')
  all_steps[arm]+=r['step_ids'];all_response[arm]+=r['response_ids']
  for mode in modes:
   if set(r['response'][mode])!=set(views):raise ValueError('Missing default-head response view')
   for view in views:
    q=r['response'][mode][view];z=np.asarray(q['logits'],float)
    if z.shape!=(30,6) or not np.isfinite(z).all() or q['n']!=30:raise ValueError('Finite complete source default logits required')
    logp=z-z.max(1,keepdims=True);logp-=np.log(np.exp(logp).sum(1,keepdims=True));ce=float((-.99*logp[np.arange(30),y]-.01*logp.mean(1)).mean());correct=int((z.argmax(1)==y).sum());report['max_CE_error']=max(report['max_CE_error'],abs(ce-q['CE']))
    if correct!=q['correct'] or abs(ce-q['CE'])>2e-5:raise ValueError('Saved source CE/accuracy differs from logits')
    a=aggregate[mode][view];a['CE']+=ce/6;a['correct']+=correct;a['n']+=30
  urban_d.append(sum(r['response']['FIRST_ORDER_LOOKAHEAD'][v]['CE']-r['response']['JOINT_STEP'][v]['CE'] for v in views[1:])/2)
  raw_d.append(r['response']['FIRST_ORDER_LOOKAHEAD']['raw']['CE']-r['response']['JOINT_STEP']['raw']['CE'])
  g=r['geometry']['As_before_after_raw_step'];co=g['cosine']
  if co is not None and (not math.isfinite(co) or abs(co)>1+1e-7):raise ValueError('Invalid measured source gradient geometry')
  geometry.append(dict(batch=r['batch'],cosine=co,angle_deg=None if co is None else math.degrees(math.acos(max(-1.,min(1.,co)))),norm_before=g['norm_a'],norm_after=g['norm_b']))
  if set(r['virtual_witnesses'])!=set(modes[1:]) or any(w['virtual_updates']!=1 or w['persistent_original_updates'] or not w['default_head_joint'] or w['unit_step']['physical_clip_calls']!=1 for w in r['virtual_witnesses'].values()):raise ValueError('Every virtual copy requires one real physical-unit joint update')
 for mode in modes:
  for view,a in aggregate[mode].items():a['accuracy']=a['correct']/a['n']
 recomputed=dict(urban_CE_delta=float(np.mean(urban_d)),raw_CE_delta=float(np.mean(raw_d)),urban_better_batches=sum(v < -1e-6 for v in urban_d),batches=6)
 expected=decision['rows'][arm]
 if any(abs(recomputed[k]-expected[k])>1e-12 for k in recomputed):raise ValueError('Fixed source decision recount differs')
 report['per_model'][arm]=dict(responses=aggregate,lookahead_minus_joint=recomputed,batch_urban_CE_deltas=urban_d,batch_raw_CE_deltas=raw_d,gradient_geometry=geometry,virtual_update_count=21,original_updates=0,NN=3094,backward_packets=910)
if all_steps['B0']!=all_steps['B1'] or all_response['B0']!=all_response['B1']:raise ValueError('Models used different source operator packets')
passed=all(v['lookahead_minus_joint']['urban_CE_delta'] < -1e-6 and v['lookahead_minus_joint']['raw_CE_delta']<=1e-6 and v['lookahead_minus_joint']['urban_better_batches']>=4 for v in report['per_model'].values())
if passed!=decision['rationale_pass']:raise ValueError('Source qualification boolean differs')
log=(Path(c['project'])/'releases/cvs_F_cleanLookahead_source_20261019_r01/probe.stdout.log').read_text(errors='replace');errors=[s for s in log.splitlines() if re.search(r'Traceback|CUDA out of memory|FloatingPointError|^Killed$|RuntimeError|ModuleNotFoundError',s)]
if errors:raise ValueError('Source log contains a technical failure')
report['source_decision_independently_recomputed']=True;report['same_source_packet_order_both_models']=True;report['full_log_errors']=errors
print(json.dumps(report))
'''

def render(x,out):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');colors=['#6082a5','#bf8d35','#178a73'];views=['raw','practical_mid_urban','practical_low_urban']
 for index,(arm,row) in enumerate(x['per_model'].items()):
  for j,mode in enumerate(('RAW_STEP','JOINT_STEP','FIRST_ORDER_LOOKAHEAD')):
   axes[index].bar(np.arange(3)+(j-1)*.24,[row['responses'][mode][v]['CE']-row['responses']['BASE'][v]['CE'] for v in views],width=.24,label=mode,color=colors[j])
  axes[index].axhline(0,color='#444444',lw=.7);axes[index].set(xticks=range(3),xticklabels=['raw','mid urban','low urban'],ylabel='Response CE change vs frozen original',title=arm+' FINAL20 · source L only');axes[index].spines[['top','right']].set_visible(False);axes[index].grid(axis='y',alpha=.15)
 axes[0].legend(frameon=False,fontsize=8);fig.savefig(out/'source_finite_response.png',dpi=180);fig.savefig(out/'source_finite_response.pdf');plt.close(fig)

def main(runtime,output):
 rt=json.loads(Path(runtime).read_text(encoding='utf8'));out=Path(output);out.mkdir(parents=True,exist_ok=False);result=remote(SCRIPT.replace('CONFIG',repr(dict(project=rt['project'],run_id=RUN))),rt)
 (out/'source_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');render(result,out)
 print(json.dumps(dict(status=result['status'],decision=result['decision'],independent_CE_max_error=result['max_CE_error'],all_12_batches_recounted=True,target_access=False)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.runtime,a.output)
