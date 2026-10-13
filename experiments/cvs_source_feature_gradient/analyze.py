"""Pure artifact summary; no input IQ/model/gradient execution."""
import argparse,copy,json,math,statistics
from pathlib import Path
from . import design as d
CODE='d4c1e9280854a3bfc988626f98224e737347260b'

def stat(values):
 values=sorted(v for v in values if v is not None)
 if not values or any(not math.isfinite(v) for v in values):raise ValueError('Missing/nonfinite actual derivative metrics')
 def q(p):
  at=(len(values)-1)*p;lo=int(at);hi=min(lo+1,len(values)-1);return values[lo]+(values[hi]-values[lo])*(at-lo)
 return dict(n=len(values),mean=statistics.mean(values),median=q(.5),q25=q(.25),q75=q(.75),minimum=values[0],maximum=values[-1])

def summarize(data):
 if data['status']!='VERIFIED' or data['run_id']!=d.RUN or data['models']!=3 or len(data['rows'])!=3:raise ValueError('Complete first-seed3arm source artifacts required')
 expected={r['row_id']:d.config(r) for r in d.rows()};summary=[];paths=[];activations=[];group_records=activation_records=0
 for item in data['rows']:
  r=item['result'];cfg=item['resolved'];batches=r['batches']
  if cfg['execution_commit']!=CODE or cfg['config']!=expected.get(r['row_id']) or r['status']!='VERIFIED' or r['seed']!=d.SEED or len(batches)!=25 or r['source_packets']!=6300 or r['packet_forwards']!=18900 or r['feature_parameters']!=176065 or r['head_fit'] or r['optimizer_steps'] or not r['parameters_buffers_unchanged'] or r['accumulated_gradients'] or r['query_read'] or r['U_truth_read'] or r['V_forward'] or r['V_gradients']:raise ValueError('Actual source-only scope/config/state differs')
  if sum(b['count'] for b in batches)!=6300 or any(b['batch']!=i+1 or b['count']!=(256 if i<24 else 156) or not b['geometry_all_visible'] or not 0<=b['accepted_correct']<=b['accepted']<=b['count'] for i,b in enumerate(batches)):raise ValueError('Actual source batch/label/gate counts differ')
  accepted=sum(b['accepted'] for b in batches);correct=sum(b['accepted_correct'] for b in batches)
  summary.append(dict(arm=r['arm'],seed=r['seed'],source_packets=6300,accepted=accepted,accepted_correct=correct,source_proxy_coverage=accepted/6300,source_proxy_precision=correct/accepted if accepted else None,teacher_correct=sum(b['teacher_correct'] for b in batches),seconds=r['seconds']))
  names=[p['path'] for p in batches[0]['parameter_paths']]
  for b in batches:
   group_records+=len(b['parameter_paths']);activation_records+=len(b['activation_paths'])
   if len(b['parameter_paths'])!=11 or sum(p['elements'] for p in b['parameter_paths'])!=176065 or [p['path'] for p in b['parameter_paths']]!=names:raise ValueError('Actual f_theta path objects differ')
   for p in b['parameter_paths']:
    for key in ('L_CE_gradient_L2','weighted_pseudo_CE_gradient_L2','weighted_entropy_gradient_L2','combined_U_gradient_L2'):
     if not math.isfinite(p[key]) or p[key]<0:raise ValueError('Invalid actual parameter norm')
    if p['L_U_cosine'] is not None and not -1-1e-10<=p['L_U_cosine']<=1+1e-10:raise ValueError('Invalid within-path cosine')
  for name in names:
   v=[next(p for p in b['parameter_paths'] if p['path']==name) for b in batches]
   paths.append(dict(arm=r['arm'],path=name,elements=v[0]['elements'],combined_U_over_L=stat([p['combined_U_over_L'] for p in v]),L_U_cosine=stat([p['L_U_cosine'] for p in v]),negative_cosine_batches=sum(p['L_U_cosine'] is not None and p['L_U_cosine']<0 for p in v),L_gradient=stat([p['L_CE_gradient_L2'] for p in v]),weighted_pseudo_gradient=stat([p['weighted_pseudo_CE_gradient_L2'] for p in v]),weighted_entropy_gradient=stat([p['weighted_entropy_gradient_L2'] for p in v])))
  records=[a for b in batches for a in b['activation_paths']]
  for name in ('time_fuse','t1','t2','t3','f1','f2','f3','pa_b1','pa_b2','pa_b3','joint160'):
   for rx in (1,3,4,6,8):
    v=[a for a in records if a['path']==name and a['receiver']==rx];n=sum(a['count'] for a in v);accepted_n=sum(a['accepted'] for a in v)
    if n!=1260:raise ValueError('Actual source RX/layer counts differ')
    fields=('per_packet_L_gradient_L2_sum','per_accepted_weighted_pseudo_gradient_L2_sum','per_packet_weighted_entropy_gradient_L2_sum')
    if any(not math.isfinite(a[k]) or a[k]<0 for a in v for k in fields):raise ValueError('Invalid receiver activation norm')
    activations.append(dict(arm=r['arm'],path=name,receiver=rx,count=n,accepted=accepted_n,L_per_packet_gradient_L2_mean=sum(a[fields[0]] for a in v)/n,weighted_pseudo_per_accepted_gradient_L2_mean=sum(a[fields[1]] for a in v)/accepted_n if accepted_n else None,weighted_entropy_per_packet_gradient_L2_mean=sum(a[fields[2]] for a in v)/n))
 return dict(status='ANALYZED',run_id=d.RUN,actual_analysis_code=CODE,model_seed_count=1,model_seed=d.SEED,arms=3,source_only=True,source_role='L_s',summary=summary,parameter_paths=paths,activation_by_receiver=activations,independent_recount=dict(source_batches=75,parameter_path_records=group_records,activation_receiver_records=activation_records,source_packets_across_models=18900,packet_forwards=56700),new_training_epochs=0,optimizer_updates=0,head_fit=False,V_cache_materialized=True,V_forward=False,V_gradients=False,U_IQ_read=False,U_truth_read=False,query_read=False,actual_artifacts=copy.deepcopy(data),interpretation='Fixed source eval old6TX sensitivities only; no actual recorded train/dropout L/U gradients, gradient causal DG attribution, newTX or U accuracy. Batch IQR is not model-seed uncertainty.')

def plot(result,path):
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 names=[p['path'] for p in result['parameter_paths'] if p['arm']=='OLD_GATE_COS50']
 labels=['Reference','Sinc (Hz)','Time early','Time middle','Time proj.','Frequency','PA early','PA middle','PA proj.','Fusion','Feature norms']
 fig,axes=plt.subplots(2,1,figsize=(12.2,7.8),sharex=True,constrained_layout=True)
 colors={'OLD_GATE_COS50':'#777777','RANDOM_ORDER':'#3274A1','PAIRED_ORDER':'#D66E32'}
 for j,(arm,color) in enumerate(colors.items()):
  x=np.arange(len(names))+(j-1)*.18;v=[next(p for p in result['parameter_paths'] if p['arm']==arm and p['path']==n) for n in names]
  for ax,key,factor in zip(axes,('combined_U_over_L','L_U_cosine'),(100,1)):
   middle=np.array([p[key]['median'] for p in v])*factor;low=middle-np.array([p[key]['q25'] for p in v])*factor;high=np.array([p[key]['q75'] for p in v])*factor-middle
   ax.errorbar(x,middle,yerr=np.stack([low,high]),fmt='o',color=color,label=arm,capsize=3,ms=4)
 axes[0].set(ylabel='Within-path weighted U / L gradient norm (%)',title='Parameter gradient ratios: median and batch IQR');axes[0].legend(ncol=3,fontsize=9)
 axes[1].axhline(0,color='#999',ls='--');axes[1].set(ylabel='L versus combined U cosine',ylim=(-1,1),xticks=range(len(names)),xticklabels=labels,xlabel='One model seed; 25 frozen source batches per arm. IQR is not model-seed uncertainty.')
 for ax in axes:ax.grid(alpha=.2)
 axes[1].tick_params(axis='x',rotation=30)
 fig.suptitle('Actual f_theta source derivatives — default head, eval mode, no model updates',fontsize=13);fig.savefig(path,dpi=160);plt.close(fig)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--figure',type=Path);a=p.parse_args();result=summarize(json.loads(a.input.read_text(encoding='utf8')));a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');
 if a.figure:plot(result,a.figure)
 print(json.dumps(dict(status=result['status'],recount=result['independent_recount'],model_seeds=1,head_fit=False)))
