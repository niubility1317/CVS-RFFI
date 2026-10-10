"""Fixed identity V pressure validation, separate from auxiliary calibration packets."""
import math
from collections import defaultdict
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .physics import factorial_batch,apply_action,sample_parameters,phase_process

def metrics(logits,y,rx,day):
 logits=logits.float();pred=logits.argmax(1)
 margin=logits.gather(1,y[:,None])-logits
 margin.scatter_(1,y[:,None],float('inf'));minimum=margin.min(1).values
 if len(y)==0 or logits.shape!=(len(y),6) or not torch.isfinite(logits).all():raise ValueError('Invalid source logits')
 risk=F.softplus(-minimum);tail=torch.topk(risk,max(1,math.ceil(len(y)*d.STRESS['tail_alpha']))).values.mean()
 cm=torch.bincount(y*6+pred,minlength=36).reshape(6,6)
 denom=cm.sum(0)+cm.sum(1);f1=torch.where(denom>0,2*cm.diag()/denom,torch.zeros_like(denom)).mean()
 groups={}
 for name,values in [('receiver',rx),('day',day),('transmitter',y)]:
  groups[name]={str(int(v)):float((pred[values==v]==y[values==v]).float().mean()) for v in values.unique()}
 return dict(count=len(y),accuracy=float((pred==y).float().mean()),macro_f1=float(f1),
  group_accuracy=groups,worst_rx=min(groups['receiver'].values()),worst_day=min(groups['day'].values()),
  worst_tx=min(groups['transmitter'].values()),margin_q10=float(torch.quantile(minimum,.1)),
  tail_risk=float(tail),mean_ce=float(F.cross_entropy(logits,y)))

def evaluate_source_stress(model,loader,device,c,output):
 modes=[(m,m.training) for m in model.modules()];model.eval()
 before={k:v.detach().clone() for k,v in model.state_dict().items()}
 try:return _evaluate_source_stress(model,loader,device,c,output,before)
 finally:
  for m,mode in modes:m.training=mode

def _evaluate_source_stress(model,loader,device,c,output,before):
 generator=torch.Generator().manual_seed(d.STRESS['seed']);collected=defaultdict(list);labels=[];receivers=[];days=[]
 applied=defaultdict(int);total=0
 with torch.no_grad():
  for step,(x,y,domain,meta) in enumerate(loader):
   x=x.to(device);b=factorial_batch(x,generator,step=0)
   # The opposite ordering uses exactly this batch's original parameters.
   tl=apply_action(b['x01'],b['linear_parameters'],'linear')
   heldout=apply_action(b['x11'],.5*b['linear_parameters'],'linear')
   pressure_p=sample_parameters('temporal',len(x),generator,x,pressure=True)
   pressure=apply_action(x,pressure_p,'temporal',phase_process(len(x),generator,x))
   noise=torch.randn(b['x11'].shape,generator=generator).to(x)
   noise=noise/noise.square().mean((1,2),keepdim=True).sqrt().clamp_min(1e-10)
   noisy=b['x11']+noise*b['x11'].square().mean((1,2),keepdim=True).sqrt()*10**(-25/20)
   views=dict(clean=x,linear=b['x10'],temporal=b['x01'],LT=b['x11'],TL=tl,noise_fixed_LT=noisy,
              LTL_heldout=heldout,curvature_pressure=pressure)
   if set(views)!=set(d.STRESS['views']):raise ValueError('Source pressure views differ from preregistration')
   for name,view in views.items():
    collected[name].append(model(view).detach().cpu())
    applied[name]+=int((view!=x).flatten(1).any(1).sum())
   labels.append(y.cpu());receivers.append(meta['rx_i'].cpu());days.append(meta['day_i'].cpu());total+=len(y)
 if total!=27000:raise ValueError('Incomplete source V pressure validation')
 y,rx,day=map(torch.cat,(labels,receivers,days))
 if set(rx.tolist())!={1,3,4,6,8}:raise ValueError('Non-source receiver in V')
 if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):raise ValueError('Source V changed model state')
 result=dict(status='SOURCE_STRESS_COMPLETE',row_id=c['row_id'],model_seed=c['model_seed'],arm=c['arm'],
   source_role='V only; not auxiliary holdout',target_access=False,model_state_unchanged=True,
   config=d.STRESS,views={name:dict(metrics(torch.cat(values),y,rx,day),applied_packets=applied[name],
    skipped_clean_copies=total-applied[name],
    heldout_composition=name=='LTL_heldout',artificial_curvature_pressure=name=='curvature_pressure')
    for name,values in collected.items()})
 d.write(Path(output)/'source_stress.json',result);return result

def source_ranking():
 """Read source-only artifacts; target scores are neither discovered nor opened."""
 groups=defaultdict(list)
 for row in d.rows():
  c=d.config(row);v=d.read(Path(c['output_root'])/'source_stress.json')
  if (v['status']!='SOURCE_STRESS_COMPLETE' or v['target_access']
      or v['row_id']!=row['row_id'] or v['model_seed']!=row['model_seed'] or v['arm']!=row['arm']
      or v['config']!=d.STRESS or set(v['views'])!=set(d.STRESS['views'])
      or any(z['count']!=27000 for z in v['views'].values())):raise ValueError('Invalid source stress')
  groups[row['arm']].append(v)
 def avg(xs):return sum(xs)/len(xs)
 base=avg([r['views']['clean']['accuracy'] for r in groups['native']]);records=[]
 for arm,items in groups.items():
  if sorted(r['model_seed'] for r in items)!=sorted(d.SEEDS):raise ValueError('Incomplete source seed matrix')
  views=[v for r in items for name,v in r['views'].items() if name!='clean']
  clean=avg([r['views']['clean']['accuracy'] for r in items]);mean=avg([v['accuracy'] for v in views]);worst=avg([v['worst_rx'] for v in views])
  costs=[]
  for row in d.rows():
   if row['arm']!=arm:continue
   p=Path(d.config(row)['output_root'])/'auxiliary_cost.json'
   if not p.is_file():raise ValueError('Missing actual auxiliary cost record')
   costs.append(d.read(p)['auxiliary_parameters'])
  records.append(dict(arm=arm,seeds=[r['model_seed'] for r in items],clean=clean,
    feasible=clean>=base-d.STRESS['clean_noninferiority_pp']/100,stress_mean=mean,stress_worst_rx=worst,
    score=(mean+worst)/2,tail_risk=avg([v['tail_risk'] for v in views]),auxiliary_parameters=avg(costs)))
 feasible=[r for r in records if r['feasible']];best=max(r['score'] for r in feasible)
 tied=[r for r in feasible if best-r['score']<=d.STRESS['noise_tie_pp']/100]
 chosen=min(tied,key=lambda r:(r['auxiliary_parameters'],r['tail_risk'],-r['score'],r['arm']))
 result=dict(status='SOURCE_RANKING_FROZEN',selected=chosen['arm'],rows=records,rule=d.STRESS,
  target_scores_consumed=False,test_scope='Fixed preregistered comparison matrix: all rows are tested independently, ranking does not select target access')
 d.write(d.BASE/'source_ranking.json',result);return result
