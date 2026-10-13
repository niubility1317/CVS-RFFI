"""Complete source step-log contributions; no IQ, checkpoint tensor or truth read."""
import argparse,gzip,json,math,statistics,time
from pathlib import Path
from . import design as d
from .predict import source_proof

def stat(x):
 if not x or any(not math.isfinite(v) for v in x):raise ValueError('Missing/nonfinite actual source values')
 return dict(n=len(x),mean=statistics.mean(x),median=statistics.median(x),minimum=min(x),maximum=max(x))

def aggregate(values):
 keys=('labeled_CE','U_CE','weighted_U_CE','weighted_entropy_residual','total_increment','combined_gradient_metric','weighted_U_to_L','total_increment_to_L')
 result={k:stat([v[k] for v in values]) for k in keys}
 result.update(steps=len(values),U_packets=sum(v['U_packets'] for v in values),U_accepted=sum(v['U_accepted'] for v in values),empty_U_steps=sum(v['U_packets']>0 and v['U_accepted']==0 for v in values))
 return result

def run():
 start=time.time();rows=[]
 for row in d.training_rows():
  c=d.config(row);root=source_proof(c);args=d.read(root/'resolved_native_args.json')
  if args.get('lambda_u')!=.16 or args.get('lambda_ent')!=.01 or args.get('use_muse_ssdg') or args.get('muse_level')!='M0':raise ValueError('Actual old-gate loss recipe differs')
  values=[]
  with gzip.open(root/'step_metrics.jsonl.gz','rt',encoding='utf8') as f:
   for line in f:
    record=json.loads(line);step=len(values)+1;epoch=(step-1)//222+1;m=record['metrics']
    if record['step']!=step or record['epoch']!=epoch:raise ValueError('Actual source step order/budget differs')
    L=float(m['train/loss_labeled']);U=float(m['train/loss_unlabeled']);closed=float(m['train/loss_closed_group']);total=float(m['train/loss']);tx=float(m['train/loss_tx_labeled']);gradient=float(m['train/grad_backbone'])
    N=int(m['train/pseudo_total']);selected=int(m['train/pseudo_selected']);weighted=.16*U;increment=closed-L;entropy=increment-weighted
    expected=0 if epoch<=33 else (124 if step%222==0 else 256)
    if N!=expected or not 0<=selected<=N or m.get('train/pseudo_truth_available',0)!=0:raise ValueError('Hidden-U actual budget differs')
    if any(not math.isfinite(v) for v in (L,U,closed,total,tx,gradient)) or L<=0 or U<0 or gradient<0:raise ValueError('Invalid source losses/combined gradient')
    if abs(total-closed)>1e-6 or abs(L-tx)>1e-6 or entropy < -3e-7:raise ValueError('Actual inactive objectives or loss decomposition differs')
    if epoch<=33 and (U!=0 or N!=0 or abs(increment)>1e-6):raise ValueError('Unexpected labeled-stage U signal')
    values.append(dict(step=step,epoch=epoch,labeled_CE=L,U_CE=U,weighted_U_CE=weighted,weighted_entropy_residual=entropy,total_increment=increment,combined_gradient_metric=gradient,weighted_U_to_L=weighted/L,total_increment_to_L=increment/L,U_packets=N,U_accepted=selected))
  if len(values)!=11100 or sum(v['U_packets'] for v in values)!=963900:raise ValueError('Incomplete actual source log')
  epochs=[dict(epoch=e,**aggregate([v for v in values if v['epoch']==e])) for e in range(1,51)]
  rows.append(dict(row_id=row['row_id'],arm=row['arm'],seed=row['model_seed'],actual_native_lambda_U=.16,actual_native_lambda_entropy=.01,label=aggregate(values[:7326]),pseudo=aggregate(values[7326:]),last10=aggregate(values[-2220:]),epochs=epochs))
 return dict(status='VERIFIED',run_id=d.RUN,models=6,steps_read=66600,rows=rows,source_only=True,IQ_read=False,checkpoint_tensors_read=False,U_truth_read=False,target_read=False,head_fit=False,optimizer_updates=0,seconds=time.time()-start,loss_reduction='Actual native F.cross_entropy on selected logits uses mean over accepted samples; no multiplication by acceptance fraction.',entropy_scope='weighted entropy recovered as closed_group minus labeled_CE minus .16*U_CE; float32 residual, not directly logged entropy.',gradient_scope='Only existing combined train/grad_backbone metric; not L/U gradient decomposition, layer gradients or proof of causal transfer.',inference='Coverage is not a loss multiplier. Scalar loss ratios do not establish parameter-gradient ratios or suggest a larger lambda automatically.')

if __name__=='__main__':
 argparse.ArgumentParser().parse_args();print(json.dumps(run()))
