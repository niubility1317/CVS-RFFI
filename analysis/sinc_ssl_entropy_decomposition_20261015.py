"""Same-row saved-source gradient decomposition; no IQ/model/truth access or updates."""
import json,math,statistics
from pathlib import Path
import torch
NAMES=('id_backbone.encoder.core.id_backbone.t1.pw.weight','id_backbone.encoder.core.id_backbone.t2.pw.weight','id_backbone.encoder.sequence_pool.views.0.local_projection.weight')

def metrics(L,C,U):
 if any(not isinstance(v,torch.Tensor) or v.ndim!=1 or not torch.isfinite(v).all() for v in (L,C,U)) or L.shape!=C.shape or L.shape!=U.shape:raise ValueError('Finite matching saved source vectors required')
 L,C,U=(v.detach().double() for v in (L,C,U));E=U-C
 norm=lambda v:float(v.norm())
 dot=lambda a,b:float(a.dot(b))
 cosine=lambda a,b:dot(a,b)/(norm(a)*norm(b)) if norm(a)>0 and norm(b)>0 else None
 nL,nC,nU,nE=map(norm,(L,C,U,E))
 return dict(L_norm=nL,weighted_U_CE_norm=nC,combined_U_norm=nU,reconstructed_entropy_norm=nE,entropy_over_L=nE/nL if nL else None,entropy_over_U_CE=nE/nC if nC else None,cos_L_U_CE=cosine(L,C),cos_L_entropy=cosine(L,E),cos_CE_entropy=cosine(C,E),cos_L_combined_U=cosine(L,U),cos_L_total=cosine(L,L+U),U_CE_zero=nC==0,entropy_nonzero=nE>0,CE_to_combined_U_sign_reversal=nC>0 and dot(L,C)>=0 and dot(L,U)<0,total_L_descent_sign_reversal=dot(L,L+C)>=0 and dot(L,L+U)<0,entropy_reconstruction_residual=norm(C+E-U))

def scalar_summary(values):
 values=sorted(v for v in values if v is not None)
 if not values:return dict(count=0,mean=None,p50=None,p95=None,minimum=None,maximum=None)
 def quantile(q):
  x=(len(values)-1)*q;i=int(x);j=min(i+1,len(values)-1);return values[i]+(x-i)*(values[j]-values[i])
 return dict(count=len(values),mean=statistics.mean(values),p50=quantile(.5),p95=quantile(.95),minimum=values[0],maximum=values[-1])

def analyze(source,epochs=(33,34)):
 source=Path(source);args=json.loads((source/'actual_native_args.json').read_text())
 if args['lambda_u']!=.16 or args['lambda_ent']!=.01 or args.get('use_muse_ssdg',False) or args['label_epochs']!=33 or args['pseudo_epochs']!=17:raise ValueError('Original native U objective or schedule differs')
 outputs=[]
 for epoch in epochs:
  values=torch.load(source/f'ssl_gradients_e{epoch}.pt',map_location='cpu',weights_only=False)
  if len(values)!=222 or [v['step'] for v in values]!=list(range((epoch-1)*222+1,epoch*222+1)) or any(v['epoch']!=epoch for v in values):raise ValueError('Complete actual diagnostic epoch required')
  per={name:[] for name in NAMES}
  for value in values:
   vectors=value['vectors']
   if set(vectors)!={'main_L_CE','weighted_U_CE','weighted_U_CE_entropy'} or any(len(a)!=3 for a in vectors.values()):raise ValueError('Fixed original gradient coordinates missing')
   for i,name in enumerate(NAMES):per[name].append(metrics(vectors['main_L_CE'][i],vectors['weighted_U_CE'][i],vectors['weighted_U_CE_entropy'][i]))
  for name,rows in per.items():
   outputs.append(dict(epoch=epoch,parameter=name,batches=len(rows),U_CE_zero_batches=sum(r['U_CE_zero'] for r in rows),entropy_nonzero_batches=sum(r['entropy_nonzero'] for r in rows),entropy_on_zero_U_CE_batches=sum(r['U_CE_zero'] and r['entropy_nonzero'] for r in rows),CE_to_combined_U_sign_reversal_batches=sum(r['CE_to_combined_U_sign_reversal'] for r in rows),total_L_descent_sign_reversal_batches=sum(r['total_L_descent_sign_reversal'] for r in rows),summary={k:scalar_summary([r[k] for r in rows]) for k in ('L_norm','weighted_U_CE_norm','combined_U_norm','reconstructed_entropy_norm','entropy_over_L','entropy_over_U_CE','cos_L_U_CE','cos_L_entropy','cos_CE_entropy','cos_L_combined_U','cos_L_total','entropy_reconstruction_residual')}))
 return dict(status='VERIFIED_SAVED_SOURCE_ANALYSIS',diagnostic_epochs=list(epochs),batch_records=222*len(epochs),layer_records=666*len(epochs),saved_input_vectors=1998*len(epochs),records=outputs,model_or_IQ_or_truth_read=False,new_gradient_calls=0,new_forwards=0,optimizer_updates=0,head_fit_updates=0,source_only=True,limitation='E = saved grad(.16UCE+.01entropy) - saved grad(.16UCE), subject to saved float32 rounding. Three actual feature parameter tensors only; no independent entropy backward, full-ftheta inference, preconditioned/cumulative AdamW contribution, pseudo-label precision or DG claim.')

def check_math():
 v=metrics(torch.tensor([1.,0.]),torch.tensor([.3,.4],dtype=torch.float64),torch.tensor([.2,.6],dtype=torch.float64))
 assert abs(v['entropy_over_L']-math.sqrt(.05))<1e-12 and v['cos_L_entropy']<0 and not v['CE_to_combined_U_sign_reversal']
 w=metrics(torch.tensor([1.,0.]),torch.zeros(2),torch.tensor([-2.,0.]));assert w['U_CE_zero'] and w['entropy_nonzero'] and w['total_L_descent_sign_reversal']
 r=metrics(torch.tensor([1.,0.]),torch.tensor([.1,.2]),torch.tensor([-.1,.2]));assert r['CE_to_combined_U_sign_reversal'] and not r['total_L_descent_sign_reversal']
 for bad in (torch.tensor([float('nan'),0.]),torch.ones(3)):
  try:metrics(torch.ones(2),bad,torch.ones(2))
  except ValueError:pass
  else:raise AssertionError('Invalid saved vector accepted')
 return dict(status='PASS',known_vector_cases=3,negative_cases=2,no_real_artifacts_read=True)
if __name__=='__main__':print(json.dumps(check_math()))
