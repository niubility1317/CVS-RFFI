import torch

METHODS=('NATIVE','COS_MEAN','CENTERED_COS','GEIV','ORTHOGONAL','OBLIQUE','RIDGE','SOFT_RIDGE')

def symmetric(a):return (a+a.T)/2

def source_scatter(z,y,rx):
 z=z.double();classes=sorted(y.unique().tolist());receivers=sorted(rx.unique().tolist());cells=[];within=[];counts=[]
 if classes!=list(range(6)) or receivers!=[1,3,4,6,8] or z.ndim!=2 or z.shape[1]!=160 or not torch.isfinite(z).all():raise ValueError('Source fullcross TX/RX identity features required')
 for tx in classes:
  means=[]
  for r in receivers:
   h=z[(y==tx)&(rx==r)]
   if len(h)<2:raise ValueError('Missing TX/RX sourcecell cannot be zeroimputed')
   mu=h.mean(0);means.append(mu);within.append((h-mu).T@(h-mu)/len(h));counts.append(len(h))
  cells.append(torch.stack(means))
 cells=torch.stack(cells);centers=cells.mean(1);mean=centers.mean(0);delta=centers-mean;identity=delta.T@delta/6;residual=(cells-centers[:,None]).reshape(-1,160);nuisance=residual.T@residual/30;error=torch.stack(within).mean(0)
 covariance=symmetric(identity+nuisance+error);scale=covariance.trace()/160;regularized=covariance+1e-3*scale.clamp_min(1e-12)*torch.eye(160,dtype=z.dtype)
 values,vectors=torch.linalg.eigh(symmetric(nuisance));keep=values>values.max().clamp_min(1e-20)*1e-10;D=vectors[:,keep]
 return dict(z=z,cells=cells,class_centers=centers,mean=mean,S_ID=symmetric(identity),S_N=symmetric(nuisance),S_E=symmetric(error),Sigma=regularized,D=D,cell_counts=counts,nuisance_rank=int(keep.sum()),identity_rank=int(torch.linalg.matrix_rank(identity,rtol=1e-9)),regularizer=1e-3*float(scale))

def operators(s):
 d=160;I=torch.eye(d,dtype=s['z'].dtype);D=s['D'];SN=s['S_N'];SE=s['S_E'];eps=1e-3*float((SN+.1*SE).trace()/d);B=symmetric(SN+.1*SE)+max(eps,1e-12)*I;L=torch.linalg.cholesky(B);Li=torch.linalg.solve_triangular(L,I,upper=False)
 eig,Q=torch.linalg.eigh(symmetric(Li@s['S_ID']@Li.T));order=torch.arange(d-1,-1,-1);eig=eig[order].clamp_min(0);V=Li.T@Q[:,order];weights=(.25+.75*eig/eig.max().clamp_min(1e-12)).sqrt();geiv=weights[:,None]*V.T
 ortho=I-D@D.T
 if D.shape[1]:
  invD=torch.linalg.solve(s['Sigma'],D);oblique=I-D@torch.linalg.solve(symmetric(D.T@invD),invD.T)
 else:oblique=I
 val,U=torch.linalg.eigh(SN/SN.trace().clamp_min(1e-12)*d);soft=(U*(1+val.clamp_min(0)).rsqrt()[None])@U.T
 return dict(NATIVE=I,COS_MEAN=I,CENTERED_COS=I,GEIV=geiv,ORTHOGONAL=ortho,OBLIQUE=oblique,RIDGE=I,SOFT_RIDGE=soft),dict(generalized_eigenvalues=eig.tolist(),GEIV_floor=.5,GEIV_full_width=160,denominator_beta=.1,denominator_ridge=max(eps,1e-12))

def fit(z,y,rx,native_weight):
 s=source_scatter(z,y,rx);maps,meta=operators(s);states={};diagnostics={};I=torch.eye(160,dtype=s['z'].dtype);mean=s['mean'];centers=s['class_centers'];onehot=torch.nn.functional.one_hot(y,6).double();onehot=onehot-onehot.mean(0)
 for name,A in maps.items():
  bias=torch.zeros_like(mean) if name in ('NATIVE','COS_MEAN') else -A@mean
  if name=='NATIVE':W=native_weight.double().clone()
  elif name in ('RIDGE','SOFT_RIDGE'):
   h=s['z']@A.T+bias;cov=h.T@h/len(h);ridge=1e-3*float(cov.trace()/160);W=torch.linalg.solve(symmetric(cov)+max(ridge,1e-12)*I,h.T@onehot/len(h)).T
  else:W=centers@A.T+bias
  states[name]=dict(A=A.float(),bias=bias.float(),weight=W.float(),classifier='ordinarycos_scale30_eps1e-4',source_gradient_updates=0,phase2_eligibility='sourcefittedmap_not_audited_forPhase2;no_source_covariance_or_samplecache_runtime')
  pairs=[]
  for a in range(6):
   for b in range(a+1,6):
    diff=centers[a]-centers[b];pairs.append(dict(TX_pair=[a,b],squared_length_ratio=float((A@diff).square().sum()/diff.square().sum().clamp_min(1e-20))))
  diagnostics[name]=dict(operator_norm=float(torch.linalg.matrix_norm(A,ord=2)),rank=int(torch.linalg.matrix_rank(A,rtol=1e-6)),idempotence_error=float((A@A-A).norm()) if name in ('ORTHOGONAL','OBLIQUE') else None,annihilation_error=float((A@s['D']).norm()) if name in ('ORTHOGONAL','OBLIQUE') else None,TX_mean_pair_retention=pairs)
 # Explicit Gaussian location workingmodel, continuous paths between TX means.
 inv=torch.linalg.solve(s['Sigma'],I);D=s['D'];effective=inv-inv@D@torch.linalg.solve(symmetric(D.T@inv@D),D.T@inv) if D.shape[1] else inv;effective=symmetric(effective);info=[]
 for a in range(6):
  for b in range(a+1,6):
   diff=centers[a]-centers[b];raw=diff@inv@diff;rem=diff@effective@diff;info.append(dict(TX_pair=[a,b],gaussian_path_information=float(raw),gaussian_effective_information=float(rem),ratio=float(rem/raw.clamp_min(1e-20))))
 report=dict(methods=list(METHODS),source_count=len(z),cells=30,cell_counts=s['cell_counts'],identity_rank=s['identity_rank'],nuisance_rank=s['nuisance_rank'],covariance_regularizer=s['regularizer'],projection_diagnostics=diagnostics,GEIV=meta,effective_Fisher_working_model='z~Gaussian(mu +tau*TXmeanDifference +D*eta,Sigma);continuous tau,not derivative ofdiscreteTX/not hardwareFisher',effective_Fisher_pairs=info,effective_Fisher_min_eigenvalue=float(torch.linalg.eigvalsh(effective).min()),source_statistics_role='trainingLonly;notVperformance/target',RIDGE_objective='source LS classdirections +relative L2 ridge;SOFT mapscoordinates before sameLS ridge. Inference normalizes W,not claiming CE optimum.',raw_source_stats_exported_to_checkpoint=False)
 return states,report

def classify(h,state):
 z=h@state['A'].to(h).T+state['bias'].to(h);return 30*torch.nn.functional.linear(torch.nn.functional.normalize(z,dim=1,eps=1e-4),torch.nn.functional.normalize(state['weight'].to(h),dim=1,eps=1e-4))
