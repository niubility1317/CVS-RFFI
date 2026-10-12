import torch,math
def metric(mu,v,target,var,kind):
 error=(mu-target).square().mean();mean_energy=target.square().mean();irreducible=var.mean();energy=mean_energy+irreducible
 finite=lambda value:float(value) if torch.isfinite(value) else None
 result=dict(count=len(mu),mean_prediction_MSE=float(error),empirical_mean_delta_energy=float(mean_energy),empirical_within_repeat_variance=float(irreducible),expected_repeat_MSE=float(error+irreducible),total_empirical_delta_energy=float(energy),normalized_mean_MSE=float(error/mean_energy) if mean_energy>1e-12 else None,normalized_repeat_MSE=float((error+irreducible)/energy) if energy>1e-12 else None,low_delta_energy=bool(energy<=1e-12),mean_cosine=float(torch.nn.functional.cosine_similarity(mu,target,dim=1,eps=1e-8).mean()),prediction_bias_l2=float((mu-target).mean(0).norm()),predicted_variance_mean=float(v.mean()))
 if kind=='noise':
  vv=v.clamp_min(1e-10);result.update(empirical_gaussian_NLL=float(.5*((mu-target).square()+var).div(vv).add(vv.log()+math.log(2*math.pi)).mean()),variance_MSE=float((v-var).square().mean()),variance_floor=1e-10,noise_repeats=4,variance_estimator='biased four-realization finite-population variance; conditional uncertainty estimated, not exact noise realization recovery')
 return result
def coverage(predicted_fit_delta,clean,L,fit,ev,rank):
 # The span is estimated only from fit finite responses. Real RX statistics use
 # independent evaluation PIDs; they are unpaired packet distributions.
 if set(fit.tolist())&set(ev.tolist()):raise ValueError('RX coverage fit/query physical leakage')
 matrix=predicted_fit_delta.reshape(-1,predicted_fit_delta.shape[-1]).double();eigen,vectors=torch.linalg.eigh(matrix.T@matrix/len(matrix))
 r=min(rank,int((eigen>eigen[-1].clamp_min(1e-24)*1e-12).sum()));basis=vectors[:,-r:] if r else vectors[:,:0];total=covered=0.;cells=0
 for tx in range(6):
  for day in (1,2,3):
   means=[]
   for rx in (1,3,4,6,8):
    idx=ev[(L['y'][ev]==tx)&(L['day'][ev]==day)&(L['rx'][ev]==rx)]
    if not len(idx):raise ValueError('Independent RX mean cell absent')
    means.append(clean[idx].double().mean(0))
   values=torch.stack(means);residual=values-values.mean(0);total+=float(residual.square().sum());covered+=float((residual@basis).square().sum()) if r else 0.;cells+=5
 return dict(independent_query_RX_cells=cells,rank=r,max_rank=rank,RX_residual_energy=total,projected_RX_residual_energy=covered,coverage_fraction=covered/total if total>1e-12 else None,unexplained_RX_residual_energy=total-covered,source_only=True,fit_eval_PID_disjoint=not bool(set(fit.tolist())&set(ev.tolist())),unpaired_RX_distributions=True,causal_counterfactual_claim=False,encoder_saw_all_source_RX=True)
