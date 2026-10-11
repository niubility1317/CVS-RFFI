import torch

def margin(logits,y):
 true=logits.gather(1,y[:,None]).flatten();others=logits.clone();others.scatter_(1,y[:,None],float('-inf'))
 return true-others.max(1).values

def summarize(predicted,actual,probabilities,tolerance):
 predicted,actual,probabilities=(v.double() for v in (predicted,actual,probabilities))
 if predicted.shape!=actual.shape or probabilities.shape!=actual.shape or actual.ndim!=2 or actual.shape[1]!=8:raise ValueError('Expected Nx8 action arrays')
 if not all(torch.isfinite(v).all() for v in (predicted,actual,probabilities)) or not torch.allclose(probabilities.sum(1),torch.ones(len(actual),dtype=torch.float64),atol=1e-6):raise ValueError('Invalid frozen action values')
 i,j=torch.triu_indices(8,8,1);delta=actual[:,i]-actual[:,j];pdelta=predicted[:,i]-predicted[:,j];eligible=delta.abs()>tolerance;ptie=pdelta.abs()<=1e-8
 concordant=((pdelta*delta)>0)&~ptie
 count=int(eligible.sum());pair_score=float(((concordant.double()+.5*ptie.double())*eligible).sum()/count) if count else None
 unique=actual.max(1).values-actual.min(1).values>tolerance;best=actual.max(1).values
 top_set=(predicted.max(1).values[:,None]-predicted)<=1e-8;top_probability=top_set.double()/top_set.sum(1,keepdim=True)
 selected=(actual*top_probability).sum(1);top_hit=(top_probability*((best[:,None]-actual)<=tolerance)).sum(1);uniform=actual.mean(1);policy=(actual*probabilities).sum(1)
 oracle_gap=(best-uniform).sum();gain=(policy-uniform).sum();zero=actual.square().mean();error=(predicted-actual).square().mean()
 return dict(count=len(actual),action_count=actual.numel(),eligible_pairs=count,pair_concordance=pair_score,informative_packets=int(unique.sum()),top1_hit_on_informative=float(top_hit[unique].mean()) if unique.any() else None,top1_actual_regret=float((best-selected).mean()),top_prediction_tie_count_mean=float(top_set.sum(1).double().mean()),uniform_expected=float(uniform.mean()),policy_expected=float(policy.mean()),policy_minus_uniform=float((policy-uniform).mean()),oracle_expected=float(best.mean()),fraction_oracle_gap_captured=float(gain/oracle_gap) if float(oracle_gap)>1e-12 else None,actual_mean=float(actual.mean()),actual_std=float(actual.std(unbiased=False)),prediction_std=float(predicted.std(unbiased=False)),within_packet_prediction_std_mean=float(predicted.std(1,unbiased=False).mean()),MSE=float(error),zero_prediction_MSE=float(zero) if float(zero)>1e-18 else 0.,MSE_relative_to_zero=float(error/zero) if float(zero)>1e-18 else None,selection_entropy_mean=float(-(probabilities*probabilities.clamp_min(1e-20).log()).sum(1).mean()),total_variation_from_uniform_mean=float(.5*(probabilities-.125).abs().sum(1).mean()))
