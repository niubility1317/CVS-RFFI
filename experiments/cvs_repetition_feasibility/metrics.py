"""Downstream source-only counts after frozen observation predictions exist."""
import math
import torch

ARMS=('copy_cycle1','visible_mean','known_reference')

def _distribution(values):
    if not len(values):return dict(mean=None,median=None,q95=None,q99=None,maximum=None,top_one_percent_fraction=None)
    total=float(values.sum());number=max(1,math.ceil(len(values)*.01))
    return dict(mean=float(values.mean()),median=float(values.median()),q95=float(torch.quantile(values,.95)),
                q99=float(torch.quantile(values,.99)),maximum=float(values.max()),
                top_one_percent_fraction=float(values.topk(number).values.sum())/total if total>0 else None)

def summarize(valid,target,predictions,labels,receivers,days):
    n=len(valid)
    if valid.dtype!=torch.bool or valid.ndim!=1 or target.shape!=(n,24) or set(predictions)!=set(ARMS):
        raise ValueError('Invalid fixed residual census shape/arms')
    if any(p.shape!=target.shape for p in predictions.values()) or not all(torch.isfinite(x).all() for x in (target,*predictions.values())):
        raise ValueError('Nonfinite or mismatched observation predictions')
    if any(x.shape!=(n,) for x in (labels,receivers,days)) or any(x.requires_grad for x in (target,*predictions.values())):
        raise ValueError('Only frozen predictions and downstream source metadata')
    energy=target.double().square().mean(1)
    errors={k:(p.double()-target.double()).square().mean(1) for k,p in predictions.items()}
    groups={'overall':torch.ones(n,dtype=torch.bool)}
    groups.update({f'RX/{k}':receivers==k for k in receivers.unique().tolist()})
    groups.update({f'day/{k}':days==k for k in days.unique().tolist()})
    groups.update({f'TX/{k}':labels==k for k in labels.unique().tolist()})
    groups.update({f'RXday/{r}/{d}':(receivers==r)&(days==d)
                   for r,d in sorted(set(zip(receivers.tolist(),days.tolist())))})
    records={}
    for name,mask in groups.items():
        use=mask&valid;count=int(mask.sum());good=int(use.sum());e=energy[use]
        base=float(errors['known_reference'][use].sum());energy_sum=float(e.sum())
        arms={}
        for arm,values in errors.items():
            val=values[use];total=float(val.sum())
            arms[arm]=dict(error_sum=total,MSE_valid=total/good if good else None,
                           MSE_full_group_zero_invalid=total/count if count else None,
                           pooled_error_over_target_energy=total/energy_sum if energy_sum>0 else None,
                           improvement_over_reference=1-total/base if base>0 else None,
                           distribution=_distribution(val))
        records[name]=dict(count=count,valid=good,invalid=count-good,valid_fraction=good/count if count else None,
                           target_energy_sum=energy_sum,target_energy=_distribution(e),arms=arms)
    return dict(records=records,packet_target_energy=energy,packet_MSE=errors,
                normalization='Per-packet mean over24 real residual coordinates; ratio of pooled sums on identical visible-valid packets. Invalid packets explicitly counted; never selected by hidden-target energy.')

def qualification(metrics,relations):
    overall=metrics['records']['overall'];geometry=relations['overall']
    benefit=overall['arms']['visible_mean']['improvement_over_reference']
    rx_benefits=[r['arms']['visible_mean']['improvement_over_reference']
                 for name,r in metrics['records'].items() if name.startswith('RX/')]
    rx_gaps=[r['gap_mean'] for r in relations['RX'].values()]
    checks=dict(visible_valid_coverage=overall['valid_fraction']>=.95,
                visible_mean_predictability=benefit is not None and benefit>=.10,
                no_RX_predictability_reversal=all(x is not None and x>=0 for x in rx_benefits),
                cross_RX_identity_gap=geometry['gap_mean'] is not None and geometry['gap_mean']>=.02,
                four_RX_identity_gap_positive=sum(x is not None and x>0 for x in rx_gaps)>=4)
    return dict(passed=all(checks.values()),checks=checks,source_only=True,target_access=False,
                parameter_scan=False,rule='>=95% visible-valid; visible mean reduces pooled MSE vs reference by>=10%, no RX negative benefit; fixed same-day cross-RX identity cosine gap>=.02 and positive in>=4/5 RX. Necessary source feasibility only, not DG efficacy or decoder benefit.')
