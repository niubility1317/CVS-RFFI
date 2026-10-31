"""Fixed-weight source counterfactuals for learned prior SIG decoder context."""
from contextlib import contextmanager
import torch

CASES=('learned','disabled','clean_SIG','shifted_SIG')


@contextmanager
def decoder_context(model,mode,target):
    if mode not in CASES:raise ValueError('Unknown fixed source context case')
    if len(model.prior)!=2 or any(not p.decoder_context or p.signature_conditioner is None for p in model.prior):
        raise ValueError('Two learned SIG decoder priors required')
    if target.ndim!=2 or target.shape[1]!=160 or not torch.isfinite(target).all():
        raise ValueError('Finite per-record source Bx160 SIG required')
    handles=[]
    try:
        for prior in model.prior:
            if mode=='disabled':
                handles.append(prior.signature_conditioner.register_forward_hook(lambda module,args,value:torch.zeros_like(value)))
            elif mode in ('clean_SIG','shifted_SIG'):
                def replace(module,args):
                    if args[0].shape!=target.shape or args[0].device!=target.device or args[0].dtype!=target.dtype:
                        raise ValueError('Matched source conditioner input required')
                    return (target,)
                handles.append(prior.signature_conditioner.register_forward_pre_hook(replace))
        yield
    finally:
        for handle in handles:handle.remove()


def decision(summary):
    checks={}
    for view,groups in summary.items():
        before=groups['all']['cases']['learned'];oracle=groups['all']['cases']['clean_SIG'];shift=groups['all']['cases']['shifted_SIG']
        checks[view]=dict(
            clean_choice_gain_vs_learned=oracle['teacher_correct']-before['teacher_correct']>=4,
            clean_choice_gain_vs_shifted=oracle['teacher_correct']-shift['teacher_correct']>=2,
            feature_cosine_no_worse=oracle['feature_cosine_error']<=before['feature_cosine_error'],
            RX_protection=all(g['cases']['clean_SIG']['teacher_correct']>=g['cases']['learned']['teacher_correct'] for k,g in groups.items() if k.startswith('rx:')),
            clean_scene_protection=groups['scene:clean']['cases']['clean_SIG']['teacher_correct']>=groups['scene:clean']['cases']['learned']['teacher_correct'],
            nonclean_scene_protection=all(g['cases']['clean_SIG']['teacher_correct']>=g['cases']['learned']['teacher_correct']-1 for k,g in groups.items() if k.startswith('scene:') and k!='scene:clean'))
    passed=set(summary)=={'1','20'} and all(all(v.values()) for v in checks.values())
    return dict(status='PASS' if passed else 'FAIL',checks=checks,
        meaning='Clean-SIG-at-conditioner utility on fixed source checkpoints only; not deployable, TX accuracy, target benefit or a recovery bound; no automatic G training')
