import copy
import json
import torch
from . import design as d
from .objective import feature_mask,selected_masks


def checks():
    torch.set_num_threads(2);torch.manual_seed(8329)
    g=torch.randn(12,96,32);mask=feature_mask(g);y=torch.arange(12)%6
    original=torch.randn(12,6);spatial=original.clone();spatial[torch.arange(12),y]-=torch.arange(12)/12
    m,r,s,drop=selected_masks(original,spatial,y,mask,91)
    assert torch.equal(m.sum(1),r.sum(1)) and torch.equal(m[~s],torch.ones_like(m[~s]))
    assert torch.equal(r,selected_masks(original,spatial,y,mask,91)[1])
    negatives=[lambda:feature_mask(g,role='U_s'),lambda:feature_mask(g,role='V'),
        lambda:feature_mask(g[:,:,:31]),lambda:feature_mask(g[:,:95]),
        lambda:feature_mask(torch.full_like(g,float('nan'))),lambda:feature_mask(torch.zeros_like(g)),
        lambda:selected_masks(original,spatial,y,mask,91,role='query'),
        lambda:selected_masks(original[:,:5],spatial,y,mask,91),
        lambda:selected_masks(original,spatial,y.float(),mask,91),
        lambda:selected_masks(original,spatial,torch.full_like(y,-1),mask,91),
        lambda:selected_masks(torch.full_like(original,float('inf')),spatial,y,mask,91),
        lambda:selected_masks(original,original,y,mask,91)]
    for bad in negatives:
        try:bad()
        except (ValueError,RuntimeError):pass
        else:raise AssertionError('Forbidden role/shape/nonfinite/undefined mask was accepted')
    c=d.config(d.rows()[0]);bad=copy.deepcopy(c);bad['feature_drop_fraction']=.5
    try:d.validate(bad)
    except ValueError:pass
    else:raise AssertionError('Unregistered coefficient change accepted')
    from experiments.cvs_receiver_dro50 import model as mapi
    from .audit import measure_batch
    model=mapi.build(d.source_config(),'cpu').eval()
    before={k:v.clone() for k,v in model.state_dict().items()};before_rng=torch.get_rng_state().clone()
    x=torch.randn(12,2,256,generator=torch.Generator().manual_seed(8421))
    result=measure_batch(model,x,y,d.MASK_SEED)
    # Input generation is deterministic; mask RNG is isolated from native RNG.
    assert torch.equal(before_rng,torch.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert all(p.grad is None for p in model.parameters())
    assert [g.numel() for g in result['vectors']['ORIGINAL']]==[3456,6912,576,960]
    assert all(torch.isfinite(g).all() for vv in result['vectors'].values() for g in vv)
    assert len(dict(model.named_modules())[c['layer']]._forward_hooks)==0
    assert result['activation_nonnegative']
    return dict(status='PASS',forbidden_or_invalid_cases=len(negatives)+1,
        exact_random_packet_and_zero_counts=True,isolated_mask_RNG=True,
        actual_SIG_TIME_t2_shape=[12,96,32],actual_native_default_head_gradients=True,
        model_state_unchanged=True,accumulated_parameter_gradients=False,
        head_fit_updates=0,real_IQ_or_checkpoint_read=False)


if __name__=='__main__':print(json.dumps(checks()))
