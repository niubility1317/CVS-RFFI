"""Native Sinc/AdamW CPU checks; no IQ data, checkpoint or classifier fitting."""
import copy
import json
import sys
from pathlib import Path
import torch
from torch import nn

NATIVE = Path(__file__).resolve().parents[1] / 'adv3b02_xuc' / 'code'
sys.path.insert(0, str(NATIVE))
from model import SincConv1d
from .coordinate import CoordinateSinc, PATH, SCALE_HZ, convert_model, convert_optimizer


class Fixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.id_backbone=nn.Module();self.id_backbone.encoder=nn.Module()
        self.id_backbone.encoder.core=nn.Module();self.id_backbone.encoder.core.id_backbone=nn.Module()
        self.id_backbone.encoder.core.id_backbone.sinc=SincConv1d(24,79,sample_rate=25e6,input_len=256)
        self.other=nn.Parameter(torch.tensor(.7))
        # An untouched existing-head-shaped sentinel; it is never fit or used
        # to train a classifier. The tested output is the native Sinc IQ output.
        self.default_class_W=nn.Parameter(torch.arange(960,dtype=torch.float32).reshape(6,160))
        self.register_buffer('other_buffer',torch.arange(7,dtype=torch.float32))
    def forward(self,x):return self.get_submodule(PATH).forward_iq_pair(x)*self.other


def _raises(callback):
    try:callback()
    except (ValueError,TypeError):return
    raise AssertionError('Expected unsupported coordinate/state/order rejection')


def check():
    torch.set_num_threads(2)
    raw=Fixture().eval();sinc=raw.get_submodule(PATH)
    initial={'low_hz_':sinc.low_hz_.detach().clone(),'band_hz_':sinc.band_hz_.detach().clone()}
    x=torch.sin(torch.arange(4*2*256,dtype=torch.float32).reshape(4,2,256)/7)
    rng=torch.get_rng_state().clone()
    centered0=convert_model(raw,initial,'CENTER_25K')
    raw0=convert_model(raw,initial,'RAW_HZ')
    with torch.no_grad():
        original_filter=sinc._filters(torch.device('cpu'),torch.float32)
        assert torch.equal(original_filter,centered0.get_submodule(PATH)._filters(torch.device('cpu'),torch.float32))
        assert torch.equal(raw(x),centered0(x)) and torch.equal(raw(x),raw0(x))
    assert torch.count_nonzero(centered0.get_submodule(PATH).low_coordinate)==0
    assert torch.count_nonzero(centered0.get_submodule(PATH).band_coordinate)==0
    opt=torch.optim.AdamW(raw.parameters(),lr=2e-4,betas=(.8,.95),eps=2e-8,weight_decay=1e-4,amsgrad=True)
    # Actual standard AdamW allocated moments from seven synthetic updates;
    # untouched W keeps genuine absent state/None gradient, rather than zeros.
    for step in range(7):
        for p in raw.parameters():p.grad=None
        sinc.low_hz_.grad=torch.full_like(sinc.low_hz_,1e-6*(step+1))
        sinc.band_hz_.grad=torch.linspace(-2e-6,3e-6,24).reshape(24,1)*(step+1)
        raw.other.grad=torch.tensor(.1*(step+1))
        opt.step()
    opt.zero_grad(set_to_none=True)
    # A visible final physical displacement, independent of any data or unit scan.
    with torch.no_grad():
        sinc.low_hz_.add_(torch.arange(24,dtype=torch.float32).reshape(24,1)*113.3)
        sinc.band_hz_.add_(torch.arange(24,dtype=torch.float32).reshape(24,1)*37.7)
    raw.eval();before={k:v.detach().clone() for k,v in raw.state_dict().items()}
    saved=copy.deepcopy(opt.state_dict());raw_named=list(raw.named_parameters())
    center=convert_model(raw,initial,'CENTER_25K')
    cs=center.get_submodule(PATH)
    assert cs.mode=='CENTER_25K' and cs.scale_hz==25000.
    assert torch.equal(cs.low_origin_hz,initial['low_hz_']) and torch.equal(cs.band_origin_hz,initial['band_hz_'])
    assert torch.equal(cs.low_coordinate,(sinc.low_hz_.detach()-initial['low_hz_'])/25000)
    assert torch.equal(cs.band_coordinate,(sinc.band_hz_.detach()-initial['band_hz_'])/25000)
    assert sum(p.numel() for p in raw.parameters())==sum(p.numel() for p in center.parameters())
    for name,value in before.items():
        if not name.startswith(PATH+'.'):
            assert torch.equal(value,center.state_dict()[name])
    assert torch.equal(sinc.t_,cs.t_) and torch.equal(sinc.window_,cs.window_)
    with torch.no_grad():
        raw_filter=sinc._filters(torch.device('cpu'),torch.float32)
        center_filter=cs._filters(torch.device('cpu'),torch.float32)
        filter_error=float((raw_filter-center_filter).abs().max())
        output_error=float((raw(x)-center(x)).abs().max())
        assert torch.allclose(raw_filter,center_filter,atol=1e-5,rtol=1e-5)
        assert torch.allclose(raw(x),center(x),atol=1e-4,rtol=1e-5)
    converted=convert_optimizer(saved,raw_named,center,'CENTER_25K')
    after=converted.state_dict()
    assert saved['param_groups']==after['param_groups']
    sinc_indices=[i for i,(name,p) in enumerate(raw_named) if name in (PATH+'.low_hz_',PATH+'.band_hz_')]
    for index,state in saved['state'].items():
        for key,value in state.items():
            expected=value*(25000 if key=='exp_avg' else 25000**2) if index in sinc_indices and key!='step' else value
            assert torch.equal(after['state'][index][key],expected)
    assert 1 not in saved['state'] and 1 not in after['state']  # untouched default_class_W
    roundtrip=convert_model(center,initial,'RAW_HZ')
    with torch.no_grad():
        no_grad_roundtrip=convert_model(center,initial,'RAW_HZ')
    assert no_grad_roundtrip.get_submodule(PATH).low_coordinate.requires_grad==cs.low_coordinate.requires_grad
    assert no_grad_roundtrip.get_submodule(PATH).band_coordinate.requires_grad==cs.band_coordinate.requires_grad
    backopt=convert_optimizer(after,list(center.named_parameters()),roundtrip,'RAW_HZ',source_mode='CENTER_25K')
    back=backopt.state_dict()
    assert saved['param_groups']==back['param_groups']
    for index,state in saved['state'].items():
        for key,value in state.items():
            if index in sinc_indices and key!='step':assert torch.allclose(value,back['state'][index][key],rtol=2e-7,atol=0)
            else:assert torch.equal(value,back['state'][index][key])
    with torch.no_grad():assert torch.allclose(roundtrip(x),raw(x),atol=1e-4,rtol=1e-5)
    # Every version in the authoritative cache contract must invalidate it.
    cache_checks=[]
    for leaf in ('low_coordinate','band_coordinate','low_origin_hz','band_origin_hz','t_','window_'):
        candidate=copy.deepcopy(center).eval().get_submodule(PATH)
        with torch.no_grad():
            old=candidate._filters(torch.device('cpu'),torch.float32)
            assert candidate._filters(torch.device('cpu'),torch.float32) is old
            value=getattr(candidate,leaf)
            if leaf=='t_':value.mul_(1.001)
            elif leaf=='window_':value.mul_(.999)
            else:value.add_(.01 if 'coordinate' in leaf else 100.)
            new=candidate._filters(torch.device('cpu'),torch.float32)
            assert new is not old and not torch.equal(new,old)
            cache_checks.append(leaf)
    # The authoritative cache tracks versions, not tensor identities. Two
    # independently created functional-call states can collide on version0.
    # Caller clearing before each branch is therefore an explicit requirement.
    temporary=copy.deepcopy(center).eval()
    params={name:p.detach().clone() for name,p in temporary.named_parameters()}
    buffers={name:b.detach().clone() for name,b in temporary.named_buffers()}
    first=dict(params);second=dict(params)
    first[PATH+'.low_coordinate']=params[PATH+'.low_coordinate']+.25
    second[PATH+'.low_coordinate']=params[PATH+'.low_coordinate']+.5
    with torch.no_grad():
        output_first=torch.func.functional_call(temporary,(first,buffers),(x,),strict=True)
        output_stale=torch.func.functional_call(temporary,(second,buffers),(x,),strict=True)
        assert torch.equal(output_first,output_stale)  # known version collision
        target=temporary.get_submodule(PATH)
        for key in ('_filter_cache_key','_filter_cache','_coordinate_cache_key'):setattr(target,key,None)
        output_fresh=torch.func.functional_call(temporary,(second,buffers),(x,),strict=True)
        independent=copy.deepcopy(center).eval();independent.load_state_dict({**second,**buffers},strict=True)
        assert not torch.equal(output_first,output_fresh)
        assert torch.equal(output_fresh,independent(x))
    _raises(lambda:convert_model(raw,initial,'CENTER_1K'))
    wrong=copy.deepcopy(raw);wrong.get_submodule(PATH).sample_rate=5e6
    _raises(lambda:convert_model(wrong,initial,'CENTER_25K'))
    _raises(lambda:convert_model(raw,{'low_hz_':torch.zeros(23,1),'band_hz_':initial['band_hz_']},'CENTER_25K'))
    fitted_origin={'low_hz_':sinc.low_hz_.detach().clone(),'band_hz_':sinc.band_hz_.detach().clone()}
    _raises(lambda:convert_model(raw,fitted_origin,'CENTER_25K'))
    groups=copy.deepcopy(saved);groups['param_groups'].append(copy.deepcopy(groups['param_groups'][0]))
    _raises(lambda:convert_optimizer(groups,raw_named,center,'CENTER_25K'))
    _raises(lambda:convert_optimizer(saved,list(reversed(raw_named)),center,'CENTER_25K'))
    incomplete=copy.deepcopy(saved);del incomplete['state'][sinc_indices[0]]['exp_avg_sq']
    _raises(lambda:convert_optimizer(incomplete,raw_named,center,'CENTER_25K'))
    absent=copy.deepcopy(saved);del absent['state'][sinc_indices[0]]
    _raises(lambda:convert_optimizer(absent,raw_named,center,'CENTER_25K'))
    malformed=copy.deepcopy(saved);malformed['state'][sinc_indices[0]]['exp_avg']=torch.zeros(23,1)
    _raises(lambda:convert_optimizer(malformed,raw_named,center,'CENTER_25K'))
    missing_group=copy.deepcopy(saved);del missing_group['param_groups'][0]['eps']
    _raises(lambda:convert_optimizer(missing_group,raw_named,center,'CENTER_25K'))
    _raises(lambda:convert_optimizer(saved,raw_named,center,'RAW_HZ'))
    assert all(torch.equal(v,raw.state_dict()[k]) for k,v in before.items())
    assert all(p.grad is None for p in raw.parameters()) and torch.equal(torch.get_rng_state(),rng)
    return dict(status='VERIFIED',torch=torch.__version__,device='cpu',dtype='float32',
        initial_native_kernel_and_output_exact=True,frozen_kernel_max_abs_error=filter_error,
        synthetic_IQ_output_max_abs_error=output_error,parameter_count_unchanged=True,
        non_Sinc_state_exact=True,actual_AdamW_moment_scale_and_roundtrip=True,
        step_group_options_and_other_state_exact=True,genuine_absent_state_preserved=True,
        cache_version_invalidations=cache_checks,unsupported_scheme_shape_order_state_rejected=True,
        constructor_origin_enforced=True,functional_call_same_version_requires_caller_cache_clear=True,
        requires_grad_preserved_under_no_grad_conversion=True,
        raw_model_and_RNG_unchanged=True,source_IQ_reads=0,checkpoint_reads=0,head_fit_updates=0,
        boundary='Fixed-final physical-equivalent coordinate hypothesis; not historical CENTER training trajectory')


if __name__=='__main__':print(json.dumps(check(),allow_nan=False,indent=2))
