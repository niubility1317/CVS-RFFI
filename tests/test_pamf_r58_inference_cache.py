import copy
import io
import pytest
import torch
from experiments.pamf_r46_stat_transport.normalization import StatisticGroupNorm
from experiments.pamf_r46_stat_transport.inference_cache import CachedStatisticGroupNorm,prepare_frozen_inference
from experiments.pamf_r46_stat_transport.models import CompactSuccessor

torch.set_num_threads(2)

def layers(channels,dtype):
    torch.manual_seed(channels)
    original=StatisticGroupNorm(4,channels).to(dtype=dtype)
    with torch.no_grad():
        original.stat_parameters.normal_(0,.4);original.weight.normal_(1,.2);original.bias.normal_(0,.1)
    original.eval().requires_grad_(False)
    cached=CachedStatisticGroupNorm(4,channels).to(dtype=dtype)
    cached.load_state_dict(original.state_dict(),strict=True);cached.eval().requires_grad_(False);cached.refresh_cache()
    return original,cached

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('shape',[(1,8,32),(3,12,256),(2,28,16)])
def test_same_arithmetic_outputs_and_input_gradients_are_exact(shape,dtype):
    old,new=layers(shape[1],dtype);x=(torch.randn(shape,dtype=dtype)*.4+.2).requires_grad_(True);z=x.detach().clone().requires_grad_(True)
    a,b=old(x),new(z);assert torch.equal(a,b)
    a.square().sum().backward();b.square().sum().backward();assert torch.equal(x.grad,z.grad)
    assert all(p.grad is None for m in (old,new) for p in m.parameters())

def test_portable_state_has_no_cache_and_loads_in_original_class():
    old,new=layers(12,torch.float32);x=torch.randn(2,12,64)
    assert tuple(old.state_dict())==tuple(new.state_dict()) and all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    assert '_inference_slopes' not in new.state_dict() and new._inference_slopes.numel()==8
    stream=io.BytesIO();torch.save(new.state_dict(),stream);stream.seek(0)
    fresh=StatisticGroupNorm(4,12).eval().requires_grad_(False);fresh.load_state_dict(torch.load(stream,weights_only=True),strict=True)
    assert torch.equal(new(x),fresh(x))

@pytest.mark.parametrize('change',['copy_weights','load_weights','dtype'])
def test_cache_refresh_after_normal_state_changes(change):
    old,new=layers(12,torch.float32)
    if change=='dtype':old=old.double();new=new.double()
    elif change=='load_weights':
        with torch.no_grad():old.stat_parameters.add_(.1)
        new.load_state_dict(old.state_dict(),strict=True)
    else:
        with torch.no_grad():old.stat_parameters.add_(.1);new.stat_parameters.copy_(old.stat_parameters)
    x=torch.randn(2,12,64,dtype=old.weight.dtype)
    with pytest.raises(ValueError):new(x)
    new.refresh_cache();assert torch.equal(old(x),new(x))

@pytest.mark.parametrize('bad',['training','grad','empty','shape'])
def test_invalid_cached_execution_rejects(bad):
    _,new=layers(12,torch.float32);x=torch.randn(2,12,64)
    if bad=='training':new.train()
    elif bad=='grad':new.stat_parameters.requires_grad_(True)
    elif bad=='empty':new._inference_slopes=torch.empty(0)
    else:x=torch.randn(2,8,64)
    with pytest.raises(ValueError):new(x)

def frozen_model():
    torch.manual_seed(58);model=CompactSuccessor('calibrated').eval().requires_grad_(False)
    with torch.no_grad():
        for module in model.modules():
            if type(module) is StatisticGroupNorm:module.stat_parameters.normal_(0,.02)
    return model

def test_complete_graph_has_exact_output_preserved_weights_and_counted_cache():
    original=frozen_model();before=copy.deepcopy(original.state_dict());cached,report=prepare_frozen_inference(original)
    x=torch.randn(2,2,256)*.1
    with torch.no_grad():assert torch.equal(original(x),cached(x))
    assert all(torch.equal(v,before[k]) for k,v in original.state_dict().items())
    assert all(torch.equal(v,before[k]) for k,v in cached.state_dict().items())
    assert report['parameters']==sum(p.numel() for p in cached.parameters())==276453
    assert report['additional_learnable_parameters']==0 and report['cache_tensor_scalars']==1344 and report['cache_tensor_bytes']==5376
    assert report['nonpersistent_cache_tensors']==168 and not report['measured_GPU_speedup'] and not report['formal_prediction_parity_verified']
    assert all(type(m) is not CachedStatisticGroupNorm for m in original.modules())
    repeated,again=prepare_frozen_inference(cached,clone=False)
    assert repeated is cached and again['cache_tensor_bytes']==5376
    with torch.no_grad():assert torch.equal(original(x),repeated(x))

@pytest.mark.parametrize('clone',[True,False])
def test_normal_preparation_inside_inference_mode_uses_versioned_norm_parameters(clone):
    with torch.inference_mode():
        original=frozen_model();x=torch.randn(1,2,256)*.1;before=original(x)
        cached,report=prepare_frozen_inference(original,clone=clone)
        assert torch.equal(before,cached(x)) and report['cache_tensor_bytes']==5376
        for module in cached.modules():
            if type(module) is CachedStatisticGroupNorm:assert not module.stat_parameters.is_inference()

@pytest.mark.parametrize('bad',['train','requires_grad','wrong_graph','clone_type'])
def test_complete_frozen_graph_required_before_mutation(bad):
    model=frozen_model()
    if bad=='train':model.train()
    elif bad=='requires_grad':next(model.parameters()).requires_grad_(True)
    elif bad=='wrong_graph':model=torch.nn.GroupNorm(4,12).eval().requires_grad_(False)
    with pytest.raises(ValueError):prepare_frozen_inference(model,clone='yes' if bad=='clone_type' else False)
    assert all(type(m) is not CachedStatisticGroupNorm for m in model.modules())
