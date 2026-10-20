import math
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.pamf_r46_source_moments import design as d
from experiments.pamf_r46_source_moments.metrics import descriptor,measure
from experiments.pamf_r46_source_moments import runner


def signal(batch=3,dtype=torch.float64):
    generator=torch.Generator().manual_seed(21)
    return torch.randn(batch,2,256,generator=generator,dtype=dtype)


def test_exact_balanced_four_phase_constellation_moments():
    z=torch.tensor([1+0j,0+1j,-1+0j,0-1j],dtype=torch.complex128).repeat(64)
    iq=torch.stack([z.real,z.imag])[None]
    got,floor=descriptor(iq)
    torch.testing.assert_close(got[0,:3],torch.tensor([1.,-1.,1.],dtype=torch.float64),rtol=0,atol=1e-12)
    assert not floor.any() and got.shape==(1,11)
    # For lag4, paired centered windows coincide: cum(z,z,z*,z*)=-1.
    assert got[0,d.FEATURES.index('lag4_real')]==pytest.approx(-1.)
    assert got[0,d.FEATURES.index('lag4_imag')]==pytest.approx(0.)


def test_global_phase_positive_amplitude_and_constant_offset_invariance():
    x=signal();z=torch.complex(x[:,0],x[:,1]);changed=2.3*z*complex(math.cos(.7),math.sin(.7))+complex(5.,-3.)
    a,_=descriptor(x);b,_=descriptor(torch.stack([changed.real,changed.imag],1))
    torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)


def test_complex_conjugation_flips_only_imaginary_lag_components():
    x=signal();a,_=descriptor(x);y=x.clone();y[:,1].neg_();b,_=descriptor(y)
    expected=a.clone()
    for i,name in enumerate(d.FEATURES):
        if name.endswith('_imag'):expected[:,i].neg_()
    torch.testing.assert_close(b,expected,rtol=1e-12,atol=1e-12)


def test_zero_and_constant_signals_are_finite_and_floor_is_reported():
    x=torch.ones(2,2,256,dtype=torch.float32);value,floor=descriptor(x)
    assert torch.equal(value,torch.zeros_like(value)) and floor.all()


def test_differentiable_metrics_have_finite_nonzero_gradient():
    x=signal(dtype=torch.float32).requires_grad_(True);target=signal(dtype=torch.float32)*1.01
    target=target+torch.sin(torch.arange(256)*.31)[None,None,:]*.01
    value=measure(x,target,target)
    value['paired_clean_moment_mse'].mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0


@pytest.mark.parametrize('bad',('shape','length','nan','integer'))
def test_invalid_IQ_rejected(bad):
    x=signal()
    if bad=='shape':x=x[:,:1]
    elif bad=='length':x=x[:,:,:9]
    elif bad=='nan':x[0,0,0]=float('nan')
    else:x=x.to(torch.int64)
    with pytest.raises(ValueError):descriptor(x)


def test_self_reference_and_moment_feature_arithmetic_close():
    x=signal();out=measure(x,x,x)
    assert len(out)==29 and all(v.shape==(3,) for v in out.values())
    assert out['teacher_moment_mse'].sum()==out['paired_clean_moment_mse'].sum()==0
    for prefix in ('paired','teacher'):
        torch.testing.assert_close(out[prefix+'_clean_moment_mse'] if prefix=='paired' else out['teacher_moment_mse'],
            torch.stack([out[prefix+'_'+name+'_squared_error'] for name in d.FEATURES],1).mean(1))


def test_unknown_F_and_target_arguments_rejected():
    required=[v for key in ('compact-parent','factor-run','complex-run','parent-action','source-cache','output') for v in ('--'+key,'placeholder')]+['--expected-commit','method']
    for flag in ('--reader-folder','--target-truth','--formal-reference'):
        with pytest.raises(SystemExit):runner.parser().parse_args(required+[flag,'forbidden'])


def test_existing_output_rejected_before_any_source_IO(tmp_path,monkeypatch):
    p=tmp_path/d.RUN;p.mkdir()
    monkeypatch.setattr(runner,'validate_sources',lambda _:pytest.fail('must not read source'))
    with pytest.raises(ValueError):runner.diagnose(SimpleNamespace(output=p))


def test_full_source_only_fixture_has_fixed_exposure_and_all_rows(tmp_path,monkeypatch):
    x=signal(256,dtype=torch.float32)
    panel={v:x.clone() for v in d.VIEWS}
    models={a:torch.nn.Identity().eval() for a in d.ARMS}
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(runner,'code_commit',lambda:'method')
    monkeypatch.setattr(runner,'validate_sources',lambda _:None)
    monkeypatch.setattr(runner,'load_model',lambda _,a:(models[a],dict(epoch=20,updates=d.SOURCE_BUDGET[a][1])))
    monkeypatch.setattr(runner,'load_panel',lambda _:(panel,np.arange(256),np.array(['L_s']*128+['U_s']*128)))
    args=SimpleNamespace(output=tmp_path/d.RUN,expected_commit='method',_owns_output=False)
    runner.diagnose(args)
    import json
    result=json.loads((args.output/'results.json').read_text(encoding='utf-8'))
    assert result['status']=='SOURCE_MOMENTS_COMPLETE' and result['new_network_versions']==0
    assert result['all_checkpoint_states_unchanged'] and result['F_weight_reads']==result['target_reads']==result['TX_label_reads']==0
    assert sum(v['actual_IQ_inputs']['center'] for v in result['rows'].values())==3072
    assert all(v['actual_IQ_inputs']==dict(smoke=2,center=768) for v in result['rows'].values())
    assert sum(len(group['metrics']) for row in result['rows'].values() for view in row['views'].values() for group in view.values())==1044
