import pytest
import torch
from experiments.pamf_stage2_inference import program,engine,design as d,runtime
from experiments.pamf_r65_four_route.stage_polyphase import StageDistilled
from experiments.pamf_action_restore.operators import Action
from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_uncertainty_restore.models import weighted_map


def model():
    torch.set_num_threads(2)
    torch.manual_seed(16)
    m=StageDistilled().eval().requires_grad_(False)
    with torch.no_grad():
        for core in (m.frozen.coarse.temporal,m.frozen.coarse.spectral,m.frozen.estimator.core,*m.prior):
            core.out.weight.normal_(0,.005)
    return m


@pytest.mark.parametrize('zero',(False,True))
def test_static_original_order_and_reuse_never_branch_on_tensor(monkeypatch,zero):
    m=model();x=torch.zeros(2,2,256) if zero else torch.randn(2,2,256)*.1
    with torch.no_grad():
        expected=m.training_outputs(x)
        def forbidden(t):raise AssertionError('Tensor-to-host branch in program')
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor,'__bool__',forbidden)
            static=program.outputs(m,x,reuse=False)
            reused=program.outputs(m,x,reuse=True)
        assert torch.equal(expected[0],static[0])
        assert torch.equal(expected[1],static[1])
        torch.testing.assert_close(expected[0],reused[0],rtol=2e-4,atol=3e-6)
        assert bool(static[-1]) and bool(reused[-1])


def test_reused_initial_residual_matches_general_real_linear_solver():
    torch.manual_seed(3);dtype=torch.complex128
    kernel=torch.randn(3,64,dtype=dtype)*.02;kernel[:,24]+=1
    a=Action(kernel,1+torch.randn(3,256,dtype=dtype)*.01,torch.randn(3,256,dtype=dtype)*.01)
    y,c,p=(torch.randn(3,256,dtype=dtype) for _ in range(3))
    variance=torch.tensor([1e-8,1e-4,.1],dtype=torch.float64)[:,None]
    raw=a.adjoint(y-a(c))
    expected=weighted_map(a,y,p,variance,steps=2,initial=c)
    got=program.reused_map(a,p,c,variance,raw)
    torch.testing.assert_close(got,expected,rtol=1e-10,atol=1e-10)
    with pytest.raises(ValueError,match='per window'):
        program.reused_map(a,p,c,variance.expand(-1,256),raw)


@pytest.mark.parametrize('reuse,count',[(False,43),(True,27)])
def test_actual_fft_count_and_unchanged_parameters(monkeypatch,reuse,count):
    m=model();state={k:v.clone() for k,v in m.state_dict().items()};calls=[]
    for name in ('fft','ifft'):
        original=getattr(torch.fft,name)
        def wrap(*a,_fn=original,**kw):calls.append(1);return _fn(*a,**kw)
        monkeypatch.setattr(torch.fft,name,wrap)
    with torch.no_grad():program.outputs(m,torch.randn(1,2,256)*.1,reuse=reuse)
    assert len(calls)==count
    assert sum(p.numel() for p in m.parameters())==266951
    assert all(torch.equal(v,state[k]) for k,v in m.state_dict().items())


@pytest.mark.parametrize('reuse',(False,True))
@pytest.mark.parametrize('bad',(float('nan'),float('inf')))
def test_public_engine_rejects_nonfinite_input(reuse,bad):
    call=engine.InferenceEngine(model(),reuse=reuse,graph=False)
    x=torch.zeros(1,2,256);x[0,0,0]=bad
    with pytest.raises(FloatingPointError):call(x)


def test_one_public_entry_for_different_current_inputs():
    m=model();call=engine.InferenceEngine(m,graph=False,reuse=False)
    first=torch.randn(2,2,256)*.1;second=first+.2
    with torch.no_grad():
        a=call(first)[0];held=a.clone();b=call(second)[0]
        assert torch.equal(a,m(first)) and torch.equal(b,m(second)) and torch.equal(a,held)
    with pytest.raises(ValueError):call(torch.zeros(1,2,255))
    with pytest.raises(ValueError,match='frozen'):engine.InferenceEngine(m.train(),graph=False)


def test_only_registered_parent_and_no_new_training(tmp_path):
    from types import SimpleNamespace
    with pytest.raises(ValueError,match='before IO'):
        runtime.validate_source(SimpleNamespace(parent_run=tmp_path,parent_action=tmp_path))
    cfg=d.configuration()
    assert cfg['training_updates']==cfg['optimizer_updates']==cfg['head_fit_updates']==0
    assert cfg['formal_count']==168000 and cfg['batch']==128 and len(cfg['views'])==7
    assert not cfg['source_selection'] and cfg['all_versions_formal'] and cfg['no_native_fallback']


def test_truth_side_rejects_GPU_and_unregistered_role_before_IO(monkeypatch):
    from types import SimpleNamespace
    from pathlib import Path
    from experiments.pamf_stage2_inference import evaluation as e
    args=SimpleNamespace(output=Path('/runs')/d.RUN/d.ARMS[0],kind=d.KIND)
    monkeypatch.setattr(e,'read_json',lambda *a:pytest.fail('Unexpected IO'))
    with pytest.raises(ValueError,match='roles'):e.reference(args,'query')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(ValueError,match='before IO'):e.score(args,'target')


def test_target_waits_for_source_completion_without_accuracy_gate(monkeypatch):
    from types import SimpleNamespace
    from pathlib import Path
    from experiments.pamf_stage2_inference import evaluation as e
    args=SimpleNamespace(output=Path('/runs')/d.RUN/d.ARMS[0],kind=d.KIND,formal_reference=Path('/formal'))
    def read(path):
        if path.name=='comparison_plan.json':return d.configuration()
        if path.name=='runtime_ready.json':return dict(status='FROZEN_RUNTIME_READY',arm=d.ARMS[0],parameters=d.PARAMETERS,new_training_updates=0,F_parameter_updates=0,head_fit_updates=0)
        if path.name=='source_complete.json':return dict(status='RUNNING')
        pytest.fail('Target opened before source completion')
    monkeypatch.setattr(e,'read_json',read)
    monkeypatch.setattr(e.np,'load',lambda *a,**k:pytest.fail('IQ or truth opened'))
    with pytest.raises(ValueError,match='Complete source'):e.reference(args,'target')


def test_formal_prediction_uses_pipeline_for_every_view_and_tail(tmp_path,monkeypatch):
    import json
    import numpy as np
    from types import SimpleNamespace
    from experiments.pamf_stage2_inference import evaluation as e
    root=tmp_path/d.RUN;root.mkdir();row=root/d.ARMS[0];row.mkdir();received=tmp_path/'inputs';received.mkdir()
    values=np.zeros((130,2,256),dtype=np.float32)
    for view in d.VIEWS:np.save(received/(view+'.npy'),values)
    args=SimpleNamespace(output=root,arm=d.ARMS[0])
    monkeypatch.setattr(e,'reference',lambda *a:(received,np.arange(130),None,None))
    monkeypatch.setattr(e,'array_tensor',lambda value,device:torch.from_numpy(value.copy()))
    monkeypatch.setattr(e.torch.cuda,'synchronize',lambda:None)
    class Pipeline:
        batches=[]
        def __call__(self,x):
            self.batches.append(len(x))
            return x+1,torch.ones(len(x)),torch.ones(len(x),160),torch.ones(len(x),6)
        def setup(self):return []
    pipeline=Pipeline();e.predict(args,'target',pipeline)
    assert pipeline.batches==[128,2]*7
    folder=row/'target_confirmation'/d.KIND/'prediction'
    done=json.loads((folder/'complete.json').read_text())
    assert done['TX_truth_reads']==done['state_updates']==0 and done['views']==list(d.VIEWS)
    assert len(e.base.prediction_arrays(folder,np.arange(130),d.KIND,'target'))==7
    assert np.array_equal(np.load(folder/'clean.restored.iq.npy'),values+1)
