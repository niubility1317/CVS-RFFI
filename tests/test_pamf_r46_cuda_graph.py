from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_r46_cuda_graph import design as d, graph, evaluation, runtime


class Restorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.))

    def training_outputs(self,x):
        return x*self.scale,torch.ones_like(x[:,0]),None,None

    def forward(self,x):
        return self.training_outputs(x)[0]


class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(160,6)

    def features(self,x):
        return x.flatten(1)[:,:160]

    def classify_features(self,x):
        return self.head(x)


def frozen():
    return Restorer().eval().requires_grad_(False),Reader().eval().requires_grad_(False)


@pytest.mark.parametrize('x',[
    torch.zeros(1,2,256,dtype=torch.float64),torch.zeros(1,2,256,dtype=torch.float16),
    torch.zeros(0,2,256),torch.zeros(1,3,256),torch.zeros(1,2,255),torch.zeros(2,256),
])
def test_invalid_iq_rejected(x):
    with pytest.raises(ValueError):
        graph.validate_iq(x)


def test_capture_rejects_cpu_before_cuda_calls():
    with pytest.raises(ValueError,match='CUDA sample'):
        graph.CapturedCall(lambda x:(x,),torch.ones(1,2,256))


@pytest.mark.parametrize('which',('restorer_training','reader_training','restorer_grad','reader_grad'))
def test_only_frozen_eval_modules(which):
    a,b=frozen()
    if which == 'restorer_training':a.train()
    elif which == 'reader_training':b.train()
    elif which == 'restorer_grad':a.requires_grad_(True)
    else:b.requires_grad_(True)
    with pytest.raises(ValueError,match='frozen'):
        graph.FrozenPipeline(a,b)


def test_pipeline_keeps_original_head_and_variance():
    a,b=frozen();x=torch.randn(3,2,256)
    iq,var,features,logits=graph.FrozenPipeline(a,b)(x)
    assert torch.equal(iq,a(x))
    assert torch.equal(var,torch.ones(3))
    assert torch.equal(logits,b.classify_features(b.features(iq)))
    assert features.shape==(3,160) and logits.shape==(3,6)
    assert torch.equal(graph.FrozenPipeline(a)(x)[0],a(x))


def fake_replay_call():
    # Exercise storage ownership/input replacement without pretending CPU is CUDA.
    call=graph.CapturedCall.__new__(graph.CapturedCall)
    call.device=torch.device('cpu');call.shape=(2,2,256)
    call.input=torch.zeros(call.shape);call.outputs=(torch.zeros(call.shape),)
    class Replay:
        count=0
        def replay(self):
            self.count+=1;call.outputs[0].copy_(call.input*3)
    call.graph=Replay()
    return call


def test_each_replay_replaces_input_and_preserves_previous_outputs():
    call=fake_replay_call();x=torch.randn(call.shape)
    first=call(x)[0];held=first.clone();second=call(x+1)[0]
    assert torch.equal(first,held) and torch.equal(first,x*3)
    assert torch.equal(second,(x+1)*3)
    assert first.data_ptr()!=call.outputs[0].data_ptr()
    assert call.graph.count==2


def test_static_shape_and_device_checked_before_replay():
    call=fake_replay_call()
    with pytest.raises(ValueError,match='static shape'):
        call(torch.zeros(1,2,256))
    call.device=torch.device('cuda:0')
    with pytest.raises(ValueError,match='device'):
        call(torch.zeros(2,2,256))
    assert call.graph.count==0


def test_graph_cache_separates_batch_shapes(monkeypatch):
    built=[]
    class Capture:
        warmup_seconds=0.;capture_seconds=0.
        def __init__(self,fn,x):built.append(tuple(x.shape));self.fn=fn
        def __call__(self,x):return self.fn(x)
    monkeypatch.setattr(graph,'CapturedCall',Capture)
    cache=graph.GraphCache(lambda x:(x*2,))
    for batch in (1,8,1,8,64):cache(torch.ones(batch,2,256))
    assert built==[(1,2,256),(8,2,256),(64,2,256)]
    assert [v['batch'] for v in cache.setup()]==[1,8,64]


def current_args():
    return SimpleNamespace(output=Path('/runs')/d.RUN/d.ARMS[0],kind=d.KIND,formal_reference=Path('/formal'))


def test_illegal_role_rejected_before_any_io(monkeypatch):
    monkeypatch.setattr(evaluation,'read_json',lambda p:pytest.fail('unexpected IO'))
    with pytest.raises(ValueError,match='roles'):
        evaluation.reference(current_args(),'query')


def test_target_waits_for_source_completion_without_truth_io(monkeypatch):
    args=current_args()
    def read(p):
        if p.name=='comparison_plan.json':return d.configuration()
        if p.name=='runtime_ready.json':return dict(status='FROZEN_RUNTIME_READY',arm=d.ARMS[0],parameters=350521,new_training_updates=0,F_parameter_updates=0,head_fit_updates=0)
        if p.name=='source_complete.json':return {'status':'RUNNING'}
        pytest.fail('target reference opened before source complete')
    monkeypatch.setattr(evaluation,'read_json',read)
    monkeypatch.setattr(evaluation.np,'load',lambda *a,**k:pytest.fail('IQ/truth opened'))
    with pytest.raises(ValueError,match='Complete source'):
        evaluation.reference(args,'target')


def test_target_does_not_require_source_accuracy_gate(monkeypatch):
    args=current_args();ids=np.arange(168000)
    def read(p):
        if p.name=='comparison_plan.json':return d.configuration()
        if p.name=='runtime_ready.json':return dict(status='FROZEN_RUNTIME_READY',arm=d.ARMS[0],parameters=350521,new_training_updates=0,F_parameter_updates=0,head_fit_updates=0)
        if p.name=='source_complete.json':return {'status':'ANALYZED'}
        if p.name=='manifest.json':return dict(status='FORMAL_REFERENCE_READY',count=168000,views=list(d.VIEWS),original_IQ_bytes_reused=True,TX_truth_reads=0)
        pytest.fail('unexpected reference read')
    monkeypatch.setattr(evaluation,'read_json',read)
    monkeypatch.setattr(evaluation.np,'load',lambda p,**k:ids)
    assert np.array_equal(evaluation.reference(args,'target')[1],ids)


def test_scorer_requires_independent_cpu_before_io(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(evaluation,'native_agreement',lambda *a:pytest.fail('prediction/truth IO'))
    with pytest.raises(ValueError,match='Independent CPU'):
        evaluation.score(SimpleNamespace(),'target')


def test_wrong_source_handle_rejected_before_any_io(monkeypatch):
    args=SimpleNamespace(teacher_run=Path('/wrong'),parent_action=Path('/wrong'))
    monkeypatch.setattr(runtime.factor,'validate_parent_root',lambda p:pytest.fail('unexpected IO'))
    with pytest.raises(ValueError,match='before IO'):
        runtime.validate_sources(args)


def test_unregistered_checkpoint_arm_rejected_before_io():
    with pytest.raises(ValueError,match='before checkpoint IO'):
        runtime.load_model(SimpleNamespace(),'unregistered','cpu')


def test_all_fixed_versions_and_seven_scenes_retained():
    cfg=d.configuration()
    assert d.ARMS==('R46_350521_graph','R18_factor625_stage_graph','R18_factor625_cond_graph')
    assert cfg['training_updates']==cfg['optimizer_updates']==cfg['head_fit_updates']==0
    assert not cfg['training_F_access'] and not cfg['source_selection']
    assert cfg['formal_count']==168000 and len(cfg['views'])==7 and cfg['batch']==256


@pytest.mark.parametrize('arm',d.ARMS)
@pytest.mark.parametrize('zero',(False,True))
def test_actual_r46_r18_arithmetic_equal_without_tensor_to_host_branch(monkeypatch,arm,zero):
    from experiments.pamf_r44_pruned_refine.models import CompactSuccessor as Parent
    from experiments.pamf_r46_conditional_factor.models import CompactSuccessor as Factor
    from experiments.pamf_r46_cuda_graph import program
    torch.set_num_threads(2);torch.manual_seed(17)
    model=(Parent('prune_refine') if arm==d.ARMS[0] else Factor(d.SOURCE_ROWS[arm])).eval().requires_grad_(False)
    x=torch.zeros(2,2,256) if zero else torch.randn(2,2,256)*.1
    with torch.no_grad():
        expected=model.training_outputs(x)
        def forbidden(t):
            raise AssertionError('Tensor converted to Python bool inside static program')
        with monkeypatch.context() as m:
            m.setattr(torch.Tensor,'__bool__',forbidden)
            got=program.outputs(model,x)
        for a,b in zip((expected[0],expected[1],expected[3]),(got[0],got[1],got[3])):
            assert torch.equal(a,b)
        for attr in ('kernel','main','image','response'):
            assert torch.equal(getattr(expected[2],attr),getattr(got[2],attr))
