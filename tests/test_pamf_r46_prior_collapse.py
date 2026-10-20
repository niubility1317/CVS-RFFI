import copy,importlib
import pytest
import torch
from torch import nn
from experiments.pamf_r17_compact.models import FactorConv

torch.set_num_threads(2)

def api():
    try:return importlib.import_module('experiments.pamf_r46_prior_collapse.collapse')
    except ModuleNotFoundError:pytest.fail('Frozen shared-prior convolution collapse is not implemented')


@pytest.mark.parametrize('stride,padding,dilation,bias',[(1,2,1,True),(2,4,2,True),(1,0,1,False),(2,0,2,False)])
def test_collapsed_pair_matches_strided_dilated_biased_function(stride,padding,dilation,bias):
    p=api();torch.manual_seed(71)
    conv=nn.Conv1d(3,7,5,stride=stride,padding=padding,dilation=dilation,bias=bias).double()
    factor=FactorConv.from_conv(conv,fraction=.625).eval().requires_grad_(False)
    merged=p.dense_pair(factor)
    x=torch.randn(4,3,53,dtype=torch.float64)
    torch.testing.assert_close(merged(x),factor(x),rtol=1e-12,atol=1e-12)
    assert not merged.training and all(not v.requires_grad for v in merged.parameters())


class PriorModel(nn.Module):
    def __init__(self,shared=True):
        super().__init__();first=FactorConv.from_conv(nn.Conv1d(4,8,5),fraction=.625)
        cores=[]
        for i in range(3):
            module=copy.deepcopy(first)
            if shared:module.left.weight=first.left.weight;module.right.weight=first.right.weight
            with torch.no_grad():module.right.bias.fill_(i*.13)
            cores.append(nn.Sequential(module,nn.GroupNorm(4,8)))
        self.prior=nn.ModuleList(cores)
        self.context=FactorConv.from_conv(nn.Conv1d(4,8,5),fraction=.625)

    def forward(self,x):return tuple(v(x) for v in self.prior)


def test_shared_kernels_remain_shared_and_stage_biases_remain_independent():
    p=api();model=PriorModel().eval().requires_grad_(False);before=copy.deepcopy(model.state_dict())
    got,report=p.collapse_prior(model)
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    assert all(isinstance(core[0],FactorConv) for core in model.prior)
    assert isinstance(got.context,FactorConv)
    assert isinstance(got.prior[0][0],nn.Conv1d)
    assert got.prior[0][0].weight is got.prior[1][0].weight is got.prior[2][0].weight
    assert len({id(core[0].bias) for core in got.prior})==3
    assert [core[0].bias[0].item() for core in got.prior]==pytest.approx([0.,.13,.26])
    assert report['factor_modules_replaced']==3 and report['unique_kernel_pairs']==1
    x=torch.randn(3,4,32)
    for a,b in zip(model(x),got(x)):torch.testing.assert_close(a,b,rtol=1e-5,atol=2e-6)


def test_equal_valued_independent_weights_do_not_become_shared():
    p=api();model=PriorModel(shared=False).eval().requires_grad_(False)
    got,report=p.collapse_prior(model)
    assert len({id(core[0].weight) for core in got.prior})==3
    assert report['unique_kernel_pairs']==3


@pytest.mark.parametrize('bad',['trainmode','gradient','leftbias','rightstride','rightpadding','nonfinite'])
def test_invalid_pair_or_trainable_model_rejected_without_input_mutation(bad):
    p=api();model=PriorModel().eval().requires_grad_(False)
    if bad=='trainmode':model.train()
    elif bad=='gradient':model.prior[0][0].left.weight.requires_grad_(True)
    elif bad=='leftbias':model.prior[0][0].left.bias=nn.Parameter(torch.zeros(5),requires_grad=False)
    elif bad=='rightstride':model.prior[0][0].right.stride=(2,)
    elif bad=='rightpadding':model.prior[0][0].right.padding=(1,)
    else:
        with torch.no_grad():model.prior[0][0].left.weight[0,0,0]=float('nan')
    ids=[id(core[0]) for core in model.prior]
    with pytest.raises(ValueError):p.collapse_prior(model)
    assert [id(core[0]) for core in model.prior]==ids and all(isinstance(core[0],FactorConv) for core in model.prior)


@pytest.mark.parametrize('arm',['complex625','complex625_identity'])
def test_complete_restorer_cost_numeric_outputs_and_state_roundtrip(arm):
    p=api();base=importlib.import_module('tests.test_pamf_r46_complex_fidelity')
    m=importlib.import_module('experiments.pamf_r46_complex_fidelity.models')
    torch.manual_seed(31);model=m.compact_from_parent(base.parent_model(),arm).eval().requires_grad_(False)
    before=copy.deepcopy(model.state_dict());got,report=p.collapse_prior(model)
    assert report['before_parameters']==278180 and report['after_parameters']==299297
    assert report['factor_modules_replaced']==84 and report['unique_kernel_pairs']==28
    assert report['new_training_updates']==report['IQ_reads']==report['F_reads']==report['TX_labels_read']==0
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    calls=[]
    def measured(module,x):
        count=[0];mac=[0];handles=[]
        def hook(layer,inputs,output):
            count[0]+=1;mac[0]+=output.numel()*(layer.in_channels//layer.groups)*layer.kernel_size[0]
        for layer in module.modules():
            if isinstance(layer,nn.Conv1d):handles.append(layer.register_forward_hook(hook))
        with torch.no_grad():result=module(x)
        for handle in handles:handle.remove()
        return result,count[0],mac[0]
    x=torch.randn(2,2,256)*.1
    a,ca,ma=measured(model,x);b,cb,mb=measured(got,x)
    torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
    assert ca-cb==84 and mb>ma
    fresh,_=p.collapse_prior(model);fresh.load_state_dict(got.state_dict(),strict=True)
    with torch.no_grad():assert torch.equal(got(x),fresh(x))
    assert got.prior[0].encoder.blocks[0][0].net[0].weight is got.prior[1].encoder.blocks[0][0].net[0].weight


def package(name):
    try:return importlib.import_module('experiments.pamf_r46_prior_collapse.'+name)
    except ModuleNotFoundError:pytest.fail('Prior-collapse formal runtime is not implemented')


@pytest.mark.parametrize('index',[0,1])
def test_expanded_full_checkpoint_strict_reload_and_conflicting_alias_rejected(tmp_path,index):
    r=package('runtime');d=package('design');base=importlib.import_module('tests.test_pamf_r46_complex_fidelity')
    m=importlib.import_module('experiments.pamf_r46_complex_fidelity.models')
    sr=importlib.import_module('experiments.pamf_r46_complex_fidelity.runtime');arm=d.ARMS[index]
    source=m.compact_from_parent(base.parent_model(),d.SOURCE_ROWS[arm]).eval().requires_grad_(False)
    source.initialization_report['seconds']=0.;payload=sr.metadata(d.SOURCE_ROWS[arm],20,19700,base.lineage(),source.initialization_report)
    model,audit=api().collapse_prior(source);path=tmp_path/'final.pt'
    r.save_checkpoint(path,model,arm,payload,audit)
    fresh,value=r.load_checkpoint(path,'cpu');assert value['new_training_updates']==0 and value['parameters']==299297
    x=torch.randn(2,2,256)*.1
    with torch.no_grad():assert torch.equal(model(x),fresh(x))
    key=next(k for k in value['model'] if k.startswith('prior.1.') and k.endswith('net.0.weight'))
    value['model'][key]=value['model'][key]+1
    with pytest.raises(ValueError):r.validate_aliases(fresh,value['model'])
    with pytest.raises(FileExistsError):r.save_checkpoint(path,model,arm,payload,audit)


@pytest.mark.parametrize('field,bad',[('new_training_updates',1),('source_commit','unknown'),('parameters',278180),('F_training_access',True)])
def test_changed_runtime_contract_rejected(field,bad):
    r=package('runtime');d=package('design');base=importlib.import_module('tests.test_pamf_r46_complex_fidelity')
    m=importlib.import_module('experiments.pamf_r46_complex_fidelity.models');sr=importlib.import_module('experiments.pamf_r46_complex_fidelity.runtime')
    arm=d.ARMS[0];source=m.compact_from_parent(base.parent_model(),d.SOURCE_ROWS[arm]).eval().requires_grad_(False)
    source.initialization_report['seconds']=0.;payload=sr.metadata(d.SOURCE_ROWS[arm],20,19700,base.lineage(),source.initialization_report)
    _,audit=api().collapse_prior(source);value=r.metadata(arm,payload,audit);r.validate_payload(value)
    value[field]=bad
    with pytest.raises(ValueError):r.validate_payload(value)


def test_unknown_frozen_source_arm_stops_before_checkpoint_io():
    r=package('runtime')
    from types import SimpleNamespace
    with pytest.raises(ValueError):r.source_root(SimpleNamespace(),'other')


@pytest.mark.parametrize('index',[0,1])
def test_each_expanded_version_runs_source_cost_and_all7_formal(tmp_path,monkeypatch,index):
    run=package('runner');d=package('design')
    from types import SimpleNamespace
    root=tmp_path/d.RUN;root.mkdir();(root/'launch.json').write_text('{"commit":"method"}')
    (root/'comparison_plan.json').write_text(__import__('json').dumps(d.configuration()))
    args=SimpleNamespace(output=root,arm=d.ARMS[index],expected_commit='method');seen=[]
    def prepare(a):(a.output/a.arm).mkdir();seen.append('prepare');return None,None,None
    monkeypatch.setattr(run,'prepare',prepare);monkeypatch.setattr(run,'acquire',lambda *a:None)
    monkeypatch.setattr(run,'prediction_pipeline',lambda *a:None)
    monkeypatch.setattr(run,'predict',lambda a,role,pipeline:seen.append('predict_'+role))
    monkeypatch.setattr(run,'profile',lambda *a:seen.append('profile'))
    monkeypatch.setattr(run,'score_subprocess',lambda a,role:seen.append('score_'+role))
    monkeypatch.setattr(run.torch.cuda,'empty_cache',lambda:None);run.worker(args)
    assert seen==['prepare','predict_source','profile','score_source','predict_target','score_target']
    assert len(d.VIEWS)==7 and (root/d.ARMS[index]/'complete.json').is_file()


def test_existing_expanded_row_rejected_without_failure_write(tmp_path,monkeypatch):
    run=package('runner');d=package('design')
    from types import SimpleNamespace
    root=tmp_path/d.RUN;root.mkdir();row=root/d.ARMS[0];row.mkdir();marker=row/'complete.json';marker.write_text('{}')
    monkeypatch.setattr(run,'acquire',lambda *a:pytest.fail('Existing row cannot allocate GPU'))
    with pytest.raises(ValueError):run.worker(SimpleNamespace(output=root,arm=d.ARMS[0]))
    assert list(row.iterdir())==[marker]


def test_complete_expanded_predictions_precede_native_comparison_and_truth(monkeypatch):
    e=package('evaluation');d=package('design')
    from pathlib import Path
    from types import SimpleNamespace
    args=SimpleNamespace(output=Path('/runs')/d.RUN,arm=d.ARMS[0],complex_run=Path('/runs')/d.COMPLEX_RUN);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(e,'reference',lambda *a:(Path('/formal'),e.np.arange(2),None,None))
    class Incomplete(Exception):pass
    def stop(*a):raise Incomplete()
    monkeypatch.setattr(e.base,'prediction_arrays',stop)
    monkeypatch.setattr(e.base,'target_baseline',lambda *a:pytest.fail('No truth before all predictions'))
    with pytest.raises(Incomplete):e.score(args,'target')
