"""Execution equivalence and pipeline boundaries, using synthetic IQ only."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import cvsrffi.practical_adapter as adapter
from cvsrffi.practical_parallel import parallel_batch, shutdown
from leo_practical import apply_leo_practical_channel_batch

VARIANTS=[('full',False,'zf'),('full',True,'zf'),('full',True,'mmse'),('residual',False,'zf')]


def args(variant):
    route,eq,method=variant
    return SimpleNamespace(practical_route=route,practical_equalization=eq,
        practical_equalizer_method=method,practical_fs_hz=25e6,practical_fc_hz=2.462e9,
        practical_receiver_seed=2027,practical_eval_cache_dir='')


@pytest.mark.parametrize('variant',VARIANTS)
@pytest.mark.parametrize('scene',adapter.PRACTICAL_ALL[1:])
def test_parallel_and_buffer_equal_reference(variant,scene,monkeypatch):
    config=args(variant)
    x=torch.randn(3,2,256,generator=torch.Generator().manual_seed(11))
    adapter.set_evaluation_context(['a','b','c'])
    before=torch.get_rng_state().clone()
    a,meta=adapter.apply_practical(x,scene,config,gen=torch.Generator().manual_seed(13),return_meta=True)
    records=copy.deepcopy(adapter._last_meta['records'])
    config.practical_buffer_output=True
    config.practical_channel_workers=2
    def forbidden(*a,**k):raise AssertionError('Unsafe ndarray ABI')
    monkeypatch.setattr(torch.Tensor,'numpy',forbidden)
    monkeypatch.setattr(torch,'from_numpy',forbidden)
    b,other=adapter.apply_practical(x,scene,config,gen=torch.Generator().manual_seed(13),return_meta=True)
    assert torch.equal(a,b)
    assert all(torch.equal(meta[k],other[k]) for k in meta)
    assert records==adapter._last_meta['records']
    assert torch.equal(before,torch.get_rng_state())


def test_parallel_rejects_duplicate_ids():
    with pytest.raises(ValueError,match='global physical'):
        parallel_batch(np.ones((2,2,256)),adapter.practical_config('practical_mid',args(VARIANTS[0])),
            workers=2,seed=7,sample_ids=['a','a'],session_ids=['s','s'],realization_namespace='source_dynamic_E2_L')


def test_fir_cache_preserves_ownership_and_reference():
    from leo_practical.channel import fractional_delay_kernel,_fractional_delay_kernel_reference
    for fraction in (0.,.125,.999999):
        old=_fractional_delay_kernel_reference(fraction,24)
        new=fractional_delay_kernel(fraction,24)
        assert np.array_equal(old,new)
        new[:]=0
        assert np.array_equal(old,fractional_delay_kernel(fraction,24))


@pytest.mark.parametrize('variant',VARIANTS)
def test_actual_channel_cached_fir_exact(variant,monkeypatch):
    import leo_practical.channel as channel
    config=args(variant)
    x=torch.randn(7,2,256,generator=torch.Generator().manual_seed(9))
    adapter.set_evaluation_context([f'fir{i}' for i in range(7)])
    for scene in adapter.PRACTICAL_ALL[1:]:
        new,_=adapter.apply_practical(x,scene,config,gen=torch.Generator().manual_seed(7))
        records=copy.deepcopy(adapter._last_meta['records'])
        with monkeypatch.context() as context:
            context.setattr(channel,'fractional_delay_kernel',channel._fractional_delay_kernel_reference)
            old,_=adapter.apply_practical(x,scene,config,gen=torch.Generator().manual_seed(7))
        assert torch.equal(new,old) and records==adapter._last_meta['records']


def test_dynamic_parallel_preserves_seed_and_input(tmp_path):
    config=args(VARIANTS[1]);config.output_dir=str(tmp_path)
    x=torch.randn(4,2,256,generator=torch.Generator().manual_seed(2))
    original=x.clone()
    adapter.set_training_context(dict(base_index=[1,2,3,4],rx_i=[1]*4,day_i=[1]*4),91,'U')
    gen=torch.Generator().manual_seed(7)
    old,_=adapter.apply_practical(x,'practical_mid',config,gen=gen)
    records=copy.deepcopy(adapter._last_meta['records']);state=gen.get_state()
    config.practical_channel_workers=2;config.practical_buffer_output=True
    gen.manual_seed(7)
    new,_=adapter.apply_practical(x,'practical_mid',config,gen=gen)
    assert torch.equal(old,new) and torch.equal(x,original)
    assert records==adapter._last_meta['records'] and torch.equal(state,gen.get_state())
    assert adapter._last_meta['cache_event']=='bypass'


def test_gradient_snapshot_preserves_actual_update():
    from cvsrffi.a1_fast_runtime import GradientSnapshot
    from scripts.train_rc4_practical import native,build_args
    a=build_args(ROOT/'configs/rc4_practical_full_noeq_20260918.json','unused','unused','unused','unused','unused','synthetic')
    ma=native.merge_checkpoint_args({'model':None,'args':{},'stats':{},'split_info':None},a,input_len=256,num_domains=15)
    m=native.build_baseline_model(native._apply_model_cli_args(ma,a),torch.device('cpu'))
    m.train();x=torch.randn(3,2,256);y=torch.tensor([0,1,2])
    output=m(x,y_tx=None,return_aux=True)
    (torch.nn.functional.cross_entropy(output['tx_logits'],y)
     +torch.nn.functional.cross_entropy(output['dom_logits'],y)).backward()
    before=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone()
    gradients={n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None}
    snapshot=GradientSnapshot.capture(m)
    assert snapshot.first_nonfinite()==native._first_nonfinite_gradient(m)
    assert snapshot.norm()==native._grad_norm(m)
    for predicate in (lambda n:'backbone' in n,lambda n:'dom' in n or 'domain' in n):
        assert snapshot.norm(predicate)==native._grad_norm(m,predicate)
    torch.nn.utils.clip_grad_norm_(m.parameters(),5.0)
    after=GradientSnapshot.capture(m)
    assert after.norm()==native._grad_norm(m)
    opt=torch.optim.SGD(m.parameters(),lr=.001);opt.step()
    optimized=copy.deepcopy(m.state_dict())
    m.load_state_dict(before)
    for n,p in m.named_parameters():p.grad=gradients.get(n)
    native._first_nonfinite_gradient(m);native._grad_norm(m)
    torch.nn.utils.clip_grad_norm_(m.parameters(),5.0);opt.step()
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in optimized.items())
    assert torch.equal(rng,torch.get_rng_state())
    first=next(p for p in m.parameters() if p.grad is not None)
    first.grad.flatten()[0]=float('nan')
    assert GradientSnapshot.capture(m).first_nonfinite()==native._first_nonfinite_gradient(m)


def test_actual_predictor_cpu_identity_subset(tmp_path,monkeypatch):
    from scripts import predict_phase1_truth_last as predict
    from scripts.train_rc4_practical import native,build_args,resolved_config
    from cvsrffi.truth_last import score_predictions
    config=build_args(ROOT/'configs/rc4_practical_full_mmse_20260918.json','unused','unused','unused','unused','unused','synthetic')
    ma=native.merge_checkpoint_args({'model':None,'args':{},'stats':{},'split_info':None},config,input_len=256,num_domains=15)
    model=native.build_baseline_model(native._apply_model_cli_args(ma,config),torch.device('cpu'))
    before=copy.deepcopy(model.state_dict())
    model.eval();x=torch.randn(5,2,256)
    with torch.no_grad():
        full=model(x,y_tx=None,return_aux=True);identity=model.forward_identity_only(x)
    assert torch.equal(full['tx_logits'],identity['tx_logits'])
    assert torch.equal(full['z_id'],identity['z_id'])
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    checkpoint=tmp_path/'model.pth';torch.save({'epoch':100,'args':resolved_config(config),'model':model.state_dict()},checkpoint)
    class Tiny(torch.utils.data.Dataset):
        def __init__(self,_):pass
        def __len__(self):return len(x)
        def __getitem__(self,i):return x[i],-1,0,{'physical_sample_id':f'id{i}'}
    monkeypatch.setattr(predict,'_OpaqueTargetDataset',Tiny)
    scenes=['clean','practical_mid','practical_mid_urban','practical_low_urban']
    common=['--checkpoint',str(checkpoint),'--input-package',str(tmp_path),'--run-id','synthetic',
        '--row-id','row','--mode','predict','--device','cpu','--num-workers','0',
        '--batch-size','3','--expected-epoch','100','--scenarios',','.join(scenes)]
    for name,extra in [('old',[]),('new',['--practical-cpu-pipeline','--identity-only'])]:
        predict.main(common+['--output-root',str(tmp_path/name)]+extra)
    old=json.loads((tmp_path/'old/predictions.json').read_text())
    new=json.loads((tmp_path/'new/predictions.json').read_text())
    assert old==new and old['record_count']==20
    truth=tmp_path/'truth.json';truth.write_text(json.dumps({'records':[{'sample_id':f'id{i}','label':i} for i in range(5)]}))
    result=score_predictions(tmp_path/'new/predictions.json',truth,output_path=tmp_path/'score.json',scenarios=scenes)
    assert result['record_count']==20


def test_bounded_profile_does_not_consume_rng(tmp_path):
    from cvsrffi.execution_profile import TrainingProfile,stage
    rng=torch.get_rng_state().clone()
    profiler=TrainingProfile(tmp_path,epoch=2,start=3,steps=1)
    profiler.begin(1,3);profiler.end()
    assert not (tmp_path/'execution_profile').exists()
    profiler.begin(2,3)
    with stage('unit_test'):torch.ones(2)+1
    profiler.end();profiler.begin(2,4)
    summary=json.loads((tmp_path/'execution_profile/summary.json').read_text())
    assert summary['steps']==1 and any(x['name']=='cvs/unit_test' for x in summary['rows'])
    assert torch.equal(rng,torch.get_rng_state())


def test_profile_closes_on_training_exception(tmp_path):
    from cvsrffi.execution_profile import TrainingProfile,close_profiles_on_exit,_active
    @close_profiles_on_exit
    def failing():
        profile=TrainingProfile(tmp_path,steps=3)
        profile.begin(1,1)
        torch.ones(2)+1
        raise RuntimeError('original training failure')
    with pytest.raises(RuntimeError,match='original training failure'):failing()
    assert not _active.get() and (tmp_path/'execution_profile/trace.json').is_file()


def test_periodic_manifest_subset_coverage(tmp_path):
    from cvsrffi.a1_periodic_target import evaluate_checkpoint
    (tmp_path/'manifest.json').write_text(json.dumps({'sample_ids':['a','b','c']}))
    output=tmp_path/'result'
    def predictor(argv):
        output.mkdir();(output/'predictions.json').write_text('{}')
    def scorer(command,env):
        assert env['CUDA_VISIBLE_DEVICES']==''
        (output/'score.json').write_text(json.dumps({'record_count':9}))
    rng=torch.get_rng_state().clone()
    evaluate_checkpoint('unused',output=output,input_package=tmp_path,truth='unused',run_id='synthetic',
        row_id='subset',device='cpu',predictor=predictor,scorer=scorer,
        scenarios='clean,practical_mid,practical_low_urban')
    assert json.loads((output/'evaluation_scope.json').read_text())['record_count']==9
    assert torch.equal(rng,torch.get_rng_state())


def teardown_module():shutdown()
