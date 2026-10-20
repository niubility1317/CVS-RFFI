import copy,json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_spatial_conditioning import design as d,models
from experiments.pamf_spatial_conditioning.models import ConditionedField,SpatialConditionedRestorer,save_checkpoint,load_checkpoint
from experiments.pamf_reception_restore.models import ComplexTFUNet
from experiments.pamf_residual_restore.models import ResidualDeterministic
from experiments.standard_ce_baselines.sigtime import SIGTimeCVS
from experiments.pamf_restoration.frozen import TripleSignaturePool
from experiments.pamf_fcrossrx_readout.frozen import FrozenCenteredSinc
torch.set_num_threads(2)


def original_architecture():
    torch.manual_seed(2);reader=SIGTimeCVS(d.RUNTIME_F_SEED)
    reader.sequence_pool=TripleSignaturePool(reader.core.id_backbone.t2.pw.out_channels)
    reader.core.id_backbone.sinc=FrozenCenteredSinc(reader.core.id_backbone.sinc)
    return reader.eval().requires_grad_(False)


def new_model(arm):return SpatialConditionedRestorer(ResidualDeterministic(ComplexTFUNet(),.1),original_architecture(),arm)


def test_field_budget_and_exact_zero_head_initialization_match_both_arms():
    outputs=[];states=[];torch.manual_seed(7);context=torch.randn(2,4,256);maps=dict(t1=torch.randn(2,72,64),t2=torch.randn(2,96,32))
    for arm in d.ARMS:
        torch.manual_seed(9);field=ConditionedField(arm)
        assert sum(p.numel() for p in field.parameters())==d.TRAINABLE_PARAMETERS
        outputs.append(field(context,maps));states.append(field.state_dict())
    assert all(torch.equal(value,torch.zeros_like(value)) for value in outputs)
    assert all(torch.equal(v,states[1][k]) for k,v in states[0].items())


def test_spatial_condition_retains_order_while_pooled_condition_discards_it():
    torch.manual_seed(11);context=torch.randn(2,4,256);maps=dict(t1=torch.randn(2,72,64),t2=torch.randn(2,96,32))
    reversal={k:value.flip(2) for k,value in maps.items()}
    for arm in d.ARMS:
        torch.manual_seed(3);field=ConditionedField(arm).eval();nn.init.normal_(field.out.weight,0,.01)
        a=field(context,maps);b=field(context,reversal)
        if arm==d.ARMS[0]:assert torch.allclose(a,b,atol=1e-6,rtol=1e-5)
        else:assert not torch.allclose(a,b,atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('arm',d.ARMS)
def test_native_condition_is_detached_but_new_field_branches_train(arm):
    torch.manual_seed(5);field=ConditionedField(arm);nn.init.normal_(field.out.weight,0,.01)
    context=torch.randn(2,4,256);maps=dict(t1=torch.randn(2,72,64,requires_grad=True),t2=torch.randn(2,96,32,requires_grad=True))
    value=field(context,maps);value.square().mean().backward()
    assert all(v.grad is None for v in maps.values())
    for module in (field.encoder,field.projections,field.fusions,field.out):
        grads=[p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads)>0


@pytest.mark.parametrize('arm',d.ARMS)
def test_whole_restorer_keeps_original_parent_F_frozen_and_batch_independent(arm):
    torch.manual_seed(7);model=new_model(arm).train();x=torch.randn(2,2,256);before={k:v.clone() for k,v in model.state_dict().items()}
    assert not model.parent.training and not model.runtime_reader.training and model.field.training
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==d.TRAINABLE_PARAMETERS
    assert sum(p.numel() for p in model.runtime_reader.parameters())==d.RUNTIME_F_PARAMETERS
    assert sum(p.numel() for p in model.parameters())==d.TOTAL_PARAMETERS
    with torch.no_grad():
        assert torch.equal(model(x),model.parent(x))
        assert torch.allclose(model(x),torch.cat([model(value[None]) for value in x]),atol=1e-6,rtol=1e-5)
    model(x).square().mean().backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in model.field.out.parameters())
    assert all(p.grad is None for p in model.parent.parameters()) and all(p.grad is None for p in model.runtime_reader.parameters())
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())


def test_original_reader_state_reconstruction_and_strict_checkpoint_roundtrip(tmp_path):
    reader=original_architecture();fresh=models.reader_from_state(reader.state_dict());x=torch.randn(2,2,256)
    with torch.no_grad():assert torch.equal(reader.features(x),fresh.features(x))
    for arm in d.ARMS:
        model=new_model(arm).eval();nn.init.normal_(model.field.out.weight,0,.01)
        path=tmp_path/(arm+'.pt');save_checkpoint(path,model,None,arm,10,2470);loaded,item=load_checkpoint(path,arm,'cpu')
        with torch.no_grad():assert torch.equal(model(x),loaded(x))
        assert item['coarse_updates']==item['correction_updates']==0 and item['field_updates']==2470
        assert item['runtime_F_seed']==d.RUNTIME_F_SEED and not any(p.requires_grad for p in loaded.parameters())


@pytest.mark.parametrize('field,value',[('TX_labels_read',1),('target_training',True),('F_optimizer_updates',1),('head_fit_updates',1),
 ('parent_corrector_epochs',0),('parent_coarse_epochs',0),('coarse_updates',1),('correction_updates',1),('field_updates',1),
 ('runtime_F_seed',2026092702),('runtime_F_commit','unknown'),('updates',2471),('epoch',11),('gradient_updates',1),('gradient_epochs',1)])
def test_checkpoint_rejects_illegal_budget_parent_runtimeF_or_labels(tmp_path,field,value):
    path=tmp_path/'m.pt';save_checkpoint(path,new_model(d.ARM),None,d.ARM,0,0)
    item=torch.load(path,weights_only=False);item[field]=value;torch.save(item,path)
    with pytest.raises(ValueError):load_checkpoint(path,d.ARM,'cpu')


@pytest.mark.parametrize('bad',['context','mapshape','mapchannels','nan','arm'])
def test_invalid_native_map_contract_is_rejected(bad):
    arm=d.ARM;context=torch.randn(1,4,256);maps=dict(t1=torch.randn(1,72,64),t2=torch.randn(1,96,32))
    if bad=='context':context=context[:,:,:255]
    elif bad=='mapshape':maps.pop('t1')
    elif bad=='mapchannels':maps['t1']=maps['t1'][:,:71]
    elif bad=='nan':maps['t2'][0,0,0]=float('nan')
    else:arm='unknown'
    with pytest.raises(ValueError):ConditionedField(arm)(context,maps)


def test_GPU_predecessor_checks_actual_owned_handles_and_ignores_reused_unrelated_PID(tmp_path):
    from experiments.pamf_spatial_conditioning.runner import previous_GPU_rows_live
    previous=tmp_path/'previous';previous.mkdir();proc=tmp_path/'proc';proc.mkdir()
    (previous/'workers.json').write_text(json.dumps(dict(workers=[dict(arm='x',pid=123)])))
    (previous/'x').mkdir();(previous/'x/target_launch.json').write_text(json.dumps(dict(pid=124)))
    for pid in (123,124):
        path=proc/str(pid);path.mkdir();(path/'stat').write_text(str(pid)+' (python) S')
        (path/'cmdline').write_bytes(('python -m experiments.pamf_tangent_guidance.runner worker --output '+str(previous)).encode())
    assert previous_GPU_rows_live(previous,proc)==[123,124]
    (proc/'123/stat').write_text('123 (python) Z');(proc/'124/cmdline').write_bytes(b'unrelated owner')
    assert previous_GPU_rows_live(previous,proc)==[]


def test_GPU_queue_does_not_launch_from_terminal_marker_while_owned_handle_is_live(tmp_path,monkeypatch):
    from experiments.pamf_spatial_conditioning import runner
    previous=tmp_path/d.GPU_PRECEDING_RUN;previous.mkdir();(previous/'complete.json').write_text('{}')
    output=tmp_path/d.RUN;output.mkdir();states=iter(([7],[]));sleeps=[]
    monkeypatch.setattr(runner,'previous_GPU_rows_live',lambda p:next(states));monkeypatch.setattr(runner.time,'sleep',lambda seconds:sleeps.append(seconds))
    runner.wait_for_GPU_predecessor(SimpleNamespace(output=output))
    result=json.loads((output/'GPU_predecessor_released.json').read_text())
    assert sleeps==[5] and result['live_owned_PIDs']==[] and result['scores_read']==0


def fixed_source(root,arm):
    from experiments.pamf_spatial_conditioning import design as d
    root=root/d.RUN;row=root/arm;(row/'prediction').mkdir(parents=True)
    values={
      root/'launch.json':dict(commit='x',fixed_target_arms=list(d.ARMS)),root/'matrix.json':d.target_matrix(),
      row/'source_frozen.json':dict(status='SOURCE_FROZEN',arm=arm,epochs=10,updates=2470,source_only=True,target_reads=0,
         checkpoint_selection='fixed_final_E10',target_policy='FIXED_COMPARISON'),
      row/'training_complete.json':dict(epochs=10,updates=2470,coarse_updates=0,correction_updates=0,field_updates=2470,trainable_parameters=925378,total_parameters=3299211,runtime_F_seed=d.RUNTIME_F_SEED,runtime_F_commit=d.RUNTIME_F_COMMIT,gradient_updates=2470,gradient_epochs=10,
         F_optimizer_updates=0,head_fit_updates=0,TX_labels_read=0,target_reads=0),
      row/'prediction/complete.json':dict(status='SOURCE_PREDICTIONS_COMPLETE',count=1800,views=d.target_matrix()['views'],
         role='source_V',TX_label_reads=0,target_reads=0,state_updates=0),
      row/'summary.json':dict(status='ANALYZED',source_only=True,truth_last=True,arm=arm,count=1800,target_reads=0,
         diagnostic_accuracy=1.,diagnostic_NMSE=100.)}
    for path,value in values.items():path.write_text(json.dumps(value),encoding='utf-8')
    return SimpleNamespace(source_run=root,arm=arm,expected_commit='x'),values


def test_both_fixed_rows_may_test_without_source_score_selection_or_other_row_completion(tmp_path):
    from experiments.pamf_spatial_conditioning.target import source_check
    for arm in d.ARMS:
        args,_=fixed_source(tmp_path/arm,arm);source_check(args)
        assert not (args.source_run/'source_decision.json').exists()


@pytest.mark.parametrize('file,field,value',[
 ('launch','fixed_target_arms',[d.ARMS[0]]),('launch','commit','other'),('source_frozen','checkpoint_selection','best_V'),
 ('source_frozen','target_reads',1),('source_frozen','target_policy','SELECTED_CANDIDATE'),('training_complete','updates',2469),('training_complete','gradient_updates',0),
 ('training_complete','coarse_updates',1),('training_complete','field_updates',0),('training_complete','runtime_F_seed',2026092702),('training_complete','total_parameters',0),('training_complete','head_fit_updates',1),('training_complete','TX_labels_read',1),('training_complete','target_reads',1),
 ('prediction/complete','TX_label_reads',1),('prediction/complete','views',['clean']),('summary','truth_last',False)])
def test_incomplete_unregistered_or_contaminated_source_row_cannot_test(tmp_path,file,field,value):
    from experiments.pamf_spatial_conditioning.target import source_check
    args,_=fixed_source(tmp_path,d.ARMS[0]);path=(args.source_run if file=='launch' else args.source_run/args.arm)/(file+'.json')
    item=json.loads(path.read_text());item[field]=value;path.write_text(json.dumps(item))
    with pytest.raises(ValueError):source_check(args)


def packets(tmp_path):
    from experiments.pamf_spatial_conditioning.target import VIEWS,SEEDS,READOUTS
    saved=tmp_path/'prediction';saved.mkdir();ids=np.array(['a','b'])
    done=dict(status='PREDICTIONS_COMPLETE',views=list(VIEWS),count=2,F_seeds=list(SEEDS),readouts=list(READOUTS),
        TX_truth_reads=0,paired_clean_in_practical_inference=False,state_updates=0)
    (saved/'complete.json').write_text(json.dumps(done));values=dict(ids=ids,seeds=np.asarray(SEEDS),**{k:np.zeros((2,3),np.uint8) for k in READOUTS})
    for view in VIEWS:
        np.savez(saved/(view+'.npz'),**values);np.save(saved/(view+'_refined_iq.npy'),np.zeros((2,2,256),np.float32))
    return saved,ids,done,values


def test_full_seven_view_three_F_packet_before_truth(tmp_path):
    from experiments.pamf_spatial_conditioning.target import VIEWS,completed_predictions
    _,ids,_,_=packets(tmp_path);assert set(completed_predictions(tmp_path,ids))==set(VIEWS)


@pytest.mark.parametrize('bad',['missingview','earlytruth','pairedclean','stateupdate','badseed','badclass','ids','missingIQ','nanIQ'])
def test_invalid_target_packets_fail_before_truth(tmp_path,bad):
    from experiments.pamf_spatial_conditioning.target import VIEWS,SEEDS,completed_predictions
    saved,ids,done,values=packets(tmp_path)
    if bad=='missingview':done['views']=list(VIEWS[:-1])
    elif bad=='earlytruth':done['TX_truth_reads']=1
    elif bad=='pairedclean':done['paired_clean_in_practical_inference']=True
    elif bad=='stateupdate':done['state_updates']=1
    elif bad=='badseed':values['seeds']=np.asarray(SEEDS[:1])
    elif bad=='badclass':values['refined'][0,0]=6
    elif bad=='ids':values['ids']=ids[::-1]
    elif bad=='missingIQ':np.save(saved/(VIEWS[-1]+'_refined_iq.npy'),np.zeros((1,2,256),np.float32))
    else:np.save(saved/(VIEWS[-1]+'_refined_iq.npy'),np.full((2,2,256),np.nan,np.float32))
    (saved/'complete.json').write_text(json.dumps(done));np.savez(saved/(VIEWS[-1]+'.npz'),**values)
    with pytest.raises(ValueError):completed_predictions(tmp_path,ids)


def test_controller_progress_can_update_through_source_target_completion(tmp_path):
    from experiments.pamf_spatial_conditioning.runner import write_stage_status
    for stage in ('source','target','complete'):
        state=dict(active={} if stage=='complete' else {d.ARMS[0]:dict(stage=stage,pid=123)},completed=stage=='complete')
        write_stage_status(tmp_path,state)
        assert json.loads((tmp_path/'stages.json').read_text())==state
        assert not (tmp_path/'stages.json.tmp').exists()
