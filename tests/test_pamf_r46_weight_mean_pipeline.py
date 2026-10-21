import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from experiments.pamf_r44_pruned_refine.models import CompactSuccessor as Parent
from experiments.pamf_r46_weight_mean import design as d,models as m,runtime as r,evaluation as e,runner
from experiments.pamf_r46_weight_mean.averaging import SourceParameterMeans
from experiments.pamf_uncertainty_restore import design as ancestor

torch.set_num_threads(2)


def lineage():
    return dict(coarse_run=ancestor.COARSE_RUN,coarse_commit=ancestor.COARSE_COMMIT,
        action_run=ancestor.ACTION_RUN,action_commit=ancestor.ACTION_COMMIT,data_contract='original7/63/30',training_TX_F_reads=0,target_training=False)


def model_info(arm,epoch=20):
    parent=Parent('prune_refine').eval().requires_grad_(False)
    with torch.no_grad():
        for part in (parent.frozen.coarse.temporal,parent.frozen.coarse.spectral,parent.frozen.estimator.core,*parent.prior):part.out.weight.normal_(std=.001)
    model=m.source_from_parent(parent);model.initialization_report['seconds']=0.
    info=SourceParameterMeans(model).metadata(arm)
    if epoch:info.update(shared_source_epochs=20,shared_source_updates=4940,ema_updates=2470,late_epoch_snapshots=5)
    return parent,model,info


@pytest.mark.parametrize('arm',d.ARMS)
@pytest.mark.parametrize('epoch',[0,20])
def test_strict_zero_probe_or_final_portable_state(tmp_path,arm,epoch):
    parent,model,info=model_info(arm,epoch)
    received=torch.randn(2,2,256)*.1;clean=(received*.9).requires_grad_(True)
    with torch.no_grad():desired=parent.stage_outputs(received)[0]
    desired=[v.requires_grad_(True) for v in desired]
    loss,_=m.objective(model.stage_outputs(received)[0],desired,clean,1);loss.mean().backward()
    assert clean.grad is None and all(v.grad is None for v in desired)
    for part in (model.frozen.coarse,model.frozen.estimator,model.prior):
        grads=[p.grad for p in part.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads)>0
    path=tmp_path/'readout.pt';r.save_checkpoint(path,model.state_dict(),arm,epoch,epoch*247,lineage(),model.initialization_report,info)
    fresh,payload=r.load_checkpoint(path,'cpu')
    assert payload['structural_smoke']==(epoch==0) and payload['optimizer'] is None
    assert payload['source_path_training_counted_once'] and sum(p.numel() for p in fresh.parameters())==273765
    with torch.no_grad():assert torch.equal(fresh(received),model(received))
    assert all(p.grad is None for p in parent.parameters())
    with pytest.raises(FileExistsError):r.save_checkpoint(path,model.state_dict(),arm,epoch,epoch*247,lineage(),model.initialization_report,info)


@pytest.mark.parametrize('field,bad',[('training_F_access',True),('F_outputs_read',1),('target_training',True),('V_training',True),
    ('parent_R46_commit','unknown'),('parent_R46_row','retain_refine'),('updates',4939),('epoch',19),('source_path_training_counted_once',False),
    ('external_student_checkpoint_reads',1),('new_calibration_IQ',1)])
def test_source_contract_and_shared_budget_negatives(field,bad):
    _,model,info=model_info('final');value=r.metadata('final',20,4940,lineage(),model.initialization_report,info)
    r.validate_payload(value);value[field]=bad
    with pytest.raises(ValueError):r.validate_payload(value)


@pytest.mark.parametrize('field,bad',[('shared_source_epochs',19),('ema_updates',2469),('late_epoch_snapshots',4),
    ('teacher_changed',True),('buffers_averaged',True),('extra_neural_forwards',1),('F_reads',1),('target_reads',1),('readout','final')])
def test_fixed_mean_metadata_cannot_be_reinterpreted(field,bad):
    _,model,info=model_info('late_mean');value=r.metadata('late_mean',20,4940,lineage(),model.initialization_report,info)
    value['parameter_readout'][field]=bad
    with pytest.raises(ValueError):r.validate_payload(value)


def source_metadata(arm):
    trained=dict(status='SOURCE_TRAINED',arm=d.SOURCE_ARM,epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        source_IQ_examples=1260000,teacher_IQ_examples=1260000,student_context_examples=1260000,parameters=273765,context_parameters=185615,
        initialization_IQ_examples=0,teacher_parameter_updates=0,context_updates=4940,prior_updates=4940,cumulative_prior_epochs=100,
        cumulative_coarse_epochs=100,cumulative_action_epochs=100,TX_labels_read=0,V_reads=0,target_reads=0,training_F_access=False,
        F_weights_read=0,F_outputs_read=0,F_parameter_updates=0,head_fit_updates=0,paired_clean_optimized=True,shared_source_training_paths=1,
        training_counted_once_across_readouts=True,readouts=list(d.ARMS),EMA_updates=2470,late_mean_snapshots=5,new_calibration_IQ=0,
        new_trainable_parameters=0,extra_training_neural_forwards=0,external_student_checkpoint_reads=0)
    complete=dict(status='SOURCE_TRAINED',source_worker_exited=True,source_training_counted_once=True,readouts=list(d.ARMS))
    _,_,info=model_info(arm)
    return trained,complete,info


@pytest.mark.parametrize('field,bad',[('epochs',19),('updates',4939),('F_outputs_read',1),('TX_labels_read',1),('U_exposures',0),
    ('target_reads',1),('shared_source_training_paths',3),('training_counted_once_across_readouts',False),('EMA_updates',2471)])
def test_actual_shared_source_budget_precedes_readout_input(monkeypatch,field,bad):
    trained,complete,info=source_metadata('late_mean');trained[field]=bad
    monkeypatch.setattr(e,'read_json',lambda p:trained if p.name=='training_complete.json' else info if p.name=='parameter_readout.json' else complete)
    with pytest.raises(ValueError):e.validate_frozen_readout(Path('/runs')/d.RUN/'late_mean')


def test_source_worker_exit_is_required(monkeypatch):
    trained,complete,info=source_metadata('final');complete['source_worker_exited']=False
    monkeypatch.setattr(e,'read_json',lambda p:trained if p.name=='training_complete.json' else info if p.name=='parameter_readout.json' else complete)
    with pytest.raises(ValueError,match='worker exit'):e.validate_frozen_readout(Path('/runs')/d.RUN/'final')


@pytest.mark.parametrize('arm',d.ARMS)
def test_every_readout_automatically_predicts_fullformal_after_source(tmp_path,monkeypatch,arm):
    args=SimpleNamespace(output=tmp_path,arm=arm);(tmp_path/arm).mkdir();seen=[]
    monkeypatch.setattr(runner,'acquire',lambda *a,**k:seen.append('acquire'))
    monkeypatch.setattr(runner,'predict',lambda a,role:seen.append('predict_'+role))
    monkeypatch.setattr(runner,'profile',lambda *a:seen.append('profile'))
    monkeypatch.setattr(runner,'score_subprocess',lambda a,role:seen.append('score_'+role))
    monkeypatch.setattr(runner.torch.cuda,'empty_cache',lambda:None);runner.worker(args)
    assert seen==['acquire','predict_source','profile','score_source','predict_target','score_target']
    assert (tmp_path/arm/'complete.json').exists()


def test_truth_closed_before_all_prediction_arrays(monkeypatch):
    args=SimpleNamespace(output=Path('/runs')/d.RUN,arm='final');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(e,'reference',lambda *a:(Path('/formal'),e.np.arange(2),None,None))
    class Incomplete(Exception):pass
    monkeypatch.setattr(e.base,'prediction_arrays',lambda *a:(_ for _ in ()).throw(Incomplete()))
    monkeypatch.setattr(e.base,'target_baseline',lambda *a:pytest.fail('No truth'))
    with pytest.raises(Incomplete):e.score(args,'target')


def test_conflicting_shared_aliases_reject():
    _,model,_=model_info('final');state=copy.deepcopy(model.state_dict())
    key=next(n for n in state if n.startswith('prior.1.') and n.endswith('net.0.weight'));state[key]=state[key]+1
    with pytest.raises(ValueError,match='Conflicting'):r.validate_aliases(model,state)


def test_controller_runs_one_source_path_then_three_gpu_bounded_readouts(tmp_path,monkeypatch):
    args=SimpleNamespace(output=tmp_path/d.RUN,expected_commit='fixed',compact_parent=Path('/M35'))
    for name in runner.PATH_ARGS:
        if not hasattr(args,name):setattr(args,name,Path('/input')/name)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');monkeypatch.setattr(runner,'validate_parent_root',lambda *a:None)
    monkeypatch.setattr(runner,'smoke',lambda *a:None);monkeypatch.setattr(runner.time,'sleep',lambda *a:None)
    launches=[];active={};maximum=[0];source_exited=[False]
    class Process:
        def __init__(self,argv,**kwargs):
            self.mode=argv[4];self.arm=argv[argv.index('--arm')+1];self.pid=100+len(launches);self.returncode=None;self.polls=0
            gpu=int(kwargs['env']['CUDA_VISIBLE_DEVICES']);self.gpu=gpu
            if self.mode=='train_source':
                assert self.arm==d.SOURCE_ARM and not launches
                for arm in d.ARMS:(args.output/arm).mkdir()
            else:
                assert source_exited[0] and gpu not in active
                active[gpu]=self;maximum[0]=max(maximum[0],len(active))
            launches.append((self.mode,self.arm,gpu))
        def poll(self):
            self.polls+=1
            if self.mode=='train_source':source_exited[0]=True;self.returncode=0;return 0
            if self.polls==1:return None
            self.returncode=0;active.pop(self.gpu,None)
            (args.output/self.arm/'complete.json').write_text('{}',encoding='utf-8');return 0
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    monkeypatch.setattr(runner,'read_json',lambda p:dict(status='SOURCE_TRAINED',readouts=list(d.ARMS),epochs=20,updates=4940))
    runner.run(args)
    assert launches==[('train_source',d.SOURCE_ARM,3),('worker','final',3),('worker','late_mean',2),('worker','step_ema',3)]
    assert maximum[0]==2 and not active and source_exited[0]
    result=json.loads((args.output/'complete.json').read_text(encoding='utf-8'))
    assert result['shared_source_training_paths']==1 and result['training_counted_once'] and result['target_rows']==list(d.ARMS)
