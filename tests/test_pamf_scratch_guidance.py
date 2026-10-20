import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from experiments.pamf_scratch_guidance import design as d,models,runner,target
from experiments.pamf_scratch_guidance.guidance import guidance
from experiments.pamf_scratch_guidance.gradients import output_gradient
from experiments.pamf_reception_restore.runner import pair_loss
torch.set_num_threads(2)


def test_fixed_40_epoch_cache_schedule_budget_and_optimizer():
    assert [d.cache_epoch(e) for e in range(1,41)]==list(range(1,21))*2
    for bad in (0,41,True,1.5):
        with pytest.raises(ValueError):d.cache_epoch(bad)
    assert d.UPDATES==40*247==9880 and d.EPOCHS<=50
    assert runner.learning_rate(0)==pytest.approx(2e-4/247)
    assert runner.learning_rate(246)==pytest.approx(2e-4)
    assert runner.learning_rate(9879)==pytest.approx(2e-5)
    for update in (-1,9880):
        with pytest.raises(ValueError):runner.learning_rate(update)
    assert not d.configuration(d.ARMS[0])['guidance_enabled'] and d.configuration(d.ARMS[1])['guidance_enabled']


def test_scratch_initialization_has_no_inherited_restorer_or_runtime_F_and_matches_both_arms(tmp_path):
    x=torch.randn(2,2,256);states=[]
    for arm in d.ARMS:
        torch.manual_seed(d.MODEL_SEED);model=models.initialize('cpu',arm);states.append(model.state_dict())
        assert sum(p.numel() for p in model.parameters())==1305958
        assert set(model._modules)=={'spectral','temporal'}
        assert torch.equal(model(x),x)
        nn.init.normal_(model.temporal.out.weight,0,.01)
        path=tmp_path/(arm+'.pt');models.save_checkpoint(path,model,None,arm,40,9880)
        fresh,item=models.load_checkpoint(path,arm,'cpu')
        assert item['initialization']=='scratch' and item['restoration_parent'] is None
        with torch.no_grad():
            assert torch.equal(model(x),fresh(x))
            assert torch.allclose(model(x),torch.cat([model(z[None]) for z in x]),atol=1e-6,rtol=1e-5)
        assert not any(p.requires_grad for p in fresh.parameters())


@pytest.mark.parametrize('key,value',[('initialization','resume'),('restoration_parent','R03'),('TX_labels_read',1),
 ('target_training',True),('F_optimizer_updates',1),('head_fit_updates',1),('F_commit','unknown'),('F_seeds',[2026092701]),('epoch',41),('updates',9881)])
def test_checkpoint_blocks_inheritance_labels_wrong_F_and_budget(tmp_path,key,value):
    path=tmp_path/'m.pt';models.save_checkpoint(path,models.initialize('cpu',d.ARMS[0]),None,d.ARMS[0],40,9880)
    item=torch.load(path,weights_only=False);item[key]=value;torch.save(item,path)
    with pytest.raises(ValueError):models.load_checkpoint(path,d.ARMS[0],'cpu')


class ToyF(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Linear(512,160);self.head=nn.Linear(160,6);self.requires_grad_(False)
    def features(self,x):return self.embedding(x.flatten(1))
    def classify_features(self,x):return self.head(x)


def test_only_guided_arm_uses_three_original_coordinate_references_and_preserves_frozen_heads():
    torch.manual_seed(4);teachers={seed:ToyF().eval() for seed in d.TEACHER_SEEDS};x=torch.randn(2,2,256)
    with torch.no_grad():references=torch.stack([f.features(x) for f in teachers.values()],1)
    estimate=(.8*x+.01).requires_grad_();identity,feature,posterior=guidance(teachers,estimate,references,d.ARMS[1])
    assert torch.allclose(identity,feature+.3*posterior)
    paired,_=pair_loss(estimate,x);gradient,stats=output_gradient(estimate,paired,identity,d.ARMS[1])
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
    assert torch.all(stats['identity_after_norm2']<=stats['wave_norm2']*(1+1e-5)+1e-10)
    assert all(p.grad is None for f in teachers.values() for p in f.parameters())
    with pytest.raises(ValueError):guidance(teachers,estimate,references,d.ARMS[0])
    with pytest.raises(ValueError):guidance({d.TEACHER_SEEDS[0]:teachers[d.TEACHER_SEEDS[0]]},estimate,references,d.ARMS[1])


def test_wave_arm_merged_gradient_is_exact_mean_waveform_gradient_without_teacher_graph():
    torch.manual_seed(6);x=torch.randn(3,2,256);estimate=(.9*x).requires_grad_();paired,_=pair_loss(estimate,x)
    direct=torch.autograd.grad(paired.mean(),estimate,retain_graph=True)[0]
    actual,stats=output_gradient(estimate,paired,torch.zeros(3),d.ARMS[0])
    assert torch.allclose(actual,direct,atol=1e-8,rtol=1e-6)
    assert stats['identity_norm2'].sum()==0 and stats['scaled'].sum()==0 and torch.all(stats['scale']==1)


def test_launcher_smoke_uses_real_teacher_interface_no_restoration_parent_or_updates(tmp_path,monkeypatch):
    torch.manual_seed(5);teachers={seed:ToyF().eval() for seed in d.TEACHER_SEEDS}
    before={s:{k:v.clone() for k,v in f.state_dict().items()} for s,f in teachers.items()}
    fake=SimpleNamespace(rx=np.array([1,1,3,3]),iq=np.random.default_rng(1).normal(size=(4,2,256)).astype(np.float32))
    monkeypatch.setattr(runner,'IQSource',lambda cache,role:fake)
    monkeypatch.setattr(runner,'load_teachers',lambda args,device:teachers)
    runner.source_smoke(SimpleNamespace(source_cache=tmp_path/'source',output=tmp_path))
    done=json.loads((tmp_path/'smoke.json').read_text());assert done['status']=='PASS' and done['restoration_weights_loaded']==0
    assert all(row['trainable_parameters']==1305958 and row['initial_equals_received'] for row in done['arms'].values())
    assert all(torch.equal(v,before[s][k]) for s,f in teachers.items() for k,v in f.state_dict().items())


def fixed_source(root,arm):
    root=root/d.RUN;row=root/arm;(row/'prediction').mkdir(parents=True)
    values={root/'launch.json':dict(commit='x',fixed_target_arms=list(d.ARMS),initialization='scratch',restoration_weights_loaded=0),
      root/'matrix.json':d.target_matrix(),row/'source_frozen.json':dict(status='SOURCE_FROZEN',arm=arm,epochs=40,updates=9880,source_only=True,target_reads=0,checkpoint_selection='fixed_final_E40',target_policy='FIXED_COMPARISON'),
      row/'training_complete.json':dict(epochs=40,updates=9880,gradient_epochs=40,gradient_updates=9880,initialization='scratch',restoration_weights_loaded=0,
        trainable_parameters=1305958,F_commit=d.F_COMMIT,F_seeds=list(d.TEACHER_SEEDS),cache_epochs=list(range(1,21))*2,
        L_physical_exposures=252000,U_physical_exposures=2268000,IQ_examples=5040000,guidance_enabled=arm==d.ARMS[1],
        teacher_training_calls=3*9880 if arm==d.ARMS[1] else 0,teacher_training_IQ_examples=3*5040000 if arm==d.ARMS[1] else 0,
        F_optimizer_updates=0,head_fit_updates=0,TX_labels_read=0,target_reads=0),
      row/'prediction/complete.json':dict(status='SOURCE_PREDICTIONS_COMPLETE',count=1800,views=list(target.VIEWS),role='source_V',TX_label_reads=0,target_reads=0,state_updates=0),
      row/'summary.json':dict(status='ANALYZED',source_only=True,truth_last=True,arm=arm,count=1800,target_reads=0,diagnostic_accuracy=0.,diagnostic_NMSE=100.)}
    for path,value in values.items():path.write_text(json.dumps(value))
    return SimpleNamespace(source_run=root,arm=arm,expected_commit='x')


@pytest.mark.parametrize('arm',d.ARMS)
def test_both_fixed_rows_test_independent_of_source_scores(tmp_path,arm):target.source_check(fixed_source(tmp_path,arm))


@pytest.mark.parametrize('key,value',[('initialization','resume'),('restoration_weights_loaded',1),('updates',2470),('F_commit','other'),('F_seeds',[2026092701]),
 ('cache_epochs',list(range(1,21))),('U_physical_exposures',567000),('teacher_training_calls',3*9880),('teacher_training_IQ_examples',1),('head_fit_updates',1),('TX_labels_read',1),('target_reads',1)])
def test_invalid_or_guide_contaminated_wave_row_cannot_test(tmp_path,key,value):
    args=fixed_source(tmp_path,d.ARMS[0]);path=args.source_run/args.arm/'training_complete.json'
    item=json.loads(path.read_text());item[key]=value;path.write_text(json.dumps(item))
    with pytest.raises(ValueError):target.source_check(args)


def test_old_GPU_owned_handles_must_exit_before_scratch_workers(tmp_path,monkeypatch):
    previous=tmp_path/d.GPU_PRECEDING_RUN;previous.mkdir();(previous/'complete.json').write_text('{}')
    output=tmp_path/d.RUN;output.mkdir();states=iter(([7],[]));sleeps=[]
    monkeypatch.setattr(runner,'previous_GPU_rows_live',lambda p:next(states));monkeypatch.setattr(runner.time,'sleep',lambda seconds:sleeps.append(seconds))
    runner.wait_for_GPU_predecessor(SimpleNamespace(output=output))
    assert sleeps==[5] and json.loads((output/'GPU_predecessor_released.json').read_text())['scores_read']==0


def test_target_truth_waits_all_seven_views_and_blocks_paired_clean(tmp_path):
    saved=tmp_path/'prediction';saved.mkdir();ids=np.array(['a','b'])
    done=dict(status='PREDICTIONS_COMPLETE',views=list(target.VIEWS),count=2,F_seeds=list(target.SEEDS),readouts=list(target.READOUTS),TX_truth_reads=0,paired_clean_in_practical_inference=False,state_updates=0)
    (saved/'complete.json').write_text(json.dumps(done))
    for view in target.VIEWS:
        np.savez(saved/(view+'.npz'),ids=ids,seeds=np.asarray(target.SEEDS),**{k:np.zeros((2,3),np.uint8) for k in target.READOUTS})
        np.save(saved/(view+'_refined_iq.npy'),np.zeros((2,2,256),np.float32))
    assert set(target.completed_predictions(tmp_path,ids))==set(target.VIEWS)
    done['paired_clean_in_practical_inference']=True;(saved/'complete.json').write_text(json.dumps(done))
    with pytest.raises(ValueError):target.completed_predictions(tmp_path,ids)
    done['paired_clean_in_practical_inference']=False;(saved/'complete.json').write_text(json.dumps(done))
    (saved/(target.VIEWS[-1]+'.npz')).unlink()
    with pytest.raises(FileNotFoundError):target.completed_predictions(tmp_path,ids)
