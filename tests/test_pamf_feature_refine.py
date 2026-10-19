import pytest
import torch
from experiments.pamf_feature_refine.models import feature_distance,enable_correction,save_checkpoint,load_checkpoint
from experiments.pamf_residual_restore.models import ResidualDeterministic
from experiments.pamf_reception_restore.models import ComplexTFUNet
import numpy as np
from experiments.pamf_feature_refine.runner import training_batch
from experiments.pamf_feature_refine.evaluation import completed_predictions,READOUTS
from experiments.pamf_fcrossrx_readout.source_diagnostic import VIEWS
import json


def test_feature_target_detached_and_estimate_receives_gradient():
    a=torch.randn(2,160,requires_grad=True);b=torch.randn(2,160,requires_grad=True)
    feature_distance(a,b).mean().backward()
    assert a.grad is not None and torch.isfinite(a.grad).all() and a.grad.abs().sum()>0
    assert b.grad is None
    assert torch.allclose(feature_distance(a.detach(),a.detach()),torch.zeros(2),atol=2e-7)


def test_only_original_correction_trainable():
    model=enable_correction(ResidualDeterministic(ComplexTFUNet(),.1));model.train()
    assert not model.coarse.training
    assert all(p.requires_grad==name.startswith('correction.') for name,p in model.named_parameters())


def test_original_clean_slots_are_never_read_as_damaged():
    clean=np.ones((3,2,256),np.float32)
    arrays=dict(residual=np.full_like(clean,2),full=np.full_like(clean,3))
    arrays['residual'][0]=0;arrays['full'][0]=0
    values,targets,route=training_batch(clean,arrays,np.array([.1,.5,.9]),np.arange(3))
    assert route.tolist()==['clean','residual','full']
    assert np.all(values[0]==1) and np.all(values[1]==2) and np.all(values[2]==3)
    assert np.all(values[3:]==1) and np.all(targets==1)


@pytest.mark.parametrize('field,value',[('TX_labels_read',1),('target_training',True),('F_optimizer_updates',1),
    ('head_fit_updates',1),('coarse_updates',1),('updates',4941),('parent_corrector_epochs',50)])
def test_invalid_checkpoints_rejected(tmp_path,field,value):
    model=ResidualDeterministic(ComplexTFUNet(),.1);path=tmp_path/'checkpoint.pt'
    save_checkpoint(path,model,None,'frozenF_feature',0,0)
    item=torch.load(path,weights_only=False);item[field]=value;torch.save(item,path)
    with pytest.raises(ValueError):load_checkpoint(path,'frozenF_feature','cpu')


@pytest.mark.parametrize('problem',['missingview','targetrole'])
def test_source_prediction_contract_fails_before_any_truth(tmp_path,problem):
    folder=tmp_path/'prediction';folder.mkdir()
    done=dict(status='SOURCE_PREDICTIONS_COMPLETE',count=2,views=list(VIEWS),readouts=list(READOUTS),
        role='source_V',target_reads=0,TX_label_reads=0,state_updates=0)
    if problem=='missingview':done['views']=['clean']
    else:done['role']='target'
    (folder/'complete.json').write_text(json.dumps(done))
    with pytest.raises(ValueError):completed_predictions(tmp_path,np.array(['a','b']))
