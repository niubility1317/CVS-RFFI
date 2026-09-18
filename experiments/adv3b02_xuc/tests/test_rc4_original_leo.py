import json
import random
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from cvsrffi.original_leo import FusedCleanSatelliteForward, ORIGINAL, WEAK, original_scenario, validate_scenarios
from cvsrffi.a1_periodic_target import due, evaluate_checkpoint
from cvsrffi.truth_last import score_predictions


def test_concat_is_one_forward_and_preserves_gradients():
    class Model(nn.Module):
        def __init__(self):super().__init__();self.fc=nn.Linear(2,3);self.calls=[]
        def forward(self,x,**kw):
            self.calls.append((x.detach().clone(),kw))
            return {'tx_logits':self.fc(x),'nested':{'z':x*2}}
    m=Model();x=torch.randn(4,2,requires_grad=True);s=torch.randn(4,2,requires_grad=True)
    y=torch.arange(4);f=FusedCleanSatelliteForward(m,s);o=f(x,y_tx=y,domain_labels=y)
    assert len(m.calls)==1 and m.calls[0][0].shape==(8,2)
    assert torch.equal(m.calls[0][1]['y_tx'],torch.cat((y,y)))
    assert torch.equal(o['nested']['z'],x*2)
    o['tx_logits'].sum().backward(retain_graph=True)
    assert x.grad.abs().sum()>0 and s.grad.abs().sum()==0
    f.satellite_output['tx_logits'].sum().backward()
    assert s.grad.abs().sum()>0


def test_original_schedules_and_due():
    from cvsrffi.muse_ssdg import select_adv3b02_u_satellite_scenario
    from training_controls import sat_channel_config_for_scenario
    from sat_channel import SatSimConfig
    for e in (1,40,41,90,91,200):
        for b in range(9):
            scene=original_scenario(select_adv3b02_u_satellite_scenario(e,b,392005))
            assert scene in ORIGINAL[1:]
            assert SatSimConfig(**sat_channel_config_for_scenario(scene)).channel_model=='legacy_full'
    assert [e for e in range(1,201) if due(e,100,200,10)]==list(range(100,201,10))
    with pytest.raises(ValueError):validate_scenarios(('clean','mixed_orbit'))


@pytest.mark.parametrize('scenes',[ORIGINAL,WEAK])
def test_truth_last_coverage_rejects_missing_scene(tmp_path,scenes):
    truth=tmp_path/'truth.json';pred=tmp_path/'pred.json'
    truth.write_text(json.dumps({'records':[{'sample_id':'opaque0','label':0}]}))
    rows=[{'sample_id':'opaque0','predicted_class':0,'scenario':s} for s in scenes]
    pred.write_text(json.dumps({'records':rows}))
    assert score_predictions(pred,truth,scenarios=scenes)['record_count']==4
    pred.write_text(json.dumps({'records':rows[:-1]}))
    with pytest.raises(ValueError):score_predictions(pred,truth,scenarios=scenes)


def test_periodic_predictor_actual_model_and_cpu_scorer(tmp_path,monkeypatch):
    from scripts import predict_phase1_truth_last as predict
    from scripts.train_rc4_original_leo import build_args,resolved_config,native
    from torch.utils.data import Dataset
    cfg=Path(__file__).resolve().parents[1]/'configs/rc4_original_leo_20260918.json'
    args=build_args(cfg,'unused','unused','unused','unused','unused','smoke')
    ma=native.merge_checkpoint_args({'model':None,'args':{},'stats':{},'split_info':None},args,input_len=256,num_domains=15)
    model=native.build_baseline_model(native._apply_model_cli_args(ma,args),torch.device('cpu'))
    checkpoint=tmp_path/'epoch100.pth';torch.save({'epoch':100,'args':resolved_config(args),'model':model.state_dict()},checkpoint)
    class TinyOpaque(Dataset):
        def __init__(self,_):self.x=torch.randn(6,2,256)
        def __len__(self):return 6
        def __getitem__(self,i):return self.x[i],-1,0,{'physical_sample_id':f'opaque{i}'}
    monkeypatch.setattr(predict,'_OpaqueTargetDataset',TinyOpaque)
    truth=tmp_path/'truth.json';truth.write_text(json.dumps({'records':[{'sample_id':f'opaque{i}','label':i} for i in range(6)]}))
    random.seed(17);np.random.seed(17);torch.manual_seed(17)
    states=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    for suffix,scenes in [('original',ORIGINAL),('weak',WEAK)]:
        out=tmp_path/suffix
        evaluate_checkpoint(checkpoint,output=out,input_package=tmp_path,truth=truth,run_id='smoke',row_id='test',device='cpu',
            expected_records=24,scenarios=','.join(scenes),expected_epoch=100)
        assert json.loads((out/'score.json').read_text())['record_count']==24
        assert not json.loads((out/'evaluation_scope.json').read_text())['feeds_training']
    assert random.getstate()==states[0]
    assert np.array_equal(np.random.get_state()[1],states[1][1])
    assert torch.equal(torch.get_rng_state(),states[2])
