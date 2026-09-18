import sys,json,random
from pathlib import Path
import numpy as np
import torch
import pytest
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from cvsrffi.practical_adapter import PRACTICAL,set_evaluation_context,apply_practical
from cvsrffi.a1_periodic_target import evaluate_checkpoint

@pytest.mark.parametrize('route,eq,method',[('full',False,'zf'),('full',True,'zf'),('residual',False,'zf'),('full',True,'mmse')])
def test_stable_practical_reorder_and_partition(route,eq,method,monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('Unsafe Torch NumPy ABI bridge')
    monkeypatch.setattr(torch.Tensor,'numpy',forbidden)
    monkeypatch.setattr(torch,'from_numpy',forbidden)
    args=SimpleNamespace(practical_route=route,practical_equalization=eq,practical_equalizer_method=method,practical_fs_hz=20e6,practical_fc_hz=2.462e9,practical_receiver_seed=2027)
    x=torch.randn(3,2,256);g=torch.Generator().manual_seed(1)
    set_evaluation_context(['a','b','c']);y,_=apply_practical(x,'practical_high',args,gen=g)
    set_evaluation_context(['c','a']);z,_=apply_practical(x[[2,0]],'practical_high',args,gen=g)
    assert torch.equal(z,y[[2,0]]) and torch.isfinite(y).all()

def test_practical_periodic_predictor_actual_model_and_cpu_scorer(tmp_path,monkeypatch):
    from scripts import predict_phase1_truth_last as predict
    from scripts.train_rc4_practical import build_args,resolved_config,native
    from torch.utils.data import Dataset
    cfg=Path(__file__).resolve().parents[1]/'configs/rc4_practical_full_zf_20260918.json'
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
    for suffix,scenes in [('practical',PRACTICAL)]:
        out=tmp_path/suffix
        evaluate_checkpoint(checkpoint,output=out,input_package=tmp_path,truth=truth,run_id='smoke',row_id='test',device='cpu',
            expected_records=24,scenarios=','.join(scenes),expected_epoch=100)
        assert json.loads((out/'score.json').read_text())['record_count']==24
        assert not json.loads((out/'evaluation_scope.json').read_text())['feeds_training']
    assert random.getstate()==states[0]
    assert np.array_equal(np.random.get_state()[1],states[1][1])
    assert torch.equal(torch.get_rng_state(),states[2])
