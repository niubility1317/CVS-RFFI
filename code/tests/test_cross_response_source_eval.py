import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cvsrffi.cross_response.source_eval import evaluate_source_response,raw_received_iq,event_metadata_from_records
from cvsrffi.cross_response.statistics import WaveformStatistics,FixedSourceNormalizer
from cvsrffi.cross_response.readouts import ResponseReadouts
from cvsrffi.cross_response.predictor import SharedResponsePredictor
from cvsrffi.cross_response.analysis import paired_joint_effect,SCENARIOS


class Dataset:
    split_source = 'ssdg_source_v_select'
    def __init__(self,p=4,q=4,k=2):
        self.index = [SimpleNamespace(tx_i=t,rx_i=r,day_i=0,eq_i=0,sig_i=s,event_id=None)
                      for t in range(p) for r in range(q) for s in range(k)]
        self.read = []
    def __len__(self): return len(self.index)
    def raw_iq(self,index):
        row = self.index[index]
        t = torch.arange(32.)
        return torch.stack((torch.cos(t*(row.tx_i+1)/8),torch.sin(t*(row.rx_i+1)/8)))*(row.rx_i+1)
    def __getitem__(self,index):
        self.read.append(index)
        row = self.index[index]
        x = self.raw_iq(index)
        return x/x.square().mean().sqrt(),row.tx_i,row.rx_i,{}


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(2)
        self.dropout = nn.Dropout(.9)
        self.count = 0
        self.fail = False
    def forward(self,x,y_tx=None,grl_lambda=1.,return_aux=False,domain_labels=None):
        assert not self.training and not self.dropout.training and not self.bn.training
        assert y_tx is None and grl_lambda==0. and return_aux
        self.count += 1
        # Even unexpected local randomness must not perturb the caller RNG.
        random.random(); np.random.rand(); torch.rand(1)
        if self.fail: raise RuntimeError('deliberate evaluator failure')
        feature = self.dropout(self.bn(x)).mean(-1)
        return {'z_id':feature,'z_dom':feature+1}


def setup():
    ds = Dataset()
    stat = WaveformStatistics('iq',input_length=32)
    normal = FixedSourceNormalizer(5).fit(torch.randn(32,5))
    return Model(),ResponseReadouts(2,2,3),SharedResponsePredictor(3,5,'bilinear',2),stat,normal,ds


def run(parts,role='source_validation',**config):
    return evaluate_source_response(*parts,dict(source_eval_max_blocks=2,data_seed=17,**config),
                                    'cpu',{r:r for r in range(4)},source_role=role)


def test_source_eval_bounded_deterministic_raw_targets_restore_flags_rng():
    parts = setup()
    model = parts[0]
    model.bn.eval()  # Preserve deliberately heterogeneous modes exactly.
    flags = [m.training for m in model.modules()]
    mean = model.bn.running_mean.clone()
    random.seed(51); np.random.seed(51); torch.manual_seed(51)
    py,np_state,cpu = random.getstate(),np.random.get_state(),torch.get_rng_state()
    first = run(parts)
    assert random.getstate()==py
    assert np.array_equal(np.random.get_state()[1],np_state[1])
    assert torch.equal(torch.get_rng_state(),cpu)
    assert flags==[m.training for m in model.modules()]
    assert torch.equal(mean,model.bn.running_mean)
    second = run(parts)
    assert first==second
    json.dumps(first,allow_nan=False)
    assert first['valid_blocks']==2 and first['backbone_forwards']==2
    assert first['valid_role_tasks']==8 and first['independent_query_tasks']==8
    assert first['changed_donor_tasks']==16 and first['shuffled_rx_tasks']==8
    assert first['constant_mse']>0 and first['response_mse']>=0
    assert all(p.grad is None for p in model.parameters())


def test_source_eval_no_target_access_and_failed_state_restoration():
    parts = setup()
    with pytest.raises(ValueError): run(parts,role='target')
    parts[-1].split_source = 'target_clean'
    with pytest.raises(ValueError): run(parts)
    assert parts[-1].read==[]
    parts[-1].split_source = 'ssdg_source_v_cal'
    parts[0].fail = True
    flags = [m.training for m in parts[0].modules()]
    cpu = torch.get_rng_state()
    with pytest.raises(RuntimeError): run(parts)
    assert torch.equal(cpu,torch.get_rng_state())
    assert flags==[m.training for m in parts[0].modules()]


def test_incomplete_blocks_are_unavailable_not_success_or_zero():
    parts = list(setup())
    parts[-1] = Dataset(p=2,q=2)
    output = run(parts)
    assert output['status']=='UNAVAILABLE' and output['response_mse'] is None
    assert output['valid_blocks']==0 and output['backbone_forwards']==0
    assert parts[-1].read==[]


def test_event_overlap_skips_candidate_without_query_leakage():
    parts = setup()
    for row in parts[-1].index:
        row.event_id = 'same-physical-event'
    output = run(parts)
    assert output['status']=='UNAVAILABLE' and output['response_mse'] is None
    assert output['skipped_blocks']==2 and output['backbone_forwards']==0
    assert parts[-1].read==[]


def test_physical_raw_reader_preserves_amplitude_and_fixed_crop():
    sample = np.arange(20,dtype=np.float32).reshape(10,2)
    base = SimpleNamespace(data=[[[[[sample]]]]],out_len=6,crop_mode='center')
    ds = SimpleNamespace(base=base,index=[SimpleNamespace(tx_i=0,rx_i=0,day_i=0,eq_i=0,sig_i=0)])
    x = raw_received_iq(ds,0)
    assert torch.equal(x,torch.tensor(sample[2:8].T.tolist()))
    base.crop_mode = 'random'
    with pytest.raises(ValueError): raw_received_iq(ds,0)
    with pytest.raises(ValueError): event_metadata_from_records([SimpleNamespace(event_id=None)],[(0,4)])
    assert event_metadata_from_records([SimpleNamespace(event_id='packet')],[(0,4)])['event_ids']==['packet']


def rows(seeds):
    return [dict(variant=v,seed=s,metrics={scene:{'accuracy':value+s*.001} for scene in SCENARIOS},
                 costs={'training_seconds':10,'peak_memory_bytes':100,'optimizer_steps':3,'physical_record_exposures':32})
            for s in seeds for v,value in (('U1',.7),('U2',.75),('U3',.76),('U4_additive',.83))]


def test_paired_joint_effect_no_single_seed_ci_and_real_scenario_cost_completeness():
    one = paired_joint_effect(rows([1]))
    assert one['complete_four_scenarios']
    assert one['scenarios']['clean']['mean']==pytest.approx(.02)
    assert one['scenarios']['clean']['confidence_interval'] is None
    many = paired_joint_effect(rows([1,2,3]),seed=6)
    assert many==paired_joint_effect(rows([1,2,3]),seed=6)
    assert many['scenarios']['clean']['confidence_interval']==pytest.approx([.02,.02])
    incomplete = rows([1,2])[:-1]
    incomplete[0]['metrics'].pop('leo_rain_weak')
    incomplete[0]['costs'].pop('peak_memory_bytes')
    report = paired_joint_effect(incomplete)
    assert report['unpaired_seeds']==[2] and not report['complete_four_scenarios']
    assert report['costs'][0]['missing']==['peak_memory_bytes']
    with pytest.raises(ValueError): paired_joint_effect(rows([1])+rows([1]))
