from experiments.pamf_physics_tri import hybrid_structure_plan as plan
import gzip
import json
import torch
import numpy as np
from torch import nn
from experiments.cric_dura_comparison import f_optimizer
from experiments.pamf_physics_tri.hybrid_update_probe import installed
from experiments.pamf_physics_tri.hybrid_paired_score import net_stats


def test_structural_matrix_has_matched_abcd_without_parameter_grid():
    rows=plan.candidate_rows()
    assert len(rows)==len({r['row_id'] for r in rows})==16
    assert {r['structure_arm'] for r in rows}=={'A','B','C','D'}
    assert {r['auxiliary_weight'] for r in rows}=={.1}
    for setting,_,seeds in plan.MATRIX:
        for seed in seeds:
            assert {r['structure_arm'] for r in rows if r['setting']==setting and r['model_seed']==seed}=={'A','B','C','D'}


class CoordinateSinc(nn.Module):
    def __init__(self):
        super().__init__();self.sample_rate=25000000.;self.scale_hz=25000.;self.mode='CENTER_25K'
        self.low_coordinate=nn.Parameter(torch.zeros(24,1));self.band_coordinate=nn.Parameter(torch.zeros(24,1))
        self.register_buffer('low_origin_hz',torch.full((24,1),50000.));self.register_buffer('band_origin_hz',torch.full((24,1),75000.))


def test_shadow_probe_observes_real_adamw_without_replacing_update(tmp_path):
    model=nn.Module();node=model
    for name in ('id_backbone','encoder','core','id_backbone'):
        child=nn.Module();node.add_module(name,child);node=child
    node.sinc=CoordinateSinc();node.cls_head=nn.Linear(160,6,bias=False);node.body=nn.Parameter(torch.tensor([.2,-.4]))
    named=list(model.named_parameters());opt=f_optimizer.build(model,model.parameters(),'CENTER_UNITS',lr=.0002,weight_decay=.01)
    base=tuple(torch.ones_like(p)*.1 for _,p in named)
    for _,p in named:p.grad=torch.ones_like(p)*.4
    path=tmp_path/'updates.jsonl.gz'
    with installed() as probe:
        probe.capture(model,named,base,50,path)
        f_optimizer.clip_physical(opt,model.parameters(),5.,clip_impl=torch.nn.utils.clip_grad_norm_)
        opt.step()
        assert probe.calls==1 and probe.pending is None
    with gzip.open(path,'rt',encoding='utf8') as f: row=json.loads(f.readline())
    assert row['step']==50 and 'same live parameters' in row['reference']
    assert any(v['aux_update_delta_norm']>0 for v in row['groups'].values())
    assert opt._sinc_unit_steps==1


def test_paired_net_corrections_and_damage_are_both_counted():
    y=np.asarray([0,1,0,1]);old=np.asarray([0,0,0,0]);new=np.asarray([1,1,0,1])
    rows=net_stats(y,old,new,np.asarray(['a','a','b','b']),np.asarray(['d','d','e','e']))
    overall=next(r for r in rows if r['dimension']=='overall')
    assert (overall['repairs'],overall['damage'],overall['net'],overall['count'])==(2,1,1,4)
    assert sum(r['net'] for r in rows if r['dimension']=='receiver')==1
