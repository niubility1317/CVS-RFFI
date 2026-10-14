import copy
import io
from pathlib import Path
import pytest
import torch
from experiments.standard_ce_baselines.common import ROOT,SIG_CONFIG,read
from experiments.standard_ce_baselines.source import validate_config
from experiments.standard_ce_baselines.dispatch import validate_spec,matrix
from experiments.standard_ce_baselines.model import build,VARIANTS
from experiments.standard_ce_baselines.sigtime import SignaturePool,path_signature


def test_eight_seed_matrix_and_reject_collision():
    spec=read(ROOT/SIG_CONFIG/'launch_spec.json')
    validate_spec(spec)
    variants,seeds,_=matrix(spec)
    assert variants==('sig_time',) and seeds==tuple(range(2026092701,2026092709))
    assert len(VARIANTS)==14 and 'sig_time' not in VARIANTS
    for row in spec['rows']:
        config=read(ROOT/SIG_CONFIG/(row['row_id']+'.json'));validate_config(config)
        assert config['model_seed']==row['model_seed'] and config['epochs']==200
    for defect in ('missing','duplicate','escape'):
        broken=copy.deepcopy(spec)
        if defect=='missing':broken['rows'].pop()
        elif defect=='duplicate':broken['rows'][-1]=broken['rows'][0]
        else:broken['rows'][0]['source_output']='/tmp/outside-source'
        with pytest.raises(ValueError):validate_spec(broken)


def test_source_rejects_inheritance_query_and_extra_strategy():
    config=read(ROOT/SIG_CONFIG/'sig_time-s2026092701.json')
    for key,value in [('checkpoint','old.pt'),('teacher','old.pt'),('target_truth','truth.json'),
                      ('extra_losses',['Fishr']),('augmentation',True),('domain_backbone',True),
                      ('epochs',20),('model_seed',2026092709)]:
        with pytest.raises(ValueError):validate_config(dict(config,**{key:value}))


def test_private_projection_rng_and_time_descriptor():
    torch.set_num_threads(2);torch.manual_seed(5)
    before=torch.random.get_rng_state().clone()
    if torch.cuda.is_available():gpu=torch.cuda.get_rng_state().clone()
    pool=SignaturePool(96,2026092701)
    assert torch.equal(before,torch.random.get_rng_state())
    if torch.cuda.is_available():assert torch.equal(gpu,torch.cuda.get_rng_state())
    t=torch.linspace(0,1,8)
    path=(t[None,:,None]*torch.arange(1.,7.)).expand(2,-1,-1)
    d=path_signature(path)
    assert d.shape==(2,28)
    assert torch.allclose(d[:,:6],torch.arange(1.,7.).expand(2,-1))
    assert torch.allclose(d[:,6],torch.ones(2))
    assert torch.allclose(d[:,7:],torch.zeros(2,21),atol=1e-6)


def test_default_head_joint_ce_updates_and_checkpoint_per_packet():
    torch.set_num_threads(2);torch.manual_seed(2026092701)
    model=build('sig_time',2026092701)
    assert sum(p.numel() for p in model.parameters())==196001
    assert sum(p.numel() for p in model.sequence_pool.parameters())==18976
    assert model.core.id_backbone.cls_head.weight.shape==(6,160)
    selected={n:p.detach().clone() for n,p in model.named_parameters()
              if n.startswith('sequence_pool.') or n.endswith('cls_head.weight')}
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    x=torch.randn(4,2,256)
    model.train();loss=torch.nn.functional.cross_entropy(model(x),torch.arange(4))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for n,p in model.named_parameters() if n in selected)
    optimizer.step()
    assert all(not torch.equal(selected[n],p) for n,p in model.named_parameters() if n in selected)
    model.eval()
    with torch.no_grad():expected=model(x)
    checkpoint=io.BytesIO();torch.save(model.state_dict(),checkpoint);checkpoint.seek(0)
    restored=build('sig_time',2026092708)
    restored.load_state_dict(torch.load(checkpoint,weights_only=True),strict=True);restored.eval()
    with torch.no_grad():
        assert torch.equal(restored(x),expected)
        assert torch.allclose(restored(x),torch.cat([restored(p[None]) for p in x]),atol=1e-4,rtol=1e-5)
        assert torch.isfinite(restored(torch.zeros(2,2,256))).all()
