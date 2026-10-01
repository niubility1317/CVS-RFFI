import copy
import json
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from experiments.cvs_identity_ce.model import IdentityOnlyCVS
from experiments.cvs_identity_ce.source import loss_for_batch, validate_config, train


@pytest.fixture
def artifact_dir():
    # Windows pytest tmpdir chmod breaks inherited workspace ACLs on this host.
    import uuid
    path=Path('E:/type10-7/local_artifacts/cvs_identity_ce_checks')/str(uuid.uuid4())
    path.mkdir(parents=True)
    return path


def test_native_identity_only_ce_updates_and_roundtrips(artifact_dir):
    torch.set_num_threads(2); torch.manual_seed(7)
    model = IdentityOnlyCVS()
    assert not hasattr(model, 'dom_backbone')
    assert not any('dom_backbone' in n or 'grl' in n for n in model.state_dict())
    x, y = torch.randn(4,2,256), torch.tensor([0,1,2,3])
    before = model.id_backbone.cls_head.head.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0002)
    calls=[]
    def augment(x,metadata): calls.append(len(x)); return x*.9
    loss, ce, sat = loss_for_batch(model,x,y,{'meta':[{}]*4},augment,79)
    assert sat is None and calls==[] and torch.equal(loss,ce)
    loss.backward(); optimizer.step()
    assert not torch.equal(before,model.id_backbone.cls_head.head.weight)
    loss,ce,sat = loss_for_batch(model,x,y,{'meta':[{}]*4},augment,80)
    assert calls==[4] and torch.allclose(loss,ce+.68*sat)
    model.eval()
    expected=model(x).detach()
    # Native head labels are absent, so CE never includes a CosFace label margin.
    assert torch.equal(expected,model.id_backbone(x,y=None,return_aux=False))
    path=artifact_dir/'checkpoint.pt';torch.save(model.state_dict(),path)
    restored=IdentityOnlyCVS().eval();restored.load_state_dict(torch.load(path,weights_only=False))
    assert torch.equal(expected,restored(x))


def test_guards_reject_inheritance_target_extra_loss_and_budget():
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'experiments/cvs_identity_ce/configs/source-s392005.json').read_text(encoding='utf-8'))
    validate_config(cfg)
    for key,value in [('checkpoint','old.pt'),('target_inputs','query.npy'),('epochs',199),
        ('extra_losses',['domain_ce']),('domain_backbone',True),('lambda_sat_cls',1.),
        ('augmentation_seed',11),('mixstyle',True)]:
        with pytest.raises(ValueError): validate_config(dict(cfg,**{key:value}))


def test_no_output_overwrite(artifact_dir):
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'experiments/cvs_identity_ce/configs/source-s392005.json').read_text(encoding='utf-8'))
    cfg['output_root']=str(artifact_dir)
    with pytest.raises(FileExistsError):train(cfg)


def test_frozen_prediction_truth_last_score_and_no_restart(artifact_dir):
    import numpy as np
    from experiments.cvs_identity_ce.predict import predict
    from experiments.cvs_identity_ce.source import write
    from comparison_suite.score import score
    torch.set_num_threads(2)
    source=artifact_dir/'source';capsule=artifact_dir/'capsule';output=artifact_dir/'prediction'
    source.mkdir();capsule.mkdir()
    classes=[str(i) for i in range(6)]
    contract=dict(role_ids={'L_s':['source-l'],'U_s':['source-u'],'V':['source-v']},
        source_rxs=[1,3,4,6,8],source_days=[1,2,3],ratios=[.07,.63,.30],split_seed=392005,
        equalized=1,out_len=256,normalize=True,classes=classes,num_classes=6)
    initial=dict(status='SCRATCH',scratch_only=True,checkpoint_sources=[],ancestors=[],
        target_access=False,target_contact=False,model_seed=392005)
    write(source/'source_contract.json',contract);write(source/'initialization.json',initial)
    write(source/'completion.json',dict(status='SOURCE_TRAINED',epoch=200))
    torch.save(dict(model=IdentityOnlyCVS().state_dict(),epoch=200,source_contract=contract,
        initialization=initial,selection='fixed_last_epoch',method='cvs_identity_ce'),source/'last.pt')
    write(capsule/'manifest.json',dict(status='VALIDATED_ONCE',classes=classes,
        channel='residual/post_sync/noeq',scenes=['practical_high','practical_mid','practical_low_urban']))
    ids=np.asarray(['target-'+str(i) for i in range(6)])
    np.savez(capsule/'index.npz',ids=ids,scenes=np.asarray([0,1,2,0,1,2]))
    for view in ('clean','satellite'):np.save(capsule/(view+'.npy'),np.random.randn(6,2,256).astype('float32'))
    cfg=dict(output_root=str(output),source_output=str(source),source_contract=str(source/'source_contract.json'),
        model_seed=392005,p1_capsule=str(capsule),device='cpu')
    with pytest.raises(ValueError):predict(dict(cfg,p1_truth='forbidden'))
    predict(cfg)
    config=artifact_dir/'predict.json';write(config,cfg)
    spec=dict(runtime_root=str(artifact_dir),p1_truth=str(artifact_dir/'truth.json'),rows=[
        dict(row_id='fixture',method='cvs_identity_ce',model_seed=392005,stage='phase12',config=str(config),output_root=str(output))])
    # No truth artifact exists during inference; independent scoring opens it later.
    assert not (artifact_dir/'truth.json').exists()
    write(artifact_dir/'truth.json',{str(s):dict(label=i,receiver=0) for i,s in enumerate(ids)})
    result=score(spec,'p1')
    assert result['status']=='SCORED_COMPLETE' and result['records']==10
    with pytest.raises(FileExistsError):predict(cfg)
