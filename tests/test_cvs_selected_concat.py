import copy
import json
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from experiments.cvs_selected_concat.model import build, VARIANT
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY
from experiments.cvs_selected_concat.augmentation import MidUrbanAugment,RECIPE
from experiments.cvs_selected_concat.source import loss_for_batch, validate_config, train


@pytest.fixture
def artifact_dir():
    # Windows pytest tmpdir chmod breaks inherited workspace ACLs on this host.
    import uuid
    path=Path('E:/type10-7/local_artifacts/cvs_identity_ce_checks')/str(uuid.uuid4())
    path.mkdir(parents=True)
    return path


def test_native_identity_only_ce_updates_and_roundtrips(artifact_dir):
    torch.set_num_threads(2); torch.manual_seed(7)
    model = build()
    assert not hasattr(model, 'dom_backbone')
    assert not any('dom_backbone' in n or 'grl' in n for n in model.state_dict())
    x, y = torch.randn(4,2,256), torch.tensor([0,1,2,3])
    before = next(model.parameters()).detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0002)
    calls=[]
    def augment(x,metadata): calls.append(len(x)); return x*.9
    loss, ce, sat = loss_for_batch(model,x,y,{'meta':[{}]*4},augment,79)
    assert sat is None and calls==[] and torch.equal(loss,ce)
    loss.backward(); optimizer.step()
    assert not torch.equal(before,next(model.parameters()))
    loss,ce,sat = loss_for_batch(model,x,y,{'meta':[{}]*4},augment,80)
    assert calls==[4] and torch.allclose(loss,ce+.68*sat)
    model.eval()
    expected=model(x).detach()
    # Native head labels are absent, so CE never includes a CosFace label margin.
    assert expected.shape==(4,6)
    path=artifact_dir/'checkpoint.pt';torch.save(model.state_dict(),path)
    restored=build().eval();restored.load_state_dict(torch.load(path,weights_only=False))
    assert torch.equal(expected,restored(x))


def test_guards_reject_inheritance_target_extra_loss_and_budget():
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'experiments/cvs_selected_concat/configs/source-s2026092701.json').read_text(encoding='utf-8'))
    validate_config(cfg)
    for key,value in [('checkpoint','old.pt'),('target_inputs','query.npy'),('epochs',199),
        ('extra_losses',['domain_ce']),('domain_backbone',True),('lambda_sat_cls',1.),
        ('augmentation_seed',11),('mixstyle',True)]:
        with pytest.raises(ValueError): validate_config(dict(cfg,**{key:value}))


def test_no_output_overwrite(artifact_dir):
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'experiments/cvs_selected_concat/configs/source-s2026092701.json').read_text(encoding='utf-8'))
    cfg['output_root']=str(artifact_dir)
    with pytest.raises(FileExistsError):train(cfg)


def test_frozen_prediction_truth_last_score_and_no_restart(artifact_dir):
    import numpy as np
    from experiments.cvs_selected_concat.predict import predict
    from experiments.cvs_selected_concat.source import write
    from comparison_suite.score import score
    torch.set_num_threads(2)
    source=artifact_dir/'source';capsule=artifact_dir/'capsule';output=artifact_dir/'prediction'
    source.mkdir();capsule.mkdir()
    classes=[str(i) for i in range(6)]
    contract=dict(role_ids={'L_s':['source-l'],'U_s':['source-u'],'V':['source-v']},
        source_rxs=[1,3,4,6,8],source_days=[1,2,3],ratios=[.07,.63,.30],split_seed=392005,
        equalized=1,out_len=256,normalize=True,classes=classes,num_classes=6)
    initial=dict(status='SCRATCH',scratch_only=True,checkpoint=None,physical_roles='EXACT_MATCH',checkpoint_sources=[],ancestors=[],
        target_access=False,target_contact=False,model_seed=392005)
    write(source/'source_contract.json',contract);write(source/'initialization.json',initial)
    completion=dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False)
    frozen=dict(status='SOURCE_SELECTION_FROZEN',variant=VARIANT,epoch=200,target_access=False,target_score_used=False)
    write(source/'completion.json',completion)
    write(source/'source_selection.json',frozen)
    root=Path(__file__).resolve().parents[1]
    resolved=json.loads((root/'experiments/cvs_selected_concat/configs/source-s2026092701.json').read_text(encoding='utf-8'))
    resolved.update(model_seed=392005,architecture_actual=build().contract(),backend_flags=FULL_FP32_POLICY,
        target_access=False,source_counts={'L_s':6300,'U_s':56700,'V':27000},steps_per_epoch=50)
    write(source/'resolved_config.json',resolved)
    torch.save(dict(model=build().state_dict(),epoch=200,source_contract=contract,
        initialization=initial,selection='fixed_last_epoch',method='cvs_selected_concat',variant=VARIANT,config=resolved),source/'last.pt')
    write(capsule/'manifest.json',dict(status='VALIDATED_ONCE',classes=classes,
        channel='residual/post_sync/noeq',scenes=['practical_high','practical_mid','practical_low_urban']))
    ids=np.asarray(['target-'+str(i) for i in range(6)])
    np.savez(capsule/'index.npz',ids=ids,scenes=np.asarray([0,1,2,0,1,2]))
    for view in ('clean','satellite'):np.save(capsule/(view+'.npy'),np.random.randn(6,2,256).astype('float32'))
    cfg=dict(output_root=str(output),source_output=str(source),source_contract=str(source/'source_contract.json'),
        model_seed=392005,p1_capsule=str(capsule),device='cpu',variant=VARIANT,numerical_policy=FULL_FP32_POLICY)
    with pytest.raises(ValueError):predict(dict(cfg,p1_truth='forbidden'))
    for j,(file,base,key,value) in enumerate([
        ('initialization.json',initial,'checkpoint','historical.pt'),
        ('completion.json',completion,'target_access',True),
        ('completion.json',completion,'target_evaluated',True),
        ('completion.json',completion,'steps',9999),
        ('source_selection.json',frozen,'target_access',True),
        ('source_selection.json',frozen,'epoch',199),
        ('resolved_config.json',resolved,'augmentation_recipe',dict(RECIPE,scenes=['practical_high'])),
        ('resolved_config.json',resolved,'steps_per_epoch',49),
        ('resolved_config.json',resolved,'backend_flags',dict(FULL_FP32_POLICY,cudnn_allow_tf32=True))]):
        write(source/file,dict(base,**{key:value}))
        rejected=artifact_dir/('rejected-'+str(j))
        with pytest.raises(ValueError):predict(dict(cfg,output_root=str(rejected)))
        assert not (rejected/'phase1_predictions.npz').exists()
        write(source/file,base)
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


def test_fresh_source_process_imports_top_level_baseline_package():
    import subprocess
    import sys
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'-m','experiments.cvs_selected_concat.source','--help'],
        cwd=root,capture_output=True,text=True)
    assert result.returncode==0,result.stderr

def test_only_mid_urban_actual_channel_reproducible_across_batch_order():
    import numpy as np
    augment=MidUrbanAugment()
    assert set(augment.configs)=={'practical_mid','practical_low_urban'}
    x=torch.randn(8,2,256);meta=[dict(sample_id='packet-'+str(i),session_id='rx:1/day:1') for i in range(8)]
    with pytest.raises(ValueError):augment(x,metadata=meta)
    for epoch in (80,90,91,200):
        augment.set_epoch(epoch);a=augment(x,metadata=meta);b=augment(x.flip(0),metadata=list(reversed(meta))).flip(0)
        assert torch.equal(a,b) and torch.isfinite(a).all()
        assert set(augment.configs)=={'practical_mid','practical_low_urban'}

def test_predictor_requires_frozen_source_and_registered_architecture(artifact_dir):
    from experiments.cvs_selected_concat.predict import predict
    with pytest.raises(ValueError,match='Fixed architecture'):
        predict(dict(variant='wrong',numerical_policy=FULL_FP32_POLICY))
    from experiments.cvs_selected_concat.publish import REMOTE
    compile(REMOTE.replace('CONFIG',repr(dict(project='x',release='x',run='x',archive='x',sha256='x',commit='x'))),'remote','exec')

def test_fixed_wrapper_exactly_preserves_residual_structure_and_gradient():
    from experiments.cvs_residual_identity.model import build as original
    torch.set_num_threads(2);torch.manual_seed(31);a=build().eval()
    torch.manual_seed(31);b=original('residual_fusion').eval()
    x=torch.randn(4,2,256)
    assert torch.equal(a(x),b(x))
    assert a.contract()['residual_head'] and a.contract()['no_dac']
    assert a.contract()['total_parameters']==164225
    F.cross_entropy(a(x),torch.arange(4)).backward()
    assert a.identity.id_backbone.cls_head.gain.grad is not None

def test_windows_publisher_git_status_command_is_bounded():
    import ast,subprocess
    from experiments.cvs_selected_concat import publish
    tree=ast.parse(Path(publish.__file__).read_text(encoding='utf-8'))
    text=ast.get_source_segment(Path(publish.__file__).read_text(encoding='utf-8'),next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='dirty' for t in n.targets)))
    assert '*prefixes' in text and '*selected' not in text
    assert len(subprocess.list2cmdline(['git','status','--porcelain','--']+['experiments/'+s+'/' for s in ['cvs_selected_concat','cvs_energy_identity','cvs_equivariant_identity','cvs_coordinate_identity','cvs_synchronized_identity','cvs_gauge_identity','cvs_rff_physics','cvs_reference_identity','cvs_residual_identity','cvs_clean_design','cvs_identity_ce','adv3b02_xuc/code']]))<32767
