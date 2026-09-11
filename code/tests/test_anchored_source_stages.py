import json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_evidence_integration import model_args
from post_stage_common import build_baseline_model
from cvsrffi.anchored_source import run_stage,FrozenPartialSystem
from cvsrffi.anchored_pipeline import FrozenAnchoredSystem


def test_explicit_source_cli_stages_and_partial_bundle(tmp_path):
    torch.set_num_threads(2);torch.manual_seed(901)
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/core90_anchored_geometry_v1.json').read_text())
    config['profile']=dict(warmup=0,repeats=1);config['cache']['batch_size']=32
    source={};roles={}
    for role,offset in (('L_s',0),('V',10)):
        ids=[f'{y}:{rx}:{day}:{i+offset}' for y in range(3) for rx in (1,3,4,6,8) for day in (1,2,3) for i in range(2)]
        roles[role]=ids;source[role]=dict(x=torch.randn(len(ids),2,256),y=torch.tensor([int(p.split(':')[0]) for p in ids]),physical_ids=ids)
        torch.save(source[role],tmp_path/(role+'.pt'))
    contract=dict(dataset_id='SYNTHETIC_TECHNICAL_TEST_ONLY',roles=roles,source_receivers=[1,3,4,6,8],source_days=[1,2,3],split_seed=392005)
    (tmp_path/'contract.json').write_text(json.dumps(contract),encoding='utf-8')
    architecture=model_args();model=build_baseline_model(architecture,torch.device('cpu'))
    torch.save(dict(model=model.state_dict(),baseline_args=vars(architecture),training_data_contract=contract,
                    checkpoint_lineage=dict(from_scratch=True,upstream=[],selection='final_only',target_feedback=False)),tmp_path/'ground.pt')
    def stage(name,candidate,output,role='L_s',cache=None,input=None):
        args=SimpleNamespace(stage=name,candidate=candidate,output=str(tmp_path/output),ground=str(tmp_path/'ground.pt'),contract=str(tmp_path/'contract.json'),
                             role=role,seed=392005,device='cpu',source=str(tmp_path/(role+'.pt')),cache=None if cache is None else str(tmp_path/cache),
                             input=None if input is None else str(tmp_path/input))
        run_stage(args,config)
        return tmp_path/output
    for candidate,prefix in (('A0','g'),('P1','e')):
        stage('cache',candidate,prefix+'Ls')
        stage('cache',candidate,prefix+'V',role='V')
        fit=stage('fit',candidate,prefix+'fit',cache=prefix+'Ls')
        input=prefix+'fit/'+('partial_evidence.pt' if candidate=='P1' else 'system_state.pt')
        stage('calibrate',candidate,prefix+'cal',cache=prefix+'V',input=input)
        export=stage('export',candidate,prefix+'export',input=prefix+'cal/system_state.pt')
        state=torch.load(export/'frozen_system.pt',weights_only=True)
        if candidate=='A0':
            system=FrozenAnchoredSystem.from_state(state)
            pred=system.predict(source['L_s']['x'][:3]);assert pred['top_class'].shape==(3,)
            stage('profile',candidate,'profile',input=prefix+'export/frozen_system.pt')
        else:
            system=FrozenPartialSystem.from_state(state)
            pred=system.predict(source['L_s']['x'][:3],'all_missing')
            assert not pred['accepted'].any() and pred['status']==['defer']*3
            pred=system.predict(source['L_s']['x'][:3],'missing_t')
            assert torch.isfinite(pred['probabilities']).all()
    with pytest.raises(FileExistsError):stage('fit','A0','gfit',cache='gLs')
    # File-backed OOF orchestration with an explicitly synthetic count trainer;
    # actual optimizer behavior is covered by the separate real-IQ fit test.
    from dataclasses import asdict
    from cvsrffi.anchored_source import cache_identity
    from cvsrffi.anchored_cache import load_source_cache,IdentityAnchor
    from cvsrffi.anchored_fit import ExpertArtifact,FitConfig
    from cvsrffi.anchored_crossfit import run_expert_oof
    from cvsrffi.anchored_geometry import AnchoredMetricHead
    rows=load_source_cache(tmp_path/'gLs',cache_identity(config,tmp_path/'ground.pt',tmp_path/'contract.json','L_s','joint'))
    anchor=IdentityAnchor(model)
    def trainer(cache,train_rx,cfg,seed,path,**kwargs):
        path.mkdir(parents=True)
        expert=ExpertArtifact(AnchoredMetricHead(anchor.w0,anchor.tau0).export_state(),tuple(train_rx),seed,asdict(cfg),[],[],'SYNTHETIC_COUNT_TEST')
        expert.save(path/'expert.pt')
        expert.predict=lambda data:data.baseline_inference_logits.clone()
        return expert
    from unittest.mock import patch
    with patch('cvsrffi.anchored_source.run_expert_oof',side_effect=lambda *a,**kw:run_expert_oof(*a,**kw,fit_fn=trainer)):
        stage('oof','A4','oof',cache='gLs')
    assert len(list((tmp_path/'oof').glob('*/expert.pt')))==6
    with patch('cvsrffi.anchored_source.fit_expert',side_effect=AssertionError('must reuse OOF final expert')):
        stage('calibrate','A4','oofcal',cache='gV',input='oof/system_state.pt')
        stage('export','A4','oofexport',input='oofcal/system_state.pt')
    final=ExpertArtifact.load(tmp_path/'oof/all_source/expert.pt')
    exported=torch.load(tmp_path/'oofexport/frozen_system.pt',weights_only=True)
    from cvsrffi.anchored_pipeline import state_identity
    assert state_identity(final.state)==state_identity(exported['expert'])
    inactive=stage('fuse','A6','a6_inactive',cache='gLs',input='oof')
    assert json.loads((inactive/'activation.json').read_text())['status']=='NOT_ACTIVATED_NO_SOURCE_UTILITY'
    assert not (inactive/'system_state.pt').exists()
    stage('fuse','A5-50','a5',cache='gLs',input='oof')
    stage('calibrate','A5-50','a5cal',cache='gV',input='a5/system_state.pt')
    stage('export','A5-50','a5export',input='a5cal/system_state.pt')
    with pytest.raises(ValueError,match='candidate'):
        stage('export','A6','wrong_candidate',input='gcal/system_state.pt')
    manifest_path=tmp_path/'efit'/'fit_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['cache_identity']['checkpoint_sha256']='c'*64
    manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
    with pytest.raises(ValueError,match='identity mismatch'):
        stage('calibrate','P1','bad_e_cal',cache='eV',input='efit/partial_evidence.pt')
