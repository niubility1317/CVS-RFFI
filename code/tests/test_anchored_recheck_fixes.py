import copy,json,sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_anchored_cache import toy_rows
from cvsrffi.anchored_crossfit import _gate_training
from cvsrffi.anchored_pipeline import FrozenAnchoredSystem
from cvsrffi.anchored_geometry import AnchoredMetricHead
from cvsrffi.anchored_fusion import action_utilities
from cvsrffi.anchored_reporting import assess_source_promotion


def complete_records():
    return [dict(candidate=candidate,group=group,value=value,weighted_net=.01 if candidate=='A6' else .002,
                 weighted_accuracy=.81 if candidate=='A6' else .802,weighted_utility=.009 if candidate=='A6' else .001,
                 rows={'all':120,'RX':24,'TX':20,'day':40,'view':60}[group])
            for candidate in ('A6','A5_inner_selected')
            for group,values in (('all',['all']),('RX',map(str,(1,3,4,6,8))),('TX',map(str,range(6))),
                                 ('day',map(str,(1,2,3))),('view',['clean'])) for value in values]


def test_raw_actions_training_deployment_exact_for_boundary_and_protected_rows():
    torch.set_num_threads(2);torch.manual_seed(602)
    rows=toy_rows(receivers=(1,3,4,6,8));n=len(rows)
    s0=torch.randn(n,6)*.4;s0[:,0]+=2;s0[:,1]+=.4
    sg=torch.randn(n,6)*.3;sg[:,2:]-=3
    p=s0.double().softmax(-1);delta=p[:,0]-p[:,1]
    e=sg.double().exp();other=e[:,2:].sum(-1)
    sg[:,1]=(((1+delta)*e[:,0]+delta*other)/(1-delta)).log().float()
    rows.baseline_inference_logits=s0
    _,threshold,_,_,training=_gate_training(rows,sg,2.,.001,0.,.9)
    state=AnchoredMetricHead(torch.randn(6,8),10.).export_state()
    deployed=[]
    for action in range(5):
        system=FrozenAnchoredSystem(None,state,fixed_action=action,protection_threshold=threshold)
        result=system.predict_scores(s0,sg,rows.quality,rows.valid)
        deployed.append(result['log_probabilities'])
        assert torch.equal(result['top_class'],training['log_probabilities'][:,action].argmax(-1))
        assert torch.equal(result['realized_alpha'],training['alpha'][:,action])
        torch.testing.assert_close(result['log_probabilities'],training['log_probabilities'][:,action],rtol=0,atol=0)
    deployed=torch.stack(deployed,1)
    assert torch.equal(action_utilities(deployed,rows.labels,s0.argmax(-1)),action_utilities(training['log_probabilities'],rows.labels,s0.argmax(-1)))


@pytest.mark.parametrize('damage',['missing_tx','duplicate_rx','wrong_rx','missing_day','nan','inf','missing_a5'])
def test_incomplete_or_corrupt_groups_never_promote(damage):
    records=complete_records()
    if damage=='missing_tx':records=[r for r in records if r['group']!='TX' or r['value']=='0']
    if damage=='duplicate_rx':records.append(copy.deepcopy(next(r for r in records if r['group']=='RX')))
    if damage=='wrong_rx':next(r for r in records if r['group']=='RX')['value']='999'
    if damage=='missing_day':records=[r for r in records if r['group']!='day']
    if damage in ('nan','inf'):records[0]['weighted_net']=float(damage)
    if damage=='missing_a5':records=[r for r in records if r['candidate']=='A6']
    result=assess_source_promotion({s:records for s in (392005,392006,392007)})
    assert not result['passed']
    assert result['status']=='INCOMPLETE_SOURCE_EVIDENCE'


def test_complete_evidence_contains_a6_vs_inner_a5_differences():
    result=assess_source_promotion({s:complete_records() for s in (392005,392006,392007)})
    assert result['passed'] and not result['confirmation']
    assert len(result['comparisons'])==3*(1+5+6+3+1)
    assert all(abs(r['accuracy_delta_A6_vs_A5']-.008)<1e-12 for r in result['comparisons'])


def aggregate_fixture(tmp_path):
    from cvsrffi.anchored_fit import write_csv
    from cvsrffi.anchored_source import cache_identity
    from dataclasses import asdict
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/core90_anchored_geometry_v1.json').read_text())
    contract={k:config[k] for k in ('source_receivers','source_days','split_seed')}
    contract['tx_mapping']=[10,20,30,40,50,60]
    cp=tmp_path/'contract.json';cp.write_text(json.dumps(contract),encoding='utf-8')
    gp=tmp_path/'ground.stub';gp.write_bytes(b'synthetic binding only, no checkpoint loaded')
    identity=asdict(cache_identity(config,gp,cp,'L_s','joint'))
    paths=[]
    for seed in config['head_seeds']:
        path=tmp_path/str(seed);path.mkdir();paths.append(str(path))
        manifest=dict(stage='fuse',status='SOURCE_STAGE_COMPLETE',candidate='A6',head_seed=seed,config=config,
                      cache_identity=identity,source_only=True,backbone_updated=False,support_used=False)
        (path/'protocol_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        (path/'activation.json').write_text(json.dumps(dict(status='SOURCE_NESTED_EVALUATED')),encoding='utf-8')
        write_csv(path/'nested_source_metrics.csv',complete_records())
    return config,cp,paths


def test_aggregate_cli_reads_three_existing_runs_without_training(tmp_path):
    import subprocess
    config,contract,paths=aggregate_fixture(tmp_path)
    script=Path(__file__).resolve().parents[1]/'scripts/aggregate_core90_anchored_source.py'
    output=tmp_path/'aggregate'
    result=subprocess.run([sys.executable,'-X','utf8',str(script),'--inputs',*paths,'--contract',str(contract),'--output',str(output)],capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stderr
    manifest=json.loads((output/'protocol_manifest.json').read_text(encoding='utf-8'))
    verdict=json.loads((output/'source_promotion.json').read_text(encoding='utf-8'))
    assert verdict['passed'] and manifest['additional_fits']==0
    assert len(manifest['inputs'])==3 and len(verdict['comparisons'])==48
    assert (output/'a6_vs_inner_a5.csv').exists()


@pytest.mark.parametrize('damage',['duplicate_seed','checkpoint','contract','config','inactive'])
def test_aggregate_rejects_unmatched_runs_before_writing(tmp_path,damage):
    from cvsrffi.anchored_aggregation import aggregate_source_runs
    config,contract,paths=aggregate_fixture(tmp_path)
    file=Path(paths[-1])/'protocol_manifest.json';manifest=json.loads(file.read_text())
    if damage=='duplicate_seed':manifest['head_seed']=config['head_seeds'][0]
    if damage=='checkpoint':manifest['cache_identity']['checkpoint_sha256']='different'
    if damage=='contract':manifest['cache_identity']['data_contract_sha256']='different'
    if damage=='config':manifest['config']['fit']['lr']=.002
    if damage=='inactive':manifest['status']='NOT_ACTIVATED_NO_SOURCE_UTILITY'
    file.write_text(json.dumps(manifest),encoding='utf-8')
    with pytest.raises(ValueError):aggregate_source_runs(paths,tmp_path/'bad',config,contract)
    assert not (tmp_path/'bad').exists()


def test_tiny_unique_h0_decision_survives_actions_utility_and_calibration():
    from cvsrffi.anchored_fusion import realize_actions
    from cvsrffi.anchored_calibration import FinalProbabilityCalibrator
    s0=torch.tensor([[0.,1e-20,-1e-20],[0.,0.,0.]])
    sg=torch.tensor([[1.,0.,0.],[0.,1.,0.]])
    valid=torch.ones(2,dtype=torch.bool);quality=torch.zeros(2,3)
    actions=realize_actions(s0,sg,protection_threshold=1.)
    assert torch.equal(actions['predictions'][:,0],s0.argmax(-1))
    u=action_utilities(actions['log_probabilities'],torch.tensor([1,0]),s0.argmax(-1),predictions=actions['predictions'])
    assert (u[:,0]==0).all()
    system=FrozenAnchoredSystem(None,AnchoredMetricHead(torch.randn(3,8),10.).export_state(),fixed_action=0)
    raw=system.predict_scores(s0,sg,quality,valid)
    assert torch.equal(raw['top_class'],s0.argmax(-1))
    system.attach_calibrator(FinalProbabilityCalibrator().fit(raw['log_probabilities'],torch.tensor([1,0]),system_identity=system.identity))
    assert torch.equal(system.predict_scores(s0,sg,quality,valid)['top_class'],s0.argmax(-1))
    rows=toy_rows();rows.baseline_inference_logits=s0[:1].repeat(len(rows),1)
    scores=sg[:1].repeat(len(rows),1)
    gate,threshold,_,_,trained=_gate_training(rows,scores,2.,.001,0.,.9)
    gated=FrozenAnchoredSystem(None,system.head.export_state(),gate_state=gate.state_dict(),protection_threshold=threshold)
    result=gated.predict_scores(rows.baseline_inference_logits,scores,rows.quality,rows.valid)
    assert torch.equal(result['top_class'],trained['predictions'][torch.arange(len(rows)),result['action']])


@pytest.mark.parametrize('damage',['impossible_baseline','impossible_utility','partition_count','missing_rows'])
def test_mathematically_impossible_groups_are_incomplete(damage):
    records=complete_records()
    for row in records:
        if damage=='impossible_baseline':row.update(weighted_accuracy=.1,weighted_net=.9)
        if damage=='impossible_utility':row['weighted_utility']=100.
        if damage=='partition_count':row['rows']=1
        if damage=='missing_rows':row['rows']=None
    verdict=assess_source_promotion({s:records for s in (392005,392006,392007)})
    assert not verdict['passed'] and verdict['status']=='INCOMPLETE_SOURCE_EVIDENCE'
