import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.pamf_backfill_restore import design as d
from experiments.pamf_backfill_restore.runtime import validate_training,validate_final,audit_contract,Adapter,reference
from experiments.pamf_backfill_restore.runner import configuration,scoring_reference
from experiments.pamf_restoration.data import write_json


def training():
    return dict(status='SOURCE_TRAINED',candidate='x',epochs=20,updates=4940,L_exposures=126000,
        U_exposures=1134000,TX_labels_read=0,F_outputs_read=0,F_weights_read=0,target_reads=0)


@pytest.mark.parametrize('key,value',[('TX_labels_read',1),('F_outputs_read',1),('F_weights_read',1),
    ('target_reads',1),('updates',4939),('epochs',19),('L_exposures',0),('U_exposures',0),('status','TRAINING'),('candidate','wrong')])
def test_rejects_wrong_training_or_permissions(key,value):
    value0=training();value0[key]=value
    with pytest.raises(ValueError):validate_training(value0,'x')


def test_original_kind_metadata_is_accepted_without_rewriting_source_status():
    value=training();value['kind']=value.pop('candidate');validate_training(value,'x')
    assert 'candidate' not in value


def test_original_single_row_R07_has_no_candidate_field():
    value=training();value.pop('candidate');validate_training(value,'learned_operator')
    with pytest.raises(ValueError):validate_training(value,'x')


@pytest.mark.parametrize('change',[dict(epoch=19),dict(updates=4939),dict(candidate='wrong')])
def test_final_checkpoint_only(change):
    value=dict(epoch=20,updates=4940,candidate='x');value.update(change)
    with pytest.raises(ValueError):validate_final(value,dict(candidate='x'))


@pytest.mark.parametrize('key,value',[('authorization','SOURCE_QUALIFIED'),('rows',[]),('new_training_updates',1),
    ('source_decisions_modified',True),('target_feedback',True),('commit','other')])
def test_user_authorized_fixed_matrix_is_required(tmp_path,key,value):
    config=configuration('a');config[key]=value;write_json(tmp_path/'resolved_config.json',config)
    with pytest.raises(ValueError):audit_contract(SimpleNamespace(output=tmp_path,expected_commit='a'))


def test_matrix_is_complete_frozen_and_reuses_comparable_results():
    assert len(d.ROWS)==42 and len(d.NEW_ROWS)==36 and len(d.BY_ID)==42
    assert {'r01_diffusion','r01_flow','r24_attention_prior','r24_tcn_prior'}<=set(d.NEW_ROWS)
    assert 'r17_unroll_unshared' not in d.NEW_ROWS
    assert all(s['checkpoint'].endswith('/final.pt') and s['commit'] for s in d.ROWS)


class Capture(torch.nn.Module):
    def forward(self,x,noise=None):return x if noise is None else noise


def test_legacy_sampler_noise_is_independent_of_batch_order():
    spec=d.BY_ID['r01_diffusion'];adapter=Adapter(Capture(),spec,'cpu');x=torch.zeros(2,2,256);ids=np.asarray(['a','b'])
    out=adapter(x,ids);reverse=adapter(x.flip(0),ids[::-1])
    assert torch.equal(out,reverse.flip(0)) and not torch.equal(out[0],out[1])


def test_primary_flow_seed_is_reproducible_without_refitting():
    spec=d.BY_ID['r20_noise_start'];x=torch.zeros(2,2,256);ids=np.asarray(['a','b'])
    a=Adapter(Capture(),spec,'cpu');b=Adapter(Capture(),spec,'cpu')
    assert torch.equal(a(x,ids),b(x,ids))
    assert torch.equal(a(x,ids),b(x,ids))


def test_missing_authorization_blocks_reference_before_any_iq_or_truth_open(tmp_path):
    write_json(tmp_path/'resolved_config.json',dict(authorization='absent'))
    args=SimpleNamespace(output=tmp_path,expected_commit='a',row=d.NEW_ROWS[0],formal_reference=tmp_path/'missing',target_truth=tmp_path/'truth')
    with pytest.raises(ValueError,match='audit matrix'):reference(args,'target')
    assert not args.formal_reference.exists() and not args.target_truth.exists()


def test_scoring_reference_restores_audit_root_without_changing_row(monkeypatch,tmp_path):
    import experiments.pamf_backfill_restore.runner as runner
    row=d.NEW_ROWS[0];adapted=SimpleNamespace(output=tmp_path/row,row=row)
    monkeypatch.setattr(runner,'reference',lambda args,role:(args.output,args.row,role))
    assert scoring_reference(adapted,'target')==(tmp_path,row,'target')


def test_cpu_truth_last_check_precedes_reference_loader(monkeypatch):
    from experiments.pamf_uncertainty_restore.evaluation import score
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    def no_open(*args):raise AssertionError('Truth/reference must remain closed')
    with pytest.raises(ValueError,match='CPU'):score(SimpleNamespace(),'target',reference_loader=no_open)


def test_new_audit_kind_scores_and_closes_without_source_promotion(tmp_path,monkeypatch):
    import experiments.pamf_backfill_restore.runner as runner
    from experiments.pamf_restoration.runner import read_json
    write_json(tmp_path/'resolved_config.json',configuration('a'));name=d.NEW_ROWS[0]
    args=SimpleNamespace(output=tmp_path,row=name,expected_commit='a')
    folder=tmp_path/name/'target_confirmation'/d.KIND;folder.mkdir(parents=True)
    summary=dict(status='ANALYZED',role='target',kind=d.KIND,count=168000,F_run=runner.REFERENCE_RUN,F_seed=2026092701,
        original_default_head=True,all_predictions_before_truth=True,records=[dict(dimension='overall',readout='restored',view=view,
        accuracy=84.0,baseline_accuracy=84.67619047619047) for view in runner.VIEWS])
    def independent_scored(adapted,role,*,reference_loader):
        assert role=='target' and adapted.kind==d.KIND
        write_json(folder/'summary.json',summary)
    monkeypatch.setattr(runner,'original_score',independent_scored);runner.score(args)
    assert read_json(tmp_path/name/'complete.json')['status']=='ANALYZED'
    assert read_json(tmp_path/name/'goal_assessment.json')['performance_targets_met']
    assert not (tmp_path/'source_decision.json').exists() and summary['kind']==d.KIND
