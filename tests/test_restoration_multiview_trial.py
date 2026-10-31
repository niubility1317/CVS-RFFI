import json
from types import SimpleNamespace
import pytest
from experiments.pamf_phase3_restoration.multiview_trial import configuration,factory,save,load,score,RUN,STREAMS
from experiments.pamf_phase3_restoration import main_split_eval as main_eval


def test_matched_single30_budget_only_changes_architecture():
    a=configuration('TF_M20');b=configuration('MSAT_M20')
    assert {k for k in a if a[k]!=b[k]}=={'arm','architecture'}
    assert a['epochs']==30 and a['updates']==7410 and a['training_stages']==1
    assert a['source_draws']==list(range(1,21))+list(range(1,11))
    assert a['ancestor_checkpoints']==[] and a['target_training_reads']==a['TX_labels_read']==a['OE_reads']==0


def test_checkpoint_cannot_change_arm_commit_or_training_budget(tmp_path):
    p=tmp_path/'state.pt';g=factory('TF_M20');commit='a'*40;save(p,g,'TF_M20',commit,0)
    load(p,'TF_M20',commit,'cpu',epoch=0)
    for arm,c,epoch in [('MSAT_M20',commit,0),('TF_M20','b'*40,0),('TF_M20',commit,30),('TF_M20',commit,1)]:
        with pytest.raises(ValueError):load(p,arm,c,'cpu',epoch=epoch)


def test_missing_population_never_reaches_truth_side(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    args=SimpleNamespace(root=tmp_path,arm='TF_M20',commit='a'*40)
    row=tmp_path/'runs'/RUN/'rows/TF_M20';(row/'prediction/cross_rx').mkdir(parents=True)
    identity=dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701,
        source_run=main_eval.F_RUN,source_row=main_eval.F_ROW,source_code_commit=main_eval.F_COMMIT)
    marker=dict(F0_identity=identity,status='PREDICTIONS_COMPLETE',run_id=RUN,arm='TF_M20',task='cross_rx',count=126000,
        code_commit=args.commit,readouts=list(STREAMS),views=list(main_eval.VIEWS),truth_read=False,query_fit=False,model_state_updates=0)
    (row/'prediction/cross_rx/complete.json').write_text(json.dumps(marker))
    with pytest.raises(FileNotFoundError,match='cross_both'):score(args,row/'score')
    assert not (row/'score').exists()
