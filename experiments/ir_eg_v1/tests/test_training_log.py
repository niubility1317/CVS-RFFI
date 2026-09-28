import csv
import json
from copy import deepcopy

import pytest

from cvsrffi.xuc_fusion.training_log import TrainingLog, epoch_metrics


def records():
    return [dict(step=i, accepted_main_step=i+1, execution_epoch=1, accepted=True,
        loss=float(i+1), grad_norm=float(2+2*i), field_evaluations=2,
        learning_rates=dict(backbone=.001/(i+1)), weights=dict(adv=.35),
        daot_rc4=dict(daot_executed=bool(i), hard_count=2+i, weighted_identity=.1),
        solver_telemetry=dict(configured=True, eligible=bool(i), executed=bool(i),
            applied=bool(i), response_norm=None if not i else .02,
            cg_status=None if not i else 'truncated_usable'),
        scenario='leo_clear_weak', source_ids=['DO_NOT_COPY'], unlabeled_ids=['HIDDEN'],
        identity_supervision_exposure=dict(L=128,U_available=256,U_HP_selected=2+i)) for i in range(2)]


def summary(epoch=1):
    return dict(epoch=epoch,accepted=2,total_step=2*epoch,mean_loss=1.5,
        elapsed_seconds=10.*epoch,terms=dict(tx=.25,adv=1.2),
        source_validation=dict(accuracy=.75,ce=.3,total=4,
            group_metrics=dict(overall=dict(macro_f1=.7,confusion=[[1,2],[3,4]]))))


def test_human_ai_metrics_agree_without_mutating_records(tmp_path):
    source=records();original=deepcopy(source);text=[]
    logger=TrainingLog(tmp_path,dict(xuc_row=dict(id='IR'),epochs=2,seed=392005),emit=text.append)
    metrics=logger.epoch(summary(),source)
    assert metrics['solver_telemetry.executed.true_count']==1
    assert metrics['solver_telemetry.executed.observed_count']==2
    assert metrics['solver_telemetry.response_norm.mean']==.02
    assert metrics['solver_telemetry.response_norm.observed_count']==1
    assert metrics['grad_norm.mean']==3.
    assert metrics['learning_rates.backbone.last']==.0005
    assert metrics['train_accuracy'] is None and metrics['target_accuracy'] is None
    saved=json.loads((tmp_path/'training_metrics.jsonl').read_text())
    assert saved==metrics
    with (tmp_path/'training_metrics.csv').open(newline='') as f:
        row=next(csv.DictReader(f))
    assert float(row['source_val_accuracy'])==saved['source_val_accuracy']
    assert row['train_accuracy']==''
    human='\n'.join(text)
    assert (tmp_path/'training.log').read_text(encoding='utf-8')==human+'\n'
    assert '[CONFIG] seed=392005' in human
    assert '[VAL-SOURCE] acc=0.75 macro_f1=0.7' in human
    assert '[IR-COUNTS]' in human and 'executed=1' in human
    assert 'DO_NOT_COPY' not in json.dumps(saved) and 'confusion' not in json.dumps(saved)
    assert source==original


def test_late_ir_columns_preserve_prior_epoch_and_resume_clock(tmp_path):
    logger=TrainingLog(tmp_path,dict(xuc_row=dict(id='IR'),epochs=3),previous_elapsed=40.,emit=lambda _:None)
    first=summary();first['elapsed_seconds']=50.
    baseline=records()
    for row in baseline:row['solver_telemetry']=None
    assert logger.epoch(first,baseline)['epoch_seconds']==10.
    second=summary(2);second['elapsed_seconds']=65.
    assert logger.epoch(second,records())['epoch_seconds']==15.
    with (tmp_path/'training_metrics.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==2 and rows[0]['solver_telemetry.executed.true_count']==''
    assert rows[1]['solver_telemetry.executed.true_count']=='1'
    with pytest.raises(FileExistsError):
        TrainingLog(tmp_path,dict(xuc_row=dict(id='IR'),epochs=3))


def test_incomplete_or_rejected_epoch_is_not_presented_as_complete():
    with pytest.raises(ValueError,match='count'):
        epoch_metrics(summary(),records()[:1],row_id='IR',epochs=2,previous_elapsed=0.)
    rows=records();rows[1]['accepted']=False
    with pytest.raises(ValueError,match='rejected'):
        epoch_metrics(summary(),rows,row_id='IR',epochs=2,previous_elapsed=0.)
