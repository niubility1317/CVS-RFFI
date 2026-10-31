from pathlib import Path
from types import SimpleNamespace
import pytest
import numpy as np
from experiments.pamf_phase3_restoration import phase3_eval as p3
from experiments.pamf_phase3_restoration import historical_eval as hist
from experiments.pamf_phase3_restoration import pipeline


@pytest.mark.parametrize('extra',['target_U','truth','TX_labels','OE_train','new_thresholds'])
def test_frozen_Phase3_interface_rejects_role_or_calibration_expansion_before_file_reads(monkeypatch,extra):
    c=dict(run_id=p3.RUNS['F0'],arm='F0',code_commit='a'*40,reference_runtime='old',
           candidate_checkpoint='final.pt',output_root='new')
    c[extra]='forbidden'
    monkeypatch.setattr(p3,'_read',lambda *a:pytest.fail('No file should be opened'))
    with pytest.raises(ValueError):p3.validate_config(c)


@pytest.mark.parametrize('defect',['status','count','private_truth_reads','target_U_reads'])
def test_incomplete_phase3_predictions_keep_truth_closed(monkeypatch,defect):
    done=dict(status=p3.STATUS,run_id=p3.RUNS['IQ'],counts=dict(known=27000,novel=3150,clean=10238,star=10238),
        all_predictions_before_truth=True,all_networks_and_thresholds_frozen=True,
        private_truth_reads=0,target_U_reads=0,U_ID_reads=0,new_updates=0,density_fit_updates=0,threshold_fit_updates=0)
    if defect=='status':done['status']='PARTIAL'
    elif defect=='count':done['counts']['star']-=1
    else:done[defect]=1
    monkeypatch.setattr(p3,'_read',lambda *a:done)
    monkeypatch.setattr(p3.reference_score,'_future_truth',lambda *a:pytest.fail('Private truth must remain closed'))
    with pytest.raises(ValueError):p3._prediction('partial')


def test_missing_TX_in_one_scene_remains_NA_instead_of_claiming_pass():
    truth=dict(y=np.arange(5),scenario=np.asarray(['low']*5))
    assert p3.paired_clean_gaps(np.arange(5),np.arange(5),truth,['low'])=={'low':None}


def test_pipeline_runs_both_prediction_processes_before_either_scorer_and_training_gets_no_truth():
    a=SimpleNamespace(root=Path('root'),arm='F0',commit='a'*40,cache=Path('source_cache'),
                      inputs=Path('thin_source.json'),reference=Path('fixed_eval_runtime'))
    commands=pipeline.commands(a);stages=[stage for stage,_ in commands]
    assert stages==['train','phase3_predict','historical_predict','phase3_score','historical_score']
    train=commands[0][1]
    assert str(a.reference) not in train and '--cache' in train and '--inputs' in train
    assert len({cmd[cmd.index('-m')+1] for _,cmd in commands})==3


def test_incomplete_historical_prediction_marker_precedes_any_truth_read(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    args=SimpleNamespace(root=Path('root'),arm='IQ',commit='a'*40)
    monkeypatch.setattr(hist,'original',lambda p:np.asarray(['public']))
    opened=[]
    def read(path):
        opened.append(path.name)
        if path.name=='truth.json':pytest.fail('Incomplete seven views opened truth')
        return {'status':'PARTIAL'}
    monkeypatch.setattr(hist,'read',read)
    with pytest.raises(ValueError):hist.score(args)
    assert opened==['complete.json']
