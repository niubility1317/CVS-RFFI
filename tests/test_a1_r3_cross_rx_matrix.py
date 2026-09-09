import math
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code/scripts'))
from run_a1_r3_cross_rx import r3_cross_rx_matrix
from run_a1_scratch_no_checkpoint import scratch_matrix, scratch_command
import run_a1_fast_v2 as runner
from SSDG import train_ssdg as train


def test_combo_is_reference_plus_clean_cross_rx_only():
    old=scratch_matrix(); reference=next(r for r in old['rows'] if r['id']=='R3_REFERENCE')
    new=r3_cross_rx_matrix(); row=new['rows'][0]
    def parse(command): return train.build_arg_parser().parse_args(command[3:])
    a=parse(scratch_command(old,project_root=Path('/p'),run_root=Path('/new'),row=reference))
    b=parse(runner.v2_command(new,project_root=Path('/p'),run_root=Path('/new'),row=row))
    def same(x,y): return x==y or (isinstance(x,float) and isinstance(y,float) and math.isnan(x) and math.isnan(y))
    diff={k for k in vars(a) if not same(getattr(a,k),getattr(b,k))}
    assert diff=={'a1_ecrs_cross_rx_weight','candidate_id','output_dir','phase2_export_path'}
    train._validate_a1_scratch_only(b); train._validate_a1_ecrs_config(b); train._validate_daot_config(b)
    assert b.use_a1_r3 and b.a1_r3_aux_scale==1 and b.a1_ecrs_cross_rx_weight==.05
    assert b.a1_ecrs_cross_rx_margin==.2 and b.a1_ecrs_cross_rx_scope=='clean'
    assert not b.a1_runtime_fast and not b.a1_ema_startup_average and not b.use_tx_rx_balanced_sampler
    assert b.epochs==200 and b.seed==392005 and b.batch_size==128 and b.muse_unlabeled_batch_size==256
    assert b.test_eval_start_epoch>b.epochs and b.test_eval_interval==0 and b.test_eval_final_window==0
    assert new['final_evaluation']=='source_only' and new['max_gpu_processes']==2


def test_source_only_completion_never_invokes_target_scoring(tmp_path,monkeypatch):
    matrix=r3_cross_rx_matrix(); row=matrix['rows'][0]
    monkeypatch.setattr(runner,'verify_checkpoint',lambda p:{'args':{'from_scratch':True}})
    def forbidden(*a,**kw): raise AssertionError('Target access')
    monkeypatch.setattr(runner,'evaluate',forbidden)
    assert runner.required_inputs(matrix,tmp_path)==[tmp_path/'Dataset_WigSig/ManySig.pkl']
    assert runner.complete_row(matrix,row,project=tmp_path,root=tmp_path,logs=tmp_path)=='SOURCE_TRAINED_PENDING_ANALYSIS'


def test_default_truth_last_completion_preserved(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'verify_checkpoint',lambda p:{'args':{'from_scratch':True}})
    called=[]; monkeypatch.setattr(runner,'evaluate',lambda *a,**kw:called.append(True))
    assert len(runner.required_inputs({},tmp_path))==3
    assert runner.complete_row({}, {'id':'old'},project=tmp_path,root=tmp_path,logs=tmp_path)=='SCORED_PENDING_ANALYSIS'
    assert called==[True]


@pytest.mark.parametrize('mode',['wrong','',None])
def test_unknown_final_evaluation_rejected(mode):
    with pytest.raises(ValueError):runner.evaluation_mode({'final_evaluation':mode})
