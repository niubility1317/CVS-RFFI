import json
from pathlib import Path
import sys
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'),str(ROOT/'code/scripts')]
from SSDG import train_ssdg as train
from cvsrffi.a1_budget_schedule import reference_epoch, reference_total, u_satellite_scenario
from cvsrffi.a1_periodic_target import due
from cvsrffi.schedule import build_stage_state
from run_a1_extended_budgets import extended_matrix
import run_a1_fast_v2 as runner


@pytest.mark.parametrize('index',[0,1,2])
def test_extension_preserves_stages_and_source_contract(index):
    matrix = extended_matrix(); row = matrix['rows'][index]
    args = train.build_arg_parser().parse_args(runner.v2_command(matrix,
        project_root=Path('/p'),run_root=Path('/r'),row=row)[3:])
    n=args.epochs; factor=n//200
    assert args.from_scratch and args.a1_scratch_only and not args.baseline_ckpt and not args.teacher_ckpt
    assert args.a1_source_screen_only and args.a1_extended_budget_mode and not args.a1_r3_budget_mode
    assert args.label_epochs == 130*factor and args.pseudo_epochs == 70*factor
    assert args.muse_final_epoch == n and reference_total(args) == 200
    assert not args.a1_budget_snapshot_epochs
    assert args.stage1_epochs+1 == args.muse_s2a_start == 16*factor+1
    assert args.stage2_epochs+1 == args.muse_s3a_start == 68*factor+1
    assert build_stage_state(args.muse_s3a_start,args)['phase']=='S3_refine_aux'
    assert max(map(int,args.daot_diagnostic_epochs.split(','))) == n
    train._validate_a1_scratch_only(args); train._validate_daot_config(args); train._muse_config_from_args(args)
    for e in range(1,n+1):
        assert 1 <= reference_epoch(args,e) <= 200
        assert u_satellite_scenario(args,e,1) in ('leo_clear_weak','leo_low_elev_weak','leo_rain_weak')
    for boundary in (40,90,160,180):
        assert reference_epoch(args,boundary*factor)==boundary
        assert reference_epoch(args,boundary*factor+1)==boundary+1
    assert reference_epoch(args,n)==200
    assert [e for e in range(1,n+1) if due(e,n//2,n,20)]==row['evaluation_epochs']
    assert args.a1_ecrs_cross_rx_scope == ('clean' if index==2 else 'clean_leo')
    assert bool(args.use_a1_r3)==(index==2)


def test_interval_boundaries_and_invalid_interval():
    assert due(400,200,400,20) and not due(410,200,400,20)
    assert due(399,200,399,20) and not due(210,200,400,20)
    with pytest.raises(ValueError): due(200,200,400,0)


def test_finalization_requires_twenty_epoch_scores_only(tmp_path,monkeypatch):
    matrix=extended_matrix(); row=matrix['rows'][0]
    monkeypatch.setattr(runner,'verify_checkpoint',lambda path,expected_epoch: {'args':{'from_scratch':True}})
    for epoch in row['evaluation_epochs']:
        folder=tmp_path/row['id']/'target_epochs'/f'E{epoch:03d}';folder.mkdir(parents=True)
        (folder/'score.json').write_text(json.dumps({'record_count':672000}),encoding='utf-8')
        (folder/'evaluation_scope.json').write_text('{}',encoding='utf-8')
    assert runner.complete_row(matrix,row,project=tmp_path,root=tmp_path,logs=tmp_path)=='EXPLORATORY_SCORED_PENDING_ANALYSIS'
    (tmp_path/row['id']/'target_epochs/E400/score.json').unlink()
    with pytest.raises(FileNotFoundError): runner.complete_row(matrix,row,project=tmp_path,root=tmp_path,logs=tmp_path)
