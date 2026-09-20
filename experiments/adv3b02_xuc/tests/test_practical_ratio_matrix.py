"""Matched scientific options, weighted schedule activation, unchanged DAOT views."""
from pathlib import Path
import json,sys
from collections import Counter
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'tools'))
from scripts.train_rc4_practical import build_args
from baseline_origin_sat_view import parse_sat_view_schedule,BaselineOriginSatViewAugment
from prepare_practical_ratio import ACCEL,SCENES

def test_all_six_recipes_match_original_except_declared_changes():
    base=json.loads((ROOT/'configs/rc4_practical_residual_noeq_20260918.json').read_text())['options']
    matrix=json.loads((ROOT/'configs/practical_ratio_matrix_20260920.json').read_text())
    assert len(matrix['rows'])==6 and len({row['id'] for row in matrix['rows']})==6
    allowed=set(ACCEL)|{'--sat_train_scenario','--sat_train_scenarios','--sat_view_schedule','--a1_periodic_target_scenarios'}
    for row in matrix['rows']:
        path=ROOT/'configs'/row['config'];opts=json.loads(path.read_text())['options']
        changes={key for key in set(base)|set(opts) if base.get(key)!=opts.get(key)}
        assert changes<=allowed
        args=build_args(path,'data','output','contract','inputs','truth','test')
        assert args.seed==392005 and args.practical_receiver_seed==2027
        assert args.daot_student_scenario=='practical_high' and args.daot_hard_scenarios=='practical_mid,practical_low_urban'
        assert args.epochs==200 and args.practical_route=='residual' and not args.practical_equalization
        assert args.a1_periodic_target_scenarios.split(',')==SCENES
        stages=parse_sat_view_schedule(args.sat_view_schedule)
        assert [(s.start_epoch,s.view_prob) for s in stages]==[(1,.3),(41,.6),(91,.8)]
        for stage in stages:
            counts=Counter(stage.scenarios);assert len(stage.scenarios)==4
            assert counts[row['easy']]/4==row['easy_probability']
            assert counts[row['hard']]/4==row['hard_probability']
        aug=BaselineOriginSatViewAugment(scenarios=[row['easy'],row['hard']],schedule=args.sat_view_schedule,p=1,seed=args.seed,apply_fn=lambda *a,**k:None)
        generator=torch.Generator().manual_seed(2027)
        chosen=Counter(aug._select_scenario(stages[-1],generator,torch.device('cpu')) for _ in range(8000))
        assert abs(chosen[row['hard']]/8000-row['hard_probability'])<.025

def test_historical_checkpoints_are_fixed_and_never_training_initializers():
    matrix=json.loads((ROOT/'configs/practical_historical_eval_matrix_20260920.json').read_text())
    assert {row['id']:row['epoch'] for row in matrix['rows']}=={'full_noeq':150,'full_zf':130,'full_mmse':160,'residual_noeq':200}
    assert matrix['scenes']==SCENES
    for row in matrix['rows']:assert row['checkpoint'].endswith(f"epoch_{row['epoch']}_ssdg.pth")
