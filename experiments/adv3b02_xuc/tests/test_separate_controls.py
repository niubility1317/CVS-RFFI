"""Regression for actual parsed native schedule serialization before training."""
import json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'code/scripts'))
from SSDG import train_ssdg as native
from train_daot_rc4_baseline import resolved_config

@pytest.mark.parametrize('seed',[392005,392006,392007])
def test_validated_native_schedule_has_lossless_json_encoding(seed):
    opts=json.loads((ROOT/'configs/separate_controls'/f'DAOT_RC4_NATIVE_s{seed}.json').read_text())['options']
    argv=[]
    for k,v in opts.items():
        argv.append(k)
        if v is not None:argv.append(str(v))
    argv+=['--output_dir','NOT_LAUNCHED']
    args=native.build_arg_parser().parse_args(argv)
    native._validate_a1_scratch_only(args);native._validate_daot_config(args)
    # train() materializes these immediately before source construction.
    args.sat_view_stages=tuple(native.parse_sat_view_schedule(args.sat_view_schedule,default_prob=args.sat_view_prob))
    with pytest.raises(TypeError,match='SatViewStage'):json.dumps(vars(args))
    saved=resolved_config(args)
    assert saved['seed']==seed and saved['from_scratch'] and saved['fasttrust_rc4']
    assert saved['sat_view_schedule']==args.sat_view_schedule
    stages=[v for v in saved.values() if isinstance(v,list) and v and isinstance(v[0],dict)]
    assert stages
    assert json.loads(json.dumps(saved))==saved
