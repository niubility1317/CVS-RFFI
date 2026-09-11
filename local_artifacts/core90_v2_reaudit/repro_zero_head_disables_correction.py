"""Read-only configuration/calibration reproducer; no model or training."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from cvsrffi.game_tracking.config import parse_args
from cvsrffi.game_tracking.runtime_control import calibrate_game_v2


args=parse_args(['--output_dir','unused','--game_control','correction','--game_max_extra_head','0'])
lag=dict(data_valid=True,coverage_valid=True,lag=dict(status='RELIABLE_LOW_GAP',quality_pass=True,
    control_ready=True,gap_normalized=.01))
cap=dict(schema='game_capability_v2',valid=True,identity_valid=True,collapsed=False,identity=.9,
    margin=.3,next_identity=.9,next_margin=.2,next_worst_tx=.8)
controller=calibrate_game_v2([lag,lag,lag],[cap,cap,cap],args)
result=dict(accepted_configuration=True,control=args.game_control,head_budget=args.game_max_extra_head,
            correction_fraction=args.game_correction_fraction,controller_is_none=controller is None)
with Path(__file__).with_suffix('.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
print(json.dumps(result))
