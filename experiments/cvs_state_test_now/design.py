"""Fixed source-completed snapshot; original training configs remain immutable."""
from pathlib import Path
from experiments.cvs_multi_state_action import design as parent
from experiments.cvs_multi_state_action.design import PROJECT,SOURCE,CAPSULE,TRUTH,SEEDS,CLASSES,FULL_FP32_POLICY,require_budget,read,write
ROOT=Path(__file__).resolve().parents[2]
RUN='20261010-phase1-state-action-completed-test-manysig-m24-r01'
RELEASE='cvs_state_test_now_20261010_r01'
BASE=Path(PROJECT)/'runs'/RUN
OWNER='codex/root/state-completed-test-20261010'
ARMS=('native','real_views','L','LT','L_EG','LT_EG')
def rows():
 return read(Path(__file__).with_name('snapshot.json'))['rows']
def config(row):
 parent.validate(row)
 return row
