"""Fixed test completion; no training, adaptation, target selection, or tuning."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RELEASE='cvs_receiver_residual_test_recovery_20261009_r01'
RUNS=[dict(run_id='20261009-phase1-receiver-residual-test-manysig-m16-r03',
    parent_run_id='20261008-phase1-receiver-residual-manysig-m16-r01',package='cvs_receiver_residual',
    source_release='cvs_receiver_residual_20261008_r01',source_commit='3712e8e4f004f21b8ffdc74ce0459ad8a2020f3d',
    disposition='DIAGNOSTIC_RNG_CONFOUNDED'),
    dict(run_id='20261009-phase1-receiver-residual-test-manysig-m16-r04',
    parent_run_id='20261008-phase1-receiver-residual-manysig-m16-r02',package='cvs_receiver_residual_v2',
    source_release='cvs_receiver_residual_v2_20261008_r02',source_commit='8a88b3a2e369644ea1d746c5e02006218ee00df6',
    disposition='FORMAL_R2_FIXED_RNG_BENCHMARK_EXPOSED')]
OWNER='codex/root/receiver-residual-test-recovery-20261009'
SCORE_RELEASE='cvs_receiver_residual_score_recovery_20261009_r01'
SCORE_RUNS=[dict(r,run_id='20261009-phase1-receiver-residual-score-manysig-m16-r0'+str(i+5),
    prediction_run_id=r['run_id']) for i,r in enumerate(RUNS)]
# Fixed dataset metadata, not inferred from prediction values or performance.
HELDOUT_RX={0:'1-1',2:'14-7',5:'2-1',7:'20-1',9:'7-14',10:'7-7',11:'8-8'}
TARGET_DAYS=('2021_03_01','2021_03_08','2021_03_15','2021_03_23')
