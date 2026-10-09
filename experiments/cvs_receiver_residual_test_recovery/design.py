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
