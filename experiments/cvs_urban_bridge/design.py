"""Zero-update source measurement preceding the authorized D/E teaching study."""
from pathlib import Path
from experiments.cvs_urban_branch50 import design as parent

ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE=parent.PROJECT,parent.SOURCE
RUN='20261015-phase1-sigtime-coupled-bridge-source-manysig-s2026092701-r01'
RELEASE='cvs_sigtime_coupled_bridge_source_20261015_r01'
BASE=Path(PROJECT)/'runs'/RUN
OWNER='codex/root/coupled-bridge-source-20261015'
SCENES=parent.SCENES
VIEWS=('full','bridge_shared','bridge_independent','no_added_noise')
MODEL_SEED=2026092701
DIAGNOSTIC_SEED=2026101504
RECEIVER_SEED=2026092701
NOISE_IMPROVEMENT_DB=6.
EXPECTED_FORWARDS=36000
read,write=parent.read,parent.write


def parent_config():return parent.config(next(v for v in parent.rows() if v['arm']=='A0'))


def configuration():
    return dict(schema='sigtime_coupled_bridge_source_v1',run_id=RUN,source_only=True,query_read=False,
        source_role='L_dev',source_count=720,parent=parent_config(),parent_selection='fixed matched A0 final50, no target ranking',
        views=list(VIEWS),scenes=list(SCENES),noise_improvement_db=NOISE_IMPROVEMENT_DB,
        diagnostic_seed=DIAGNOSTIC_SEED,receiver_seed=RECEIVER_SEED,
        receiver_state='full-view frozen',AGC_gain='full-view frozen',
        expected_forward_packets=EXPECTED_FORWARDS,optimizer_updates=0,head_fit_updates=0,
        training_epochs_added=0,V_forward_packets=0,U_sample_reads=0,
        scope='Coupled physics/default-head/actual SIG112 source measurements; not a new training performance result',
        subsequent_design='D/E must retain direct clean-to-urban joint anchor; both get intermediate CE+clean anchor and matched extra forwards; E alone adds grouped urban-to-HQ SIG. Not implemented by this source run.')
