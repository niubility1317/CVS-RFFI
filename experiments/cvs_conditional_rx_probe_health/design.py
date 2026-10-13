from pathlib import Path

from experiments.cvs_conditional_rx_adversary import design as parent
from experiments.cvs_source_pseudotarget_health import design as probe_design

ROOT = Path(__file__).resolve().parents[2]
PROJECT = parent.PROJECT
SOURCE = parent.SOURCE
RUN = '20261013-source-conditional-rx-probe-health-manysig-m9-r01'
RELEASE = 'cvs_conditional_rx_probe_health_20261013_r01'
OWNER = 'codex/root/conditional-rx-probe-health-20261013'
BASE = Path(PROJECT) / 'runs' / RUN
SEEDS = parent.SEEDS
ARMS = parent.ARMS
RX = probe_design.RX
DAYS = probe_design.DAYS
CLASSES = 6
BATCH_SIZE = 256
FEATURES = probe_design.FEATURES
DIMENSIONS = (96, 192, 160)
ROLE_COUNTS = {'L_s': 6300, 'V': 27000}
PROBE = dict(probe_design.PROBE)
FULL_FP32_POLICY = parent.FULL_FP32_POLICY
read = parent.read
write = parent.write


def rows():
    return parent.rows()


def source_config(row):
    if row not in rows():
        raise ValueError('Unregistered frozen conditional RX probe row')
    return parent.config(row)


def config(row):
    source = source_config(row)
    return dict(run_id=RUN, **row, launch_owner=OWNER,
                output=(BASE / row['row_id']).as_posix(),
                source_run_id=parent.RUN, source_row_id=row['row_id'],
                source_checkpoint=(Path(source['output']) / 'FINAL.pth').as_posix(),
                source_config=source, all_frozen_seeds=list(SEEDS),
                all_frozen_arms=list(ARMS), roles=dict(ROLE_COUNTS),
                input='source L_s/V clean IQ only; U cache never opened',
                identity_updates=0, target_access=False, U_cache_opened=False,
                U_truth_read=False, batch_size=BATCH_SIZE,
                features=list(FEATURES), dimensions=list(DIMENSIONS),
                probe=dict(PROBE), source_V='known source RX health diagnostic only',
                source_performance_gate=False, target_metrics=False,
                probe_selection=False, auxiliary_adversary_state_loaded=False,
                raw_artifact='source/raw_source.pt: probabilities/descriptors/labels/RX/day/IDs/folds',
                purpose='all fixed FINAL30 identities remain frozen; fresh independent RX linear probes only')


def validate(c):
    row = next((r for r in rows() if r['row_id'] == c.get('row_id')), None)
    if row is None or c != config(row):
        raise ValueError('Fixed frozen conditional RX probe recipe differs')
    return c
