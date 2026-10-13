from pathlib import Path

from experiments.cvs_repeated_rx_margin import design as parent

ROOT = Path(__file__).resolve().parents[2]
PROJECT = parent.PROJECT
SOURCE = parent.SOURCE
RUN = '20261013-source-pseudotarget-health-manysig-m3-r01'
RELEASE = 'cvs_source_pseudotarget_health_20261013_r01'
OWNER = 'codex/root/source-pseudotarget-health-20261013'
BASE = Path(PROJECT) / 'runs' / RUN
SEEDS = parent.SEEDS
RX = parent.RX
DAYS = parent.DAYS
CLASSES = 6
THRESHOLD = .95
BATCH_SIZE = 256
FULL_FP32_POLICY = parent.FULL_FP32_POLICY
read = parent.read
write = parent.write
FEATURES = ('time_down', 't2.pool', 'joint160')
ROLE_COUNTS = {'L_s': 6300, 'V': 27000, 'U_s': 56700}
PROBE = dict(train_folds=[0, 1], diagnostic_fold=2, steps=100,
             learning_rate=.01, weight_decay=.001, normalization_eps=1e-4,
             seed_offset=31337, classifier='ordinary Linear RX5',
             fit_role='L_s only', selection=False, identity_gradients=False)


def rows():
    return [dict(row_id=f'MATCHED_BASE-s{seed}', model_seed=seed) for seed in SEEDS]


def source_config(row):
    if row not in rows():
        raise ValueError('Unregistered source health row')
    return parent.config(next(r for r in parent.rows()
                              if r['arm'] == 'MATCHED_BASE'
                              and r['model_seed'] == row['model_seed']))


def config(row):
    source = source_config(row)
    return dict(run_id=RUN, **row, launch_owner=OWNER,
                output=(BASE / row['row_id']).as_posix(),
                source_run_id=parent.RUN, source_row_id=source['row_id'],
                source_checkpoint=(Path(source['output']) / 'FINAL.pth').as_posix(),
                source_config=source, all_frozen_seeds=list(SEEDS),
                roles=dict(ROLE_COUNTS), input='source clean IQ only',
                identity_updates=0, target_access=False, U_truth_read=False,
                peer='fixed T=1 mean probability of the other two seeds',
                confidence_threshold=THRESHOLD, batch_size=BATCH_SIZE,
                probe=dict(PROBE), source_V='health diagnostic only',
                source_performance_gate=False, target_metrics=False,
                purpose='source diagnostic; does not update or promote identity models')


def validate(c):
    row = next((r for r in rows() if r['row_id'] == c.get('row_id')), None)
    if row is None or c != config(row):
        raise ValueError('Fixed source health recipe differs')
    return c
