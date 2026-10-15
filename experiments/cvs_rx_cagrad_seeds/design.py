from copy import deepcopy
from pathlib import Path
from experiments.cvs_rx_cagrad50 import design as first
from experiments.cvs_backbone_fishr_seeds import design as models

ROOT = Path(__file__).resolve().parents[2]
for name in ('PROJECT', 'SOURCE', 'CAPSULE', 'TRUTH', 'CLASSES', 'FULL_FP32_POLICY',
             'read', 'write', 'ARMS', 'BASELINE_ARM', 'PRIMARY', 'FEATURE_CONTROL', 'PAIRS',
             'READOUTS', 'VIEWS', 'QUERY_COUNT', 'EPOCHS', 'LABEL_EPOCHS', 'PSEUDO_EPOCHS',
             'STEPS', 'UPDATES', 'BASELINE_EVAL_RUN', 'STYLE_LAYERS', 'FEATURE_PARAMETER_COUNT',
             'VERIFY_INITIAL_EVAL_EQUAL', 'PARAMETERS', 'DIAGNOSTIC_EPOCHS', 'RX', 'INITIAL_CONTROL'):
    globals()[name] = getattr(first, name)
RUN = '20261015-phase1-rx-cagrad-oldgate-cos-e50-clean-manysig-s0203-m6-r01'
RELEASE = 'cvs_rx_cagrad_seeds_20261015_r01'
OWNER = 'codex/root/rx-cagrad-seeds-20261015'
BASE = Path(PROJECT) / 'runs' / RUN
SEEDS = (2026092702, 2026092703)
baseline_config = first.baseline_config


def rows():
    return [dict(row_id=f'{a}-s{s}', arm=a, model_seed=s) for s in SEEDS for a in ARMS]


def training_rows():
    return [r for r in rows() if r['arm'] != BASELINE_ARM]


def config(row):
    if row not in rows():
        raise ValueError('Unregistered remaining receiver-risk row')
    skeleton = dict(row, arm=BASELINE_ARM if row['arm'] == BASELINE_ARM else 'SIG_TIME',
                    row_id=f"{BASELINE_ARM if row['arm'] == BASELINE_ARM else 'SIG_TIME'}-s{row['model_seed']}")
    c = models.config(skeleton); c.pop('backbone_fishr')
    template = first.config(next(r for r in first.rows() if r['arm'] == row['arm']))
    for key in ('feature_combination', 'receiver_gradient', 'hypothesis_basis', 'source_falsification', 'goal'):
        c[key] = deepcopy(template[key])
    c.update(run_id=RUN, **row, output_root=(BASE / row['row_id'] / 'source').as_posix(),
             launch_owner=OWNER, source_qualification_run=first.RUN, source_decision_commit='PENDING_SOURCE_ONLY_DECISION', first_seed_retrained=False,
             source_expansion_rule='Only original first-source PASS permits both unchanged arms at fixed02/03',
             combined_model_seeds=[2026092701, *SEEDS])
    return c


def validate(c):
    row = next((r for r in rows() if r['row_id'] == c.get('row_id')), None)
    if row is None or c != config(row):
        raise ValueError('Remaining fixed receiver-risk recipe differs')


def make_args(c, device='cuda:0'):
    validate(c)
    if c['baseline_reuse']:
        raise ValueError('Historical baseline cannot be retrained')
    old = models.config(next(r for r in models.training_rows()
                             if r['arm'] == 'SIG_TIME' and r['model_seed'] == c['model_seed']))
    a = models.make_args(old, device)
    a.run_id, a.candidate_id, a.output_dir = RUN, c['row_id'], c['output_root']
    return a


def validate_qualification(q):
    if (q.get('status') != 'PASS' or not q.get('source_only')
            or q.get('target_metrics_read_for_decision') is not False
            or q.get('fixed_rules') != first.config(first.training_rows()[0])['source_falsification']
            or q.get('predefined_remaining_seeds') != list(SEEDS)
            or q.get('expand_both_arms_unchanged') is not True):
        raise ValueError('Original first-seed source qualification missing, failed or changed')
    return q


def require_source():
    return validate_qualification(read(first.BASE / 'source_qualification.json'))

# Resolve the same portable registration handle in this release context.
def read(path):return first.parent.read(path,root=ROOT)
