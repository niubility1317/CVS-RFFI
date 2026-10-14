from copy import deepcopy
from pathlib import Path
from experiments.cvs_backbone_fishr50 import design as parent

ROOT = Path(__file__).resolve().parents[2]
for name in ('PROJECT', 'SOURCE', 'CAPSULE', 'TRUTH', 'CLASSES', 'FULL_FP32_POLICY',
             'read', 'write', 'READOUTS', 'VIEWS', 'QUERY_COUNT', 'EPOCHS', 'LABEL_EPOCHS',
             'PSEUDO_EPOCHS', 'STEPS', 'UPDATES', 'BASELINE_EVAL_RUN', 'STYLE_LAYERS', 'PARAMETERS'):
    globals()[name] = getattr(parent, name)
RUN = '20261015-phase1-receiver-dro-oldgate-cos-e50-clean-manysig-s01-m4-r01'
RELEASE = 'cvs_receiver_dro50_20261015_r01'
OWNER = 'codex/root/receiver-dro50-20261015'
BASE = Path(PROJECT) / 'runs' / RUN
SEEDS = (2026092701,)
ARMS = ('OLD_GATE_COS50', 'SIG_TIME', 'RX_BALANCED', 'RX_DRO')
BASELINE_ARM, FEATURE_CONTROL, PRIMARY = ARMS[0], 'SIG_TIME', 'RX_DRO'
FEATURE_PARAMETER_COUNT, VERIFY_INITIAL_EVAL_EQUAL = 195041, True
PAIRS = [('RX_DRO', 'RX_BALANCED'), ('RX_DRO', 'SIG_TIME'), ('RX_BALANCED', 'SIG_TIME'),
         ('RX_DRO', BASELINE_ARM), ('SIG_TIME', BASELINE_ARM)]
RX = (1, 3, 4, 6, 8)
STEP_SIZE = .01
DIAGNOSTIC_EPOCHS = (1, 17, 33, 50)
baseline_config = parent.baseline_config


def rows():
    return [dict(row_id=f'{a}-s{s}', arm=a, model_seed=s) for s in SEEDS for a in ARMS]


def training_rows():
    return [r for r in rows() if r['arm'] != BASELINE_ARM]


def config(row):
    if row not in rows():
        raise ValueError('Unregistered receiver-risk row')
    skeleton = next(r for r in parent.rows() if r['arm'] ==
                    (BASELINE_ARM if row['arm'] == BASELINE_ARM else 'SIG_TIME'))
    c = deepcopy(parent.config(skeleton)); c.pop('backbone_fishr')
    c.update(run_id=RUN, **row, output_root=(BASE / row['row_id'] / 'source').as_posix(), launch_owner=OWNER)
    c['receiver_risk'] = dict(mode='ERM' if row['arm'] == 'SIG_TIME' else
        'BALANCED' if row['arm'] == 'RX_BALANCED' else 'DRO' if row['arm'] == 'RX_DRO' else 'historical',
        role='L_s', groups=list(RX), step_size=STEP_SIZE, initial_group_weights='uniform1/5',
        weight_update='logq += .01 * detached current within-RX mean CE; softmax after update',
        risk='sum q_RX * within-RX mean original default CE smoothing.01',
        absent_RX='zero risk and no logq increment; renormalize applied q over present RX',
        balanced_control='equal mean over present RX; no adversarial weight update',
        ERM_control='original scalar CE returned unchanged, same group calculations for observation',
        enable='all11100 native L steps from epoch1; no extra warmup, reset or clipping',
        original_U='same old batch_neighbor/ownEMA/.16UCE/.01entropy; no reweighting or label access',
        optimizer='original AdamW lr.0002 WD.0001 cosine; no new optimizer/head/forward',
        parameters=list(PARAMETERS), diagnostic_epochs=list(DIAGNOSTIC_EPOCHS),
        reference='https://github.com/kohpangwei/group_DRO/blob/master/loss.py',
        scope='GroupDRO-style receiver grouping adaptation; not a reproduction of original datasets or regularization study')
    c['hypothesis_basis'] = ('Prior source covariance penalties optimized their source objective without stable DG gain; '
        'test source identity-risk allocation instead of additional feature invariance. Receiver groups are legal '
        'L metadata; original default head remains jointly trained. Original GroupDRO warns that risk weighting '
        'alone can fail; preserve WD to isolate objective and do not claim expected target gain.')
    c['source_falsification'] = dict(final5_unweighted_L_CE_max=.5, feature_difference_min=1e-6,
        native_steps=11100, actual_U_packets=963900, required_gradient_epochs=[17, 33, 50],
        required_gradient_parameters=list(PARAMETERS),
        DRO_final5_training_worst_RX_CE_max_relative_to_BALANCED=1.,
        DRO_final5_mean_max_group_weight_min=.2001,
        source_only=True, expansion='If exact source budget/defaultfeature/body-gradient health and fixed risk '
        'thresholds pass, extend all three unchanged arms to02/03 before reading new target summaries; '
        'otherwise no expansion. Source V health only, no epoch or candidate selection.')
    c['goal'] = dict(primary=PRIMARY, clean_three_seed_mean=.84,
                     first_seed_is_not_goal_completion=True, feedback_to_training=False)
    return c


def validate(c):
    row = next((r for r in rows() if r['row_id'] == c.get('row_id')), None)
    if row is None or c != config(row):
        raise ValueError('Fixed receiver-risk configuration differs')


def make_args(c, device='cuda:0'):
    validate(c)
    if c['baseline_reuse']:
        raise ValueError('Historical baseline cannot be retrained')
    old = parent.config(next(r for r in parent.rows() if r['arm'] == 'SIG_TIME'))
    a = parent.make_args(old, device)
    a.run_id, a.candidate_id, a.output_dir = RUN, c['row_id'], c['output_root']
    return a
