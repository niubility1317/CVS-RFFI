from pathlib import Path
import math
from experiments.cvs_sigtime_rdrop_legacy import design as original
from experiments.cvs_rx_u_increment20 import design as fold
from .mechanism import ARMS

ROOT = Path(__file__).resolve().parents[2]
PROJECT, SOURCE, FULL_FP32_POLICY, read, write = (getattr(original, k) for k in ('PROJECT', 'SOURCE', 'FULL_FP32_POLICY', 'read', 'write'))
RUN = '20261016-phase1-SIGTIME-rdrop-margin-envelope-sourcefold-s2026092701-m4-r01'
RELEASE = 'cvs_SIGTIME_rdrop_margin_envelope_source20_20261016_r01'
OWNER = 'codex/root/SIGTIME-RDrop-margin-envelope-20261016'
MODEL_RELEASE = PROJECT + '/releases/cvs_originalSIGTIME_rdrop20_50_20261015_r01'
MODEL_COMMIT = 'b3ac442e2d33a5b315fb3251c30e0ad7083566bc'
DEPENDENCY_RELEASE = original.DEPENDENCY_RELEASE
DEPENDENCY_COMMIT = original.DEPENDENCY_COMMIT
FOLD_RELEASE = original.FOLD_RELEASE
BASE = Path(PROJECT) / 'runs' / RUN
SEED, FIT, HELD, UPDATES = 2026092701, (3, 4, 6, 8), 1, 4440


def native_config():
    return original.config(next(r for r in original.rows() if r['epochs'] == 20 and r['mode'] == 'RDROP' and r['model_seed'] == SEED))


def rows():
    return [dict(row_id=a + '-RX1-s' + str(SEED), arm=a, model_seed=SEED) for a in ARMS]


def config(row):
    if row not in rows():
        raise ValueError('Fixed four source RDrop rows required')
    return dict(native_config(), schema='SIGTIME_RDrop_margin_envelope_v1', run_id=RUN, **row,
        launch_owner=OWNER, output_root=(BASE / row['row_id'] / 'source').as_posix(),
        method='cvs_rdrop_margin_envelope', stage='source_RX_fold_full_training',
        test_views=[], readouts=['FINAL'], fold=fold.config()['fold'], updates=UPDATES, source_falsification=None,
        training_recipe='Unchanged original SIG_TIME RAW_HZ/normalaug/LS.01/defaultcos joint; scratch20 13L+7oldU,true RX1 exclusion,222steps',
        training_mechanism=dict(arm=row['arm'], alpha=1., label_smoothing=.01,
            FULL='Exact original symmetric KL with both probability arguments differentiated',
            MEAN='Detached arithmetic-mean probability teacher r=(p1+p2)/2; mean KL(r||p1),KL(r||p2)',
            CONFIDENCE='Detached envelope true-class mass; conditional nontrue-class shape from MEAN teacher; same mean forward KL',
            ENVELOPE='a_vc=z_vc-z_vy; detached r=softmax(min(a_1,a_2)); same mean forward KL',
            objective='Mean original smoothed CE of same post-augmentation L128 two views + alpha*arm consistency',
            scope='Only actual visible L labels; per-packet teacher. Native independent dropout at every site; original U,EMA,augmentation and default joint head unchanged',
            interpretation='Envelope true-class probability >= either view. This is a target property,not a guarantee of CE-aligned parameter gradients,AdamW displacement,calibration or generalization',
            default_head_joint=True, KL_default_W_joint=True, original_hidden_U_recipe_unchanged=True,
            augmentation_redraws=0, extra_parameters=0, extra_optimizer_steps=0, extra_L_identity_forwards=568320),
        source_selection=dict(primary='ENVELOPE', controls=list(ARMS[:-1]), held_RX=HELD,
            readout='Fixed final20 native joint head,1260 held source L', final_L_CE_max=.5,
            minimum_mixed_envelope_fraction=.1, errors_strictly_fewer=True, mean_CE_no_higher=True,
            on_pass='Separately preregister fixed RX3 exclusion confirmation with same arms/seed/E20 before any main target experiment',
            on_fail='Close fixed envelope recipe;no alpha,temperature,layer,seed,epoch rescue', target_access=False),
        baseline_reference='Fresh original FULL RDrop,MEAN detached ensemble,confidence-matched teacher. All scratch,not inherited source or target models',
        budget_authorization='Current only SIG_TIME+RDrop; independent consistency-target hypothesis. First predefined source seed01,four scratch20 arms',
        cost_claim='4scratch20/80model-epoch/17760updates;each568320extraL/396368hiddenU observations/1260heldL/144unusedL anchors;0target',
        goal=dict(clean_three_seed_mean=.84, source_only=True, target_access=False, global_novelty_claim=False, goal_completed=False))


def validate(c):
    row = next((r for r in rows() if r['row_id'] == c.get('row_id')), None)
    if row is None or c != config(row):
        raise ValueError('Fixed source envelope matrix differs')
    return c


def make_args(c, device='cuda:0'):
    validate(c)
    a = original.make_args(native_config(), device)
    a.run_id, a.candidate_id, a.output_dir = RUN, c['row_id'], c['output_root']
    a.source_fold_held_rx, a.source_fold_training_receivers = HELD, ','.join(map(str, FIT))
    if (a.epochs, a.label_epochs, a.pseudo_epochs, a.label_smoothing, a.use_aug, a.batch_size) != (20, 13, 7, .01, True, 128) or a.use_concat_sat_channel_aug or a.baseline_ckpt or a.teacher_ckpt:
        raise ValueError('Original native recipe and own scratch required')
    return a


source_validation = fold.source_validation


def qualify(results):
    if set(results) != set(ARMS):
        raise ValueError('All four complete source results required')
    for r in results.values():
        h = r['exports']['FINAL']
        if h['count'] != 1260 or not math.isfinite(h['mean_CE']):
            raise ValueError('Complete finite held source readout required')
    main = results['ENVELOPE']['exports']['FINAL']
    healthy = all(r['source_health'] for r in results.values())
    active = results['ENVELOPE']['mixed_envelope_fraction'] >= .1
    contrasts = {a: dict(error_difference=main['errors'] - results[a]['exports']['FINAL']['errors'],
                        CE_difference=main['mean_CE'] - results[a]['exports']['FINAL']['mean_CE']) for a in ARMS[:-1]}
    return dict(qualified=healthy and active and all(v['error_difference'] < 0 and v['CE_difference'] <= 0 for v in contrasts.values()),
                all_source_healthy=healthy, signal_active=active, contrasts=contrasts, target_access=False, goal_completed=False)


def request():
    return dict(run_id=RUN, model_release=MODEL_RELEASE, model_code_commit=MODEL_COMMIT,
                dependency_release=DEPENDENCY_RELEASE, fold_release=FOLD_RELEASE,
                rows=[config(r) for r in rows()], source_only=True)
