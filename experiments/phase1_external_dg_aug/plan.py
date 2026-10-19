"""User-requested augmentation/PL revision; no result-dependent choices."""
from copy import deepcopy

from experiments.phase1_external_dg.plan import SEEDS, VIEWS, make_plan as old_plan, native_row, resnet_row

RUN = '20261019-phase1-external-dg-concat-pl-m219-r01'
COST_RUN = '20261019-phase1-external-dg-concat-pl-cost-m96-r01'
REPLACES = '20261019-phase1-external-dg-suite-m168-r01'
PARENT = '20261019-phase1-pamf-dg-nativebudget-m24-r01'


def common(row, pl=False):
    row.setdefault('stages', 1)
    row.update(satellite_train_augmentation=True, concat_sat_ce_only=True,
        lambda_sat_cls=.68, lambda_sat_cons=0., sat_ce_start=1,
        sat_scene_family='LEO_WEAK', augmentation_seed=2027,
        U_s_use=bool(pl), training_regime='pl' if pl else 'supervised',
        pl_threshold=.95 if pl else None, pl_weight=1. if pl else 0.,
        pl_u_to_l_batch_ratio=1 if pl else 0, teacher=False,
        initialization='scratch', default_head_joint_training=True, separate_head_training=False)
    return row


def make_plan():
    old = old_plan()
    rows = []
    for family in ('ibrff', 'sdg', 'riei', 'drift'):
        for seed in SEEDS:
            for variant in ('full', 'ce'):
                rows.append(common(native_row('F0', 'F0', family, variant, seed)))
    rows += [common(deepcopy(r)) for r in old['rows'] if r['budget_kind'] != 'frozen_eval']
    for family in ('resnet18', 'resnet50'):
        for seed in SEEDS:
            row = resnet_row('strongce', 'F0', 'erm', seed, updates=None, batch=128, epochs=200)
            row.update(family=family, row_id='strongce-%s-erm-s%d' % (family, seed))
            rows.append(common(row))
    supervised = list(rows)
    for row in supervised:
        if row['group'] == 'F0' or (row['group'] == 'strongce' and row['family'] == 'resnet18'):
            copied = common(deepcopy(row), pl=True)
            copied.update(group=row['group'] + '-pl', protocol_id='F0_PL',
                row_id=row['row_id'].replace('-s202609', '-pl-s202609'),
                supervised_control_row_id=row['row_id'])
            rows.append(copied)
    # Finish the main factorial contrasts first; each PL row retains augmentation.
    order = {'F0': 0, 'F0-pl': 0, 'strongce': 1, 'strongce-pl': 1}
    families = {'ibrff': 0, 'sdg': 1, 'riei': 2, 'drift': 3, 'resnet18': 4, 'resnet50': 5}
    rows.sort(key=lambda r: (order.get(r['group'], 2), families[r['family']],
                            r['model_seed'], r['variant'], r['training_regime'], r['row_id']))
    protocols = deepcopy(old['protocols'])
    protocols['F0_PL'] = deepcopy(protocols['F0'])
    protocols['F0_PL'].update(protocol_id='F0_PL', U_s_use=True)
    plan = dict(run_id=RUN, replaces_run_id=REPLACES, parent_run_id=PARENT,
        allowed_gpus=[0, 1], exclude_pamf=True, views=list(VIEWS), protocols=protocols,
        rows=rows, new_training_rows=219, supervised_rows=192, pl_rows=27,
        frozen_eval_rows=0, no_target_feedback=True, optional_candidate_search=False,
        cost_rows=96, cost_run_id=COST_RUN,
        dependency='original 24-row dispatcher terminal and actual process gone; withdrawn old expansion never resumed')
    return validate_plan(plan)


def validate_plan(plan):
    if plan['allowed_gpus'] != [0, 1] or not plan['exclude_pamf']:
        raise ValueError('Two physical GPUs and non-PAMF scope required')
    rows = plan['rows']
    if len(rows) != 219 or len({r['row_id'] for r in rows}) != 219:
        raise ValueError('Incomplete/duplicate augmented matrix')
    lookup = {r['row_id']: r for r in rows}
    if sum(r['U_s_use'] for r in rows) != 27:
        raise ValueError('Plain PL factorial size mismatch')
    for r in rows:
        if r['family'] not in ('riei', 'drift', 'ibrff', 'sdg', 'resnet18', 'resnet50') or r['model_seed'] not in SEEDS:
            raise ValueError('Unauthorized architecture/seed')
        if r.get('stages') != (2 if r['family'] == 'ibrff' else 1):
            raise ValueError('Native stage count differs')
        required = dict(satellite_train_augmentation=True, concat_sat_ce_only=True,
            lambda_sat_cls=.68, lambda_sat_cons=0., sat_ce_start=1,
            sat_scene_family='LEO_WEAK', augmentation_seed=2027, teacher=False,
            initialization='scratch', default_head_joint_training=True, separate_head_training=False)
        if any(r.get(k) != v for k, v in required.items()):
            raise ValueError('Concat/default-head training contract differs')
        if r['U_s_use']:
            if (r['pl_threshold'], r['pl_weight'], r['pl_u_to_l_batch_ratio'], r['protocol_id']) != (.95, 1., 1, 'F0_PL'):
                raise ValueError('Unregistered plain PL recipe')
            control = lookup.get(r.get('supervised_control_row_id'))
            ignore = {'row_id', 'group', 'protocol_id', 'U_s_use', 'training_regime',
                      'pl_threshold', 'pl_weight', 'pl_u_to_l_batch_ratio', 'supervised_control_row_id'}
            if control is None or any(control.get(k) != v for k, v in r.items() if k not in ignore):
                raise ValueError('PL control differs in an additional factor')
        elif r['pl_threshold'] is not None or r['pl_weight'] != 0 or r['pl_u_to_l_batch_ratio'] != 0:
            raise ValueError('Supervised row cannot use PL')
    for p in plan['protocols'].values():
        if set(p['source_rxs']) & set(p['target_rxs']):
            raise ValueError('Source/target RX overlap')
    return plan
