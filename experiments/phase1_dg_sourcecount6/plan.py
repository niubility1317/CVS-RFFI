"""Result-independent source-count matrix, excluding PAMF and ResNet controls."""
from copy import deepcopy
from experiments.phase1_external_dg.plan import SEEDS, VIEWS, native_row
from experiments.phase1_external_dg_aug.plan import common

RUN = '20261021-phase1-dg-sourcecount1to6-m432-r01'
PARENT = '20261019-phase1-external-dg-concat-pl-m219-r03'
SOURCE_ORDER = (1, 3, 4, 6, 8, 11)
TARGET = (0, 2, 5, 7, 9, 10)
FAMILIES = ('riei', 'drift', 'ibrff', 'sdg')


def counts(m, budget):
    if budget == 'natural':
        return dict(L_s=1260*m, U_s=11340*m, V=5400*m)
    return dict(L_s=1080, U_s=12600*m-1080, V=5400)


def make_plan():
    protocols, rows = {}, []
    for budget in ('natural', 'fixed'):
        for m in range(1, 7):
            regimes = ('supervised', 'pl') if budget == 'natural' else ('supervised',)
            for regime in regimes:
                pid = 'm%d-%s-%s' % (m, budget, regime)
                protocols[pid] = dict(protocol_id=pid, dataset='ManySig',
                    source_rxs=sorted(SOURCE_ORDER[:m]), source_days=[1, 2, 3],
                    target_rxs=list(TARGET), target_days=[0, 1, 2, 3],
                    label_per_thousand=70, fixed_labels=None if budget=='natural' else 1080,
                    fixed_validation=None if budget=='natural' else 5400,
                    split_seed=392005, equalized=1, out_len=256, normalize=True,
                    U_s_use=regime=='pl', diagnostic_day=None, expected_counts=counts(m, budget))
                group = pid
                for family in FAMILIES:
                    for seed in SEEDS:
                        for variant in ('ce', 'full'):
                            row = common(native_row(group, pid, family, variant, seed), pl=regime=='pl')
                            row.update(source_count=m, label_budget=budget,
                                source_subset_id='historical-prefix-plus-rx11',
                                reuse_parent_row_id=None)
                            if regime == 'pl':
                                row['supervised_control_row_id'] = row['row_id'].replace('-pl-', '-supervised-')
                            if budget=='natural' and m==5:
                                parent = 'F0-%s-%s%s-s%d' % (family, variant,
                                    '-pl' if regime=='pl' else '', seed)
                                row['reuse_parent_row_id'] = parent
                            rows.append(row)
    # First expose every count using seed01 CE/full supervision, then PL and
    # remaining seeds. This order is fixed before any new target prediction.
    rows.sort(key=lambda r: (r['label_budget']!='natural', r['model_seed'],
        r['U_s_use'], r['source_count'], FAMILIES.index(r['family']), r['variant']))
    return validate_plan(dict(run_id=RUN, parent_run_id=PARENT, allowed_gpus=list(range(8)),
        exclude_pamf=True, exclude_resnet=True, source_order=list(SOURCE_ORDER),
        fixed_target_rxs=list(TARGET), views=list(VIEWS), protocols=protocols, rows=rows,
        expected_rows=432, new_training_rows=384, reused_rows=48,
        natural_rows=288, fixed_budget_rows=144, supervised_rows=288, pl_rows=144,
        no_target_feedback=True, source_subset_selection='fixed historical order; append RX11; no score ranking'))


def validate_plan(plan):
    rows = plan['rows']; ids = {r['row_id'] for r in rows}
    if len(rows)!=432 or len(ids)!=432 or plan['allowed_gpus']!=list(range(8)):
        raise ValueError('Incorrect matrix size or authorized GPU set')
    if not plan['exclude_pamf'] or not plan['exclude_resnet']:
        raise ValueError('PAMF and standalone ResNet are excluded')
    lookup={r['row_id']:r for r in rows}
    expected = {'riei':(100,64,.0001,1), 'drift':(100,256,.0001,1),
                'ibrff':(200,128,.001,2), 'sdg':(200,32,.001,1)}
    for r in rows:
        p=plan['protocols'][r['protocol_id']];m=r['source_count']
        if r['family'] not in FAMILIES or r['variant'] not in ('ce','full') or r['model_seed'] not in SEEDS:
            raise ValueError('Unregistered method or seed')
        if (r['epochs'],r['batch_size'],r['lr'],r['stages'])!=expected[r['family']]:
            raise ValueError('Native budget changed')
        required=dict(satellite_train_augmentation=True,lambda_sat_cls=.68,lambda_sat_cons=0.,
            concat_sat_ce_only=True,sat_ce_start=1,teacher=False,
            default_head_joint_training=True,separate_head_training=False,initialization='scratch')
        if any(r.get(k)!=v for k,v in required.items()):raise ValueError('Matched training recipe changed')
        if p['source_rxs']!=sorted(SOURCE_ORDER[:m]) or p['target_rxs']!=list(TARGET):
            raise ValueError('Source subset or fixed target changed')
        if set(p['source_rxs'])&set(TARGET) or p['expected_counts']!=counts(m,r['label_budget']):
            raise ValueError('Invalid physical roles/budget')
        if r['U_s_use']:
            control=lookup.get(r.get('supervised_control_row_id'))
            if r['label_budget']!='natural' or control is None:
                raise ValueError('PL lacks a matched supervised control')
            ignore={'row_id','protocol_id','group','training_regime','U_s_use','pl_threshold',
                'pl_weight','pl_u_to_l_batch_ratio','supervised_control_row_id','reuse_parent_row_id'}
            if any(control.get(k)!=v for k,v in r.items() if k not in ignore):
                raise ValueError('PL changes an additional factor')
            if (r['pl_threshold'],r['pl_weight'],r['pl_u_to_l_batch_ratio'])!=(.95,1.,1):
                raise ValueError('Ordinary PL recipe changed')
        if bool(r['reuse_parent_row_id'])!=(m==5 and r['label_budget']=='natural'):
            raise ValueError('Only identical five-source rows may reuse frozen predictions')
    return plan
