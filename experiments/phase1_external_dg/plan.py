"""Deterministic, result-independent expansion of the approved comparison design."""
from copy import deepcopy

RUN = '20261019-phase1-external-dg-suite-m168-r01'
PARENT = '20261019-phase1-pamf-dg-nativebudget-m24-r01'
SEEDS = (2026092701, 2026092702, 2026092703)
VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
         'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')
RX_F0 = (1, 3, 4, 6, 8)
TARGET_F0 = (0, 2, 5, 7, 9, 10, 11)


def protocol(name, source, *, dataset='ManySig', days=(1, 2, 3),
             label_per_thousand=70, fixed_labels=None, diagnostic_day=None):
    return dict(protocol_id=name, dataset=dataset, source_rxs=sorted(source),
        source_days=list(days), target_rxs=list(TARGET_F0) if name.startswith('count-') else
        [i for i in range(12 if dataset == 'ManySig' else 32) if i not in source],
        target_days=[0, 1, 2, 3], label_per_thousand=label_per_thousand,
        fixed_labels=fixed_labels, diagnostic_day=diagnostic_day, split_seed=392005,
        equalized=1, out_len=256, normalize=True, U_s_use=False)


def native_row(group, proto, family, variant, seed):
    epochs, batch, lr = {'riei': (100, 64, .0001), 'drift': (100, 256, .0001),
                         'ibrff': (200, 128, .001), 'sdg': (200, 32, .001)}[family]
    return dict(row_id='%s-%s-%s-s%d' % (group, family, variant, seed), group=group,
        protocol_id=proto, family=family, variant=variant, model_seed=seed,
        split_seed=392005, epochs=epochs, batch_size=batch, lr=lr, weight_decay=0.,
        stages=2 if family == 'ibrff' else 1, budget_kind='native', initialization='scratch',
        U_s_use=False, satellite_train_augmentation=False,
        default_head_joint_training=True, separate_head_training=False)


def resnet_row(group, proto, route, seed, *, updates=5000, batch=150, epochs=None):
    return dict(row_id='%s-resnet18-%s-s%d' % (group, route, seed), group=group,
        protocol_id=proto, family='resnet18', variant='ce', route=route,
        model_seed=seed, split_seed=392005, epochs=epochs, updates=updates,
        batch_size=batch, lr=.0002, lr_min=1e-6, weight_decay=.0001,
        optimizer='AdamW', budget_kind='epochs' if epochs is not None else 'updates',
        balanced_sampling=epochs is None, optimizer_reset_step=1500 if group == 'generic' else None,
        mixstyle_probability=.5, mixstyle_alpha=.1, fishr_lambda=1000., fishr_ema=.95,
        fishr_warmup=1500, coral_lambda=1., initialization='scratch',
        U_s_use=False, satellite_train_augmentation=False,
        default_head_joint_training=True, separate_head_training=False)


def make_plan():
    protocols = {'F0': protocol('F0', RX_F0)}
    rows = []
    for name, source in [('F1', [0, 2, 5, 7, 9]), ('F2', [1, 3, 4, 10, 11]),
                         ('chrono', list(RX_F0)), ('ManyRx', list(range(0, 32, 2)))]:
        kwargs = dict(dataset='ManyRx') if name == 'ManyRx' else {}
        if name == 'chrono': kwargs.update(days=(0, 1, 2), diagnostic_day=3)
        protocols[name] = protocol(name, source, **kwargs)
        for family in ('ibrff', 'sdg', 'riei', 'drift'):
            for seed in SEEDS:
                for variant in ('full', 'ce'):
                    rows.append(native_row(name, name, family, variant, seed))
    for route in ('erm', 'mixstyle', 'mixstyle_id', 'fishr', 'coral'):
        for seed in SEEDS:
            rows.append(resnet_row('generic', 'F0', route, seed))
    chains = ((1, 3, 4, 6, 8), (4, 6, 8, 1, 3), (8, 1, 3, 4, 6))
    registered = set()
    for chain_index, chain in enumerate(chains):
        for m in (1, 3, 5):
            for budget in ('natural', 'fixed'):
                source = tuple(sorted(chain[:m]))
                # Identical data and training recipes are trained once only.
                key = (source, 'natural' if m == 1 else budget)
                if key in registered: continue
                registered.add(key)
                name = 'count-c%d-m%d-%s' % (chain_index + 1, m, key[1])
                if m == 5 and budget == 'natural':
                    pid = 'F0'
                else:
                    pid = name
                    protocols[pid] = protocol(pid, source,
                        fixed_labels=1260 if key[1] == 'fixed' else None)
                for seed in SEEDS:
                    rows.append(resnet_row(name, pid, 'erm', seed, updates=10000, batch=180))
    for label, rate in [('label01', 7), ('label05', 35)]:
        protocols[label] = protocol(label, RX_F0, label_per_thousand=rate)
        for seed in SEEDS:
            rows.append(resnet_row(label, label, 'erm', seed, updates=None, batch=128, epochs=200))
            for variant in ('full', 'ce'):
                rows.append(native_row(label, label, 'ibrff', variant, seed))
    for architecture in ('resnet18', 'resnet50'):
        for seed in SEEDS:
            rows.append(dict(row_id='legacy-%s-s%d' % (architecture, seed),
                group='legacy_practical', protocol_id='F0', family=architecture,
                variant='ce', model_seed=seed, split_seed=392005, budget_kind='frozen_eval',
                initialization='verified_historical_scratch', views=list(VIEWS[1:]),
                default_head_joint_training=True, separate_head_training=False))
    plan = dict(run_id=RUN, parent_run_id=PARENT, allowed_gpus=[0, 1],
        dependency='parent dispatcher terminal and process gone before GPU workers start',
        protocols=protocols, rows=rows, views=list(VIEWS), exclude_pamf=True,
        parent_rows=24, new_training_rows=162, frozen_eval_rows=6,
        no_target_feedback=True, optional_candidate_search=False)
    validate_plan(plan)
    return plan


def validate_plan(plan):
    if plan['allowed_gpus'] != [0, 1] or not plan['exclude_pamf']:
        raise ValueError('Two physical GPUs and non-PAMF scope required')
    rows = plan['rows']
    if len(rows) != 168 or len({r['row_id'] for r in rows}) != len(rows):
        raise ValueError('Incomplete or duplicate fixed comparison matrix')
    if sum(r['budget_kind'] != 'frozen_eval' for r in rows) != 162:
        raise ValueError('Training matrix size mismatch')
    if any(r['model_seed'] not in SEEDS or r['family'] not in
           ('riei', 'drift', 'ibrff', 'sdg', 'resnet18', 'resnet50') for r in rows):
        raise ValueError('Unauthorized architecture or seed')
    for p in plan['protocols'].values():
        if set(p['source_rxs']) & set(p['target_rxs']):
            raise ValueError('Source/target RX overlap')
        if p['dataset'] not in ('ManySig', 'ManyRx') or p['split_seed'] != 392005:
            raise ValueError('Data recipe mismatch')
        if p['label_per_thousand'] not in (7, 35, 70) or p['U_s_use']:
            raise ValueError('Hidden labels cannot be enabled')
    return plan
