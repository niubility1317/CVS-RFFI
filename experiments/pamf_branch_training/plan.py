"""Single-seed falsification of labeled and unlabeled branch objectives."""

RUN = '20261110-phase1-pamf-branch-relations-f0-s2026092701-m4-r01'
SEED = 2026092701
VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
         'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')
WEAK_VIEWS = ('leo_clear_weak', 'leo_low_elev_weak', 'leo_rain_weak')


def candidate_rows(seed=SEED):
    definitions = (('sup_l', 'sup', .05, 0.), ('pl_l', 'pl', .05, 0.),
                   ('pl_u', 'pl', 0., .05), ('pl_lu', 'pl', .05, .05))
    return [dict(row_id=name + '-s' + str(seed), regime=regime, l_weight=wl, u_weight=wu,
                 model_seed=seed, parent_regime=regime, structure='full', permuted=False,
                 signature_views=3, protocol_id='F0', epochs=50, updates=11100,
                 new_updates=6660, source_rxs=[1, 3, 4, 6, 8], source_days=[1, 2, 3],
                 num_classes=6, views=list(VIEWS), weak_views=list(WEAK_VIEWS))
            for name, regime, wl, wu in definitions]


def validate_candidate(row, seed=SEED):
    if row not in candidate_rows(seed):
        raise ValueError('Only a preregistered F0 single-seed candidate is allowed')
    return row


def configuration(context, row, project, commit, *, seed=SEED):
    """Use the historical same-contract F20/F21 scratch E20 as the sole parent."""
    from experiments.pamf_paper_completion import plan as paper
    validate_candidate(row, seed)
    paper_row = next(r for r in paper.make_plan()['rows']
                     if r['data_setting'] == 'main_split' and r['family'] == 'factorial'
                     and r['method'] == row['regime'] and r['model_seed'] == seed)
    cfg = paper.configuration(context, paper_row, project, commit)
    parent_key = 'F20' if row['parent_regime'] == 'sup' else 'F21'
    cfg.update(run_id=RUN, row_id=row['row_id'], algorithm_id='pamf_branch_relations_v1',
               branch_row=row, branch_relation=dict(l_weight=row['l_weight'], u_weight=row['u_weight']),
               shared_prefix=context['sources'][str(seed)][parent_key] + '/segment20',
               output_root=project + '/runs/' + RUN + '/F/' + row['row_id'] + '/source')
    if row['u_weight']:
        cfg['hard_pl'] = dict(method='HPL2', threshold=.95, weight=1.)
        cfg['hard_pl_label_smoothing'] = 0.
    return cfg
