"""Two matched source-only F0 candidates: physically measured direction on/off."""
from experiments.pamf_branch_training.plan import SEED, VIEWS, WEAK_VIEWS

RUN = '20261110-phase1-pamf-ptbr-f0-s2026092701-m2-r01'


def candidate_rows(seed=SEED):
    return [dict(row_id='ptbr-beta' + tag + '-s' + str(seed), model_seed=seed, beta=beta,
                 protocol_id='F0', regime='pl', structure='full', signature_views=3, permuted=False,
                 source_rxs=[1, 3, 4, 6, 8], source_days=[1, 2, 3], num_classes=6,
                 epochs=50, updates=11100, new_updates=6660, l_weight=.05, u_weight=.05,
                 L_limit=64, U_limit=128, echo_magnitude=[.04, .1], echo_delays=[1, 2, 3],
                 physical_forward_mode='eval_no_grad_restore_modes_rng',
                 positives='same_pseudo_or_label_same_day_other_rx',
                 negatives='different_pseudo_or_label_same_day_same_rx',
                 views=list(VIEWS), weak_views=list(WEAK_VIEWS))
            for tag, beta in (('0', 0.), ('05', .5))]


def validate_candidate(row, seed=SEED):
    if row not in candidate_rows(seed):
        raise ValueError('Only the registered fixed-beta source-only PTBR candidates are allowed')
    return row


def configuration(context, row, project, commit, *, seed=SEED):
    from experiments.pamf_branch_training import plan as previous
    validate_candidate(row, seed)
    source_row = next(x for x in previous.candidate_rows(seed) if x['row_id'].startswith('pl_lu-'))
    c = previous.configuration(context, source_row, project, commit, seed=seed)
    c.pop('branch_row')
    c.pop('branch_relation')
    c.update(run_id=RUN, row_id=row['row_id'], algorithm_id='pamf_ptbr_v1', physical_row=row,
             physical_relation=dict(beta=row['beta'], l_weight=row['l_weight'], u_weight=row['u_weight']),
             output_root=project + '/runs/' + RUN + '/F/' + row['row_id'] + '/source')
    return c
