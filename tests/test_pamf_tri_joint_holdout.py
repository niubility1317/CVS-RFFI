import numpy as np

from experiments.pamf_repl_source_adapt import joint_holdout as trial
from experiments.pamf_repl_source_adapt.folds import fold_positions


def test_entire_receiver_and_date_disappear_from_all_training_roles():
    rules = trial.folds([2, 4, 6, 7, 9], [1, 2, 3])
    rule = rules['rx2_day1']
    rx = np.repeat([2, 4, 6, 7, 9], 3*128)
    day = np.tile(np.repeat([1, 2, 3], 128), 5)
    meta = {r: {'rx_i': rx, 'day_i': day} for r in ('L_s', 'U_s', 'V')}
    meta['L_s']['y'] = np.arange(len(rx)) % 2
    split = fold_positions(meta, rule['excluded'], score_groups=rule['scored'])
    for indices in split['train'].values():
        assert ((rx[indices] != 2) & (day[indices] != 1)).all()
    assert set(split['held_v']) == {(2, 1), (2, 2), (2, 3)}


def test_source_decision_requires_improvement_in_joint_shift():
    rules = trial.folds([2, 4, 6], [1, 2, 3])

    def evidence(positive_joint):
        return [
            (rule, [
                dict(rx=rx, date=day, count=1800, weight=weight, gain=8,
                     damage=(2 if positive_joint or day != rule['date'] else 9) if weight == 3. else 8)
                for rx, day in rule['scored'] for weight in (4.5, 3.)
            ])
            for rule in rules.values()
        ]

    assert trial.choose(evidence(False))['weight'] == 6.
    assert trial.choose(evidence(True))['weight'] == 3.


def test_exact_zero_joint_shift_is_noninferior():
    rules = trial.folds([2, 4, 6], [1, 2, 3])
    evidence = [(rule, [
        dict(rx=rx, date=day, count=1800, weight=weight, damage=5,
             gain=5+(joint_net if day==rule['date'] else 10) if weight==3. else 5)
        for rx, day in rule['scored'] for weight in (4.5, 3.)
    ]) for rule, joint_net in zip(rules.values(), (1,2,-3), strict=True)]

    assert trial.choose(evidence)['weight'] == 3.
