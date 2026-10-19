"""Only existing source V and fixed-final, matching-contract checkpoints."""
from collections import Counter
import numpy as np

METHODS = ('csil', 'mopc_hr')
SCENES = ('practical_low_suburban', 'practical_mid_urban', 'practical_low_urban')
CLASSES = ('14-10', '14-7', '20-15', '20-19', '6-15', '8-20')
N = 1800


def provenance(method, current, contract, init, done, config):
    if method not in METHODS or contract['role_ids'] != current['role_ids']:
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
    if contract.get('classes') != list(CLASSES) or config.get('method') != method:
        raise ValueError('Class map or external method differs')
    if (init.get('status') != 'SCRATCH' or init.get('checkpoint') is not None
            or init.get('ancestors') != [] or init.get('checkpoint_sources') != []
            or not init.get('scratch_only') or init.get('model_seed') != 2026092701
            or init.get('selection') != 'fixed_last_epoch'):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED')
    if (init.get('target_contact') is not False or init.get('target_access') is not False
            or done.get('target_access') is not False or done.get('target_evaluated') is not False):
        raise ValueError('CHECKPOINT_TARGET_CONTAMINATED')
    if done.get('epoch') != 200 or done.get('steps') != 9800 or done.get('status') != 'SOURCE_TRAINED':
        raise ValueError('Only original fixed final200 checkpoint')
    if config.get('source_counts') != dict(L_s=6300, U_s=56700, V=27000):
        raise ValueError('External source role counts differ')


def panel_contract(ids, y, rx, day, roles, current):
    if (roles.get('role') != 'V' or roles.get('V_panel') != N
            or roles.get('U_truth_used') is not False or roles.get('target_access') is not False
            or len(ids) != N or len(set(ids)) != N or list(ids) != roles['physical_ids']
            or not set(ids).issubset(current['role_ids']['V'])):
        raise ValueError('Only original, ordered source V1800 panel')
    expected = Counter({(t, r, d): 20 for t in range(6) for r in (1, 3, 4, 6, 8) for d in (1, 2, 3)})
    if Counter(zip(y.tolist(), rx.tolist(), day.tolist())) != expected:
        raise ValueError('Balanced TX/RX/day panel differs')


def paired(teacher, student, truth):
    teacher, student, truth = map(np.asarray, (teacher, student, truth))
    if teacher.shape != student.shape or teacher.shape != truth.shape or truth.ndim != 1 or not len(truth):
        raise ValueError('Aligned nonempty predictions required')
    tc, sc = teacher == truth, student == truth
    repair, damage = int((tc & ~sc).sum()), int((~tc & sc).sum())
    return dict(n=len(truth), teacher_accuracy_pct=float(tc.mean()*100),
                student_accuracy_pct=float(sc.mean()*100), repair=repair, damage=damage,
                both_wrong=int((~tc & ~sc).sum()), both_correct=int((tc & sc).sum()),
                delta_pp=(repair-damage)*100/len(truth), oracle_union_pct=float((tc | sc).mean()*100))


def qualification(rows):
    """Fixed source condition, applied separately to both existing channel draws."""
    checks = []
    for draw in (0, 1):
        low = rows[(draw, SCENES[2])]
        mid = rows[(draw, SCENES[1])]
        checks.append(low['delta_pp'] >= 1.0 and (low['delta_pp'] + mid['delta_pp'])/2 >= .5)
    return dict(pass_both_draws=all(checks), draw_pass=checks,
                rule='Each draw: low delta >=1pp and mid/low mean delta >=0.5pp; no target feedback',
                automatic_training=False, automatic_budget_extension=False)
