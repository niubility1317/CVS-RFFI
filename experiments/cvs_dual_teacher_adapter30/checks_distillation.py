"""Pure-tensor distillation checks; no checkpoints, IQ, models or remote work."""
import torch
from torch.nn import functional as F
from .distillation import masked_teacher_kl, objective


def run():
    dtype = torch.float64
    n = 8
    student = torch.linspace(-1.5, 1.5, n * 6, dtype=dtype).reshape(n, 6).requires_grad_()
    teacher = student.detach().flip(1).clone().requires_grad_()
    labels = teacher.detach().argmax(1)
    labels = labels.clone(); labels[1::2] = (labels[1::2] + 1) % 6
    kwargs = dict(temperature=4., teacher_scale=6., student_scale=1., teacher_correct_mask=True,
                  temperature_compensation=True)
    value, details = masked_teacher_kl(student, teacher, labels, **kwargs)
    gradient, teacher_gradient = torch.autograd.grad(value, (student, teacher), allow_unused=True)
    mask = (teacher.detach().argmax(1) == labels).to(dtype)
    target = (teacher.detach() * 6 / 4).softmax(1)
    expected = mask[:, None] * (student.detach().div(4).softmax(1) - target) * 4 / n
    if not torch.allclose(gradient, expected, atol=1e-12, rtol=1e-12) or teacher_gradient is not None:
        raise AssertionError('KL direction, multiplier, T-squared gradient or stop-gradient differs')
    lt, ls = (teacher.detach() * 6 / 4).log_softmax(1), (student.detach() / 4).log_softmax(1)
    expected_value = (mask * (lt.exp() * (lt - ls)).sum(1)).sum() * 16 / n
    if details['accepted'] != 4 or details['denominator'] != n or not torch.equal(value.detach(), expected_value):
        raise AssertionError('Teacher-correct denominator must remain all N')
    identical, _ = masked_teacher_kl(student, student.detach(), labels,
        temperature=4., teacher_scale=1., student_scale=1., teacher_correct_mask=False)
    identical_gradient, = torch.autograd.grad(identical, student)
    if identical.item() != 0. or not torch.allclose(identical_gradient, torch.zeros_like(student), atol=1e-15):
        raise AssertionError('Identical clean Tc/student native logits must give zero KL')
    wrong_labels = (teacher.detach().argmax(1) + 1) % 6
    zero, zero_info = masked_teacher_kl(student, teacher, wrong_labels, **kwargs)
    zero_gradient, = torch.autograd.grad(zero, student)
    if zero.item() != 0. or zero_info['accepted'] != 0 or not torch.equal(zero_gradient, torch.zeros_like(student)):
        raise AssertionError('All wrong teacher entries must give graph-connected zero KL')
    unmasked, info = masked_teacher_kl(student, teacher, wrong_labels,
        **dict(kwargs, teacher_correct_mask=False))
    if info['accepted'] != n or unmasked.item() <= 0:
        raise AssertionError('Disabling teacher-correct masking must retain every teacher entry')
    all_correct_labels = teacher.detach().argmax(1)
    opposite = -student.detach().clone().requires_grad_()
    _, student_true = masked_teacher_kl(student, teacher, all_correct_labels, **kwargs)
    _, student_opposite = masked_teacher_kl(opposite, teacher, all_correct_labels, **kwargs)
    if student_true['accepted'] != n or student_opposite['accepted'] != n:
        raise AssertionError('Teacher mask must not depend on student decisions')
    clean = student.detach().clone().requires_grad_()
    satellite = student.detach().roll(2, 1).clone().requires_grad_()
    options = dict(clean_temperature=4., satellite_temperature=4., clean_teacher_scale=1.,
        clean_student_scale=1., satellite_teacher_scale=6., satellite_student_scale=1.,
        clean_KD_weight=.25, satellite_KD_weight=.25, teacher_correct_mask=True)
    total, record = objective(clean, satellite, labels, teacher.detach(), teacher.detach().roll(1, 1), **options)
    expected_total = (.5 * F.cross_entropy(clean, labels, label_smoothing=.01)
        + .5 * F.cross_entropy(satellite, labels, label_smoothing=.01)
        + .25 * masked_teacher_kl(clean, teacher.detach(), labels, temperature=4.,
              teacher_scale=1., student_scale=1.)[0]
        + .25 * masked_teacher_kl(satellite, teacher.detach().roll(1, 1), labels, temperature=4.,
              teacher_scale=6., student_scale=1.)[0])
    if not torch.equal(total, expected_total) or record['mixed_teacher_target']:
        raise AssertionError('Clean/satellite CE and teacher targets must remain separate')
    ce_only, _ = objective(clean, satellite, wrong_labels, teacher.detach(), teacher.detach(),
                           **dict(options, clean_KD_weight=0., satellite_KD_weight=0.))
    gc, gs = torch.autograd.grad(ce_only, (clean, satellite))
    smoothed = F.one_hot(wrong_labels, 6).to(dtype) * .99 + .01 / 6
    if (not torch.allclose(gc, .5 * (clean.softmax(1) - smoothed) / n, atol=1e-12, rtol=1e-12)
            or not torch.allclose(gs, .5 * (satellite.softmax(1) - smoothed) / n, atol=1e-12, rtol=1e-12)):
        raise AssertionError('True CE must supervise all N in both slots, including wrong-teacher items')
    # Positive scale conversion: CSIL 5*cos+5 becomes 30*cos+30.
    cosine = torch.linspace(-.8, .8, n * 6, dtype=dtype).reshape(n, 6)
    common, _ = masked_teacher_kl(30*cosine, 5*cosine+5, labels, temperature=4.,
        teacher_scale=6., student_scale=1., teacher_correct_mask=False)
    if abs(common.item()) > 1e-12:
        raise AssertionError('Native scale conversion must preserve the common cosine distribution')
    noas_options = dict(options, clean_CE_weight=1., satellite_CE_weight=0., clean_KD_weight=0., satellite_KD_weight=0.)
    noas, noas_info = objective(clean, None, wrong_labels, teacher.detach(), teacher.detach(), **noas_options)
    noas_gradient, = torch.autograd.grad(noas, clean)
    if (not torch.allclose(noas_gradient, (clean.softmax(1)-smoothed)/n, atol=1e-12, rtol=1e-12)
            or noas_info['satellite']['student_forward'] or noas_info['As_CE'] != 0.):
        raise AssertionError('NoAs must use one all-N clean CE and no satellite student forward')
    negatives = 0
    def reject(fn):
        nonlocal negatives
        try:
            fn()
        except ValueError:
            negatives += 1
        else:
            raise AssertionError('Invalid distillation input accepted')
    for change in ({'temperature':0.}, {'temperature':float('nan')}, {'teacher_scale':0.},
                   {'teacher_scale':-1.}, {'student_scale':float('inf')},
                   {'teacher_correct_mask':1}, {'temperature_compensation':0}):
        reject(lambda change=change: masked_teacher_kl(student, teacher, labels, **dict(kwargs, **change)))
    reject(lambda: masked_teacher_kl(student[:, :5], teacher[:, :5], labels, **kwargs))
    reject(lambda: masked_teacher_kl(student[:0], teacher[:0], labels[:0], **kwargs))
    reject(lambda: masked_teacher_kl(student, teacher, torch.full_like(labels,-1), **kwargs))
    reject(lambda: masked_teacher_kl(student, teacher, labels.float(), **kwargs))
    broken = teacher.detach().clone(); broken[0, 0] = float('nan')
    reject(lambda: masked_teacher_kl(student, broken, labels, **kwargs))
    reject(lambda: objective(clean, satellite, labels, teacher, teacher,
                            **dict(options, clean_KD_weight=-.25)))
    reject(lambda: objective(clean, None, labels, teacher, teacher,
                            **dict(noas_options, satellite_CE_weight=.5, clean_CE_weight=.5)))
    reject(lambda: objective(clean, None, labels, teacher, teacher,
                            **dict(noas_options, satellite_KD_weight=.25)))
    return dict(status='PASS', protocol_negatives=negatives, KL_direction='teacher||student',
        all_N_denominator=True, teacher_only_correctness_mask=True, teacher_stop_gradient=True,
        identical_clean_KD_zero=True, zero_accepted_KD_zero=True, both_all_N_CE_gradients=True,
        separate_teacher_targets=True, native_scale_conversion=True, NoAs_single_CE_gradient=True, CP_reads=0, IQ_reads=0,
        model_forward_packets=0, optimizer_updates=0, remote_actions=0)


if __name__ == '__main__':
    print(run())
