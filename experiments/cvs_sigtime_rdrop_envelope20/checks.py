"""Focused no-data tests for the new target and actual original model hook."""
import copy
import json
import sys
import torch
from torch.nn import functional as F
from . import design as d, model as api, mechanism as f


def reject(fn):
    try:
        fn()
    except (ValueError, RuntimeError, TypeError, KeyError):
        return
    raise AssertionError('Illegal source RDrop action accepted')


def main():
    torch.set_num_threads(2)
    torch.manual_seed(19)
    negative = 0
    y = torch.arange(48) % 6
    a = (torch.randn(48, 6, dtype=torch.float64) * 3).requires_grad_()
    b = torch.randn_like(a, requires_grad=True)
    original = api.original.terms(a, b, y, 1.)[2]
    targets, mixed = f.teachers(a, b, y)
    assert all(not t.requires_grad for t in targets.values())
    env, conf = targets['ENVELOPE'], targets['CONFIDENCE']
    assert torch.equal(env.gather(1, y[:, None]), conf.gather(1, y[:, None]))
    for row in range(48):
        for competitor in range(6):
            margin = env[row, y[row]] - env[row, competitor]
            reference = max(a[row, y[row]] - a[row, competitor], b[row, y[row]] - b[row, competitor])
            assert torch.allclose(margin, reference, atol=4e-15, rtol=1e-14)
    assert bool(mixed.any())
    for arm in d.ARMS:
        loss, _ = f.consistency(a, b, y, arm, f.new_audit())
        grad = torch.autograd.grad(loss, (a, b), retain_graph=True)
        assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in grad)
        if arm == 'FULL':
            assert torch.equal(loss, original)
            assert all(torch.equal(g, z) for g, z in zip(grad, torch.autograd.grad(original, (a, b), retain_graph=True)))
        else:
            target = targets[arm]
            assert all(torch.allclose(g, (v.softmax(1) - target.exp()) / (2 * len(y)), atol=1e-17, rtol=1e-13) for g, v in zip(grad, (a, b)))
        swapped, _ = f.consistency(b, a, y, arm, f.new_audit())
        shifted, _ = f.consistency(a + 17, b - 23, y, arm, f.new_audit())
        perm = torch.tensor([2, 5, 0, 3, 1, 4])
        permuted, _ = f.consistency(a[:, perm], b[:, perm], torch.argsort(perm)[y], arm, f.new_audit())
        assert all(torch.allclose(loss, z, atol=2e-14, rtol=1e-13) for z in (swapped, shifted, permuted))
        same, _ = f.consistency(a, a, y, arm, f.new_audit())
        assert abs(float(same.detach())) < 1e-15
        assert all(g.abs().max() < 1e-16 for g in torch.autograd.grad(same, (a,), retain_graph=True))
    for aa, bb, yy, arm in [(a, b, y - 1, 'FULL'), (a, b, y + 1, 'FULL'), (a, b, y.float(), 'FULL'),
                           (a, b, y[:1], 'FULL'), (a, b[:, :5], y, 'FULL'), (a.float(), b, y, 'FULL'),
                           (a * float('nan'), b, y, 'FULL'), (a, b, y, 'INVALID')]:
        reject(lambda: f.consistency(aa, bb, yy, arm, f.new_audit()))
        negative += 1
    from .provenance import validate_args
    for row in d.rows():
        c = d.config(row)
        args = vars(d.make_args(c, 'cpu'))
        validate_args(args, c)
        for key, value in [('use_aug', False), ('label_smoothing', 0.), ('source_fold_held_rx', None),
                           ('source_fold_training_receivers', '1,3,4,6,8'), ('baseline_ckpt', 'unknown.pt'),
                           ('teacher_ckpt', 'unknown.pt'), ('epochs', 50)]:
            bad = dict(args)
            bad[key] = value
            reject(lambda: validate_args(bad, c))
            negative += 1
        for key, value in [('model_seed', 2026092702), ('test_views', ['clean']), ('head_fit_updates', 1), ('fold', {})]:
            bad = dict(c)
            bad[key] = value
            reject(lambda: d.validate(bad))
            negative += 1
    device = 'cuda:0' if '--cuda' in sys.argv else 'cpu'
    if device != 'cpu':
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    models = [api.build(d.config(row), device) for row in d.rows()]
    states = [m.state_dict() for m in models]
    assert all(set(s) == set(states[0]) and all(torch.equal(v, s[k]) for k, v in states[0].items()) for s in states[1:])
    torch.manual_seed(2026101601)
    aug = torch.randn(128, 2, 256, device=device) * .7 + .13
    y = (torch.arange(128, device=device) // 8) % 6
    domains = torch.arange(128, device=device) % 12
    get_rng = torch.get_rng_state if device == 'cpu' else torch.cuda.get_rng_state
    ends, CE, logits, stats, norms, grads = [], [], [], {}, {}, {}
    names = ('id_backbone.encoder.core.id_backbone.t1.pw.weight',
             'id_backbone.encoder.core.id_backbone.t2.pw.weight',
             'id_backbone.encoder.sequence_pool.local_projection.weight')
    for row, m in zip(d.rows(), models):
        m.train()
        torch.manual_seed(2026101602)
        with m._phase1_labeled_feature_context():
            out = m(aug, y_tx=y, domain_labels=domains, return_aux=True)
        assert torch.equal(m._legacy_actual_L_input, aug)
        ce = F.cross_entropy(out['tx_logits'], y, label_smoothing=.01)
        correction, metrics = m._source_labeled_feature_hook(model=m, labeled_output=out, labels=y, extra=None,
            domains=domains, optimizer_step=1, identity_loss=ce, labeled_count=128)
        assert m._legacy_actual_L_input is None and m._envelope_calls == m._legacy_calls == m._feature_l_steps == 1 and m._legacy_extra_packets == 128
        named = dict(m.named_parameters())
        gs = torch.autograd.grad(ce + correction, [named[n] for n in names] + [api.classifier(m).weight])
        assert all(torch.isfinite(g).all() and g.norm() > 0 for g in gs)
        ends.append(get_rng().clone())
        CE.append(metrics['rdrop_envelope/paired_CE'])
        logits.append(out['tx_logits'].detach())
        stats[row['arm']], norms[row['arm']], grads[row['arm']] = m._envelope_audit.copy(), [float(g.norm()) for g in gs], gs
        teacher = copy.deepcopy(m).eval()
        assert teacher._legacy_original_labeled_scope.__self__ is teacher and teacher._legacy_actual_L_input is None
        reject(lambda: teacher._source_labeled_feature_hook(model=teacher, labeled_output=out, labels=y, extra=None,
            domains=domains, optimizer_step=2, identity_loss=ce, labeled_count=128))
        negative += 1
    assert all(torch.equal(v, CE[0]) for v in CE[1:]) and all(torch.equal(v, logits[0]) for v in logits[1:]) and all(torch.equal(v, ends[0]) for v in ends[1:])
    # Fresh original model, same seeds/input/RNG and unmodified native objective.
    native = api.original.build(d.native_config(), device).train()
    torch.manual_seed(2026101602)
    with native._phase1_labeled_feature_context():
        first = native(aug, y_tx=y, domain_labels=domains, return_aux=True)['tx_logits']
    second = native.forward_identity_only(aug, domain_labels=domains)['tx_logits']
    value = api.original.terms(first, second, y, 1.)[0]
    named = dict(native.named_parameters())
    reference = torch.autograd.grad(value, [named[n] for n in names] + [api.classifier(native).weight])
    assert torch.equal(get_rng(), ends[0]) and torch.equal(first.detach(), logits[0])
    assert all(torch.allclose(g, r, atol=1e-7, rtol=1e-5) for g, r in zip(grads['FULL'], reference))
    fixture = {arm: dict(source_health=True, mixed_envelope_fraction=.3,
               exports={'FINAL': dict(count=1260, errors=100, mean_CE=1.)}) for arm in d.ARMS}
    assert not d.qualify(fixture)['qualified']
    fixture['ENVELOPE']['exports']['FINAL']['errors'] = 99
    assert d.qualify(fixture)['qualified']
    for key in ('CE', 'signal', 'health', 'confidence'):
        bad = copy.deepcopy(fixture)
        if key == 'CE': bad['ENVELOPE']['exports']['FINAL']['mean_CE'] = 1.01
        elif key == 'signal': bad['ENVELOPE']['mixed_envelope_fraction'] = .09
        elif key == 'health': bad['FULL']['source_health'] = False
        else: bad['CONFIDENCE']['exports']['FINAL']['errors'] = 98
        assert not d.qualify(bad)['qualified']
    print(json.dumps(dict(status='PASS', device=device, negative_cases=negative, qualification_negative_cases=5,
        FP64_native_KL_and_gradient_exact=True, envelope_margin_identity=True, detached_teacher_gradient=True,
        class_permutation_and_logit_shift_invariance=True, confidence_control_same_true_mass=True,
        actual_initial_state_tensors=len(states[0]), actual_CE_logits_native_RNG_equal=True,
        FULL_actual_native_gradient_close=True, post_augmentation_input_reused=True,
        actual_parameter_gradient_norms=norms, envelope_audits=stats, synthetic_identity_forwards=1280,
        real_IQ_reads=0, checkpoint_loads=0, optimizer_steps=0, head_fit_updates=0, target_access=False)))


if __name__ == '__main__':
    main()
