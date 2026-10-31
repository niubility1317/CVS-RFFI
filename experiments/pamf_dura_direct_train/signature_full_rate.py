"""Scratch single E30 with an IQ residual branch at the original sample rate."""
import json
import numpy as np
import torch
from . import signature_context_prior as parent
from experiments.pamf_dura_efficiency.signature_full_rate import attach, ADDED, PARAMETERS, MACS, SEED_OFFSET

base = parent.base
RUN = '20261031-phase1-DURA-SIGfullRatePlain-E30-s2026092701-r01'
CONDITIONED_RUN = '20261031-phase1-DURA-SIGfullRateContext-E30-s2026092701-r01'
MODULE = 'experiments.pamf_dura_direct_train.signature_full_rate'
previous_factory = None


def smoke(args, p):
    clean, roles = base.source_training(p)
    targets, reuse = parent.reuse_targets(p, None, clean, roles)
    folder = p['cache'] / 'epochs/01'
    variance = np.load(folder / 'variance.npy', mmap_mode='r', allow_pickle=False)
    take = np.flatnonzero((roles == 'L_s') & (variance > np.float32(1e-8)))[:2]
    if len(take) != 2:
        raise ValueError('Two actual nonclean source L_s required')
    values = {k: base.tensor(np.load(folder / (k + '.npy'), mmap_mode='r', allow_pickle=False)[take], 'cpu')
              for k in ['received', 'kernel', 'coefficients', 'variance']}
    x, target = base.tensor(clean[take], 'cpu'), base.tensor(targets[take], 'cpu')
    student, original = base.scratch_model(2026092701), previous_factory(2026092701)
    if any(not torch.equal(v, student.state_dict()[k]) for k, v in original.state_dict().items()):
        raise ValueError('Original main G initial parameters changed')
    with torch.no_grad():
        torch.testing.assert_close(student(values['received']), original(values['received']), atol=0, rtol=0)
    reader, rng_info = parent.parent.load_teacher_preserving_rng(p, 'cpu')
    before = {k: v.clone() for k, v in reader.state_dict().items()}
    with torch.no_grad():
        features, parts = parent.parent.components(reader, x)
        torch.testing.assert_close(parts['signature'], target, atol=3e-5, rtol=2e-5)
        reader.classify_features(features)
    generator = torch.Generator().manual_seed(2026103004)
    probes = [torch.complex(torch.randn(2, 256, generator=generator), torch.randn(2, 256, generator=generator)) for _ in range(2)]
    loss, terms, restored = parent.objective(student, reader, values['received'], x,
        base.supervised_action(values['kernel'], values['coefficients']), values['variance'], probes, target)
    selected = {f'prior{i}_SIG': core.signature_projection.weight for i, core in enumerate(student.prior)}
    grads = torch.autograd.grad(terms[parent.METRICS[1]].sum(), list(selected.values()), retain_graph=True)
    if not all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads):
        raise ValueError('Original source-SIG auxiliary gradients missing')
    actual = torch.autograd.grad(terms[parent.parent.GUIDE].sum(), restored, retain_graph=True)[0]
    iqnorm = 2 * (restored.detach() - x).flatten(1).norm(dim=1) / (512 * x.square().mean((1, 2)).clamp_min(1e-4))
    if not actual.abs().sum() > 0 or not torch.isfinite(actual).all() or (actual.flatten(1).norm(dim=1) > parent.CAP * iqnorm * (1 + 2e-5)).any():
        raise ValueError('Original output-SIG IQ gradient cap failed')
    loss.mean().backward()
    for prefix in ['frozen.coarse.', 'frozen.estimator.', 'prior.']:
        gs = [v.grad for k, v in student.named_parameters() if k.startswith(prefix) and v.grad is not None]
        if not gs or not all(torch.isfinite(g).all() for g in gs) or not any(g.abs().sum() > 0 for g in gs):
            raise ValueError('Joint G gradient missing: ' + prefix)
    output_norms = []
    for core in student.prior:
        branch = core.full_rate
        g = branch.output.weight.grad
        if g is None or not torch.isfinite(g).all() or not g.abs().sum() > 0:
            raise ValueError('Actual full-rate output gradient missing')
        output_norms.append(float(g.norm()))
        dormant = [branch.input[0].weight, *[b[0].weight for b in branch.blocks]]
        if branch.conditioner is not None:
            dormant.append(branch.conditioner.weight)
        if any(p.grad is None or not torch.isfinite(p.grad).all() or not torch.equal(p.grad, torch.zeros_like(p.grad)) for p in dormant):
            raise ValueError('Zero-output-initialized full-rate internal dormancy differs')
    calls, shapes, handles = [], [], []
    for core in student.prior:
        handles.append(core.full_rate.register_forward_hook(lambda *args: calls.append('full_rate')))
        for block in core.full_rate.blocks:
            handles.append(block.register_forward_hook(lambda _, args, out: shapes.append(list(out.shape))))
        if core.full_rate.conditioner is not None:
            handles.append(core.full_rate.conditioner.register_forward_hook(lambda *args: calls.append('conditioner')))
    try:
        with torch.no_grad():
            expected = student(values['received']).clone()
    finally:
        for handle in handles:
            handle.remove()
    mode = student.prior[0].full_rate_mode
    if calls.count('full_rate') != 2 or calls.count('conditioner') != (2 if mode == 'conditioned' else 0) or shapes != [[2, 12, 256]] * 6:
        raise ValueError('Actual full-rate inference path differs')
    checkpoint = p['output'] / 'smoke-scratch_joint.pt'
    parent.f_guided.save_model(checkpoint, student, 'scratch_joint', 0, 0, args.commit)
    loaded = base.load_model(checkpoint, 'cpu', arm='scratch_joint', commit=args.commit, epoch=0)
    from experiments.pamf_stage2_inference.engine import TensorPipeline
    with torch.no_grad():
        if not torch.equal(loaded(values['received']), expected):
            raise ValueError('Strict full-rate E0 reload changed IQ')
        optimized, valid = TensorPipeline(loaded, None, reuse=True, native=False)(values['received'])
        if not valid:
            raise ValueError('Optimized full-rate inference invalid')
        torch.testing.assert_close(optimized, expected, atol=3e-5, rtol=3e-5)
        scores = reader(optimized)
    if scores.shape != (2, 6) or not torch.isfinite(scores).all() or any(p.grad is not None for p in reader.parameters()) or any(not torch.equal(v, before[k]) for k, v in reader.state_dict().items()):
        raise ValueError('Frozen original F/defaulthead changed')
    base.write_json(p['output'] / 'smoke.json', dict(status='PASS', real_nonclean_Ls=2,
        G_inputs=12, F_full_inputs=4, F_prefix_inputs=2, F_input_VJP_inputs=2, smoke_aux_parameter_VJP_inputs=2,
        full_rate_mode=mode, main_initial_parameters_and_IQ_matched=True, full_rate_output_gradient_norms=output_norms,
        full_rate_internal_initial_zero_gradient_expected=True, actual_full_rate_inference_calls=2,
        actual_conditioner_calls=calls.count('conditioner'), actual_block_shapes=shapes,
        strict_E0_reload=True, native_optimized_match=True, G_parameters=sum(p.numel() for p in student.parameters()),
        SIG_IQ_gradient_cap=.5, aux_gradient_outside_output_cap=True, F_unchanged=True, F_updates=0,
        TX_labels_read=0, V_reads=0, target_reads=0, G_initialization_checkpoint_loads=0,
        source_cache_reuse=reuse, teacher_load=rng_info))
    print(json.dumps(dict(event='REAL_SOURCE_SIG_FULL_RATE_E30_SMOKE_PASS', mode=mode)), flush=True)


def configure(*, mode='plain'):
    global previous_factory
    if mode not in PARAMETERS:
        raise ValueError('Fixed plain or conditioned full-rate path required')
    parent.configure()
    previous_factory = base.scratch_model
    previous_configuration, previous_command, previous_paths = base.configuration, base.command, base.paths
    run = RUN if mode == 'plain' else CONDITIONED_RUN
    module = MODULE if mode == 'plain' else 'experiments.pamf_dura_direct_train.signature_full_rate_context'
    def factory(seed):
        return attach(previous_factory(seed), seed, mode)
    def configuration():
        value = previous_configuration()
        value.update(run=run, recipe='scratch_E30_full_rate_' + mode, parameters=PARAMETERS[mode], ConvLinear_MACs=MACS[mode],
            full_rate_mode=mode, full_rate_parameters=ADDED[mode], full_rate_width=12, full_rate_dilations=[2, 4, 8],
            full_rate_seed_offset=SEED_OFFSET, full_rate_zero_output_init=True,
            full_rate_conditioner_zero_init=mode == 'conditioned', full_rate_conditioner_bound=.1,
            full_rate_conditioner_dimension=72 if mode == 'conditioned' else 0,
            full_rate_input='same9channelpriorinput', full_rate_inside_existing_SCALE=True,
            full_rate_common_branch_initialization_matched=True, full_rate_no_new_prior_CG_FFT=True,
            source_mechanism_evidence_run='20261031-diagnostic-DURA-source-SIG-generalization-s2026092701-m3-r01',
            source_mechanism_evidence_commit='f19f47ffd1c021477898a08e9115a16d07ecbdea',
            source_mechanism_evidence='deeper_predictors_improve_all6_sourceV_but_output_SIG_gain_small_mixed_and_IQ_often_worse;no_causal_decoder_claim',
            matched_control=parent.RUN, matched_control_commit='4452e09285b650536ab5e9dcc2acbfd9268c00e8',
            matched_full_rate_plain_run=RUN, matched_full_rate_conditioned_run=CONDITIONED_RUN,
            CPU_scorer_coordination_lock='runs/' + RUN + '/score.lock')
        return value
    def command(args, task, arm=None, role=None):
        return [module if x == parent.MODULE else x for x in previous_command(args, task, arm, role)]
    def paths(root):
        value = previous_paths(root)
        value['score_lock'] = root / 'runs' / RUN / 'score.lock'
        return value
    base.RUN, base.scratch_model, base.configuration = run, factory, configuration
    base.command, base.paths, base.smoke = command, paths, smoke


if __name__ == '__main__':
    configure()
    base.main()
