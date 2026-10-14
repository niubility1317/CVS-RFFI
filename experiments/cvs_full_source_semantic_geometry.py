"""Frozen default-model geometry over the complete authorized visible L pool."""
import torch


def run(request):
    from pathlib import Path
    import time
    from experiments.cvs_pa_recomposition import design as d
    from experiments.cvs_pa_recomposition import provenance
    from experiments.cvs_source_semantic_geometry import measure
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_phase1_stack.fast_execution import configure
    from experiments.cvs_equivariant_identity.precision import numerical_context
    from experiments.cvs_receiver_residual_capacity4.control import lock
    from experiments.cvs_multi_state_action.dispatch import capacity

    row = next(r for r in d.rows() if r['row_id'] == request['row_id'])
    if row['arm'] != 'ZERO':
        raise ValueError('Only all three predefined ZERO source controls')
    c = d.config(row)
    p = provenance.source_proof(c)
    out = p.parent / 'source_semantic_geometry_full_L6300'
    if (out / 'summary.json').exists():
        return d.read(out / 'summary.json')
    if out.exists():
        raise FileExistsError('Incomplete existing analysis; preserve')
    torch.set_num_threads(2)
    while True:
        with lock(Path(d.PROJECT) / 'runs/receiver_residual_capacity4.lock'):
            ready = [(len(s['pids']), -s['free_mb'], g)
                     for g, s in capacity().items()
                     if len(s['pids']) < 3 and s['free_mb'] >= 12000]
            if ready:
                gpu = min(ready)[2]
                device = torch.device(f'cuda:{gpu}')
                torch.zeros(1, device=device)
                torch.cuda.synchronize()
                break
        time.sleep(3)
    with numerical_context(d.FULL_FP32_POLICY):
        model = provenance.own_model(c, device)
        with installed(c) as native:
            context = native._build_ssdg_wisig_data(
                configure(d.make_args(c, str(device))), device)
            from cvsrffi.xuc_fusion.native import role_ids_from_native
            if (context['named_test_loaders'] or
                    role_ids_from_native(context) != d.read(d.SOURCE)['role_ids']):
                raise ValueError('Only unchanged original source roles')
        dataset = context['train_loader'].dataset
        if len(dataset) != 6300 or getattr(dataset, 'hide', True):
            raise ValueError('Only complete visible source L6300')
        if model.training or any(p.grad is not None for p in model.parameters()):
            raise ValueError('Frozen evaluation with no training gradients')
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        vectors, labels, receivers, days, physical_ids = [], [], [], [], []
        with torch.no_grad():
            for start in range(0, len(dataset), 128):
                samples = [dataset[i] for i in range(start, min(start + 128, len(dataset)))]
                x = torch.stack([s[0] for s in samples]).to(device)
                z = model.forward_identity_only(x)['id_feat_joint']
                if z.shape != (len(samples), 160) or not torch.isfinite(z).all():
                    raise ValueError('Actual default forward joint160 differs')
                vectors.append(z.detach().cpu())
                for _, y, _, meta in samples:
                    labels.append(int(y))
                    receivers.append(int(meta['rx_i']))
                    days.append(int(meta['day_i']))
                    physical_ids.append([int(meta[k]) for k in
                                         ('eq_i', 'rx_i', 'day_i', 'sig_i', 'base_index')])
        if any(not torch.equal(value, model.state_dict()[key]) for key, value in before.items()):
            raise ValueError('Source analysis changed parameter or buffer')
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError('Source analysis changed live model gradients')
        features = torch.cat(vectors)
        metadata = dict(labels=labels, receivers=receivers, days=days,
                        physical_ids=physical_ids)
        result = measure(features, **metadata)
        result.update(run_id=d.RUN, row_id=row['row_id'], model_seed=row['model_seed'],
                      arm='ZERO', source_role='L_s', source_current_final_epoch=50,
                      source_execution_commit=d.read(p / 'resolved_config.json')['commit'],
                      analysis_code_commit=request['analysis_code_commit'],
                      extra_IQ_forwards=6300, model_parameters_buffers_gradients_unchanged=True,
                      head_fit_updates=0, optimizer_updates=0, EMA_updates=0,
                      V_gradient_updates=0, U_forward_packets=0, U_truth_read=False,
                      target_access=False, source_cells=90, packets_per_cell=70,
                      within_cell_order='Original dataset order; ordinal-matched references, '
                                        'no hard mining, no target-based selection', GPU=gpu)
        out.mkdir()
        torch.save(dict(features=features, **metadata), out / 'visible_L6300_features.pt')
        d.write(out / 'summary.json', result)
        return result
