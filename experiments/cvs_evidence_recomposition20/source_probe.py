"""L-only gradients, V-only readout, no live update or new fitted head."""
import copy
import json
import os
import sys
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .capture import forward
from .objective import objective
from experiments.cvs_sigtime_heldcredit.held_credit import physical_ids, fresh_parameter_caches
from experiments.cvs_sigtime_heldcredit.source_probe import kernel_changes, norm
from experiments.cvs_u_schedule50.counterfactual import simulate, _equal, _rng_snapshot
from experiments.cvs_crossrx_sharpness20.source_features import sample, geometry


def protected_names(model):
    encoder = model.id_backbone.encoder
    objects = [encoder.core.id_backbone.cls_head.weight, encoder.core.id_backbone.cls_head.gain,
               encoder.response_gain, encoder.sequence_pool.gain_raw]
    names = [n for n, p in model.named_parameters() if any(p is v for v in objects)]
    if len(names) != 4:
        raise ValueError('Original default W and three deployment gains required')
    return names


def gradient_difference(a, b):
    return tuple(None if x is None and y is None else -y if x is None else x if y is None else x-y
                 for x, y in zip(a, b))


def total_gradient(base, aux):
    return tuple(None if x is None and y is None else y if x is None else x if y is None else x+y
                 for x, y in zip(base, aux))


def v_readout(model, bank, fixed_weight):
    x, y, rx, day = bank
    with fresh_parameter_caches(model), torch.no_grad(), torch.autocast(device_type='cpu', enabled=False):
        z = model.id_backbone.encoder.features(x.float()).detach().clone()
    logits = 30. * F.linear(F.normalize(z.double(), dim=1, eps=1e-4),
                           F.normalize(fixed_weight.double(), dim=1, eps=1e-4))
    return z, dict(geometry(z, y, rx, day), pre_step_default_W_CE=float(F.cross_entropy(logits, y)))


def virtual_readout(model, names, after, bank, fixed_weight, protected):
    temp = copy.deepcopy(model).eval().requires_grad_(False)
    target = dict(temp.named_parameters()); source = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in zip(names, after):
            target[name].copy_(source[name] if name in protected else value)
    # Fix pre-step W AND gains here to exclude clipping-induced readout changes.
    return v_readout(temp, bank, fixed_weight)


def l_bank(dataset):
    if len(dataset) != 6300 or getattr(dataset, 'hide', True):
        raise ValueError('Visible source L6300 required')
    metadata, cells = [], {}
    for i, it in enumerate(dataset.index):
        pid = (int(it.eq_i), int(it.rx_i), int(it.day_i), int(it.sig_i), int(dataset.selected[i]))
        key = (int(it.tx_i), int(it.rx_i), int(it.day_i))
        if key[0] not in range(6) or key[1] not in (1,3,4,6,8) or key[2] not in (1,2,3):
            raise ValueError('Source L roles differ')
        metadata.append(dict(y=key[0], id=pid)); cells.setdefault(key, []).append(i)
    physical_ids([r['id'] for r in metadata], 6300)
    if len(cells) != 90 or any(len(v) != 70 for v in cells.values()):
        raise ValueError('Fixed source 90x70 L cells differ')
    batches = {}
    for ordinal in d.ORDINALS:
        indices = [cells[k][ordinal] for k in sorted(cells)]
        xs, ys, ids = [], [], []
        for i in indices:
            x, y, _, meta = dataset[i]
            pid = tuple(int(meta[k]) for k in ('eq_i','rx_i','day_i','sig_i','base_index'))
            if pid != metadata[i]['id'] or int(y) != metadata[i]['y']:
                raise ValueError('Actual L IQ metadata differs')
            xs.append(x); ys.append(int(y)); ids.append(pid)
        batches[ordinal] = (torch.stack(xs), torch.tensor(ys), ids)
    if len({pid for _, _, ids in batches.values() for pid in ids}) != 360:
        raise ValueError('Four L cohorts must contain distinct packets')
    return batches


def run(request):
    d.validate(request); out = Path(d.OUTPUT)
    if out.exists():
        raise FileExistsError('Preserve existing assay output, no relaunch')
    out.mkdir(parents=True, exist_ok=False); torch.set_num_threads(2); started = time.perf_counter()
    runtime = dict(pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv, device='cpu', GPU_allocations=0)
    d.write(out/'runtime.json', runtime); print('RUNTIME '+json.dumps(runtime), flush=True)
    from experiments.cvs_crossrx_sharpness20 import design as parent, model as api
    from experiments.cvs_crossrx_sharpness20.provenance import source_proof, validate_checkpoint
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_phase1_stack.fast_execution import configure
    c = parent.config(next(r for r in parent.rows() if r['row_id'] == request['source_row']))
    source = source_proof(c)  # Source roles/ancestry/final selection before either deserialization.
    ck = torch.load(source/'final_ssdg.pth', map_location='cpu', weights_only=False)
    validate_checkpoint(ck, c)
    initial = torch.load(source/'initial_smoke.pt', map_location='cpu', weights_only=False)
    if set(initial) != {'model', 'scratch_only'} or initial['scratch_only'] is not True:
        raise ValueError('Own initial scratch checkpoint differs')
    model = api.build(c, torch.device('cpu')).eval()
    args = parent.make_args(c, 'cpu')
    named = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    names, parameters = [n for n,p in named], [p for n,p in named]
    if len(parameters) != 188 or len(ck['optimizer']['param_groups']) != 1:
        raise ValueError('Original native AdamW traversal differs')
    protected = protected_names(model)
    head = api.classifier(model)
    # Do not infer weight order from traversal order.
    wi = next(i for i,p in enumerate(parameters) if p is head.weight)
    for label, state in [('INITIAL', initial['model']), ('FINAL20', ck['model'])]:
        model.load_state_dict(state, strict=True)
        with torch.no_grad(): seen = forward(model, torch.zeros(2,2,256))
        if seen['logits'].shape != (2,6):
            raise ValueError('Real-checkpoint no-query smoke failed')
        print('SMOKE PASS '+label+' own checkpoint/direct four branches/query_read=false', flush=True)
    with installed(c) as native:
        context = native._build_ssdg_wisig_data(configure(parent.make_args(c, 'cpu')), torch.device('cpu'))
        from cvsrffi.xuc_fusion.native import role_ids_from_native
        if context['named_test_loaders'] or role_ids_from_native(context) != parent.read(parent.SOURCE)['role_ids']:
            raise ValueError('Original source-only role IDs differ')
    batches = l_bank(context['train_loader'].dataset)
    v_dataset = context['val_loader'].dataset; bank = sample(v_dataset)
    l_indices = set(int(i) for i in context['train_loader'].dataset.selected)
    if l_indices.intersection(int(i) for i in v_dataset.selected):
        raise ValueError('Source L/V physical base-index separation differs')
    records = []; temporary_steps = 0; network_forwards = 4
    for label in d.STATES:
        model.load_state_dict(initial['model'] if label == 'INITIAL' else ck['model'], strict=True)
        model.eval().requires_grad_(True)
        opt = torch.optim.AdamW(parameters, lr=float(args.lr), weight_decay=float(args.weight_decay))
        if label == 'FINAL20': opt.load_state_dict(ck['optimizer'])
        if label == 'INITIAL' and opt.state:
            raise ValueError('Initial diagnostic must use explicitly fresh empty moments')
        before = copy.deepcopy(model.state_dict()); opt_before = copy.deepcopy(opt.state_dict()); rng_before = _rng_snapshot()
        fixed_weight = head.weight.detach().clone()
        z_before, v_before = v_readout(model, bank, fixed_weight); network_forwards += 180
        for ordinal in d.ORDINALS:
            x, y, ids = batches[ordinal]; components = forward(model, x); network_forwards += 90
            ce = F.cross_entropy(components['logits'], y)
            base = torch.autograd.grad(ce, parameters, retain_graph=True, allow_unused=True)
            if base[wi] is None or norm([base[wi]]) <= 0:
                raise ValueError('Main CE must jointly train default W')
            modes = {}; gradients = {}; zs = {}
            for mode in d.MODES:
                result = objective(model, components, y, ids, mode, role='L_s')
                gradient = torch.autograd.grad(result['loss'], parameters, retain_graph=True, allow_unused=True)
                if any(g is not None for n,g in zip(names,gradient) if n in protected):
                    raise ValueError('Auxiliary must detach original default W and all three gains')
                gradients[mode] = gradient
                proposal = simulate(opt, base, total_gradient(base,gradient), names=names, max_grad_norm=5.)
                temporary_steps += 2
                z, v = virtual_readout(model, names, proposal['total_after'], bank, fixed_weight, protected)
                zs[mode] = z; network_forwards += 180
                summary = result['summary']
                summary.update(auxiliary_gradient_norm=norm(gradient), protected_auxiliary_gradients_None=True,
                    gradient_norm_by_branch={part:norm([g for n,g in zip(names,gradient) if token in n])
                        for part,token in [('Sinc','.sinc.'),('base_time','.id_backbone.t'),('PA','pa_'),
                            ('reference','.response_projection.'),('SIG_TIME','.sequence_pool.')]},
                    virtual_AdamW=proposal['summary'], physical_Sinc=kernel_changes(model,names,proposal),
                    V_after=v, V_before=v_before,
                    V_feature_RMS_from_pre_step=float((z.double()-z_before.double()).square().mean().sqrt()),
                    V_pre_step_W_and_gains_fixed=True)
                modes[mode] = summary
            relative = norm(gradient_difference(gradients['SWAP_PA'],gradients['SELF']))/max(norm(gradients['SELF']),1e-12)
            record = dict(state=label,ordinal=ordinal,current_L_CE=float(ce.detach()),
                learning_rate=float(opt.param_groups[0]['lr']),optimizer_moments='fresh hypothetical initial' if label=='INITIAL' else 'actual saved final20',
                modes=modes,SWAP_SELF_gradient_relative_difference=relative,
                SWAP_SELF_V_identity_margin_gain=modes['SWAP_PA']['V_after']['identity_margin']-modes['SELF']['V_after']['identity_margin'],
                SWAP_SELF_V_preW_CE_reduction=modes['SELF']['V_after']['pre_step_default_W_CE']-modes['SWAP_PA']['V_after']['pre_step_default_W_CE'],
                SWAP_SELF_V_feature_RMS=float((zs['SWAP_PA'].double()-zs['SELF'].double()).square().mean().sqrt()))
            records.append(record); d.write(out/(label+'-ordinal'+str(ordinal)+'.json'),record)
            print('SOURCE_CELL '+json.dumps({k:v for k,v in record.items() if k!='modes'}),flush=True)
        if (not _equal(before,model.state_dict()) or not _equal(opt_before,opt.state_dict())
            or any(p.grad is not None for p in model.parameters()) or not _equal(rng_before,_rng_snapshot())):
            raise ValueError('Assay modified live model/optimizer/gradients/RNG')
    gate = d.source_gate(records)
    if temporary_steps != 48 or network_forwards != 5404:
        raise ValueError('Fixed diagnostic compute counts differ')
    summary = dict(status='SOURCE_SCREEN_PASSED' if gate['passed'] else 'SOURCE_SCREEN_NOT_PASSED',
        run_id=d.RUN,analysis_code_commit=request['analysis_code_commit'],source_falsification=d.GATE,gate_result=gate,
        checkpoint_source=dict(run_id=parent.RUN,row_id=c['row_id'],epoch=20,seed=2026092701,
            source_roles='EXACT_MATCH',ancestors=[],selection='preregistered BASE_CE firstseed, own initial and final20; no target ranking'),
        source_L_IQ_reads=360,source_V_IQ_reads=180,source_L_metadata_packets=6300,source_V_metadata_packets=27000,
        network_IQ_forwards=network_forwards,temporary_AdamW_steps=temporary_steps,
        actual_optimizer_updates=0,head_fit_updates=0,EMA_updates=0,U_IQ_reads=0,U_truth_read=False,
        target_access=False,V_gradient=False,live_parameters_buffers_optimizer_gradients_RNG_unchanged=True,
        previously_trained_L=True,V_known_source_RX=True,reference_branch_preserved=True,default_head_main_joint=True,
        CPU_diagnostic_not_contemporaneous_CUDA_step=True,records=records,runtime=runtime,
        wall_seconds=time.perf_counter()-started,
        limitation='Two fixed states, four L90 cohorts, V180 known source RX. INITIAL moments are explicitly fresh hypothetical; FINAL20 uses saved moments. Virtual eval fixes pre-step W/all gains. No training or unseen-RX efficacy, novelty proof, U evidence or target access.')
    d.write(out/'summary.json',summary); print('SOURCE_COMPLETE '+json.dumps(gate),flush=True)
    return summary
