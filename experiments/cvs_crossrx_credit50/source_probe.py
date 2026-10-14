"""Fixed own-final SIG_TIME source feature and hypothetical update diagnosis."""
import json
import math
import os
import sys
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from .capture import capture
from .objective import objective
from .credit import assess

RUN = '20261014-phase1-SIGTIME-crossRX-credit-sourceL-manysig-s2026092701-r01'
RELEASE = 'cvs_SIGTIME_credit_sourceL_20261014_r01'
PROJECT = '/home/szu2070436088/2510044040/CV-SincNet'


def run(request):
    expected = PROJECT + '/runs/' + RUN + '/SIG_TIME-s2026092701/source_diagnostic'
    if (request['run_id'] != RUN or request['output_root'] != expected
            or request['source_role'] != 'L_s' or request['model_seed'] != 2026092701
            or request.get('source_release') != PROJECT+'/releases/cvs_feature_sequence50_20261013_r01'):
        raise ValueError('Only fixed first-seed SIG_TIME source L and owned output')
    out = Path(expected)
    if out.exists():
        raise FileExistsError('Preserve existing source probe, no relaunch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    runtime = dict(pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv, device='cpu', GPU_allocations=0,
                   run_id=RUN, analysis_code_commit=request['analysis_code_commit'])
    (out/'runtime.json').write_text(json.dumps(runtime, indent=2)+'\n')
    print('RUNTIME '+json.dumps(runtime), flush=True)
    from experiments.cvs_feature_sequence50 import design as d
    from experiments.cvs_feature_sequence50.provenance import source_proof
    from experiments.cvs_feature_sequence50.model import build, classifier
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_phase1_stack.fast_execution import configure
    from experiments.cvs_source_semantic_geometry import measure
    c = d.read(Path(request['source_release'])/'experiments/cvs_feature_sequence50/configs/SIG_TIME-s2026092701.json')
    if c['arm']!='SIG_TIME' or c['model_seed']!=2026092701 or c['row_id']!='SIG_TIME-s2026092701':
        raise ValueError('Only preregistered fixed SIG_TIME first seed')
    p = source_proof(c)  # Complete source contract/ancestry/selection before load.
    model = build(c, 'cpu')
    ck = torch.load(p/'final_ssdg.pth', map_location='cpu', weights_only=False)
    if (ck['epoch'] != 50 or ck['run_id'] != d.RUN or ck['candidate_id'] != c['row_id']
            or ck['checkpoint_selection'] != 'final_only' or ck['a1_ema_successful_updates'] != 11100):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: actual fixed final student')
    model.load_state_dict(ck['model'], strict=True)
    model.eval()
    if 'optimizer' not in ck or len(ck['optimizer']['param_groups']) != 1:
        raise ValueError('Saved native one-group AdamW required; never fabricate moments')
    parameters = [v for v in model.parameters() if v.requires_grad]
    opt = torch.optim.AdamW(parameters)
    if len(parameters) != len(ck['optimizer']['param_groups'][0]['params']):
        raise ValueError('Native trainable traversal/optimizer order differs')
    opt.load_state_dict(ck['optimizer'])
    by_id = {id(v): name for name, v in model.named_parameters()}
    names = [by_id[id(v)] for v in parameters]
    for param in parameters:
        state = opt.state.get(param, {})
        for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            if key in state and state[key].shape != param.shape:
                raise ValueError('Saved optimizer moment/parameter shape mismatch')
    group = opt.param_groups[0]
    if not math.isclose(float(group['lr']),1e-6,rel_tol=1e-12) or float(group['weight_decay']) != 1e-4:
        raise ValueError('Actual fixed final50 optimizer LR/decay differs')
    head = classifier(model)
    wnames = [name for name, v in model.named_parameters() if v is head.weight]
    if len(wnames) != 1 or head.weight.shape != (6, 160):
        raise ValueError('Original default classifier differs')
    default_w = wnames[0]
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad(), capture(model) as observed:
        zero = model.forward_identity_only(torch.zeros(2, 2, 256))
    if (zero['tx_logits'].shape != (2, 6) or zero['id_feat_joint'].shape != (2, 160)
            or observed['z'].shape != (2, 160) or not torch.isfinite(zero['tx_logits']).all()):
        raise ValueError('Real-checkpoint no-query smoke failed')
    print('SMOKE PASS fixed own student checkpoint; query_read=false', flush=True)
    with installed(c) as native:
        context = native._build_ssdg_wisig_data(configure(d.make_args(c, 'cpu')), torch.device('cpu'))
        from cvsrffi.xuc_fusion.native import role_ids_from_native
        if context['named_test_loaders'] or role_ids_from_native(context) != d.read(d.SOURCE)['role_ids']:
            raise ValueError('Exact original source-only physical roles required')
    dataset = context['train_loader'].dataset
    if len(dataset) != 6300 or getattr(dataset, 'hide', True):
        raise ValueError('Complete visible L6300 required')
    xs, mids, joints, labels, rx, days, ids = [], [], [], [], [], [], []
    with torch.no_grad():
        for begin in range(0, 6300, 128):
            samples = [dataset[i] for i in range(begin, min(begin+128, 6300))]
            x = torch.stack([s[0] for s in samples])
            with capture(model) as observed:
                output = model.forward_identity_only(x)
            xs.append(x)
            mids.append(observed['z'].detach())
            joints.append(output['id_feat_joint'].detach())
            for _, y, _, meta in samples:
                labels.append(int(y)); rx.append(int(meta['rx_i'])); days.append(int(meta['day_i']))
                ids.append([int(meta[k]) for k in ('eq_i','rx_i','day_i','sig_i','base_index')])
            if begin % 1024 == 0:
                print(f'SOURCE_L_FEATURES {min(begin+len(samples),6300)}/6300', flush=True)
    xs, mids, joints = torch.cat(xs), torch.cat(mids), torch.cat(joints)
    geometry = {}
    for kind, vectors in (('SIG_TIME_pre_gain160', mids), ('actualjoint160', joints)):
        result = measure(vectors, labels, rx, days, ids)
        result['feature'] = kind
        result['limitation'] += ' SIG_TIME branch geometry does not equal the whole deployment geometry.'
        geometry[kind] = result
    cells = {}
    for i, key in enumerate(zip(labels, rx, days)):
        cells.setdefault(key, []).append(i)
    if len(cells) != 90 or any(len(v) != 70 for v in cells.values()):
        raise ValueError('Original balanced source physical cells differ')
    yt, rt, dt = map(torch.tensor, (labels, rx, days))
    probes, probe_indices, IQ_forward_count = [], [], 6302
    for ordinal in (0, 17, 35, 52):
        index = torch.tensor([cells[key][ordinal] for key in sorted(cells)])
        x, y, r, day = xs[index], yt[index], rt[index], dt[index]
        opaque_ids = [json.dumps(ids[i], separators=(',',':')) for i in index.tolist()]
        with capture(model) as observed:
            output = model.forward_identity_only(x)
        base_loss = F.cross_entropy(output['tx_logits'], y)
        base_grads = torch.autograd.grad(base_loss, parameters, retain_graph=True, allow_unused=True)
        order = objective(observed['z'], y, r, day, opaque_ids, 'ORDER')
        supcon = objective(observed['z'], y, r, day, opaque_ids, 'SUPCON')
        supcon_grads = torch.autograd.grad(supcon['weighted_loss'], parameters, retain_graph=True, allow_unused=True)
        utilities, gradients = {}, {}
        for receiver in (1, 3, 4, 6, 8):
            auxiliary = torch.autograd.grad(order['loss_by_RX'][str(receiver)], parameters,
                                            retain_graph=True, allow_unused=True)
            if auxiliary[names.index(default_w)] is not None:
                raise ValueError('Auxiliary W gradient must be None')
            gain_names = [i for i, name in enumerate(names) if name.endswith('sequence_pool.gain_raw')]
            if len(gain_names) != 1 or auxiliary[gain_names[0]] is not None:
                raise ValueError('Pre-gain loss must not train auxiliary gain')
            gradients[str(receiver)] = float(torch.sqrt(sum(
                v.detach().double().square().sum() for v in auxiliary if v is not None)))
            result = assess(model, opt, base_grads, auxiliary, x, y, r, receiver,
                            default_weight_name=default_w)
            utilities[str(receiver)] = result['summary']
            IQ_forward_count += result['summary']['extra_IQ_forwards']
        IQ_forward_count += 90
        norms = lambda grads: float(torch.sqrt(sum(v.detach().double().square().sum() for v in grads if v is not None)))
        record = dict(cell_ordinal=ordinal, source_count=90, base_L_CE=float(base_loss),
                      base_gradient_norm=norms(base_grads), SUPCON_gradient_norm=norms(supcon_grads),
                      SUPCON_auxiliary_W_None=supcon_grads[names.index(default_w)] is None,
                      ORDER=order['summary'], SUPCON=supcon['summary'],
                      ORDER_gradient_norm_by_RX=gradients, credit_by_RX=utilities,
                      main_default_W_gradient_nonzero=bool(base_grads[names.index(default_w)].abs().sum() > 0))
        probes.append(record); probe_indices.append(index.tolist())
        print('SOURCE_CREDIT_PROBE '+json.dumps(dict(ordinal=ordinal, gradients=gradients,
              credits={k:dict(mean=v['mean_credit'],accepted=v['accepted']) for k,v in utilities.items()})),flush=True)
    if any(not torch.equal(v, model.state_dict()[k]) for k,v in before.items()) or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Source probe changed parameters/buffers/live gradients')
    summary = dict(status='SOURCE_SIGTIME_CREDIT_COMPLETE',run_id=RUN,
                   analysis_code_commit=request['analysis_code_commit'], source_role='L_s',
                   source_count=6300, source_IQ_packet_reads=6300, network_IQ_forwards=IQ_forward_count,
                   checkpoint_source=dict(run_id=d.RUN,row_id=c['row_id'],epoch=50,seed=2026092701,
                                          source_execution_commit=d.read(p/'resolved_config.json')['commit'],
                                          source_roles='EXACT_MATCH',checkpoint=str(p/'final_ssdg.pth'),
                                          ancestors=[],source_selection='fixed final50',target_training_selection=False),
                   optimizer=dict(parameter_count=len(parameters),state_count=len(opt.state),
                                  lr=float(group['lr']),weight_decay=float(group['weight_decay']),
                                  betas=list(group['betas']),eps=float(group['eps']),one_native_group=True,
                                  hypothetical_final_state_only=True),
                   geometry=geometry,probes=probes,temporary_AdamW_steps=40,
                   optimizer_updates=0,head_fit_updates=0,EMA_updates=0,V_gradient_updates=0,
                   U_IQ_reads=0,U_truth_read=False,target_access=False,
                   live_parameters_buffers_gradients_unchanged=True,runtime=runtime,
                   wall_seconds=time.perf_counter()-started,
                   limitation='Four fixed balanced seen-L cohorts and hypothetical final50 L-only CE steps; '
                              'not historical/U-containing training updates, unseenRX validation, efficacy or novel training result.')
    torch.save(dict(mid_features=mids,joint_features=joints,labels=labels,receivers=rx,days=days,
                    physical_ids=ids,probe_indices=probe_indices),out/'visible_L6300_features.pt')
    d.write(out/'summary.json',summary)
    print('SOURCE_SIGTIME_CREDIT_COMPLETE '+json.dumps(dict(status=summary['status'],
          network_IQ_forwards=IQ_forward_count,temporary_AdamW_steps=40,
          accepted_groups=sum(v['accepted'] for q in probes for v in q['credit_by_RX'].values()))),flush=True)
    return summary
