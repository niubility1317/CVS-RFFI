"""Fixed, frozen source-V feasibility for additive evidence candidate sets.

No source-V gradients enter a model, and no U or target observations are read.
The final own EMA is used only at its actual final epoch, never as an earlier
training teacher.  Labels enter only the downstream diagnostic census.
"""
from copy import deepcopy
from pathlib import Path
import math
import time
import torch

BETA = .5
PROBABILITY = .9
COEFFICIENT = .16
SOURCE_COUNT = 27000


def census(candidates, accepted, masses, truth, rx, day):
    """Labels are diagnostic only; candidates/masks/masses already exist."""
    n = len(truth)
    if candidates.shape != (n, 6) or accepted.shape != (n,):
        raise ValueError('Diagnostic shapes differ')
    sizes = candidates.sum(1)
    retained = candidates[torch.arange(n), truth]
    selectors = {'overall': torch.ones(n, dtype=torch.bool)}
    selectors.update({f'RX/{k}': rx == k for k in rx.unique().tolist()})
    selectors.update({f'day/{k}': day == k for k in day.unique().tolist()})
    selectors.update({f'TX/{k}': truth == k for k in range(6)})
    selectors.update({f'RXday/{r}/{d}': (rx == r) & (day == d)
                      for r, d in sorted(set(zip(rx.tolist(), day.tolist())))})
    result = {}
    for name, select in selectors.items():
        count = int(select.sum())
        groups = {}
        for subset, mask in [('all', select), ('hard_accepted', select & accepted),
                             ('complement', select & ~accepted)]:
            total = int(mask.sum())
            record = dict(count=total, size_histogram={str(k): int((mask & (sizes == k)).sum())
                                                       for k in range(1, 7)},
                          true_class_retained=int((mask & retained).sum()),
                          true_class_excluded=int((mask & ~retained).sum()),
                          truth_retention=float(retained[mask].double().mean()) if total else None)
            for arm, log_mass in masses.items():
                active = mask & (log_mass < math.log(PROBABILITY)) & (sizes < 6)
                multi = active & (sizes > 1)
                record[arm] = dict(active=int(active.sum()), active_multiclass=int(multi.sum()),
                                   active_true_excluded=int((active & ~retained).sum()),
                                   multiclass_active_true_excluded=int((multi & ~retained).sum()),
                                   mean_candidate_mass=float(log_mass[mask].exp().double().mean()) if total else None,
                                   mean_hinge_full_group=float(torch.relu(math.log(PROBABILITY)-log_mass[mask]).double().mean()) if total else None)
            groups[subset] = record
        result[name] = dict(count=count, subsets=groups)
    return result


def qualification(records):
    """Fixed source-only falsification thresholds, before any source run."""
    comp = records['overall']['subsets']['complement']
    checks = dict(complement_exists=comp['count'] > 0,
                  complement_truth_retention=(comp['truth_retention'] or 0) >= .98,
                  worst_RX_complement_truth_retention=min(
                      (z['subsets']['complement']['truth_retention'] or 0)
                      for name, z in records.items() if name.startswith('RX/')) >= .95,
                  S0_active_multiclass=comp['S0']['active_multiclass'] >= 100,
                  S1_active_multiclass=comp['S1']['active_multiclass'] >= 100,
                  S0_active_truth_retention=comp['S0']['active'] > 0 and
                      comp['S0']['active_true_excluded']/comp['S0']['active'] <= .05,
                  S1_active_truth_retention=comp['S1']['active'] > 0 and
                      comp['S1']['active_true_excluded']/comp['S1']['active'] <= .05,
                  nonvacuous_sets=sum(comp['size_histogram'][str(k)] for k in range(1, 6)) >= comp['count'] * .5)
    return dict(passed=all(checks.values()), checks=checks, source_only=True,
                parameter_scan=False, target_access=False,
                rule='Retain >=98% true TX in original-hard complement and >=95% in every RX; '
                     'at least100 active multiclass sets and >=95% active truth retention in each arm; >=50% complement non-full sets. '
                     'Failure closes this fixed beta/pi feasibility, without tuning them to create gradients.')


def run(request):
    from experiments.cvs_pa_recomposition import design as d, provenance
    from experiments.cvs_pa_recomposition.features import default_head
    from experiments.cvs_evidence_relation.components import parts
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_phase1_stack.fast_execution import configure
    from experiments.cvs_equivariant_identity.precision import numerical_context
    from experiments.cvs_receiver_residual_capacity4.control import lock
    from experiments.cvs_multi_state_action.dispatch import capacity
    from experiments.cvs_source_label_gate.audit import gate
    from experiments.cvs_trusted_correction.order import order
    from experiments.cvs_evidence_set.objective import candidate_set, set_loss

    row = next(r for r in d.rows() if r['row_id'] == request['source_row'])
    if row['arm'] != 'ZERO' or row['model_seed'] != d.SEEDS[0]:
        raise ValueError('First fixed ZERO source seed only; no target-ranked checkpoint')
    c = d.config(row)
    output = Path(request['output_root'])
    if (output/'summary.json').exists():
        return d.read(output/'summary.json')
    if output.exists():
        raise FileExistsError('Partial existing source analysis; preserve')
    source = provenance.source_proof(c)
    checkpoint = torch.load(source/'final_ssdg.pth', map_location='cpu', weights_only=False)
    if checkpoint['ema_model'] is None or checkpoint['epoch'] != 50 or checkpoint['a1_ema_successful_updates'] != 11100:
        raise ValueError('Actual own final EMA required')
    torch.set_num_threads(2)
    started = time.perf_counter()
    with numerical_context(d.FULL_FP32_POLICY):
        student = provenance.own_model(c, 'cpu')
        teacher = deepcopy(student)
        teacher.load_state_dict(checkpoint['ema_model'], strict=True)
        teacher.eval().requires_grad_(False)
        before = [{k:v.detach().cpu().clone() for k,v in obj.state_dict().items()}
                  for obj in (student, teacher)]
        while True:
            with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                ready = [(len(z['pids']), -z['free_mb'], gpu)
                         for gpu, z in capacity().items()
                         if len(z['pids']) < 3 and z['free_mb'] >= 12000]
                if ready:
                    gpu = min(ready)[2]
                    device = torch.device(f'cuda:{gpu}')
                    student.to(device); teacher.to(device)
                    torch.cuda.synchronize(device)
                    break
            time.sleep(3)
        with installed(c) as native:
            args = configure(d.make_args(c, str(device)))
            context = native._build_ssdg_wisig_data(args, device)
            from cvsrffi.xuc_fusion.native import role_ids_from_native
            if context['named_test_loaders'] or role_ids_from_native(context) != d.read(d.SOURCE)['role_ids']:
                raise ValueError('Original source-only roles required')
            dataset = context['val_loader'].dataset
            if len(dataset) != SOURCE_COUNT or getattr(dataset, 'hide', False) or 'v_select' not in dataset.split_source:
                raise ValueError('Only existing complete visible source V proxy')
            if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in student.id_backbone.modules()):
                raise ValueError('Dropout source proxy cannot update identity normalization')
            metadata = {k:torch.tensor([int(getattr(i,k)) for i in dataset.index], dtype=torch.long)
                        for k in ('rx_i','day_i','eq_i','sig_i')}
            metadata['base_index'] = torch.tensor(dataset.selected.tolist(), dtype=torch.long)
            indices = order(metadata, c['model_seed']+99431, 'PAIRS')
            if len(indices.unique()) != SOURCE_COUNT:
                raise ValueError('Every authorized V physical packet exactly once')
            vectors = {'S0':[], 'S1':[]}
            candidate_masks=[]; acceptance=[]; labels=[]; masses={'S0':[], 'S1':[]}; pseudo=[]
            with torch.random.fork_rng(devices=[gpu]), torch.no_grad():
                torch.manual_seed(c['model_seed']+99437+50)
                for obj in (student,teacher):
                    smoke = obj.forward_identity_only(torch.zeros(2,2,256,device=device))
                    if smoke['tx_logits'].shape != (2,6) or not torch.isfinite(smoke['tx_logits']).all():
                        raise ValueError('Real-checkpoint no-query smoke failed')
                student.train();teacher.eval()
                for begin in range(0, SOURCE_COUNT, 256):
                    ix = indices[begin:begin+256]
                    samples = [dataset[int(i)] for i in ix]
                    x = torch.stack([z[0] for z in samples]).to(device)
                    meta = {k:z[ix].to(device) for k,z in metadata.items()}
                    domains = torch.searchsorted(torch.tensor([1,3,4,6,8],device=device),meta['rx_i'])*3+meta['day_i']-1
                    weak = teacher.forward_identity_only(x)
                    weak_parts = parts(teacher, weak)
                    candidates = candidate_set(weak['id_feat_cls'], weak_parts['PA'], weak_parts['ref'],
                                               default_head(teacher).weight, beta=BETA)['candidates']
                    strong = student.forward_identity_only(native._strong_augment(x,.015))
                    strong_parts = parts(student, strong)
                    current_pseudo, masks = gate(native,args,weak['tx_logits'],strong['tx_logits'],(None,meta),domains)
                    # One fixed PA weakening; same forward and original hard gate.
                    student_vectors={'S0':strong_parts['joint'],
                                     'S1':strong_parts['joint']-BETA*strong_parts['PA']}
                    for arm,z in student_vectors.items():
                        loss=set_loss(z,default_head(student).weight,candidates,probability=PROBABILITY,
                                      coefficient=COEFFICIENT,eligible_mask=~masks['accepted'])
                        vectors[arm].append(z.cpu());masses[arm].append(loss['tensors']['log_candidate_mass'].cpu())
                    candidate_masks.append(candidates.cpu());acceptance.append(masks['accepted'].cpu());pseudo.append(current_pseudo.cpu())
                    # Truth is first consumed after all candidate/gate/mass computations.
                    labels.append(torch.tensor([int(z[1]) for z in samples],dtype=torch.long))
                    if begin % 4096 == 0:print(f'SOURCE_V_SEEN {min(begin+len(ix),SOURCE_COUNT)}/{SOURCE_COUNT}',flush=True)
            student.eval()
        for obj, saved in zip((student, teacher), before):
            if any(not torch.equal(value,obj.state_dict()[key].detach().cpu()) for key,value in saved.items()) or any(p.grad is not None for p in obj.parameters()):
                raise ValueError('Source feasibility changed model parameters/buffers/gradients')
        candidates=torch.cat(candidate_masks);accepted=torch.cat(acceptance);truth=torch.cat(labels)
        rx=metadata['rx_i'][indices];day=metadata['day_i'][indices]
        log_masses={k:torch.cat(z) for k,z in masses.items()}
        records=census(candidates,accepted,log_masses,truth,rx,day)
        local_gradients={}
        weight=default_head(student).weight.detach().cpu()
        for arm, values in vectors.items():
            z=torch.cat(values).detach().requires_grad_(True)
            loss=set_loss(z,weight,candidates,probability=PROBABILITY,coefficient=COEFFICIENT,eligible_mask=~accepted)
            grad=torch.autograd.grad(loss['weighted_loss'],z)[0]
            local_gradients[arm]=dict(weighted_loss=float(loss['weighted_loss']),gradient_norm=float(grad.norm()),
                                      nonzero_packets=int((grad.abs().sum(1)>0).sum()),denominator=SOURCE_COUNT,
                                      parameters_updated=False,limitation='Derivative only on detached source feature copies; not an actual AdamW update or efficacy result.')
        result=dict(status='SOURCE_FEASIBILITY_COMPLETE',run_id=request['run_id'],source_row=row['row_id'],
                    analysis_code_commit=request['analysis_code_commit'],source_execution_commit=d.read(source/'resolved_config.json')['commit'],
                    source_checkpoint=str(source/'final_ssdg.pth'),source_role='V',source_current_final_epoch=50,
                    own_EMA_successful_updates=11100,source_count=SOURCE_COUNT,beta=BETA,probability=PROBABILITY,
                    coefficient=COEFFICIENT,student_change='S1 fixed PA increment half weakening only; reference unchanged',
                    GPU=gpu,records=records,qualification=qualification(records),local_feature_gradients=local_gradients,
                    teacher_top1_accuracy=float((torch.cat(pseudo)==truth).double().mean()),
                    model_parameters_buffers_gradients_unchanged=True,head_fit_updates=0,optimizer_updates=0,
                    EMA_updates=0,V_gradient_updates=0,U_forward_packets=0,U_truth_read=False,target_access=False,
                    extra_V_IQ_forwards=54000,noquery_smoke_packets=4,augmentation_RNG_restored=True,
                    labels_enter_candidates_or_gate=False,wall_seconds=time.perf_counter()-started,
                    limitation='Final seen-source V proxy with current final own EMA. Does not establish hidden U accuracy, early training signal, unseen RX transfer or Phase2 deployment.')
        output.mkdir(parents=True,exist_ok=False)
        torch.save(dict(source_indices=indices,candidates=candidates,accepted=accepted,truth=truth,rx=rx,day=day,
                        teacher_top1=torch.cat(pseudo),log_candidate_masses=log_masses),output/'source_census.pt')
        d.write(output/'summary.json',result)
        return result
