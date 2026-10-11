"""Fixed source-L direction audit. No optimization, selection, or target access."""
from contextlib import contextmanager
import itertools
import math
import time
import torch
from torch import nn
from torch.nn import functional as F
from experiments.cvs_e20_cross_style.style import _paths, _metadata
from experiments.cvs_e20_budget_fishr.multilevel import weak_mixstyle_u, _parameter_groups

LEVELS = ('time_down', 't2.pool')
LEVEL_SETS = (('time_down',), ('t2.pool',), LEVELS)


def cosine(a, b):
    """Real Euclidean parameter-coordinate cosine; zero vectors are undefined."""
    na, nb = float(a.norm()), float(b.norm())
    return max(-1., min(1., float(a.dot(b))/(na*nb))) if na > 0 and nb > 0 else None


def _flat_grad(loss, parameters):
    values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    pieces = []
    for p, g in zip(parameters, values):
        g = torch.zeros_like(p) if g is None else g.detach()
        if g.is_complex():
            g = torch.view_as_real(g)
        pieces.append(g.reshape(-1).to(device='cpu', dtype=torch.float64))
    result = torch.cat(pieces)
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite identity gradient')
    return result


def _groups(model):
    full = tuple(p for p in model.base.id_backbone.parameters() if p.requires_grad)
    if not full:
        raise ValueError('Missing trainable identity parameters')
    groups = dict(full_identity=full, **_parameter_groups(model))
    if any(not value for value in groups.values()):
        raise ValueError('Missing native shallow/middle parameter group')
    return groups


def _group_slices(parameters, groups):
    positions = {}; start = 0
    for p in parameters:
        end = start+p.numel()*(2 if p.is_complex() else 1)
        positions[id(p)] = (start, end); start = end
    return {name: tuple(positions[id(p)] for p in group) for name, group in groups.items()}


def _view(vector, slices):
    return torch.cat([vector[a:b] for a, b in slices])


def _direction_report(vectors, slices):
    result = {}
    for group, locations in slices.items():
        g = {name: _view(v, locations) for name, v in vectors.items()}
        result[group] = dict(norms={name: float(v.norm()) for name, v in g.items()},
            pairwise_cosine={a+'|'+b: cosine(g[a], g[b]) for a, b in itertools.combinations(g, 2)})
    return result


@contextmanager
def style_hooks(model, y, rx, days, ids, seed, levels, strength, p=.5):
    """Manual native hooks also work in eval; private donor RNG, no flag changes."""
    y, rx, days, ids = _metadata(y, rx, days, ids)
    levels = tuple(levels)
    if levels not in LEVEL_SETS or not 0 <= strength <= 1:
        raise ValueError('Invalid fixed style diagnostic configuration')
    modules, paths = _paths(model); handles = []
    detail = dict(selected_levels=list(levels), hook_order=[], levels={}, strength=float(strength),
                  p=float(p), alpha=.1, donor_rules_verified=True)
    try:
        for level in levels:
            def hook(module, inputs, value, level=level):
                if level in detail['levels']:
                    raise RuntimeError('Repeated native layer in one audit forward')
                layer_seed = int(seed)+LEVELS.index(level)*1709
                mixed, stats = weak_mixstyle_u(value, y, rx, days, ids, layer_seed,
                                                p=p, alpha=.1, eta=strength)
                donor = stats.pop('donors')
                for i, j in enumerate(donor):
                    if i != j and (y[i] != y[j] or rx[i] == rx[j] or days[i] != days[j] or ids[i] == ids[j]):
                        raise ValueError('Illegal physical donor')
                # Aggregate only: no physical IDs or sample arrays leave the audit.
                stats.update(seed=layer_seed, path=paths[level], donor_assignments=sum(i != j for i, j in enumerate(donor)),
                             legal_donor_assignments=sum(i != j for i, j in enumerate(donor)))
                detail['levels'][level] = stats
                detail['hook_order'].append(level)
                return mixed
            handles.append(modules[paths[level]].register_forward_hook(hook))
        yield detail
        if tuple(detail['hook_order']) != levels:
            raise RuntimeError('Native diagnostic hook order differs')
    finally:
        for handle in handles:
            handle.remove()


def _metrics(logits, y):
    if logits.shape != (len(y), 6) or not torch.isfinite(logits).all():
        raise ValueError('Expected finite six-class logits')
    logits = logits.detach(); mask = F.one_hot(y, 6).bool()
    margin = logits.gather(1, y[:, None]).squeeze(1)-logits.masked_fill(mask, -torch.inf).max(1).values
    return dict(ce=float(F.cross_entropy(logits, y)), accuracy=float((logits.argmax(1) == y).float().mean()),
                margin_mean=float(margin.mean()), margin_q10=float(torch.quantile(margin.float(), .1)))


def _comparison(clean, styled, y):
    a, b = clean.detach(), styled.detach()
    ca, cb = a.argmax(1) == y, b.argmax(1) == y
    kl = F.kl_div(b.log_softmax(1), a.softmax(1), reduction='batchmean')
    return dict(styled=_metrics(b, y), repaired=int((~ca & cb).sum()), broken=int((ca & ~cb).sum()),
                persistent_wrong=int((~ca & ~cb).sum()), persistent_correct=int((ca & cb).sum()),
                changed_prediction=int((a.argmax(1) != b.argmax(1)).sum()), clean_to_styled_KL=max(0., float(kl)),
                logit_delta_rms=float((b-a).square().mean().sqrt()),
                max_abs_logit_delta=float((b-a).abs().max()))


def _variant_key(levels, strength):
    return '+'.join(levels)+'@'+str(float(strength))


def _rng_seed(seed):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _dropout_control(model, x, y, rx, days, ids, seed, strengths, level_sets):
    """Only dropout training flags active; same-mask replay isolates perturbation."""
    modes = [(m, m.training) for m in model.modules()]
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        model.eval()
        drops = [m for m in model.modules() if isinstance(m, nn.modules.dropout._DropoutNd)]
        for module in drops:
            module.training = True
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            _rng_seed(seed); clean = model(x)
            _rng_seed(seed); replay = model(x)
            replay_exact = torch.equal(clean, replay)
            replay_close = torch.allclose(clean, replay, rtol=2e-4, atol=2e-4)
            if not replay_close:
                raise AssertionError('Same-mask no-hook replay not numerically equivalent')
            _rng_seed(seed+190013); independent = model(x)
            rows = []
            for strength in strengths:
                for levels in level_sets:
                    _rng_seed(seed)
                    with style_hooks(model, y, rx, days, ids, seed+390017, levels, strength) as detail:
                        styled = model(x)
                    style_delta = styled-clean; mask_delta = independent-clean
                    # An additional independent-mask styled readout quantifies the
                    # interaction; additive decompositions are not presumed exact.
                    _rng_seed(seed+190013)
                    with style_hooks(model, y, rx, days, ids, seed+390017, levels, strength):
                        styled_independent = model(x)
                    interaction = (styled_independent-independent)-style_delta
                    rows.append(dict(variant=_variant_key(levels, strength), style=detail,
                        same_mask_comparison=_comparison(clean, styled, y),
                        independent_mask_comparison=_comparison(clean, styled_independent, y),
                        style_delta_rms=float(style_delta.square().mean().sqrt()),
                        mask_delta_rms=float(mask_delta.square().mean().sqrt()),
                        interaction_delta_rms=float(interaction.square().mean().sqrt()),
                        delta_cosine=cosine(style_delta.flatten().double().cpu(), mask_delta.flatten().double().cpu())))
        return dict(same_mask_replay_exact=replay_exact, same_mask_replay_close=replay_close,
                    same_mask_replay_max_abs_delta=float((replay-clean).abs().max()),
                    same_mask_seed_replayed=True, replay_rtol=2e-4, replay_atol=2e-4,
                    dropout_modules=len(drops), dropout_only_train=True,
                    persistent_statistics_eval=True, clean=_metrics(clean, y),
                    independent_mask=_comparison(clean, independent, y), variants=rows)
    finally:
        for module, mode in modes:
            module.training = mode


def _aggregate(records):
    summary = {}
    for key in sorted({row['variant'] for record in records for row in record['styles']}):
        selected = [row for record in records for row in record['styles'] if row['variant'] == key]
        def average(values):
            values = [v for v in values if v is not None]
            return sum(values)/len(values) if values else None
        groups = {}
        for group in selected[0]['directions']:
            template = selected[0]['directions'][group]
            groups[group] = dict(mean_norms={name: average([r['directions'][group]['norms'][name] for r in selected]) for name in template['norms']},
                mean_cosines={name: average([r['directions'][group]['pairwise_cosine'][name] for r in selected]) for name in template['pairwise_cosine']},
                cosine_valid_rounds={name: sum(r['directions'][group]['pairwise_cosine'][name] is not None for r in selected) for name in template['pairwise_cosine']})
        summary[key] = dict(rounds=len(selected), mean_styled_ce=average([r['comparison']['styled']['ce'] for r in selected]),
            mean_styled_accuracy=average([r['comparison']['styled']['accuracy'] for r in selected]),
            repaired=sum(r['comparison']['repaired'] for r in selected), broken=sum(r['comparison']['broken'] for r in selected),
            directions=groups)
    return summary


def audit(model, L, config, device):
    """Run the prespecified 36 x 6 source-L diagnostic; return aggregates/rounds.

    Every gradient is an autograd.grad result, never a parameter.grad update.
    Full identity includes Hz parameters and the classifier. The shallow/middle
    groups match the original standard-parameter Fishr calibration (exclude Hz).
    Raw styled-CE alignment includes the clean task gradient: the increment is
    separately reported and must be used for style-specific interpretations.
    """
    if L.get('role') not in ('L', 'L_s'):
        raise ValueError('This audit accepts only source L; U/V/target prohibited')
    rounds = config.get('rounds', 36); strengths = tuple(config.get('strengths', (.3, 1.)))
    level_sets = tuple(tuple(v) for v in config.get('level_sets', LEVEL_SETS))
    if rounds != 36 or strengths != (.3, 1.) or level_sets != LEVEL_SETS:
        raise ValueError('The fixed 36-pair six-variant diagnostic table differs')
    seed = int(config['model_seed'])
    if seed != model.multilevel.seed:
        raise ValueError('Model and audit sampling seeds differ')
    mc = dict(config['multilevel'], fishr_levels=list(LEVELS), fishr_estimator='cross')
    coefficients = dict(model.multilevel.fixed_coefficients)
    if set(coefficients) != set(LEVELS) or any(not math.isfinite(v) or v < 0 for v in coefficients.values()):
        raise ValueError('Missing fixed source-calibrated coefficients')
    groups = _groups(model); parameters = groups['full_identity']; slices = _group_slices(parameters, groups)
    before = {n: v.detach().clone() for n, v in model.state_dict().items()}
    modes = [(m, m.training) for m in model.modules()]
    gradients = [(p, None if p.grad is None else p.grad.detach().clone()) for p in model.parameters()]
    rng_before = [torch.random.get_rng_state().clone()]+(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    records = []; controls = []; seen_days = set(); tick = time.perf_counter()
    resource = dict(model_forwards=0, source_packet_readouts=0)
    def count_forward(module, args, result):
        resource['model_forwards'] += 1
        resource['source_packet_readouts'] += len(args[0])
    counter = model.register_forward_hook(count_forward)
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices):
            for i in range(rounds):
                # A source-private step namespace, distinct from training/calibration.
                step = 4000004+4*i
                ia, ib, day, _ = model.multilevel.pair_indices(L, step)
                index = torch.cat((ia, ib)); x = L['x'][index].to(device); y = L['y'][index].to(device)
                rx = L['rx'][index]; days = L['day'][index]; ids = [L['ids'][j] for j in index.tolist()]
                logits = []
                handle = model.register_forward_hook(lambda module, args, result: logits.append(result))
                try:
                    _, terms, pair_detail, _ = model.multilevel._pair(model, L, step, device, mc)
                finally:
                    handle.remove()
                if len(logits) != 2:
                    raise AssertionError('Expected exactly two clean pair forwards')
                split_clean = torch.cat(logits).detach()
                # Reference and styled forwards use the identical 120-occurrence
                # batch shape, avoiding split-vs-combined floating-point residuals
                # being interpreted as a style-specific gradient.
                clean = model(x); ce = F.cross_entropy(clean, y)
                split_delta = float((clean.detach()-split_clean).abs().max())
                if not torch.allclose(clean.detach(), split_clean, rtol=2e-4, atol=2e-4):
                    raise AssertionError('Eval split/combined readouts are not numerically equivalent')
                fishr = {level: terms[level]*coefficients[level]/2 for level in LEVELS}
                vectors = dict(clean_CE=_flat_grad(ce, parameters),
                    Fishr_shallow=_flat_grad(fishr[LEVELS[0]], parameters),
                    Fishr_middle=_flat_grad(fishr[LEVELS[1]], parameters))
                vectors['Fishr_both'] = vectors['Fishr_shallow']+vectors['Fishr_middle']
                record = dict(round=i, day=day, pair=pair_detail, clean=_metrics(clean, y),
                    split_combined_clean_max_abs_difference=split_delta, split_combined_clean_close=True,
                    fishr_losses={k: float(v.detach()) for k, v in terms.items()},
                    weighted_fishr_losses={k: float(v.detach()) for k, v in fishr.items()},
                    baseline_directions=_direction_report(vectors, slices), styles=[])
                detached_clean = clean.detach()
                # Release clean and higher-order graphs before six extra readouts.
                del ce, clean, logits, terms, fishr
                style_seed = seed+step*701+870001
                for strength in strengths:
                    for levels in level_sets:
                        with style_hooks(model, y, rx, days, ids, style_seed, levels, strength) as detail:
                            styled = model(x)
                        styled_ce = F.cross_entropy(styled, y)
                        v = dict(vectors, styled_CE=_flat_grad(styled_ce, parameters))
                        v['style_increment'] = v['styled_CE']-v['clean_CE']
                        for name in ('shallow', 'middle', 'both'):
                            v['Fishr_'+name+'_plus_style_increment'] = v['Fishr_'+name]+v['style_increment']
                        record['styles'].append(dict(variant=_variant_key(levels, strength), style=detail,
                            comparison=_comparison(detached_clean, styled, y),
                            style_increment_loss=float(styled_ce.detach())-record['clean']['ce'],
                            directions=_direction_report(v, slices)))
                        del styled, styled_ce, v
                records.append(record)
                if day not in seen_days:
                    control = _dropout_control(model, x, y, rx, days, ids, seed+step*97, strengths, level_sets)
                    controls.append(dict(day=day, round=i, **control)); seen_days.add(day)
            day_counts = {int(day): sum(row['day'] == day for row in records) for day in seen_days}
            if len(day_counts) != 3 or set(day_counts.values()) != {12}:
                raise AssertionError('Audit does not cover twelve pairs per source day')
            changed = [n for n, v in model.state_dict().items() if not torch.equal(v, before[n])]
            if changed:
                raise AssertionError('Audit mutated persistent state: '+','.join(changed))
            for p, value in gradients:
                if (p.grad is None) != (value is None) or (value is not None and not torch.equal(p.grad, value)):
                    raise AssertionError('Audit mutated parameter.grad')
    finally:
        counter.remove()
        with torch.no_grad():
            for n, v in model.state_dict().items():
                v.copy_(before[n])
        for module, mode in modes:
            module.training = mode
    rng_after = [torch.random.get_rng_state()]+(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    if not all(torch.equal(a, b) for a, b in zip(rng_before, rng_after)):
        raise AssertionError('Audit changed global RNG')
    return dict(schema='cvs_e20_direction_audit_v1', status='COMPLETE', source_role='L_s', target_access=False,
        U_access=False, V_access=False, optimizer_steps=0, rounds=rounds, variants_per_round=6, day_counts=day_counts,
        strengths=list(strengths), level_sets=[list(v) for v in level_sets], fixed_coefficients=coefficients,
        group_parameters={k: sum(p.numel() for p in v) for k, v in groups.items()},
        group_definition=dict(full_identity='all trainable model.base.id_backbone parameters including classifier and Hz; excludes unused nuisance network',
            time_down='identity time_fuse standard parameters excluding Hz', **{'t2.pool':'identity t2 standard parameters excluding Hz'}),
        gradient_semantics='autograd.grad; fixed coefficient divided by 2; style increment = styled CE gradient minus clean CE gradient',
        interpretation='Raw styled CE alignment includes clean task gradient; it does not establish style-specific benefit.',
        records=records, summary=_aggregate(records), dropout_controls=controls, resources=resource,
        state_unchanged=True, training_flags_restored=True, RNG_preserved=True, parameter_grad_unchanged=True,
        seconds=time.perf_counter()-tick)
