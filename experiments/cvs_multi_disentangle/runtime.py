"""Alternating, training-only disturbance supervision of the native identity.

The native identity optimizer/EMA/forward and update count are unchanged. A
checked insertion adds the auxiliary objective before native finite checks and
backward. Separate networks never become native model parameters or query state.
"""
from contextlib import contextmanager
from collections import defaultdict
import inspect
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from experiments.cvs_phase1_stack.runtime import installed as native_installed
from experiments.cvs_multi_disentangle.model import (
    AuxiliaryNetworks, INTERMEDIATE_DIM, intermediate,
    identity_from_intermediate, classify_intermediate,
)
from experiments.cvs_multi_disentangle.physics import factorial_batch


ARM_PATHS = {
    'legacy_cosine': (), 'fixed_interventions': ('linear', 'temporal'),
    'unified': ('linear', 'temporal', 'receiver'),
    'linear': ('linear',), 'temporal': ('temporal',), 'receiver': ('receiver',),
    'multi': ('linear', 'temporal', 'receiver'),
    'multi_interaction': ('linear', 'temporal', 'receiver', 'interaction'),
}


@contextmanager
def deterministic_forward(module):
    """Gradients remain enabled; extra forwards do not alter running buffers."""
    states = [(m, m.training) for m in module.modules()]
    module.eval()
    try:
        yield
    finally:
        for m, training in states:
            m.training = training


@contextmanager
def frozen_parameters(module):
    states = [(p, p.requires_grad) for p in module.parameters()]
    for p, _ in states:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, enabled in states:
            p.requires_grad_(enabled)


def normalized_error(prediction, target):
    # Detached per-coordinate target scale prevents large intermediate channels
    # from dominating while retaining an absolute error near zero perturbations.
    scale = target.detach().square().mean().clamp_min(1e-4)
    return (prediction - target).square().mean() / scale


def finite_grad_norm(module):
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    if any(not bool(torch.isfinite(g).all()) for g in grads):
        raise FloatingPointError('Nonfinite auxiliary gradient; preserve run, no retry')
    return float(torch.stack([g.detach().norm() for g in grads]).norm()) if grads else 0.


class ReceiverStatistics:
    """Bounded-age source-L sufficient statistics; no packet/sample cache.

    Key=(TX,RX,quality bin,response RMS bin,absolute CFO bin). Groups are
    distributional comparisons, never declared same-emission counterfactuals.
    """
    def __init__(self, recipe):
        self.recipe = recipe
        self.groups = {}

    def key_rows(self, labels, receivers, views):
        quality = torch.bucketize(views[:, 24].contiguous(), views.new_tensor(
            self.recipe.get('receiver_quality_thresholds', [.25, .75])))
        response = (views[:, :24].square().mean(1).sqrt() >=
                    self.recipe.get('receiver_response_rms_threshold', 1.)).long()
        temporal = (views[:, 27].abs() >=
                    self.recipe.get('receiver_abs_cfo_threshold', .1)).long()
        rows = torch.stack((labels, receivers, quality, response, temporal), 1)
        return [tuple(int(v) for v in row) for row in rows.detach().cpu().tolist()]

    def update(self, labels, receivers, views, h, step):
        if bool((labels < 0).any()) or bool((receivers < 0).any()):
            raise ValueError('Receiver statistics require visible source-L labels/RX')
        max_age = int(self.recipe.get('receiver_max_age_steps', 888))
        self.groups = {k: v for k, v in self.groups.items() if step-v['step'] <= max_age}
        current = defaultdict(list)
        for i, key in enumerate(self.key_rows(labels, receivers, views)):
            current[key].append(i)
        for key, indices in current.items():
            v = views[indices].detach().mean(0).cpu()
            feature = h[indices].detach().mean(0).cpu()
            old = self.groups.get(key)
            # Cap historical effective count: EMA anchors drift during training.
            previous = min(old['count'], 32) if old else 0
            weight = len(indices)/(previous+len(indices))
            self.groups[key] = dict(view=v if old is None else old['view'].lerp(v, weight),
                h=feature if old is None else old['h'].lerp(feature, weight),
                count=previous+len(indices), step=int(step))
        return dict(current)

    def pairs(self, device, limit, current_keys=()):
        minimum = int(self.recipe.get('receiver_min_group_count', 2))
        valid = {k: v for k, v in self.groups.items() if v['count'] >= minimum}
        result, relations = [], defaultdict(list)
        current_keys = set(current_keys)
        keys = sorted(valid)
        for key in keys:
            for target in keys:
                if key[0] == target[0] and key[1] != target[1] and key[2:] == target[2:]:
                    a, b = valid[key], valid[target]
                    relation=(key[1],target[1],*key[2:])
                    relations[relation].append(dict(source=key, target=target,
                        v0=a['view'].to(device), v1=b['view'].to(device),
                        h0=a['h'].to(device), h1=b['h'].to(device)))
        # Keep cross-TX sets together, prioritizing relations with genuinely
        # observed current groups. Do not fill missing groups with fake pairs.
        order=sorted(relations,key=lambda r:(
            -sum(p['source'] in current_keys for p in relations[r]),
            -len(relations[r]),
            -max(valid[p['source']]['step'] for p in relations[r]),r))
        for relation in order:
            result.extend(relations[relation][:limit-len(result)])
            if len(result)>=limit: break
        return result

    def state_dict(self):
        return dict(groups=self.groups, scope='source_L_aggregate_statistics_only',
                    sample_cache=False, target_access=False)


class MultiTrainer:
    def __init__(self, c, model):
        self.config = c
        self.output_root = Path(c['output_root'])
        self.recipe = c['multi_recipe']
        self.arm = c['arm']
        self.paths = ARM_PATHS[self.arm]
        self.device = next(model.parameters()).device
        self.generator = torch.Generator(device='cpu').manual_seed(int(c['model_seed'])+39071)
        self.auxiliary = None
        self.optimizer = None
        self.statistics = ReceiverStatistics(self.recipe)
        self.updates = 0
        self.last = {}
        self.epoch_totals = defaultdict(float)
        self.epoch_samples = 0
        self.fit_gradients = {}
        self.fit_steps = {k:0 for k in self.paths}
        self.identity_steps = {k:0 for k in self.paths}
        self.receiver_relations_total = 0
        self.exposure_totals = defaultdict(int)
        self.training_seconds = 0.
        self.identity_parameters = sum(p.numel() for p in model.parameters())
        if self.paths and self.arm != 'fixed_interventions':
            # Creating training auxiliaries must not change native initialization,
            # loader RNG, augmentations or dropout streams.
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []):
                torch.manual_seed(int(c['model_seed'])+11939)
                if self.arm == 'unified':
                    from experiments.cvs_multi_disentangle.model import UnifiedAuxiliaryNetworks
                    self.auxiliary = UnifiedAuxiliaryNetworks(
                        rank=self.recipe['rank'], state_dim=self.recipe['code_dim'])
                else:
                    self.auxiliary = AuxiliaryNetworks(rank=self.recipe['rank'],
                        state_dim=self.recipe['code_dim'], active=self.paths,
                        include_interaction='interaction' in self.paths)
            self.auxiliary.to(self.device)
            for module in self.auxiliary.modules():
                if hasattr(module,'radius'):
                    module.radius=float(self.recipe['residual_bound'])
            self.optimizer = torch.optim.AdamW(self.auxiliary.parameters(),
                lr=self.recipe['auxiliary_lr'], weight_decay=1e-4)
        self.cost = dict(identity_parameters=self.identity_parameters,
            auxiliary_parameters=0 if self.auxiliary is None else sum(p.numel() for p in self.auxiliary.parameters()),
            inference_auxiliary_parameters=0, auxiliary_optimizer='separate AdamW',
            auxiliary_active_paths=list(self.paths), auxiliary_data='source_L_only',
            native_optimizer_excludes_auxiliaries=True, resume_supported=False)

    def _predict(self, kind, batch, h):
        target = {'linear': 'x10', 'temporal': 'x01', 'interaction': 'x11'}[kind]
        if kind == 'interaction':
            return self.auxiliary.predict(kind, batch['x00'], batch[target], h,
                descriptor=dict(x_linear=batch['x10'], x_temporal=batch['x01']))
        return self.auxiliary.predict(kind, batch['x00'], batch[target], h)

    def _receiver_views(self, x):
        # All RX arms store the same fixed sufficient statistics: response and
        # receive moments (37), followed by ordered temporal observations (160).
        # This enables explicit L/T subtraction without storing raw packets.
        branch=self.auxiliary.branch('receiver')
        if self.arm=='unified':
            return branch.view(x)
        return torch.cat((branch.views.receiver(x),branch.views.temporal(x).flatten(1)),dim=1)

    def _receiver_predict(self, pairs):
        v0 = torch.stack([p['v0'] for p in pairs])
        v1 = torch.stack([p['v1'] for p in pairs])
        h0 = torch.stack([p['h0'] for p in pairs])
        if self.arm=='unified':
            return self.auxiliary.branch('receiver').forward_views(v0,v1,h0,kind='receiver')
        return self.auxiliary.branch('receiver').forward_views(v0[:,:37],v1[:,:37],h0)

    @torch.no_grad()
    def _receiver_main_actions(self,pairs):
        """Detached group-conditional L/T approximation, not physical recovery."""
        h0=torch.stack([p['h0'] for p in pairs])
        v0=torch.stack([p['v0'] for p in pairs])
        v1=torch.stack([p['v1'] for p in pairs])
        delta=torch.zeros_like(h0)
        for kind in ('linear','temporal'):
            if kind not in self.paths: continue
            branch=self.auxiliary.branch(kind)
            if self.arm=='unified':
                part=branch.forward_views(v0,v1,h0,kind=kind)['delta']
            else:
                a,b=(v0[:,:29],v1[:,:29]) if kind=='linear' else (
                    v0[:,37:].reshape(-1,8,20),v1[:,37:].reshape(-1,8,20))
                part=branch.forward_views(a,b,h0)['delta']
            delta=delta+part
        return delta.detach()

    def _fit(self, batch, anchors, pairs, step):
        fit_start=time.perf_counter()
        losses, fitted = [], []
        self.optimizer.zero_grad(set_to_none=True)
        for kind in self.paths:
            if kind == 'receiver':
                if not pairs:
                    continue
                pred = self._receiver_predict(pairs)
                observed = torch.stack([p['h1']-p['h0'] for p in pairs])
                explained = self._receiver_main_actions(pairs)
                target = observed-explained
                self.last.update(receiver_residual_target=1.,
                    receiver_main_subtraction_enabled=float('linear' in self.paths or 'temporal' in self.paths),
                    receiver_main_action_norm=float(explained.norm(dim=1).mean()),
                    receiver_residual_target_norm=float(target.norm(dim=1).mean()),
                    receiver_target_requires_grad=float(target.requires_grad))
            else:
                pred = self._predict(kind, batch, anchors['x00'])
                target = (anchors['x11']-anchors['x10']-anchors['x01']+anchors['x00']
                          if kind == 'interaction' else anchors['x10' if kind == 'linear' else 'x01']-anchors['x00'])
            fit = normalized_error(pred['delta'], target)
            regression = fit.new_zeros(())
            if kind in ('linear', 'temporal'):
                regression = F.mse_loss(pred['parameters'], batch[kind+'_parameters'])
            losses.append(fit+regression)
            fitted.append(kind)
            self.last[kind+'_fit_loss'] = float(fit.detach())
            self.last[kind+'_parameter_loss'] = float(regression.detach())
        if not losses:
            self.last['auxiliary_fit_skipped_no_groups'] = 1.
            return
        fit_loss = torch.stack(losses).mean()
        if not bool(torch.isfinite(fit_loss)):
            raise FloatingPointError('Nonfinite auxiliary fit loss')
        fit_loss.backward()
        self.last['auxiliary_gradient_norm'] = finite_grad_norm(self.auxiliary)
        if self.arm != 'unified':
            for kind in self.paths:
                self.fit_gradients[kind] = finite_grad_norm(self.auxiliary.branch(kind))
                self.last[kind+'_gradient_norm'] = self.fit_gradients[kind]
        warmup=int(self.recipe['warmup_epochs'])
        fraction = min(1., max(0., (step-warmup*222)/((200-warmup)*222)))
        lr = self.recipe['auxiliary_lr_min'] + .5*(self.recipe['auxiliary_lr']-
            self.recipe['auxiliary_lr_min'])*(1+math.cos(math.pi*fraction))
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.optimizer.step()
        for kind in fitted: self.fit_steps[kind]+=1
        self.updates += 1
        self.last.update(auxiliary_fit_loss=float(fit_loss.detach()), auxiliary_lr=lr,
                         auxiliary_successful_updates=float(self.updates),
                         auxiliary_fit_seconds=time.perf_counter()-fit_start)

    def _receiver_identity(self, identity, h, current, pairs):
        """Only group-level relational geometry; no packet correspondence/CE."""
        minimum = int(self.recipe.get('receiver_min_group_count', 2))
        matched = {}
        for pair in pairs:
            key = pair['source']
            if key not in current or self.statistics.groups[key]['count'] < minimum:
                continue
            # Grouping relation=(source RX,target RX,observable strata).
            relation = (key[1], pair['target'][1], *key[2:])
            matched.setdefault(relation, []).append(pair)
        terms = []
        for group in matched.values():
            if len(terms) >= int(self.recipe.get('receiver_relations_max', 8)):
                break
            unique = {p['source'][0]: p for p in group}
            keys = sorted(unique)
            for i, left_tx in enumerate(keys):
                for right_tx in keys[i+1:]:
                    pair_a, pair_b = unique[left_tx], unique[right_tx]
                    hs = torch.stack([h[current[p['source']]].mean(0) for p in (pair_a, pair_b)])
                    with torch.no_grad():
                        selected=[pair_a,pair_b]
                        delta = self._receiver_predict(selected)['delta']+self._receiver_main_actions(selected)
                        target = torch.stack([p['h1']-p['h0'] for p in (pair_a,pair_b)])
                        reliability = (1+normalized_error(delta, target)).reciprocal()
                    zs = F.normalize(identity_from_intermediate(identity, hs), dim=1)
                    zt = F.normalize(identity_from_intermediate(identity, hs+delta.detach()), dim=1)
                    # Preserve class-pair geometry across the learned RX action.
                    relation_loss = ((zs[0]*zs[1]).sum()-(zt[0]*zt[1]).sum()).square()
                    terms.append(reliability*relation_loss)
                    if len(terms) >= int(self.recipe.get('receiver_relations_max', 8)):
                        break
                if len(terms) >= int(self.recipe.get('receiver_relations_max', 8)):
                    break
        self.last['receiver_relations'] = float(len(terms))
        self.last['receiver_identity_skipped_no_relation'] = float(not terms)
        return torch.stack(terms).mean() if terms else h.sum()*0.

    def loss(self, model, ema_model, x, y, receivers, epoch, step):
        self.last = dict(active=0., epoch=float(epoch), native_update=float(step),
            auxiliary_successful_updates=float(self.updates), source_L_only=1.,
            identity_aux_weight=float(self.recipe['identity_aux_weight']),
            paired_consistency_weight=float(self.recipe['paired_consistency_weight']),
            receiver_identity_weight=float(self.recipe['receiver_identity_weight']),
            cadence=float(self.recipe['auxiliary_every_steps']))
        if not self.paths or epoch <= self.recipe['warmup_epochs'] or step % self.recipe['auxiliary_every_steps']:
            return None
        if ema_model is None:
            raise RuntimeError('Multi disentanglement requires own current-run native EMA')
        if x.shape[0] != y.numel() or receivers.numel() != y.numel():
            raise ValueError('Auxiliary source-L IQ/label/RX alignment mismatch')
        if x.is_cuda: torch.cuda.synchronize(x.device)
        start = time.perf_counter()
        count = min(len(x), int(self.recipe['auxiliary_batch_size']))
        indices = torch.randperm(len(x), generator=self.generator)[:count].to(x.device)
        x, y, receivers = x[indices], y[indices], receivers[indices]
        identity, teacher = model.id_backbone, ema_model.id_backbone
        devices = [x.device.index] if x.is_cuda else []
        # Aux-only views never advance the native dropout/augmentation streams.
        with torch.random.fork_rng(devices=devices), deterministic_forward(identity):
            batch = factorial_batch(x, self.generator)
            needed = {'x00'}
            if any(k in self.paths for k in ('linear','interaction')): needed.add('x10')
            if any(k in self.paths for k in ('temporal','interaction')): needed.add('x01')
            if 'interaction' in self.paths: needed.add('x11')
            with torch.no_grad(), deterministic_forward(teacher):
                anchors = {k: intermediate(teacher, batch[k]).detach() for k in sorted(needed)}
                teacher_z = identity_from_intermediate(teacher, anchors['x00']).detach()
            current, pairs = {}, []
            if 'receiver' in self.paths:
                with torch.no_grad():
                    views = self._receiver_views(x)
                    current = self.statistics.update(y, receivers, views, anchors['x00'], step)
                    pairs = self.statistics.pairs(x.device, max(32, 4*int(self.recipe.get('receiver_relations_max',8))),current)
                self.last.update(receiver_groups=float(len(self.statistics.groups)), receiver_pairs=float(len(pairs)),
                    receiver_fit_skipped_no_groups=float(not pairs))
            if self.optimizer is not None:
                self._fit(batch, anchors, pairs, step)
            h = intermediate(identity, x)
            terms = []
            freeze = frozen_parameters(self.auxiliary) if self.auxiliary is not None else _empty_context()
            with freeze:
                for kind in self.paths:
                    if kind == 'receiver':
                        relational = self._receiver_identity(identity, h, current, pairs)
                        self.last['receiver_relational_loss'] = float(relational.detach())
                        if self.last.get('receiver_relations',0)>0:
                            self.identity_steps[kind]+=1
                            self.receiver_relations_total+=int(self.last['receiver_relations'])
                        terms.append(float(self.recipe['receiver_identity_weight'])*relational)
                        continue
                    target_key = {'linear':'x10','temporal':'x01','interaction':'x11'}[kind]
                    with torch.no_grad():
                        if self.arm == 'fixed_interventions':
                            delta = anchors[target_key]-anchors['x00']
                            reliability = delta.new_tensor(1.)
                        else:
                            predicted = self._predict(kind, batch, anchors['x00'])['delta']
                            truth_delta = (anchors['x11']-anchors['x10']-anchors['x01']+anchors['x00']
                                if kind == 'interaction' else anchors[target_key]-anchors['x00'])
                            reliability = (1+normalized_error(predicted, truth_delta)).reciprocal()
                            # Interaction is trained on factorial residual; its
                            # identity intervention uses the full composed action.
                            delta = predicted
                            if kind == 'interaction':
                                delta = (delta+self._predict('linear',batch,anchors['x00'])['delta']+
                                         self._predict('temporal',batch,anchors['x00'])['delta'])
                    moved = h+delta.detach()
                    moved_z = identity_from_intermediate(identity, moved)
                    moved_logits = classify_intermediate(identity, moved)
                    actual_logits = identity(batch[target_key])
                    ce = .5*(F.cross_entropy(moved_logits,y)+F.cross_entropy(actual_logits,y))
                    consistency = (1-F.cosine_similarity(moved_z,teacher_z,dim=1)).mean()
                    term = float(self.recipe['identity_aux_weight'])*reliability*(
                        ce+float(self.recipe['paired_consistency_weight'])*consistency)
                    terms.append(term)
                    self.identity_steps[kind]+=1
                    self.last.update({kind+'_identity_ce':float(ce.detach()),
                        kind+'_consistency_loss':float(consistency.detach()),
                        kind+'_reliability':float(reliability),
                        kind+'_delta_norm':float(delta.norm(dim=1).mean())})
            result = torch.stack(terms).sum()/len(self.paths)
            if not bool(torch.isfinite(result)):
                raise FloatingPointError('Nonfinite identity auxiliary objective')
        elapsed = time.perf_counter()-start
        self.training_seconds += elapsed
        self.last.update(active=1., identity_aux_loss=float(result.detach()),
            auxiliary_batch_size=float(count), active_paths=float(len(self.paths)),
            auxiliary_step_seconds=elapsed, receiver_samplewise_counterfactual=0.,
            ema_extra_packet_forwards=float(count*len(needed)),
            identity_extra_packet_forwards=float(count*(1+sum(k!='receiver' for k in self.paths))),
            actual_intervention_view_packets=float(count*sum(k!='receiver' for k in self.paths)),
            cuda_peak_allocated_bytes=float(torch.cuda.max_memory_allocated(x.device)) if x.is_cuda else 0.,
            cuda_peak_reserved_bytes=float(torch.cuda.max_memory_reserved(x.device)) if x.is_cuda else 0.)
        self.epoch_samples += 1
        for key in ('ema_extra_packet_forwards','identity_extra_packet_forwards','actual_intervention_view_packets'):
            self.exposure_totals[key]+=int(self.last[key])
        for key, value in self.last.items():
            self.epoch_totals[key] += float(value)
        return result

    def execution(self):
        return dict(fit_steps=dict(self.fit_steps),identity_steps=dict(self.identity_steps),
            receiver_relations_total=self.receiver_relations_total,
            exposure_totals=dict(self.exposure_totals),source_L_only=True,
            target_access=False)

    def checkpoint(self, epoch):
        return dict(schema='multi_disentangle_auxiliary_state_v1', epoch=int(epoch),
            config=self.config, auxiliary=None if self.auxiliary is None else self.auxiliary.state_dict(),
            optimizer=None if self.optimizer is None else self.optimizer.state_dict(),
            private_generator=self.generator.get_state(), statistics=self.statistics.state_dict(),
            auxiliary_successful_updates=self.updates, resume_supported=False,
            checkpoint_sources=[], teacher_origin='own native EMA from same scratch run',
            target_access=False, inference_uses_auxiliary=False, cost=self.cost,
            mechanism_execution=self.execution())

    def annotate(self, rows):
        if not rows:
            return
        epoch = int(rows[-1]['epoch'])
        values = dict(self.last)
        if self.epoch_samples:
            values.update({k: v/self.epoch_samples for k,v in self.epoch_totals.items()})
        values.update(auxiliary_successful_updates=self.updates,
            auxiliary_training_seconds=self.training_seconds,
            auxiliary_active_steps_epoch=self.epoch_samples,
            auxiliary_parameters=self.cost['auxiliary_parameters'],
            training_parameters=self.identity_parameters+self.cost['auxiliary_parameters'],
            inference_parameters=self.identity_parameters,
            metric_scope='active auxiliary steps in this epoch; no target observations')
        values.update({k+'_fit_steps_total':v for k,v in self.fit_steps.items()})
        values.update({k+'_identity_steps_total':v for k,v in self.identity_steps.items()})
        values['receiver_relations_total']=self.receiver_relations_total
        values.update({k+'_total':v for k,v in self.exposure_totals.items()})
        self.cost['mechanism_execution']=self.execution()
        rows[-1].update({'multi_'+k:v for k,v in values.items()})
        output = self.output_root
        output.mkdir(parents=True,exist_ok=True)
        torch.save(self.checkpoint(epoch),output/('auxiliary_final.pth' if epoch==200 else 'auxiliary_latest.pth'))
        (output/'auxiliary_cost.json').write_text(json.dumps(self.cost,indent=2),encoding='utf-8')
        resolved=output/'resolved_config.json'
        if resolved.is_file():
            data=json.loads(resolved.read_text(encoding='utf-8'))
            data['multi_disentangle_cost']=self.cost
            resolved.write_text(json.dumps(data,indent=2),encoding='utf-8')
        print('MULTI '+json.dumps(dict(values,epoch=epoch,arm=self.arm)),flush=True)
        self.epoch_totals.clear()
        self.epoch_samples=0


@contextmanager
def _empty_context():
    yield


def instrument_train(native, callback):
    """One checked insertion; fail closed if the upstream native loop changes."""
    original=native.train
    code=inspect.getsource(original)
    marker='            loss_is_finite = bool(torch.isfinite(loss.detach()).item())'
    if code.count(marker)!=1:
        raise RuntimeError('Native multi objective insertion point is not unique')
    insertion=(
        '            multi_loss = _multi_disentangle_step(model, ema_model, r3_x_l,\n'
        '                y_l[:labeled_clean_count], receiver_l_base, epoch, r3_optimizer_steps)\n'
        '            if multi_loss is not None:\n'
        '                loss = loss + multi_loss\n')
    # Use native globals so the production data/writer hooks and synthetic
    # diagnostics see exactly the same function bindings as original training.
    native.__dict__['_multi_disentangle_step']=callback
    exec(compile(code.replace(marker,insertion+marker),
        inspect.getsourcefile(original)+'::multi_disentangle','exec'),native.__dict__)
    return original


@contextmanager
def installed(c, training=True):
    if c['multi_recipe'].get('resume') or c.get('checkpoint_sources'):
        raise ValueError('Auxiliary resume/initialization is unsupported; scratch-only run')
    with native_installed(c) as native:
        if not training:
            yield native
            return
        build=native.build_baseline_model
        writer=native._write_ssdg_epoch_telemetry
        detach=native._detach_log_mapping
        state={}
        old_annotation=getattr(native,'_multi_disentangle_annotate',None)
        old_step=getattr(native,'_multi_disentangle_step',None)
        def multi_build(args,device):
            model=build(args,device)
            state['trainer']=MultiTrainer(c,model)
            # Native merge_checkpoint_args returns architecture-only arguments;
            # output_dir is normally absent there. The registered row remains
            # authoritative (synthetic checks explicitly replace its output).
            state['trainer'].output_root=Path(getattr(args,'output_dir',c['output_root']))
            output=state['trainer'].output_root
            if output.is_dir():
                (output/'auxiliary_cost.json').write_text(
                    json.dumps(state['trainer'].cost,indent=2),encoding='utf-8')
            print('MULTI_CONFIG '+json.dumps(dict(recipe=c['multi_recipe'],**state['trainer'].cost)),flush=True)
            return model
        def step(*args):
            return state['trainer'].loss(*args)
        def annotate(rows):
            if 'trainer' in state: state['trainer'].annotate(rows)
        def telemetry(cp,jp,rows):
            annotate(rows)
            return writer(cp,jp,rows)
        def step_metrics(values):
            if 'train/loss' in values and 'trainer' in state:
                values=dict(values,**{'train/multi_'+k:v for k,v in state['trainer'].last.items()})
            return detach(values)
        native.build_baseline_model=multi_build
        native._write_ssdg_epoch_telemetry=telemetry
        native._detach_log_mapping=step_metrics
        native._multi_disentangle_annotate=annotate
        original_train=instrument_train(native,step)
        try:
            yield native
        finally:
            native.train=original_train
            native.build_baseline_model=build
            native._write_ssdg_epoch_telemetry=writer
            native._detach_log_mapping=detach
            if old_annotation is None: del native._multi_disentangle_annotate
            else: native._multi_disentangle_annotate=old_annotation
            if old_step is None: del native._multi_disentangle_step
            else: native._multi_disentangle_step=old_step
