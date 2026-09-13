"""Full pinned CORE90 objective with clean-only X/U in every field evaluation."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from cvsrffi.a1_ecrs_cross_rx import cross_rx_triplet_loss
from cvsrffi.cross_response.tensor_ops import normalized_identity_interaction_loss
from cvsrffi.cross_response.roles import mixstyle_allowed_mask
from cvsrffi.game_tracking.step_context import slice_batch
from cvsrffi.game_tracking.legacy.objective import labeled_terms


@contextmanager
def labeled_forward_context(model, plan, device, views=2):
    if plan is None:
        yield
        return
    allowed = mixstyle_allowed_mask(plan, views=views, device=device)
    saved = []
    try:
        for module in model.modules():
            if module.__class__.__name__ == 'MixStyle1D':
                saved.append((module, '_cross_response_allowed', getattr(module, '_cross_response_allowed', None)))
                module._cross_response_allowed = allowed
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                saved.append((module, 'training', module.training))
                module.training = False
        yield
    finally:
        for module, key, value in reversed(saved):
            setattr(module, key, value)


class FusionObjective:
    def __init__(self, model, args, proto, row, dr=None):
        self.model, self.args, self.proto, self.row = model, args, proto, row
        self.dr=dr

    def __call__(self, ctx):
        n = len(ctx.y)
        with labeled_forward_context(self.model, ctx.grid_plan, ctx.x.device):
            combined = self.model(torch.cat((ctx.x, ctx.satellite)), y_tx=torch.cat((ctx.y, ctx.y)),
                                  grl_lambda=1., return_aux=True, domain_labels=torch.cat((ctx.domain, ctx.domain)))
        ctx.forward_calls += 1
        out, sat = slice_batch(combined, 0, n, 2*n), slice_batch(combined, n, 2*n, 2*n)
        loss, terms = labeled_terms(out, ctx.y, ctx.domain, self.args, ctx.epoch,
                                    ctx.batch_index, dict(ctx.weights), self.proto)
        sat_ce = F.cross_entropy(sat['tx_logits'], ctx.y) if ctx.epoch >= self.args.sat_cons_start_epoch else out['tx_logits'].sum()*0
        loss = loss + ctx.weights['sat_cls'] * sat_ce
        terms['sat_cls'] = sat_ce
        if ctx.strong is not None and self.dr is None:
            strong = self.model(ctx.strong, return_aux=True)
            ctx.forward_calls += 1
            if ctx.strong_mask is None:
                agreement = strong['tx_logits'].detach().argmax(1) == ctx.pseudo if self.args.pseudo_strong_agreement else torch.ones_like(ctx.base_mask)
                ctx.strong_mask = (ctx.base_mask & agreement).detach()
            selected = ctx.strong_mask
            ce = F.cross_entropy(strong['tx_logits'][selected], ctx.pseudo[selected]) if selected.any() else strong['tx_logits'].sum()*0
            prob = strong['tx_logits'].softmax(1)
            ent = -(prob*prob.clamp_min(1e-8).log()).sum(1).mean()
            loss = loss + self.args.lambda_u*ce + self.args.lambda_ent*ent
            terms.update(unlabeled_ce=ce, unlabeled_entropy=ent)
            ctx.pseudo_selected = int(selected.sum())
        z = out['z_id']
        telemetry = dict(x_enabled=self.row['x_enabled'], u_enabled=self.row['u_enabled'],
                         legal_anchors=0, normalized_valid_blocks=0, normalized_unavailable_blocks=0)
        weighted = {}
        if self.dr is not None:
            extra,dr_terms=self.dr.objective(ctx,out)
            loss=loss+extra
            terms.update(dr_terms)
        if self.row['x_enabled']:
            x, count = cross_rx_triplet_loss(z, ctx.y, ctx.rx, ctx.day, torch.zeros_like(ctx.y), margin=self.row['x_margin'])
            weighted['x'] = self.row['lambda_x']*x
            loss = loss + weighted['x']
            terms['x_cross_rx'] = x
            telemetry['legal_anchors'] = count
        if self.row['u_enabled']:
            if ctx.grid_plan is None:
                raise ValueError('normalized interaction requires labeled complete grid')
            blocks = []
            for block in ctx.grid_plan.blocks:
                c = block.candidate
                zi = z[list(block.batch_positions)].reshape(len(c.tx_ids), len(c.rx_ids), c.k, -1)
                value, _ = normalized_identity_interaction_loss(zi.float(), min_norm=1e-8)
                if value is None:
                    telemetry['normalized_unavailable_blocks'] += 1
                else:
                    blocks.append(value)
            telemetry['normalized_valid_blocks'] = len(blocks)
            u = torch.stack(blocks).mean() if blocks else z.sum()*0
            weighted['u'] = self.row['lambda_u_interaction']*u
            loss = loss + weighted['u']
            terms['u_normalized'] = u
        ctx.field_components.append(sorted(terms))
        if ctx.origin_features is None:
            ctx.origin_features = z.detach().clone()
            ctx.origin_terms = {k:float(v.detach()) for k,v in terms.items()}
            if ctx.audit_gradients:
                grads = {}
                for key, value in weighted.items():
                    grad = torch.autograd.grad(value, z, retain_graph=True, allow_unused=True)[0] if value.requires_grad else None
                    telemetry[key+'_weighted_identity_grad_norm'] = 0. if grad is None else float(grad.detach().norm())
                    if grad is not None: grads[key] = grad.detach().flatten()
                if set(grads) == {'x','u'}:
                    telemetry['xu_gradient_cosine'] = float(F.cosine_similarity(grads['x'],grads['u'],dim=0))
            ctx.fusion_telemetry = telemetry
        return loss
