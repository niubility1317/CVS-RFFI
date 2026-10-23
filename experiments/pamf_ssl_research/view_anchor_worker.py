"""Fixed-E30 continuation, real-parent source smoke, then frozen raw predictions."""
import argparse
import gc
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write


def smoke(rt, *, paired=False, u_cric=False, u_mixstyle=False, u_gradprotect=False, u_cmr=False, u_dualanchor=False):
    from experiments.cric_dura_comparison.bootstrap import activate, numerical_policy
    activate(rt)
    numerical_policy()
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison.f_data import context
    from experiments.cric_dura_comparison.f_train import arguments, original_config, class_dimension
    from experiments.cvs_phase1_stack.runtime import installed as native_installed
    from experiments.cric_dura_comparison import t1_model
    from .train import installed
    from .view_anchor import PARENT_ALGORITHM
    from .view_anchor_train import source_parents

    c = read(rt['rows'][0]['train_config'])
    if u_dualanchor:
        from .u_dualanchor import validate as validate_dual
        parents, _ = source_parents(c, validate_config=validate_dual)
    elif u_cmr:
        from .u_cmr import validate as validate_u_cmr
        parents, _ = source_parents(c, validate_config=validate_u_cmr)
    elif u_gradprotect:
        from .u_gradprotect import validate as validate_u_gradprotect
        parents, _ = source_parents(c, validate_config=validate_u_gradprotect)
    elif u_mixstyle:
        from .u_mixstyle import validate as validate_u_mixstyle
        parents, _ = source_parents(c, validate_config=validate_u_mixstyle)
    elif u_cric:
        from .u_cric import validate as validate_u_cric
        parents, _ = source_parents(c, validate_config=validate_u_cric)
    elif paired:
        from .paired_u import validate as validate_pair
        parents, _ = source_parents(c, validate_config=validate_pair)
    else:
        parents, _ = source_parents(c)
    parent = parents[-1]['folder']
    cfg = dict(c, segment_epochs=20, epoch_offset=30, step_offset=6660, parent_source=parent)
    if u_cric:
        cfg['u_cric_smoke'] = True
    if u_mixstyle:
        cfg['u_mixstyle_smoke'] = True  # Cover both hooks; formal recipe retains p=.5.
    if u_gradprotect:
        cfg['u_gradprotect_smoke'] = True
    if u_cmr:
        cfg['u_cmr_smoke'] = True
    if u_dualanchor:
        cfg['u_dualanchor_smoke'] = True
    payload = torch.load(Path(parent) / 'final_ssdg.pth', map_location='cpu', weights_only=False)
    device = 'cuda:0'
    ctx = context(c)
    holder = {}
    torch.manual_seed(c['model_seed'])
    with installed(cfg, holder, parent_algorithm=PARENT_ALGORITHM), class_dimension(cfg), native_installed(original_config(c['model_seed'])) as native:
        a = arguments(cfg, device)
        a.input_len = 256
        a.baseline_ckpt = a.teacher_ckpt = ''  # Build architecture; load the validated payload below.
        a.from_scratch = a.a1_scratch_only = True
        if a.use_unlabeled or a.use_ema_teacher:
            raise ValueError('Native U loop must remain disabled')
        model = t1_model.install(native.build_baseline_model(a, device), cfg).to(device)
        model.load_state_dict(payload['model'], strict=True)
        model.eval()
        with torch.no_grad():
            logits = model(ctx['train_loader'].dataset.x[:2].to(device))
            if logits.shape != (2, 6) or not bool(torch.isfinite(logits).all()):
                raise ValueError('Real E30 checkpoint no-query inference failed')
        model.train()
        model._cd_U_loader = ctx['unlabeled_loader']
        model._cd_U_iter = iter(ctx['unlabeled_loader'])
        batch = next(iter(ctx['train_loader']))
        x, y, domain = [value.to(device) for value in batch[:3]]
        meta = {key:value.to(device) if torch.is_tensor(value) else value for key,value in batch[3].items()}
        with model._phase1_labeled_feature_context():
            output = model(x, return_aux=True, domain_labels=domain)
        ce = F.cross_entropy(output['tx_logits'], y, label_smoothing=.01)
        correction, metrics = model._source_labeled_feature_hook(model=model, labeled_output=output,
            labels=y, extra=(None, meta), domains=domain, optimizer_step=1,
            identity_loss=ce, labeled_count=128, labeled_inputs=x)
        total = ce + correction
        total.backward()
        objective = holder['objective']
        head = model.id_backbone.encoder.core.id_backbone.cls_head.weight
        if u_dualanchor:
            expected = objective._smoke_expected
            gradient_matches = all((p.grad is None if expected[n] is None else
                p.grad is not None and torch.allclose(p.grad,expected[n],atol=1e-5,rtol=1e-5))
                for n,p in model.named_parameters() if p.requires_grad)
            constraints = all(v['applied_U_norm']<=v['full_L_norm']*(1+1e-6)+1e-8
                and v['full_dot_after']>=-1e-6*max(1.,v['full_L_norm']*v['applied_U_norm'])
                and v['raw_dot_after']>=-1e-6*max(1.,v['raw_L_norm']*v['applied_U_norm'])
                for v in objective._smoke_detail.values())
            mechanism_ok = (objective.u_dualanchor_calls==objective.u_dualanchor_raw_calls==1
                and objective.pending_gradient_loss is None and objective.pending_raw_anchor is None
                and gradient_matches and constraints and objective.physical_seen==482078
                and objective._smoke_raw_value>0 and all(v['raw_L_norm']>0 for v in objective._smoke_detail.values()))
        elif u_cmr:
            mechanism_ok = (objective.u_cmr_calls==1 and objective.u_cmr_seen==256
                and objective.u_cmr_eligible>0 and not objective._raw_context_active
                and all(count>0 for count in objective.u_cmr_pairs.values())
                and all(h.grad is not None and bool(torch.isfinite(h.grad).all()) and float(h.grad.norm())>0
                        for h in objective._smoke_maps.values()))
        elif u_gradprotect:
            expected = objective._smoke_expected
            gradient_matches = all((p.grad is None if expected[n] is None else
                p.grad is not None and torch.allclose(p.grad,expected[n],atol=1e-5,rtol=1e-5))
                for n,p in model.named_parameters() if p.requires_grad)
            constraints = all(v['applied_U_norm']<=v['L_norm']*(1+1e-6)+1e-8
                and v['dot_after']>=-1e-6*max(1.,v['L_norm']*v['applied_U_norm'])
                for v in objective._smoke_detail.values())
            mechanism_ok = (objective.u_gradprotect_calls==1 and objective.pending_gradient_loss is None
                and gradient_matches and constraints and objective.physical_seen==482078)
        elif u_mixstyle:
            mechanism_ok = (objective.u_mixstyle_calls == 1 and objective.u_mixstyle_seen == 256
                            and objective.u_mixstyle_eligible > 0 and objective.physical_seen == 482078
                            and all(count == objective.u_mixstyle_eligible for count in objective.u_mixstyle_applied.values())
                            and not objective._style_context_active
                            and all(h.grad is not None and bool(torch.isfinite(h.grad).all())
                                    and float(h.grad.norm()) > 0 for h in objective._smoke_maps.values()))
        elif u_cric:
            mechanism_ok = (float(metrics['SSL_U_CRIC_loss']) > 0 and objective.u_cric_calls == 1
                            and objective.u_cric_seen == 256 and objective.physical_seen == 482078
                            and all(count > 0 for count in objective.u_cric_anchors.values())
                            and all(h.grad is not None and bool(torch.isfinite(h.grad).all())
                                    and float(h.grad.norm()) > 0 for h in objective._smoke_maps.values()))
        elif paired:
            mechanism_ok = (float(metrics['SSL_pair_physical_feature']) > 0
                            and objective.paired_calls == 1 and objective.paired_clean_seen == 256
                            and objective.paired_physical_seen == 256 and objective.student_U_views == 964412
                            and objective.physical_seen == 482206)
        else:
            mechanism_ok = (float(metrics['SSL_view_anchor_loss']) > 0 and objective.view_anchor_calls == 1
                            and objective.view_anchor_physical_seen == 128)
        if (not bool(torch.isfinite(total)) or head.grad is None or not bool(torch.isfinite(head.grad).all())
                or metrics['T1_L_packets'] != 768 or not mechanism_ok
                or objective.seen != 964156
                or any(parameter.grad is not None for parameter in objective.teacher.parameters())):
            raise ValueError('Real E30 student/own-EMA physical-view backward failed')
        objective.after_update(model)
        state = objective.state_dict()
        objective.load_state_dict(state)
        if objective.ema_updates != 6661:
            raise ValueError('Own E30 teacher state did not continue')
        checked = dict(parent_epoch=30, parent_updates=6660, EMA_updates_after_smoke=6661,
            U_seen=objective.seen, physical_U_seen=objective.physical_seen,
            L_forward_packets=768, finite_backward=True, teacher_gradient=False)
        if u_dualanchor:
            checked.update(u_dualanchor_calls=objective.u_dualanchor_calls,raw_L_anchor_calls=objective.u_dualanchor_raw_calls,
                raw_L_CE=objective._smoke_raw_value,protected_gradient_matches=gradient_matches,
                both_anchor_constraints=constraints,group_details=objective._smoke_detail,no_extra_student_forward=True)
        elif u_cmr:
            checked.update(u_cmr_calls=objective.u_cmr_calls,extra_raw_U_seen=objective.u_cmr_seen,
                moment_eligible=objective.u_cmr_eligible,moment_pairs=objective.u_cmr_pairs,
                shallow_middle_gradients=True,feature_hooks_removed=not objective._raw_context_active,
                extra_student_forward=True)
        elif u_gradprotect:
            checked.update(u_gradprotect_calls=objective.u_gradprotect_calls,
                protected_gradient_matches=gradient_matches,group_constraints=constraints,
                group_details=objective._smoke_detail,no_extra_student_forward=True)
        elif u_mixstyle:
            checked.update(u_mixstyle_seen=256,u_mixstyle_eligible=objective.u_mixstyle_eligible,
                u_mixstyle_applied=objective.u_mixstyle_applied,shallow_middle_gradients=True,
                source_smoke_force_both_hooks=True,formal_probability_per_level=.5,feature_hooks_removed=True)
        elif u_cric:
            checked.update(u_cric_loss=float(metrics['SSL_U_CRIC_loss']), u_cric_seen=256,
                u_cric_anchors=objective.u_cric_anchors, shallow_middle_gradients=True, feature_hooks_removed=True)
        elif paired:
            checked.update(student_U_views=objective.student_U_views, paired_clean_seen=256,
                           paired_physical_seen=256, physical_feature_loss=float(metrics['SSL_pair_physical_feature']))
        else:
            checked.update(view_anchor_loss=float(metrics['SSL_view_anchor_loss']), view_anchor_physical_seen=128)
    write(Path(rt['run_root']) / 'smoke.json', dict(status='PASS', real_checkpoint=parent + '/final_ssdg.pth',
        target_reads=0, U_truth_reads=0, method='UDAP' if u_dualanchor else 'UCMR' if u_cmr else 'UGP' if u_gradprotect else 'UMIX' if u_mixstyle else 'UCRIC' if u_cric else 'PAIREDU' if paired else 'VIEWANCHOR',
        checked=checked, immediate_continuation=True))


def train_and_predict(row):
    c = read(row['train_config'])
    from experiments.cric_dura_comparison.bootstrap import activate, numerical_policy
    activate(c)
    numerical_policy()
    from .view_anchor_train import train
    train(c)
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    from .source_diagnostics import run as source_diagnostics
    source_diagnostics(c, Path(c['output_root']) / 'source_V_diagnostic.json')
    gc.collect()
    torch.cuda.empty_cache()
    from experiments.cric_dura_comparison.evaluation import predict
    predict(read(row['prediction_config']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('smoke', 'row', 'score'))
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    value = read(args.config)
    if args.action == 'smoke':
        smoke(value)
    elif args.action == 'row':
        train_and_predict(value)
    else:
        from experiments.cric_dura_comparison.evaluation import score
        score(value)


if __name__ == '__main__':
    main()
