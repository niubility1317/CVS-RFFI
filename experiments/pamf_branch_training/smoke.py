"""Exercise all four objectives on real source data and E20 checkpoints."""
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from .plan import candidate_rows, configuration
from .source import checked_parent
from .training_scope import installed


def smoke(rt):
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison import f_data, f_train, t1_model
    from experiments.cvs_phase1_stack.runtime import installed as native_scope

    context = read(rt['context'])
    checks = []
    for row in candidate_rows():
        c = configuration(context, row, rt['project'], rt['code_commit'])
        checked_parent(c)
        payload = torch.load(Path(c['shared_prefix']) / 'final_ssdg.pth', map_location='cpu', weights_only=False)
        cfg = dict(c, segment_epochs=10, epoch_offset=20, step_offset=4440,
                   parent_source=c['shared_prefix'])
        holder = {}
        torch.manual_seed(c['model_seed'])
        torch.cuda.reset_peak_memory_stats()
        with installed(cfg, holder, branch_u_enabled=bool(row['u_weight'])):
            loaders = f_data.context(cfg)
            if (loaders['unlabeled_loader'] is not None) != c['U_use']:
                raise ValueError('Unlabeled loader violates registered permissions')
            with f_train.class_dimension(cfg), native_scope(f_train.original_config(c['model_seed'])) as native:
                args = f_train.arguments(cfg, 'cuda:0')
                net = t1_model.install(native.build_baseline_model(args, 'cuda:0'), cfg).cuda()
            net.load_state_dict(payload['model'], strict=True)
            net.eval()
            with torch.no_grad():
                logits = net(loaders['train_loader'].dataset.x[:2].cuda())
            if logits.shape != (2, 6) or not bool(torch.isfinite(logits).all()):
                raise ValueError('Real E20 checkpoint cannot classify source input')
            net.train()
            net._cd_U_loader = loaders['unlabeled_loader']
            net._cd_U_iter = iter(net._cd_U_loader) if net._cd_U_loader else None
            batch = next(iter(loaders['train_loader']))
            x, y, domain = [v.cuda() for v in batch[:3]]
            with net._phase1_labeled_feature_context():
                output = net(x, return_aux=True, domain_labels=domain)
            ce = F.cross_entropy(output['tx_logits'], y, label_smoothing=.01)
            correction, metrics = net._source_labeled_feature_hook(
                model=net, labeled_output=output, labels=y, extra=(None, batch[3]),
                domains=domain, optimizer_step=1, identity_loss=ce,
                labeled_count=128, labeled_inputs=x)
            loss = ce + correction
            loss.backward()
            head = net.id_backbone.encoder.core.id_backbone.cls_head.weight
            if (not bool(torch.isfinite(loss)) or head.grad is None or not bool(torch.isfinite(head.grad).all())
                    or float(head.grad.norm()) <= 0 or metrics['T1_L_packets'] != 768
                    or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in net.parameters())):
                raise ValueError('Source backward/default head failed')
            if row['l_weight'] and (net._branch_l_calls != 1 or metrics['branch_L_cross_anchors'] <= 0):
                raise ValueError('L branch relation failed to execute')
            if c['U_use'] and net._cd_U_seen != 256:
                raise ValueError('Ordinary PL sample count differs')
            if row['u_weight']:
                state = holder['objective'].state_dict()
                if state['calls'] != 1 or state['seen'] != 256 or state['EMA_updates'] or state['U_truth_read']:
                    raise ValueError('U branch relation failed to execute')
            checks.append(dict(row=row['row_id'], finite_backward=True,
                               L_branch_calls=net._branch_l_calls if row['l_weight'] else 0,
                               U_seen=net._cd_U_seen, U_branch_calls=holder['objective'].calls if row['u_weight'] else 0,
                               peak_allocated_bytes=torch.cuda.max_memory_allocated()))
            del net, head, loss, ce, correction, output, x, y, domain, logits, loaders, payload, holder
        torch.cuda.empty_cache()
    write(Path(rt['run_root']) / 'smoke.json', dict(status='PASS', real_checkpoint=True,
          query_reads=0, U_truth_reads=0, optimizer_updates=0, checks=checks))
