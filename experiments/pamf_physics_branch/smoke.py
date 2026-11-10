"""One real-E20, source-only backward check for both matched PTBR rows."""
from pathlib import Path
from experiments.cric_dura_comparison.io import read, write
from .plan import candidate_rows, configuration
from .source import checked_parent
from .training_scope import installed


def smoke(rt):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison import f_data, f_train, t1_model
    from experiments.cvs_phase1_stack.runtime import installed as native_scope
    from .source_diagnostic import as_view, direction_overlap, identity_basis
    context, checks = read(rt['context']), []
    for row in candidate_rows():
        c = configuration(context, row, rt['project'], rt['code_commit'])
        checked_parent(c)
        payload = torch.load(Path(c['shared_prefix']) / 'final_ssdg.pth', map_location='cpu', weights_only=False)
        cfg = dict(c, segment_epochs=10, epoch_offset=20, step_offset=4440, parent_source=c['shared_prefix'])
        torch.manual_seed(c['model_seed'])
        torch.cuda.reset_peak_memory_stats()
        holder = {}
        with installed(cfg, holder):
            loaders = f_data.context(cfg)
            with f_train.class_dimension(cfg), native_scope(f_train.original_config(c['model_seed'])) as native:
                net = t1_model.install(native.build_baseline_model(f_train.arguments(cfg, 'cuda:0'), 'cuda:0'), cfg).cuda()
            net.load_state_dict(payload['model'], strict=True)
            net.eval()
            with torch.no_grad():
                logits = net(loaders['train_loader'].dataset.x[:2].cuda())
            if logits.shape != (2, 6) or not bool(torch.isfinite(logits).all()):
                raise ValueError('Actual parent/source input failed')
            net.train()
            net._cd_U_loader = loaders['unlabeled_loader']
            net._cd_U_iter = iter(net._cd_U_loader)
            batch = next(iter(loaders['train_loader']))
            x, y, domain = [v.cuda() for v in batch[:3]]
            with net._phase1_labeled_feature_context():
                output = net(x, return_aux=True, domain_labels=domain)
            ce = F.cross_entropy(output['tx_logits'], y, label_smoothing=.01)
            correction, metrics = net._source_labeled_feature_hook(
                model=net, labeled_output=output, labels=y, extra=(None, batch[3]), domains=domain,
                optimizer_step=1, identity_loss=ce, labeled_count=128, labeled_inputs=x)
            loss = ce + correction
            loss.backward()
            state = holder['objective'].state_dict()
            head = net.id_backbone.encoder.core.id_backbone.cls_head.weight
            # Exercise the source-only As diagnostic with two real L samples;
            # this is wiring verification, not a representative source study.
            meta = {k: np.asarray(v.detach().cpu().tolist()) for k, v in batch[3].items() if torch.is_tensor(v)}
            probe = np.asarray(x[:2].detach().cpu().tolist(), dtype=np.float64)
            sample = as_view(probe, meta, np.arange(2), 'practical_low_suburban', 'L_s')
            basis, rank = identity_basis(torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([0, 1]), classes=2)
            fraction, valid = direction_overlap(torch.tensor([[1., -1.]]), basis)
            if (not bool(torch.isfinite(loss)) or head.grad is None or not bool(torch.isfinite(head.grad).all())
                    or float(head.grad.norm()) <= 0 or metrics['T1_L_packets'] != 768
                    or net._physical_L_calls != 1 or net._physical_L_samples != 64
                    or metrics['physical_L_valid_anchors'] <= 0 or metrics['physical_U_valid_anchors'] <= 0
                    or state['calls'] != 1 or state['seen'] != 256 or state['U_truth_read'] or state['EMA_updates']
                    or state['physical_U']['selected_samples'] != 128
                    or not 0 < state['physical_U']['selected_accepted'] <= 128
                    or sample.shape != (2, 2, 256) or not np.isfinite(sample).all()
                    or rank != 1 or not bool(valid.all()) or not bool(torch.isfinite(fraction).all())
                    or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in net.parameters())):
                raise ValueError('Physical L/U source integration or finite backward failed')
            correction_value = float(metrics['physical_L_distance_correction'])
            if row['beta'] == 0 and correction_value != 0.:
                raise ValueError('Matched beta-zero direction control must reduce to uncorrected distance')
            checks.append(dict(row=row['row_id'], finite_backward=True, L_physical_samples=64,
                               L_anchors=metrics['physical_L_valid_anchors'], U=state,
                               U_anchors=metrics['physical_U_valid_anchors'],
                               L_distance_correction=correction_value,
                               peak_allocated_bytes=torch.cuda.max_memory_allocated()))
            del net, head, loss, ce, correction, output, x, y, domain, logits, loaders, payload, holder
        torch.cuda.empty_cache()
    write(Path(rt['run_root']) / 'smoke.json', dict(status='PASS', real_checkpoint=True,
          query_reads=0, U_truth_reads=0, optimizer_updates=0, checks=checks))
