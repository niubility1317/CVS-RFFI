"""Random-input CPU check of the real full PAMF architecture; no data/checkpoint."""
import argparse
import json
from pathlib import Path


def run(context, frozen_root, full_step=False):
    import torch
    from experiments.cric_dura_comparison.bootstrap import activate
    roots = {name: str(Path(frozen_root) / name) for name in ('F', 'AF', 'BASE', 'FOLD', 'G')}
    activate({'frozen_sources': roots})
    from experiments.cric_dura_comparison import f_train, t1_model
    from experiments.cvs_phase1_stack.runtime import installed as native_scope
    from .plan import candidate_rows, configuration
    from .training_scope import installed
    from .objective import physical_directions
    torch.set_num_threads(2)
    c = configuration(json.loads(Path(context).read_text(encoding='utf-8-sig')),
                      candidate_rows()[0], '/synthetic', 'synthetic')
    cfg = dict(c, segment_epochs=10, epoch_offset=20, step_offset=4440, parent_source=c['shared_prefix'])
    holder = {}
    with installed(cfg, holder), f_train.class_dimension(cfg), native_scope(f_train.original_config(c['model_seed'])) as native:
        net = t1_model.install(native.build_baseline_model(f_train.arguments(cfg, 'cpu'), 'cpu'), cfg).cpu()
        net.train()
        x = torch.randn(4, 2, 256)
        directions, norms = physical_directions(net, x, 20261110)
        if set(directions) != {'time', 'frequency', 'pa', 'path', 'reference'}:
            raise ValueError('Full real architecture branches missing')
        if any(not bool(torch.isfinite(v).all()) for v in directions.values()):
            raise FloatingPointError('Physical direction contains nonfinite values')
        step_result = None
        if full_step:
            import numpy as np
            from torch.nn import functional as F
            from .source_diagnostic import as_view
            def batch(n, hidden):
                index = torch.arange(n)
                rx = torch.tensor([1, 3, 4, 6, 8])[index.remainder(5)]
                meta = dict(rx_i=rx, day_i=torch.ones(n, dtype=torch.long), eq_i=torch.ones(n, dtype=torch.long),
                            sig_i=index, base_index=index, branch_record_id=index,
                            tx_label_visible=torch.full((n,), not hidden, dtype=torch.bool))
                y = torch.full((n,), -1, dtype=torch.long) if hidden else index.div(5, rounding_mode='floor').remainder(6)
                return torch.randn(n, 2, 256), y, torch.zeros(n, dtype=torch.long), meta
            u = batch(256, True)
            net._cd_U_loader = [u]
            net._cd_U_iter = iter(net._cd_U_loader)
            x, y, domain, meta = batch(128, False)
            with net._phase1_labeled_feature_context():
                output = net(x, return_aux=True, domain_labels=domain)
            ce = F.cross_entropy(output['tx_logits'], y, label_smoothing=.01)
            correction, metrics = net._source_labeled_feature_hook(model=net, labeled_output=output,
                labels=y, extra=(None, meta), domains=domain, optimizer_step=1,
                identity_loss=ce, labeled_count=128, labeled_inputs=x)
            loss = ce + correction
            loss.backward()
            state = holder['objective'].state_dict()
            metadata = {k: np.asarray(v.tolist()) for k, v in meta.items()}
            as_input = np.asarray(x[:2].tolist(), dtype=np.float64)
            as_output = as_view(as_input, metadata, np.arange(2), 'practical_low_suburban', 'L_s')
            if (not bool(torch.isfinite(loss)) or metrics['T1_L_packets'] != 768
                    or metrics['physical_L_valid_anchors'] <= 0 or state['calls'] != 1
                    or state['seen'] != 256 or state['U_truth_read']
                    or state['physical_U']['selected_samples'] != 128
                    or as_output.shape != (2, 2, 256) or not np.isfinite(as_output).all()
                    or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in net.parameters())):
                raise ValueError('Actual architecture L/U synthetic step failed')
            step_result = dict(L_anchors=metrics['physical_L_valid_anchors'], U_seen=state['seen'],
                               U_accepted=state['accepted'], finite_backward=True)
        print(json.dumps(dict(status='PASS', input='random synthetic IQ; no data or checkpoint',
              full_deploy_parameters=sum(p.numel() for p in net.id_backbone.encoder.parameters()),
              direction_norm_mean={k: float(v.mean()) for k, v in norms.items()}, full_step=step_result)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', required=True)
    parser.add_argument('--frozen-root', required=True)
    parser.add_argument('--full-step', action='store_true')
    args = parser.parse_args()
    run(args.context, args.frozen_root, args.full_step)
