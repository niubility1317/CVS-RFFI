"""Source-only true-label physical continuation and frozen seven-scene evaluation."""
import argparse
from functools import partial
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_physics_branch import runner as physical
from experiments.pamf_physics_branch.tri_identity_scope import installed
from experiments.pamf_physics_route import prepare as routes
from experiments.pamf_physics_route import runner as dispatching
from .plan import candidate_rows, configuration
from .source import parent, train


def smoke(rt, *, rows=candidate_rows, configure=configuration, check_parent=parent, scope=installed,
          ssl=True, check_pairing=False):
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison import f_train, t1_model
    from experiments.cvs_phase1_stack.runtime import installed as native_scope
    from experiments.pamf_repl_view_shuffle.source import assert_structure

    ctx, checks = read(rt['context']), []
    for row in rows():
        c = configure(ctx, row, rt['project'], rt['code_commit'])
        seed = c['model_seed']
        prefix = check_parent(c)
        cp = torch.load(Path(prefix['folder']) / 'final_ssdg.pth', map_location='cpu', weights_only=False)
        cfg = dict(c, epoch_offset=20, step_offset=4440, segment_epochs=10, hard_pl_smoke=True)
        holder = {}
        torch.manual_seed(seed)
        torch.cuda.reset_peak_memory_stats()
        with scope(cfg, holder):
            loaders = f_train.context(cfg)
            with f_train.class_dimension(cfg), native_scope(f_train.original_config(seed)) as native:
                a = f_train.arguments(cfg, 'cuda:0')
                a.input_len = 256
                net = t1_model.install(native.build_baseline_model(a, 'cuda:0'), cfg).cuda()
            graph = assert_structure(net, cfg) if ssl else dict(view_count=3, parameters=232993, order=list(range(8)))
            net.load_state_dict(cp['model'], strict=True)
            net.train()
            if ssl:
                net._cd_U_loader = loaders['unlabeled_loader']
                net._cd_U_iter = iter(net._cd_U_loader)
            elif loaders['unlabeled_loader'] is not None:
                raise ValueError('Supervised source must not construct a U loader')
            batch = next(iter(loaders['train_loader']))
            x, y, domain = [v.cuda() for v in batch[:3]]
            metadata = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch[3].items()}
            with net._phase1_labeled_feature_context():
                output = net(x, return_aux=True, domain_labels=domain)
            ce = F.cross_entropy(output['tx_logits'], y, label_smoothing=.01)
            correction, metrics = net._source_labeled_feature_hook(
                model=net, labeled_output=output, labels=y, extra=(None, metadata), domains=domain,
                optimizer_step=1, identity_loss=ce, labeled_count=128, labeled_inputs=x)
            loss = ce + correction
            loss.backward()
            state = holder['objective'].state_dict() if ssl else dict(seen=0, student_views=0)
            if ssl and (graph['order'] != [0, 5, 3, 6, 2, 1, 4, 7] or state['calls'] != 1
                        or state['seen'] != 256 or state['student_views'] != 1536
                        or state['U_gradient_guard']['calls'] != 1 or state['U_truth_read'] or state['EMA_updates']):
                raise ValueError('Complete original Tri-SIG-TIME U smoke differs')
            if (a.seed != seed or a.use_unlabeled or a.use_ema_teacher or not bool(torch.isfinite(loss))
                    or (not ssl and net._cd_U_seen)
                    or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in net.parameters())):
                raise ValueError('Finite source smoke with exact stage options required')
            if check_pairing:
                pair = state['U_branch_pairing']
                if (pair['calls'] != 1 or pair['extra_eval_forwards'] != 2 or pair['probed'] <= 0
                        or pair['changed_vs_random'] <= 0 or (pair['beta'] > 0 and pair['mean_eligible_discount'] <= 0)):
                    raise ValueError('Actual U branch probes and eligible partner replacements required')
                details = dict(U_branch_pairing=pair)
            else:
                if net._physical_L_calls != 1 or net._physical_L_samples != 64 or metrics['physical_L_valid_anchors'] <= 0:
                    raise ValueError('Physical L smoke did not execute the registered relation')
                details = dict(L_anchors=metrics['physical_L_valid_anchors'])
            checks.append(dict(row=row['row_id'], graph=graph, finite_backward=True,
                               U_seen=state['seen'], U_views=state['student_views'],
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(), **details))
            del net, loss, correction, ce, output, loaders, cp, holder
        torch.cuda.empty_cache()
    write(Path(rt['run_root']) / 'smoke.json', dict(status='PASS', checks=checks, real_checkpoint=True,
          optimizer_updates=0, query_reads=0, U_truth_reads=0))


def worker(c, kind):
    if kind in ('predict', 'score'):
        return physical.worker(c, kind)
    if kind not in ('train', 'smoke'):
        raise ValueError('Only source and practical evaluation stages are allowed')
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    return train(c) if kind == 'train' else smoke(c)


def dispatch(rt):
    prepare = partial(routes.prepare, rows_for_run=lambda _: candidate_rows(), configure=configuration,
                      source_diagnostic=False, tri_dura=True)
    launch = partial(dispatching.launch, module='experiments.pamf_physics_tri.runner')
    dispatching.dispatch(rt, prepare_run=prepare, launch_worker=launch)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['dispatch', 'worker'])
    p.add_argument('--kind')
    p.add_argument('--config', required=True)
    a = p.parse_args()
    dispatch(read(a.config)) if a.action == 'dispatch' else worker(read(a.config), a.kind)
