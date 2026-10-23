"""Train and freeze before reading a prediction configuration."""
import argparse
import gc
from pathlib import Path
from experiments.cric_dura_comparison.io import read, write


def smoke(rt):
    from experiments.cric_dura_comparison.bootstrap import activate, numerical_policy
    activate(rt)
    numerical_policy()
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison.f_data import context
    from experiments.cric_dura_comparison.f_train import arguments, original_config, class_dimension
    from experiments.cric_dura_comparison.frozen_inference import load_F
    from experiments.cvs_phase1_stack.runtime import installed as native_installed
    from .train import installed
    from experiments.cric_dura_comparison import t1_model
    device = 'cuda:0'
    first = read(rt['rows'][0]['train_config'])
    contract = read(first['source_contract'])
    reader, _ = load_F(rt['baseline_source'], contract, device)
    ctx = context(first)
    with torch.no_grad():
        x = ctx['train_loader'].dataset.x[:2].to(device)
        logits = reader.classify_features(reader.features(x))
        if logits.shape != (2, 6) or not torch.isfinite(logits).all():
            raise ValueError('Real-checkpoint no-query smoke failed')
    real_teacher_state = {k: v.detach().clone() for k, v in reader.state_dict().items()}
    del reader
    torch.cuda.empty_cache()
    checked = []
    for row in rt['rows'][:2]:
        c = dict(read(row['train_config']), step_offset=4440)
        holder = {}
        torch.manual_seed(c['model_seed'])
        with installed(c, holder), class_dimension(c), native_installed(original_config(c['model_seed'])) as native:
            a = arguments(c, device)
            a.input_len = 256
            if a.use_unlabeled or a.use_ema_teacher:
                raise ValueError('Native U loop must be disabled for the new adapter')
            model = t1_model.install(native.build_baseline_model(a, device), c).train()
            # Bounded smoke only: exercise nonzero U gradients without a 13-epoch warmup.
            # Real experiment initialization remains scratch, as train.py specifies.
            holder['objective'].teacher.load_state_dict(real_teacher_state, strict=True)
            model._cd_U_loader = ctx['unlabeled_loader']
            model._cd_U_iter = iter(ctx['unlabeled_loader'])
            batch = next(iter(ctx['train_loader']))
            x, y, domain = [v.to(device) for v in batch[:3]]
            meta = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch[3].items()}
            with model._phase1_labeled_feature_context():
                output = model(x, return_aux=True, domain_labels=domain)
            ce = F.cross_entropy(output['tx_logits'], y, label_smoothing=.01)
            correction, metrics = model._source_labeled_feature_hook(model=model, labeled_output=output,
                labels=y, extra=(None, meta), domains=domain, optimizer_step=1,
                identity_loss=ce, labeled_count=128, labeled_inputs=x)
            total = ce + correction
            total.backward()
            head = model.id_backbone.encoder.core.id_backbone.cls_head.weight
            if (not torch.isfinite(total) or head.grad is None or not torch.isfinite(head.grad).all()
                    or metrics['T1_L_packets'] != 768 or holder['objective'].seen != 256
                    or holder['objective'].effective <= 0 or float(metrics['SSL_loss']) <= 0):
                raise ValueError('Actual supervised/U/default-head smoke failed')
            holder['objective'].after_update(model)
            state = holder['objective'].state_dict()
            holder['objective'].load_state_dict(state)
            checked.append(dict(method=c['ssl_method'], L_forward_packets=metrics['T1_L_packets'],
                U_IQ_samples=state['seen'], physical_U_views=state['physical_seen'],
                EMA_updates=state['ema_updates'], U_truth_read=False, finite_backward=True))
        del holder, model, output, total, correction, ce, head, state
        gc.collect()
        torch.cuda.empty_cache()
    write(Path(rt['run_root']) / 'smoke.json', dict(status='PASS', real_checkpoint=rt['baseline_source'],
        target_reads=0, U_truth_reads=0, methods=checked, immediate_continuation=True))


def train_and_predict(row):
    c = read(row['train_config'])
    from experiments.cric_dura_comparison.bootstrap import activate, numerical_policy
    activate(c)
    numerical_policy()
    from .train import train
    train(c)
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    from .source_diagnostics import run as source_diagnostics
    source_diagnostics(c, Path(c['output_root']) / 'source_V_diagnostic.json')
    if c['ssl_method'] == 'PSM':
        for arm, baseline in row['baseline_source_configs'].items():
            source_diagnostics(baseline, Path(c['output_root']) / ('source_V_' + arm + '.json'))
    gc.collect()
    torch.cuda.empty_cache()
    # The source model, teacher and all three segments are already immutable here.
    from experiments.cric_dura_comparison.evaluation import predict
    predict(read(row['prediction_config']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('smoke', 'row', 'score'))
    p.add_argument('--config', required=True)
    a = p.parse_args()
    value = read(a.config)
    if a.action == 'smoke':
        smoke(value)
    elif a.action == 'row':
        train_and_predict(value)
    else:
        from experiments.cric_dura_comparison.evaluation import score
        score(value)


if __name__ == '__main__':
    main()
