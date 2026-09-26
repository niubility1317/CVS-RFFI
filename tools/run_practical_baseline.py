"""Run one predeclared source-only baseline, from scratch, into a fresh output."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'code')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    if cfg['method'] not in {'cvcnn_ce', 'riei_fd', 'drift', 'poster', 'radionet'}:
        raise ValueError('Unsupported method')
    options = cfg['options']
    for name in ('--source_only', '--use_source_ssl_split', '--practical_residual_noeq',
                 '--concat_sat_ce_only', '--use_concat_sat_channel_aug'):
        if name not in options or options[name] is not None:
            raise ValueError(f'Missing mandatory switch: {name}')
    if '--use_augmentation_consistency' in options or '--eval_sat_channel' in options:
        raise ValueError('No target evaluator or additional consistency route in this matrix')
    out = Path(options['--output_dir'])
    out.mkdir(parents=True, exist_ok=False)
    cfg['runtime'] = dict(pid=os.getpid(), cwd=str(ROOT), python=sys.executable,
                          visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    (out / 'resolved_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    (out / 'initialization.json').write_text(json.dumps(dict(
        scratch_only=True, checkpoint_sources=[], model_seed=int(options['--seed']),
        selection='fixed_final_epoch', target_contact=False)), encoding='utf-8')
    import torch
    torch.set_num_threads(2)
    module_name = 'cvcnn_ce' if cfg['method'] in {'poster', 'radionet'} else cfg['method']
    module = importlib.import_module(f'baselines.{module_name}.train_cvs')
    original = module.run_validation_gated_training

    def checked_training(**kw):
        if not kw.get('source_only') or kw['named_test_loaders'] or kw.get('extra_test_fn'):
            raise ValueError('Source-only runtime boundary violated')
        model = kw['model']
        # A real, newly initialized checkpoint is saved/reloaded. It is never a
        # historical training source; smoke reads only two labeled source IQs.
        smoke = out / 'initial_smoke.pt'
        torch.save({'model': model.state_dict(), 'scratch_only': True}, smoke)
        payload = torch.load(smoke, map_location=kw['device'], weights_only=False)
        model.load_state_dict(payload['model'], strict=True)
        model.eval()
        dataset = kw['train_loader'].dataset
        x = torch.stack([dataset[i]['iq'] for i in range(min(2, len(dataset)))]).to(kw['device'])
        from baselines.common.cvs_trainer import logits_from_output
        with torch.no_grad():
            scores = logits_from_output(model(x))
        if scores.shape != (len(x), 6) or not torch.isfinite(scores).all():
            raise ValueError('Initial checkpoint no-query smoke failed')
        print('[SMOKE] initial checkpoint roundtrip and source-only forward PASS', flush=True)
        return original(**kw)

    module.run_validation_gated_training = checked_training
    argv = []
    for name, value in options.items():
        argv.append(name)
        if value is not None:
            argv.append(str(value))
    sys.argv = [module.__file__, *argv]
    module.main()
    ckpt = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
    if ckpt['epoch'] != int(options['--epochs']):
        raise ValueError('Fixed-epoch budget was not completed')
    (out / 'completion.json').write_text(json.dumps(dict(status='SOURCE_TRAINED',
        epoch=ckpt['epoch'], checkpoint=str(out / 'last.pt'), target_evaluated=False)), encoding='utf-8')


if __name__ == '__main__':
    main()
