"""Original practical DAOT/RC4 recipe with source-only training and final selection."""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from SSDG import train_ssdg as native
from scripts.train_daot_rc4_baseline import resolved_config
from cvsrffi.xuc_fusion.native import role_ids_from_native
from cvsrffi.game_tracking.runtime import json_write


def build_args(config, dataset, output, source_contract=None, target_inputs='', target_truth='', run_id=''):
    if target_inputs or target_truth:
        raise ValueError('Source training cannot accept target paths')
    doc = json.loads(Path(config).read_text(encoding='utf-8'))
    opts = dict(doc['options'])
    opts.update({'--wisig_pkl': str(dataset), '--output_dir': str(output),
                 '--candidate_id': doc['id'], '--run_id': run_id})
    argv = []
    for key, value in opts.items():
        argv.append(key)
        if value is not None:
            argv.append(str(value))
    args = native.build_arg_parser().parse_args(argv)
    native._validate_a1_scratch_only(args)
    native._validate_daot_config(args)
    native._resolve_sat_training_mode(args)
    checks = {
        'scratch': args.from_scratch and args.a1_scratch_only and not args.baseline_ckpt and not args.teacher_ckpt and not args.rc4_use_anchor,
        'method': args.fasttrust_rc4 and args.use_adv3b02_daot_stn and not args.use_a1_r3,
        'daot': args.daot_ablation == 'A1' and args.daot_teacher_view_count == 2 and args.daot_student_scenario == 'practical_high',
        'source_only': args.a1_source_screen_only and args.a1_periodic_target_start == 0 and args.a1_periodic_target_interval == 0 and not args.a1_final_weak_reference and not args.a1_periodic_target_inputs and not args.a1_periodic_target_truth,
        'final': args.epochs == 200 and args.checkpoint_selection == 'final_only',
        'concat': args.concat_sat_fused_ce_only and args.sat_training_mode == 'concat_masked' and args.concat_sat_ce_only and args.lambda_sat_cons == 0 and args.lambda_sat_cls == .68,
        'channel': args.practical_route == 'residual' and not args.practical_equalization and args.rc4_satellite_family == 'practical',
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise ValueError('Invalid matched recipe: ' + ','.join(failed))
    return args


def main():
    p = argparse.ArgumentParser()
    for name in ('config', 'dataset', 'output', 'source-contract', 'run-id', 'commit'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--validate-only', action='store_true')
    a = p.parse_args()
    args = build_args(a.config, a.dataset, a.output, run_id=a.run_id)
    if a.validate_only:
        print(json.dumps(resolved_config(args), ensure_ascii=False))
        return
    output = Path(a.output)
    if output.exists():
        raise FileExistsError(output)
    import torch
    expected = json.loads(Path(a.source_contract).read_text(encoding='utf-8'))
    startup = dict(argv=sys.argv, parsed_launcher_args=vars(a), resolved_args=resolved_config(args),
                   commit=a.commit, python=sys.executable, python_version=platform.python_version(),
                   torch_version=torch.__version__, cuda_version=torch.version.cuda,
                   cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   pid=os.getpid(), cwd=os.getcwd(), timestamp=time.time(),
                   method_version='DAOT_A1_FastTrust_RC4_original_residual_noeq',
                   checkpoint_sources=[], selection='final_epoch_200')
    print('STARTUP ' + json.dumps(startup, ensure_ascii=False), flush=True)
    original = native._build_ssdg_wisig_data
    def checked(*values, **kwargs):
        ctx = original(*values, **kwargs)
        if ctx['named_test_loaders']:
            raise ValueError('Training constructed target loaders')
        if role_ids_from_native(ctx) != expected['role_ids']:
            raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: physical source roles')
        classes = ctx['class_id_to_tx']
        classes = [str(classes[i]) for i in range(len(classes))]
        json_write(output / 'source_contract.json', dict(expected, native_role_comparison='EXACT_MATCH', classes=classes))
        json_write(output / 'resolved_config.json', resolved_config(args))
        json_write(output / 'startup.json', startup)
        json_write(output / 'initialization.json', dict(scratch_only=True, checkpoint_sources=[],
                   target_training_contact=False, target_contact=False, source_roles='EXACT_MATCH',
                   ema_origin='this_run_student', daot=True, rc4=True, seed=args.seed))
        return ctx
    native._build_ssdg_wisig_data = checked
    start = time.time()
    code = native.train(args)
    if code:
        raise RuntimeError(code)
    payload = torch.load(output / 'final_ssdg.pth', map_location='cpu', weights_only=False)
    if payload['epoch'] != 200 or not payload['args']['from_scratch'] or payload['args']['baseline_ckpt']:
        raise ValueError('Invalid final checkpoint')
    json_write(output / 'completion.json', dict(status='TRAINING_COMPLETE', epochs=200,
               checkpoint='final_ssdg.pth', elapsed_seconds=time.time()-start, target_evaluated=False,
               commit=a.commit, method_version=startup['method_version']))


if __name__ == '__main__':
    main()
