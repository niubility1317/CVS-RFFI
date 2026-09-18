"""Single registered scratch DAOT+RC4 experiment; no inherited teacher."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from SSDG import train_ssdg as native
from scripts.train_daot_rc4_baseline import resolved_config
from cvsrffi.xuc_fusion.native import role_ids_from_native
from cvsrffi.game_tracking.runtime import json_write


def build_args(config, dataset, output, source_contract, target_inputs, target_truth, run_id):
    doc = json.loads(Path(config).read_text(encoding='utf-8'))
    opts = dict(doc['options'])
    opts.update({'--wisig_pkl':str(dataset), '--output_dir':str(output), '--candidate_id':doc['id'],
        '--run_id':run_id, '--a1_periodic_target_inputs':str(target_inputs), '--a1_periodic_target_truth':str(target_truth)})
    argv=[]
    for key,value in opts.items():
        argv.append(key)
        if value is not None: argv.append(str(value))
    args = native.build_arg_parser().parse_args(argv)
    native._validate_a1_scratch_only(args)
    native._validate_daot_config(args)
    native._resolve_sat_training_mode(args)
    assert args.from_scratch and not args.baseline_ckpt and not args.teacher_ckpt and not args.rc4_use_anchor
    assert args.fasttrust_rc4 and args.use_adv3b02_daot_stn and not args.use_a1_r3
    assert args.daot_ablation == 'A1' and args.daot_teacher_view_count == 2
    assert args.daot_student_scenario == 'clear_leo' and '_weak' not in args.daot_hard_scenarios
    assert args.a1_source_screen_only and args.epochs == 200 and args.concat_sat_fused_ce_only
    assert args.sat_training_mode == 'concat_masked' and args.concat_sat_ce_only
    assert args.rc4_satellite_family == 'original_leo'
    assert args.lambda_sat_cons == 0 and args.lambda_sat_cls == 0.68
    assert args.a1_periodic_target_start == 100 and args.a1_periodic_target_interval == 10
    assert '_weak' not in args.sat_train_scenarios and '_weak' not in args.sat_view_schedule
    return args


def main():
    p=argparse.ArgumentParser()
    for name in ('config','dataset','output','source-contract','target-inputs','target-truth','run-id'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--validate-only', action='store_true')
    a=p.parse_args()
    args=build_args(a.config,a.dataset,a.output,a.source_contract,a.target_inputs,a.target_truth,a.run_id)
    if a.validate_only:
        print(json.dumps(resolved_config(args),ensure_ascii=False));return
    output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    expected=json.loads(Path(a.source_contract).read_text(encoding='utf-8'))
    original=native._build_ssdg_wisig_data
    def checked(*values,**kwargs):
        ctx=original(*values,**kwargs)
        if ctx['named_test_loaders']:raise ValueError('Training must not construct target loaders')
        if role_ids_from_native(ctx)!=expected['role_ids']:raise ValueError('Physical source roles mismatch')
        output.mkdir(parents=True,exist_ok=True)
        json_write(output/'source_contract.json',dict(expected,native_role_comparison='EXACT_MATCH'))
        json_write(output/'resolved_config.json',resolved_config(args))
        json_write(output/'initialization.json',dict(scratch_only=True,checkpoint_sources=[],target_training_contact=False,
            source_roles='EXACT_MATCH',ema_origin='this_run_student',daot=True,rc4=True))
        return ctx
    native._build_ssdg_wisig_data=checked
    code=native.train(args)
    if code:raise RuntimeError(code)
    for epoch in range(100,201,10):
        if not (output/'target_epochs'/f'E{epoch:03d}'/'evaluation_scope.json').is_file():
            raise ValueError(f'Missing fixed evaluation E{epoch}')
    if not (output/'final_weak_reference'/'evaluation_scope.json').is_file():
        raise ValueError('Missing final weak reference')
    json_write(output/'completion.json',dict(status='ARTIFACTS_COMPLETE',epochs=200,
        periodic_epochs=list(range(100,201,10)),scenarios=['clean','clear_leo','low_elev_leo','rain_leo'],
        final_weak_reference=True,feeds_training=False))

if __name__=='__main__':main()
