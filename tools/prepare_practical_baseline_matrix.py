"""Create the explicitly fixed baseline source-training matrix and registry spec."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = '20260927-phase1-baselines-practical-manysig-m5-r01'
REMOTE = '/home/szu2070436088/2510044040/CV-SincNet'
RELEASE = REMOTE + '/releases/baselines_practical_20260927_r01'
PYTHON = '/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
SEEDS = [392005, 2026092701, 2026092702, 2026092703, 2026092704]


def main():
    spec = json.loads((ROOT / 'baseline_registry_template_20260927.json').read_text(encoding='utf-8'))
    config_dir = ROOT / 'configs' / 'baselines_practical_20260927'
    config_dir.mkdir(parents=True, exist_ok=False)
    contract = REMOTE + '/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
    spec.update(run_id=RUN, group_id='phase1-external-practical-residual-matched',
                display_name='外部对比方法：统一残差信道、五模型种子源域训练', kind='comparison', stage='Phase1',
                description='8条方法路线，固定L/U/V与200轮预算；5个模型种子。392005是历史优化参考，其余4个在本次目标评分前固定。',
                authorization='2026-09-27用户要求执行其他对比方法；确认Phase1/2 residual_noeq，加入POSTER和RadioNet，并包含CVS历史优化种子。',
                tags=['baseline', 'practical', 'residual_noeq', 'poster', 'radionet', 'source_only'],
                comparison_group_id='manysig-six-source-rx13468-day123-luv076330-residual-noeq-v1')
    spec['code'] = dict(commit='release_git_commit_recorded_in_launch_manifest',
                        checkout=str(ROOT), environment=PYTHON, cwd=RELEASE)
    spec['data'].update(dataset=REMOTE + '/Dataset_WigSig/ManySig.pkl', version='existing_N607_ManySig_6tx_12rx_4day',
        representation='equalized=1; center256; unit packet RMS; fs=25000000Hz',
        contract_ref=contract, physical_ids_ref=contract + '#role_ids', label_map_ref='each row/source_contract.json#classes',
        source_receivers=[1,3,4,6,8], target_receivers=[0,2,5,7,9,10,11], source_days=[1,2,3], target_days=[0,1,2,3],
        tx_sets_ref='ManySig.tx_list: all6; actual physical class strings exported at runtime',
        roles={'L_s': .07, 'U_s': .63, 'V': .30}, train_ratio=.1,
        leo_config_ref='baselines/common/practical_source.py:PracticalResidualAugment')
    spec['permissions'].update(regime='source_only', query_use='not_loaded_in_this_run',
        claim_scope='Source training only; no target results. Historical optimized seed is exploratory, fresh seeds reported separately.')
    spec['checkpoint'].update(initialization='scratch', sources=[], contract_check_ref=contract,
        provenance_verdict='SCRATCH_NO_INHERITANCE', selection_rule='fixed final epoch200; source best also retained but not substituted')
    spec['execution'].update(host='N607', launch_owner='codex/root/practical-baselines-20260927',
        gpu_policy='one lane per GPU0..7; five seeds sequential per lane; never more than2 total training jobs/GPU',
        remote_run_root=f'{REMOTE}/runs/{RUN}', remote_log_root=f'{REMOTE}/logs/{RUN}',
        local_artifact_root=f'automation_reports/CV-SincNet/{RUN}',
        launch_command=f'{PYTHON} tools/launch_practical_baselines.py --spec automation_reports/CV-SincNet/{RUN}/experiment.json --detach --commit RELEASE_COMMIT',
        stop_rule='Stop affected lane on nonzero exit/missing final artifact/protocol violation; never stop for low accuracy; no automatic retry or checkpoint fallback.')
    spec['expected_artifacts'] = ['each row/initialization.json', 'each row/source_contract.json',
        'each row/resolved_config.json', 'each row/initial_smoke.pt', 'each row/last.pt',
        'each row/metrics.json', 'each row/completion.json', 'launch.json', 'state.json']
    spec['metrics_plan'].update(metric_names=['source_V_accuracy', 'source_training_loss', 'pseudo_coverage', 'wall_time'],
        dimensions=['method','model_seed','epoch'], prediction_ref=None, scorer_ref=None)
    spec['notes'] = ['Phase2 and final target prediction/scoring are separate dependent runs, not claimed complete here.',
        'User explicitly overrides legacy LEO_WEAK for this experiment; Phase2 support/query will use residual_noeq too.',
        'Fixed reference channel augmentation seed2027 and receiver_seed2027 for all methods. Sample gates are ID/epoch deterministic.',
        'POSTER/RadioNet use parity-tested PyTorch ports, author architecture/freeze scope. Source budget matched200 differs from native defaults.',
        'CE arms do not use U. PL arms use threshold.9, start150, weight1. RIEI PL updates both classifiers and feature extractor with disjoint optimizers.',
        'Prior target exploration exists; seed392005 is not a fresh blind confirmatory replicate. No per-method seed selection.',
        'data seed is null: source role allocation is deterministic contiguous contract. support/evaluation seeds null in source-only run.']
    methods = [(m, pl) for m in ['cvcnn_ce','riei_fd','drift'] for pl in [False,True]] + [('poster',False),('radionet',False)]
    spec['rows'] = []
    for gpu, (method, pl) in enumerate(methods):
        for seed in SEEDS:
            row_id = f'{method}-' + ('pl-' if pl else 'ce-') + f's{seed}'
            output = f'{REMOTE}/runs/{RUN}/{row_id}'
            options = {'--source_only':None, '--use_source_ssl_split':None, '--source_contract':contract,
                '--wisig_pkl':spec['data']['dataset'], '--wisig_equalized':'1', '--wisig_out_len':256,
                '--wisig_train_rxs':'1,3,4,6,8', '--wisig_test_rxs':'0,2,5,7,9,10,11',
                '--wisig_train_days':'1,2,3', '--wisig_test_days':'0,1,2,3', '--wisig_domain':'rx_day',
                '--wisig_split_seed':392005, '--wisig_labeled_ratio':.07, '--wisig_unlabeled_ratio':.63,
                '--wisig_source_val_ratio':.30, '--seed':seed, '--epochs':200, '--batch_size':128,
                '--eval_batch_size':256, '--num_workers':0, '--no_train_drop_last':None,
                '--device':'cuda:0', '--output_dir':output, '--practical_residual_noeq':None,
                '--practical_fs_hz':25000000, '--practical_receiver_seed':2027, '--sat_view_seed':2027,
                '--use_concat_sat_channel_aug':None, '--concat_sat_ce_only':None,
                '--concat_sat_start_epoch':80, '--lambda_sat_cls':.68, '--lambda_sat_cons':0,
                '--no_test_on_val_improve':None}
            if pl:
                options.update({'--use_pseudo_labels':None, '--pseudo_start_epoch':150,
                                '--pseudo_threshold':.90, '--pseudo_margin':0, '--lambda_pseudo':1})
            if method in {'poster','radionet'}:
                options.update({'--author_backbone':method, '--lr':.001})
            cfg = dict(method=method, row_id=row_id, gpu=gpu, model_seed=seed, options=options)
            config_ref = f'configs/baselines_practical_20260927/{row_id}.json'
            (ROOT / config_ref).write_text(json.dumps(cfg, indent=2), encoding='utf-8')
            spec['rows'].append(dict(row_id=row_id, method=method + ('+PL' if pl else '') + '+residual_noeq_CE',
                purpose='historical_seed_replication' if seed==392005 else 'predeclared_fresh_model_seed',
                gpu=gpu, config_ref=config_ref, resolved_config_ref=output+'/resolved_config.json',
                data_overrides={}, seeds=dict(model=seed, split=392005, data=None, augmentation=2027, support=None, evaluation=None),
                seed_notes='Data roles deterministic; support/evaluation not performed. Receiver hardware seed2027 shared.',
                k=None, scenario='practical_high,practical_mid,practical_low_urban; residual/post_sync/noeq',
                optimizer='AdamW+cosine' if method=='cvcnn_ce' else 'Adam',
                lr=.0002 if method=='cvcnn_ce' else (.001 if method in {'poster','radionet'} else .0001),
                epochs=200, fl_rounds=None, budget_ref=config_ref, output_root=output,
                log_path=f'{REMOTE}/logs/{RUN}/{row_id}.log',
                command=f'{PYTHON} tools/run_practical_baseline.py --config {config_ref}',
                expected_artifacts=['last.pt','metrics.json','completion.json']))
    path = ROOT / 'practical_baselines_spec_20260927.json'
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
