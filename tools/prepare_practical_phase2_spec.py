import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = '20260927-phase2-practical-data-manytx-s2026092705-r01'
REMOTE = '/home/szu2070436088/2510044040/CV-SincNet'


def main():
    coverage = json.loads((ROOT/'phase2_coverage_20260927.json').read_text(encoding='utf-8'))
    source, target = coverage['datasets']['ManySig'], coverage['datasets']['ManyTx']
    receivers = [source['rx'][i] for i in [0,2,5,7,9,10,11]]
    candidates = [c for c in target['tx'] if c not in source['tx'] and
                  all(sum(target['counts'][r][c])>=198 for r in receivers)]
    selected = sorted(candidates,key=lambda c:hashlib.sha256(f'2026092705/new-class/{c}'.encode()).hexdigest())[:20]
    if len(selected)!=20:
        raise ValueError('Insufficient common new classes')
    cfg = dict(output_root=REMOTE+'/runs/'+RUN,manytx_path=REMOTE+'/Dataset_WigSig/ManyTx.pkl',
        old_classes=source['tx'],new_classes=selected,source_receivers=[source['rx'][i] for i in [1,3,4,6,8]],
        target_receivers=receivers,fs_hz=25000000,data_seed=2026092705,augmentation_seed=2026092707,
        receiver_seed=2027,evaluation_seed=2026092706,support_seeds=[2026092711,2026092712,2026092713,2026092714,2026092715],
        scenarios=['practical_high','practical_mid','practical_low_urban'], shots=[1,5,10,20],new_counts=[0,2,5,10,20])
    config_ref='configs/phase2_practical_data_20260927.json'
    with (ROOT/config_ref).open('x',encoding='utf-8') as f:
        json.dump(cfg,f,indent=2)
    spec=json.loads((ROOT/'baseline_registry_template_20260927.json').read_text(encoding='utf-8'))
    spec.update(run_id=RUN,group_id='phase2-practical-residual-shared-data',display_name='Phase2残差信道共享数据：7RX、6旧类+20新类',
        description='只按物理元数据覆盖选类，不读取模型或性能；36support池+30query/scene/class/RX，单物理记录单观测。',
        kind='diagnostic',stage='Phase2-data-builder',authorization='用户2026-09-27确认所有Phase2 support/query统一residual_noeq，并要求执行域泛化/适应/注册对比。',
        tags=['phase2','data-builder','practical','residual_noeq'],comparison_group_id='manytx-6old20new-7targetrx-residual-noeq-v1')
    python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
    release=REMOTE+'/releases/phase2_practical_data_20260927_r01'
    spec['code']=dict(commit='release_git_commit_recorded_in_builder_launch',checkout=str(ROOT),environment=python,cwd=release)
    spec['data'].update(dataset=cfg['manytx_path'],version='existing_N607_ManyTx150tx18rx4day',
        representation='equalized1 IQ256 fs25MHz; residual_noeq added once',contract_ref=config_ref,
        physical_ids_ref=cfg['output_root']+'/score_only/truth.json', label_map_ref=config_ref,
        source_receivers=cfg['source_receivers'],target_receivers=receivers,
        source_days=[1,2,3],target_days=[0,1,2,3],tx_sets_ref=config_ref,
        roles={'L_s':None,'U_s':None,'V':None},capsule_id='assigned_by_builder_on_completion',
        split_id='2100 splits emitted by builder',validation_ref=cfg['output_root']+'/capsule/manifest.json',
        support_query_ref=cfg['output_root']+'/capsule/splits',leo_config_ref=config_ref)
    spec['checkpoint'].update(initialization='not_applicable_data_builder',sources=[],
        contract_check_ref=config_ref,provenance_verdict='NO_CHECKPOINT_READ',selection_rule='no model access')
    spec['permissions'].update(regime='builder_outside_predictor',query_use='builder splits; truth exported only to isolated scorer directory',
        claim_scope='Data generation/validation only. No model performance.')
    command=f'{python} tools/build_practical_phase2_data.py --config {config_ref}'
    spec['execution'].update(host='N607',launch_owner='codex/root/practical-phase2-builder-20260927',gpu_policy='CPU only; no training process',
        remote_run_root=cfg['output_root'],remote_log_root=release+'/builder.log',local_artifact_root=f'automation_reports/CV-SincNet/{RUN}',
        launch_command=command,stop_rule='Fail on overlap/missing records/schema mismatch/nonfinite IQ; preserve partial output; no overwrite/retry.')
    spec['rows']=[dict(row_id='shared-received',method='practical residual_noeq builder',purpose='shared data',
        config_ref=config_ref,resolved_config_ref=cfg['output_root']+'/builder_report.json',data_overrides={},
        seeds=dict(model=None,split=2026092705,data=2026092705,augmentation=2026092707,support=None,evaluation=2026092706),
        seed_notes='model null: no model; support null: this builder row emits all five split seeds2026092711..2026092715 explicitly listed in config; receiver hardware2027; K nested',k=cfg['shots'],scenario=cfg['scenarios'],
        optimizer=None,lr=None,epochs=None,fl_rounds=None,budget_ref=config_ref,output_root=cfg['output_root'],
        log_path=release+'/builder.log',command=command,expected_artifacts=['capsule/manifest.json','capsule/received.npz','capsule/splits','score_only/truth.json','builder_report.json'])]
    spec['expected_artifacts']=spec['rows'][0]['expected_artifacts']
    spec['metrics_plan'].update(metric_names=['physical_count','split_count','validation_checks'],dimensions=['RX','scene','class','K','support_draw'])
    spec['notes']=['Source RX physically disjoint from all target records, even though ManySig/ManyTx are related subsets.',
        'Old and new target IQ both come from ManyTx to avoid family confounding.',
        'Selected20 new classes from97 eligible classes by fixed hash ordering of identity strings; no scores inspected.',
        '36,036 observations; 2,100 splits. Per-scene query IDs fixed across K/support draws; no new channel draws across methods.',
        'Real checkpoint smoke not applicable: builder never loads checkpoint or runs predictor.',
        'Final data capsule inherits no prior LEO_WEAK payload or VALIDATED_ONCE claim. Validate new payload once.']
    with (ROOT/'practical_phase2_data_spec_20260927.json').open('x',encoding='utf-8') as f:
        json.dump(spec,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(new_classes=selected,eligible=len(candidates),splits=2100)))


if __name__=='__main__':
    main()
