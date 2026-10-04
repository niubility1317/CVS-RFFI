"""Materialize the user-fixed, four-seed original-mechanism factorial."""
import copy
from experiments.cvs_phase1_overlay.contract import *


def main():
    spec=copy.deepcopy(read(ROOT/'experiments/cvs_selected_concat/configs/experiment_spec.json'))
    runroot=PROJECT+'/runs/'+RUN
    owner='codex/root/reference-overlay-r1-20261004'
    spec.update(run_id=RUN,group_id='cvs-reference-original-phase1-overlay',
        display_name='reference_response 原 Phase1 机制叠加：LEO×MixStyle 四 seed 消融',
        description='固定用户指定 reference_response，从零进行 CE/LEO/MixStyle/LEO+MixStyle 的 2×2 消融；完整源域矩阵冻结后，全部固定行完成 clean 和卫星测试及独立评分。',
        authorization='用户：使用 reference_response 逐步探索加上原来的各种 CVS 方法机制；先给出计划；按你设计的，再细一点，发布实验。',
        aliases=[],tags=['cvs','reference_response','phase1','factorial','leo','mixstyle','scratch','benchmark_informed'],
        parent_run_ids=['20261002-phase1-cvs-reference-identity-manysig-m4-r01','20260927-phase1-cvs-daot-rc4-practical-manysig-m5-r01'],
        replaces_run_id=None,status='PLANNED',rows=[])
    spec['code'].update(checkout=str(ROOT),cwd=PROJECT+'/releases/'+RELEASE)
    spec['code'].pop('architecture_base_commit',None)
    spec['checkpoint'].update(initialization='scratch',sources=[],provenance_verdict='SCRATCH_NO_INHERITANCE',
        selection_rule='User-fixed reference_response; every row own epoch200. All16 fixed comparisons tested. Continuation arm chosen by fourseed mean of 0.5V+0.5worstRX before any new query; exact ties fixed arm order.')
    spec['permissions']['claim_scope']='BENCHMARK_INFORMED_FIXED_DESIGN: user-fixed architecture on exposed benchmark; no fresh blind architecture-selection claim. No test feedback into tuning, selection or selective rerun.'
    spec['data']['leo_config_ref']='experiments/cvs_phase1_overlay/contract.py:LEO; native ConcatSatChannelAugment and apply_practical'
    spec['execution'].update(launch_owner=owner,remote_run_root=runroot,remote_log_root=PROJECT+'/logs/'+RUN,
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_command='python -m experiments.cvs_phase1_overlay.publish --output local_artifacts/'+RELEASE,
        gpu_policy='Queue <=8 concurrent jobs, least occupied first, <=2 total jobs/GPU and >=12GB free; no healthy job intervention')
    spec['metrics_plan'].update(primary='Fourseed mean/sampleSD accuracy, Macro-F1, worstRX; seed-paired LEO/MixStyle simple effects and interaction in percentage points; all views/RX/TX and negative results retained',
        scorer_ref='comparison_suite.score:p1 then independent bincount recount; all16 predictions complete before truth',
        prediction_ref=runroot+'/<arm>-s<seed>/prediction/phase1_predictions.npz')
    spec['metrics_plan']['dimensions']=['arm','model_seed','view','receiver','transmitter']
    spec['test_completion_plan']=dict(views=['clean','satellite']+SCENES,query_count=168000,
        registered_classes=6,rows=16,capsule=CAPSULE,physical_id_index=CAPSULE+'/index.npz',truth=TRUTH,
        architecture_fixed_before_training=True,weight_selection='own_E200',target_feedback=False,
        target_tx_ids=CLASSES,target_rx_indices=[0,2,5,7,9,10,11],scene_partition='one satellite observation per ID; three disjoint scene strata')
    spec['notes']=[
        'BENCHMARK_INFORMED_FIXED_DESIGN. reference_response fixed explicitly by user; design parents are not checkpoint ancestors.',
        'L/U/V=6300/56700/27000; sourceRX1,3,4,6,8 days1,2,3. U unused. Exact existing physical split392005; no target training/validation.',
        'All16 scratch E200x50=10000 optimizer steps; B128 drop_lastFalse; AdamW2e-4 wd1e-4 cosine1e-6; FP32 TF32False; no clipping.',
        'Original MixStyle1D p.18 alpha.1 strength.70, time_down and t1; sameTXcrossRXday, missing eligible partner skip. No eval MixStyle.',
        'Native batch-level LEO sampler: E1-40 high p.30; E41-90 mid/urban p.60; E91-200 all3 p.80. Native practical residual/post_sync/noeq.',
        'LEO concatenation startsE1. CleanCE weight1 throughout; satelliteCE weight.68 onlyE80 onward. Consistency0. Early satellite forward can affect MixStyle pairing/RNG.',
        'Augmentation RNG2027 paired across modelseeds and LEO arms; receiverseed2027; source opaque IDs mapped to source:<ID>. Original sampler/physics retained, historical integer-ID random realization not claimed identical.',
        'No domain/PL/teacher/EMA/DAOT/RC4/prototype/other auxiliary loss in round1. Future stages require their own complete preregistration.',
        'All16 source rows must complete before source-only continuation freeze and any query. Four-arm comparisons are fixed and all are tested regardless of source rank.',
        'Existing VALIDATED_ONCE capsule reused; clean168000 and pairedsatellite168000, three scene partitions. Predictions precede independent truth scoring/recount.',
        'No Phase2 support/adaptation/newclass results: A/B/C,K,H not applicable. Native fullCVS uses different U-based iteration budget; no equal-compute historical fullCVS claim.',
        'Full measured stepJSONL and epoch text/compactJSONL/CSV; actual gradient/mechanism counts/sourceRX; actual epochtime/peak plus synthetic CE-only resource timings.',
        'Technical failure preserves all artifacts; no automatic retrain or low-score stop. Whole pipeline ends ANALYZED only after complete scoring and independent recount.'
    ]
    spec['expected_artifacts']=['initialization.json','initial_smoke.pt','resolved_config.json','source_contract.json','step_metrics.jsonl','epoch_metrics.jsonl','epoch_metrics.csv','last.pt','source_selection.json','resource_profile.json','source/completion.json','source_matrix_frozen.json','phase1_predictions.npz','phase1_complete.json','phase1_scored_results.json','phase1_summary.json','scoring_p1_complete.json','independent_recount.json','paired_effects.json','analysis.md','completion.json']
    runtime=dict(run_id=RUN,launch_owner=owner,runtime_root=runroot,log_root=spec['execution']['remote_log_root'],rows=[],p1_truth=TRUTH)
    for seed in SEEDS:
        for arm in ARMS:
            rid=arm+'-s'+str(seed);source=runroot+'/'+rid+'/source';pred=runroot+'/'+rid+'/prediction'
            srcref='experiments/cvs_phase1_overlay/configs/source-'+rid+'.json'
            prref='experiments/cvs_phase1_overlay/configs/predict-'+rid+'.json'
            src=dict(method='cvs_phase1_overlay',variant='reference_response',arm=arm,model_seed=seed,
                source_contract=SOURCE,dataset=PROJECT+'/Dataset_WigSig/ManySig.pkl',output_root=source,device='cuda:0',
                epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,weight_decay=.0001,split_seed=392005,
                augmentation_seed=2027 if 'leo' in arm else None,receiver_seed=2027 if 'leo' in arm else None,
                drop_last=False,domain_backbone=False,extra_losses=[],selection='fixed_last_epoch',
                numerical_policy=FULL_FP32_POLICY,mixstyle_recipe=MIXSTYLE,leo_recipe=LEO)
            validate_config(src)
            cfg=dict(method='cvs_phase1_overlay',variant='reference_response',arm=arm,model_seed=seed,
                source_output=source,output_root=pred,device='cuda:0',source_contract=SOURCE,p1_capsule=CAPSULE,numerical_policy=FULL_FP32_POLICY)
            write(ROOT/srcref,src);write(ROOT/prref,cfg)
            sc=PROJECT+'/releases/'+RELEASE+'/'+srcref;pc=PROJECT+'/releases/'+RELEASE+'/'+prref
            spec['rows'].append(dict(row_id=rid,method=arm,purpose='fixed_factorial_original_mechanism',gpu=None,
                config_ref=srcref,prediction_config_ref=prref,resolved_config_ref=source+'/resolved_config.json',data_overrides={},
                seeds=dict(model=seed,split=392005,data=None,augmentation=2027 if 'leo' in arm else None,support=None,evaluation=None),
                seed_notes='Model/loader/MixStyle RNG=modelseed; fixedphysicalsplit; nativeLEO RNG2027 and receiver2027; immutable evaluation observations.',
                k=None,scenario=','.join(['clean']+SCENES),optimizer='AdamW+cosine',lr=.0002,epochs=200,fl_rounds=None,
                budget_ref='E200x50=10000updates;6300L/epoch;LEO arms additional6300forward samples/epoch fromE1',
                output_root=source,prediction_root=pred,log_path=runtime['log_root']+'/source-'+rid+'.log',
                command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_phase1_overlay.source --config '+sc,
                expected_artifacts=spec['expected_artifacts']))
            runtime['rows'].append(dict(row_id=rid,method=arm,arm=arm,model_seed=seed,stage='phase12',source_output=source,source_config=sc,config=pc,output_root=pred))
    write(ROOT/'experiments/cvs_phase1_overlay/configs/experiment_spec.json',spec)
    write(ROOT/'experiments/cvs_phase1_overlay/configs/launch_spec.json',runtime)
    print(RUN)


if __name__=='__main__':main()
