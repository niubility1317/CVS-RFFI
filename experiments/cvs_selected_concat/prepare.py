"""Four fixed seeds; one source-fixed architecture; same practical concat recipe."""
import copy
import json
from pathlib import Path
from experiments.cvs_selected_concat.model import VARIANT
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY
from experiments.cvs_selected_concat.augmentation import RECIPE
from experiments.cvs_identity_ce.prepare import PROJECT,write
ROOT=Path(__file__).resolve().parents[2]
RUN='20261002-phase1-cvs-selected-concat-manysig-m4-r01'
RELEASE='cvs_selected_concat_20261002_r01'
SEEDS=[2026092701,2026092702,2026092703,2026092704]

def main():
    spec=copy.deepcopy(json.loads((ROOT/'experiments/cvs_identity_ce/configs/experiment_spec.json').read_text(encoding='utf-8')))
    spec.update(run_id=RUN,group_id='cvs-fixed-residual-midurban-concat-ce',display_name='CVS 已探索测试较优身份网络：practical 拼接星地增强',
        description='按用户测试性能优先的补充，固定既有clean测试较优的residual_fusion结构；从零训练，仅mid/low urban训练增强，clean+sat拼接，普通身份CE，无域骨干；最终性能由测试集报告。',
        authorization='用户2026-10-02：使用探索过最好的神经网络，使用拼接星地信道增强跑实验；用户补充：这次只使用mid low urban环境增强，而且具体性能看测试集不是源域；按已说明的residual_fusion固定网络和matched practical增强口径执行，发布后默认测试。',
        aliases=[],tags=['cvs','identity_only','ce_only','concat','practical','residual_noeq','user_fixed_architecture','benchmark_informed','performance_priority','scratch'],
        parent_run_ids=['20261001-phase1-cvs-residual-identity-manysig-m8-r01'],replaces_run_id=None,status='PLANNED',rows=[])
    spec['code'].update(checkout=str(ROOT),cwd=PROJECT+'/releases/'+RELEASE,architecture_base_commit='130d6e743d7d0747e8befb8287dc1d62af9e4c9f')
    spec['checkpoint'].update(initialization='scratch',sources=[],provenance_verdict='SCRATCH_NO_INHERITANCE',selection_rule='Architecture fixed by user-requested best tested network on exposed benchmark; not source-only or fresh blind architecture selection; each own E200 final checkpoint. No historical weight load.')
    spec['permissions'].update(claim_scope='BENCHMARK_INFORMED_FIXED_DESIGN: user-requested fixed residual_fusion on exposed benchmark; historical test performance informs design, so no target-free architecture-selection or fresh blind-test claim. No feedback from this run for tuning, seed exclusion, checkpoint selection or selective rerun.')
    spec['data']['leo_config_ref']='experiments/cvs_selected_concat/augmentation.py:MidUrbanAugment/RECIPE'
    spec['execution'].update(launch_owner='codex/root/cvs-selected-concat-20261002',remote_run_root=PROJECT+'/runs/'+RUN,remote_log_root=PROJECT+'/logs/'+RUN,
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,launch_command='python -m experiments.cvs_selected_concat.publish --output local_artifacts/'+RELEASE,
        gpu_policy='four legal slots, at most2 total training jobs/GPU and >=12GB free; no unrelated interference')
    spec['metrics_plan'].update(primary='Four fixed modelseed means/sampleSD for accuracy and Macro-F1; full seed/view/RX confusion matrices; paired differences against fixed historical CVS identity CE and CVCNN CE; no selection from scores',
        scorer_ref='comparison_suite.score:p1; all4 predictions fixed before independent scorer opens truth',prediction_ref=PROJECT+'/runs/'+RUN+'/<row>/prediction/phase1_predictions.npz')
    spec['notes']=[
        'User requested performance judged on test rather than source and previously best explored network; fixed residual_fusion as explicitly stated assumption. Historical clean mean78.4543% informs this user-fixed design. BENCHMARK_INFORMED_FIXED_DESIGN, not TARGET_FREE_ARCHITECTURE_SELECTION or fresh blind confirmation.',
        'One architecture, four seeds2026092701..04;164225 parameters; own scratch only. Historical source parent supplies design/metrics, not weights/optimizer/EMA/teacher/prototypes.',
        'Exact source_contract L/U/V6300/56700/27000, sourceRX1,3,4,6,8/day1,2,3; U unused,6TX, equalized1/center256/unitRMS/25MHz.',
        'E200,batch128,no drop_last,50steps/epoch,10000steps/model. AdamW lr2e-4/wd1e-4/cosine1e-6,no clipping. FullFP32 with cuDNN/matmulTF32false; explicit loader RNG followsmodelseed.',
        'Existing matched PracticalResidualAugment seed2027/receiverseed2027,residual/post_sync/noeq. E1..79clean only; E80..90mid/urban p.60; E91..200mid/urban p.80. BeforeE80 no augmentation call.',
        'At E80..200 generate satellite batch including unchanged unselected packets; concatenateclean+sat, one forward and one optimizer update: cleanCE+.68satCE; no consistency/domain/MixStyle/PL/DAOT/RC4/EMA/extra loss.',
        'Native received time/frequency/PA branches and bounded additive residual fusion; no arbitrary channel/RXinvariance or uniquely identified TXhardwareclaim.',
        'Actual config/parametercount/backendflags/50stepbudget, fullstep JSONL, fulltext epoch and compact JSONL/CSV, CEweights/LR/gradientusage/V/worstRX/time/peakmemory and synthetic resourceprofile.',
        'Frozen E200 source marker before prediction; existing VALIDATED_ONCE capsule reused, no data rebuild. Clean168000+same pairedsatellite168000 split into3 disjoint practicalscene strata; not3 full independentLEOviews.',
        'Fixed benchmark already tested; final scores are descriptive paired evidence, not fresh blindconfirmation. Negative results retained. No Phase2/SFT/newclass; no test feedback or selective rerun.',
        'User2026-10-02 restricts training augmentation to mid/low urban only; high never selected in training. Test clean and all3 practical scenes as requested; performance assessed on tests, not source scores.',
        'Earlier noaugmentation crossphase prototype preserved unlaunched; this newexperiment does not resume or modify prior healthy/source jobs.'
    ]
    runtime=dict(run_id=RUN,launch_owner=spec['execution']['launch_owner'],runtime_root=spec['execution']['remote_run_root'],log_root=spec['execution']['remote_log_root'],rows=[],
        p1_truth=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json')
    spec['test_completion_plan']=dict(views=['clean','satellite','practical_high','practical_mid','practical_low_urban'],
        query_count=168000,registered_classes=6,rows=4,capsule=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule',
        physical_id_index=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule/index.npz',truth=runtime['p1_truth'],
        architecture_fixed_before_training=True,weight_selection='own_E200',target_feedback=False)
    spec['expected_artifacts']=['initial_smoke.pt','initialization.json','source_contract.json','resolved_config.json','step_metrics.jsonl','epoch_metrics.jsonl','epoch_metrics.csv','last.pt','source_selection.json','resource_profile.json','source/completion.json','phase1_predictions.npz','phase1_complete.json','phase1_scored_results.json','phase1_summary.json','scoring_p1_complete.json','completion.json']
    for seed in SEEDS:
        rid=VARIANT+'-s'+str(seed);source=runtime['runtime_root']+'/'+rid+'/source';pred=runtime['runtime_root']+'/'+rid+'/prediction'
        srcref='experiments/cvs_selected_concat/configs/source-s'+str(seed)+'.json';predref='experiments/cvs_selected_concat/configs/predict-s'+str(seed)+'.json'
        src=dict(method='cvs_selected_concat',variant=VARIANT,model_seed=seed,source_contract=spec['data']['contract_ref'],dataset=spec['data']['dataset'],output_root=source,device='cuda:0',epochs=200,batch_size=128,
            lr=.0002,lr_min=1e-6,weight_decay=.0001,augmentation_seed=2027,receiver_seed=2027,split_seed=392005,drop_last=False,sat_ce_start=80,lambda_sat_cls=.68,concat_sat_ce_only=True,lambda_sat_cons=0.,domain_backbone=False,mixstyle=False,extra_losses=[],selection='fixed_last_epoch',numerical_policy=FULL_FP32_POLICY,augmentation_recipe=RECIPE)
        cfg=dict(method='cvs_selected_concat',variant=VARIANT,model_seed=seed,source_output=source,output_root=pred,device='cuda:0',source_contract=spec['data']['contract_ref'],p1_capsule=spec['test_completion_plan']['capsule'],numerical_policy=FULL_FP32_POLICY)
        write(ROOT/srcref,src);write(ROOT/predref,cfg)
        srcpath=PROJECT+'/releases/'+RELEASE+'/'+srcref;predpath=PROJECT+'/releases/'+RELEASE+'/'+predref
        command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_selected_concat.dispatch --source-config '+srcpath+' --predict-config '+predpath
        spec['rows'].append(dict(row_id=rid,method='cvs_selected_concat',purpose='user_fixed_residual_CVS_midurban_concat',gpu=None,config_ref=srcref,prediction_config_ref=predref,resolved_config_ref=source+'/resolved_config.json',data_overrides={},
            seeds=dict(model=seed,split=392005,data=None,augmentation=2027,support=None,evaluation=None),seed_notes='Exact physical split; model/loader RNG=modelseed; augmentation deterministic ID/epoch seed2027,receiverseed2027; no support; immutable evaluation observations.',
            k=None,scenario='clean,practical_high,practical_mid,practical_low_urban',optimizer='AdamW+cosine',lr=.0002,epochs=200,fl_rounds=None,budget_ref='E200x50=10000steps;clean6300/epoch;sat6300/epoch E80..200',output_root=source,prediction_root=pred,log_path=runtime['log_root']+'/'+rid+'.log',command=command,expected_artifacts=spec['expected_artifacts']))
        runtime['rows'].append(dict(row_id=rid,method='cvs_selected_concat',model_seed=seed,stage='phase12',source_config=srcpath,config=predpath,output_root=pred))
    write(ROOT/'experiments/cvs_selected_concat/configs/experiment_spec.json',spec)
    write(ROOT/'experiments/cvs_selected_concat/configs/launch_spec.json',runtime)
    print(RUN)

if __name__=='__main__': main()
