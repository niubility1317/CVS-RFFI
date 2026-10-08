"""Materialize all16 fixed rows and their experiment preregistration."""
import copy
from experiments.cvs_receiver_residual import design as d


def main():
    spec=copy.deepcopy(d.read(d.ROOT/'experiments/cvs_phase1_overlay/configs/experiment_spec.json'))
    base=d.PROJECT+'/runs/'+d.RUN
    spec.update(run_id=d.RUN,group_id='cvs-cross-tx-displacement-branch-contribution',
        display_name='reference_response：跨TX共同位移补偿×分支贡献预测，CE拼接四seed消融',
        description='固定强骨干与最基础CE+原生clean/satellite拼接；四组baseline/displacement/contribution/combined，从零训练，固定E200，全部测试并保留负结果。',
        authorization='用户2026-10-08：按你设计的完成，其他原来的使用最基础的CE+拼接星地信道增强(最基础的)。按前述两条主线的四组设计落实。',
        status='PLANNED',rows=[],parent_run_ids=[],aliases=['receiver_residual','共同位移','分支贡献'],
        tags=['cvs','reference_response','phase1','factorial','ce_concat','scratch','displacement','contribution','benchmark_exposed'])
    spec['code'].update(checkout=str(d.ROOT),cwd=d.PROJECT+'/releases/'+d.RELEASE,
        commit='release_commit.txt fixes actual Git commit; independently verify remote OID before publish')
    spec['checkpoint'].update(initialization='scratch',sources=[],provenance_verdict='SCRATCH_NO_INHERITANCE',
        selection_rule='All16 fixed comparisons ownE200. Source preference mean4(.5V+.5worstRX), exact ties fewer parameters; target never used.')
    spec['data'].update(leo_config_ref='experiments/cvs_phase1_overlay/contract.py:LEO; native practical3 concat retained',
        validation_ref=d.VIEWS_ROOT+'/manifest.json',physical_ids_ref=d.SOURCE+'#role_ids; '+d.VIEWS_ROOT+'/index.npz',
        roles=dict(L_s=.07,U_s=.63,V=.30,U_use='unused',centroid_targets='L only; V read-only metrics'))
    spec['permissions']['claim_scope']='BENCHMARK_EXPOSED_FIXED_DESIGN; source-only training and current-run selection; no fresh blind benchmark or formal Phase2 claim.'
    spec['execution'].update(launch_owner='codex/root/receiver-residual-20261008',remote_run_root=base,
        remote_log_root=d.PROJECT+'/logs/'+d.RUN,local_artifact_root='automation_reports/CV-SincNet/'+d.RUN,
        launch_command='python -m experiments.cvs_receiver_residual.publish --output local_artifacts/'+d.RELEASE,
        gpu_policy='max4 own workers; start only on wholly idle GPU, pre-CUDA visible reservation, <=2 total/GPU, >=12GB free; preserve healthy jobs')
    spec['metrics_plan'].update(primary='Accuracy/Macro-F1, fourseed mean/sampleSD, worstRX, all RX/TX, paired seed effects/interaction; all negative results',
        dimensions=['arm','model_seed','view','receiver','transmitter'],
        scorer_ref='evaluate.py score in separate process after all16x7 predictions, independent bincount recount',
        prediction_ref=base+'/<row>/prediction/predictions.npz')
    spec['test_completion_plan']=dict(rows=16,views=list(d.VIEWS),query_count=d.COUNT,registered_classes=6,
        capsule=d.VIEWS_ROOT,physical_id_index=d.VIEWS_ROOT+'/index.npz',truth=d.TRUTH,
        target_tx_ids=d.CLASSES,target_rx_indices=[0,2,5,7,9,10,11],weight_selection='all ownE200',
        all_source_rows_frozen_before_query=True,scene_partition='each of six full scenes same168000 physical IDs',target_feedback=False)
    spec['mechanism_recipe']=d.RECIPE
    spec['notes']=[
        'Common recipe: reference_response scratch, pure identity CE, native clean/satellite concat E1; satellite CE .68 from E80; no consistency. Existing staged practical3 residual/post_sync/noeq sampler preserved.',
        'No pseudo labels, U loss, EMA teacher, MixStyle, GRL, global orthogonality, old domain backbone, Fishr, DAOT, RC4 or checkpoint inheritance.',
        'E200x50=10000 updates,6300 L exposure/epoch, B128/drop_lastFalse, AdamW2e-4 wd1e-4 cosine1e-6, FP32/TF32False, no gradient clipping.',
        'E1-40 new residuals disabled; E41-60 linear ramp; thereafter full strength. Zero output initialization preserves exact backbone forward. At most25% feature norm correction and25% perbranch gain delta.',
        'Each epoch E40-200 all arms perform same detached current-feature pass on L only,210 physical samples per TX/RX cell. No EMA. Stats refresh next epoch; no prototype/state transferred to inference. Extra6300 readonlyL forwards/epoch explicitly counted.',
        'Displacement target coordinatewise median of other5TX cellmean differences; supervised TX excluded. Within-TX average predicted correction across RX penalized toward zero. Additive common-shift hypothesis tested by L-only geometry diagnostics; not assumed true.',
        'Contribution supervision detached CE(z-minus-branch)-CE(z) at the current base classifier; clipped[-2,2]. Predict from perpacket29 received-response observables. Residual gates around true TF/PA/reference contributions; no confidence gating or claim of causal attribution.',
        'Source q may contain TX information. CrossTX exclusion and class gauge reduce shortcuts but do not prove identifiability. Auxiliary losses cleanL only; satellite CE still trains full deployed path.',
        'Identical backbone/loader/augmentation seeds; added-head initialization preserves global backbone RNG and is identical between single/combined arms. Different modelseeds do not alter physical split.',
        'All16 ownE200 models/source preference frozen before any target access; all fixed arms tested regardless of source ranking. No targetfeedback, tuning or selective rerun. Exposed historicalbenchmark explicitly disclosed.',
        'Reuse existing VALIDATED_ONCE clean+six practical residual views. All16x7 predictions before truth; separate scorer with independent metric recount. Seven complete views are not disjoint formalLEO strata.',
        'Phase2 support/adaptation/newclasses,K,H: N/A. No claim that benefit is established before independent scoring. Actual timings/parameters/peaks logged; synthetic resource profile scope disclosed.'
    ]
    spec['expected_artifacts']=['initial_smoke.pt','initialization.json','source_contract.json','resolved_config.json',
        'step_metrics.jsonl','epoch_metrics.jsonl','epoch_metrics.csv','centroid_diagnostics.jsonl','last.pt',
        'source_selection.json','resource_profile.json','source/completion.json','source_matrix_frozen.json',
        'prediction/provenance.json','prediction/predictions.npz','prediction/complete.json','scores.json','scores.csv',
        'summary.json','summary.csv','paired_results.json','resources.json','analysis.md','scoring_complete.json','completion.json']
    launch=dict(run_id=d.RUN,launch_owner=spec['execution']['launch_owner'],max_active=4,per_gpu_limit=2,rows=[])
    for c in d.rows():
        rid=d.row_id(c['arm'],c['model_seed']);ref='experiments/cvs_receiver_residual/configs/'+rid+'.json'
        d.validate_config(c);d.write(d.ROOT/ref,c);launch['rows'].append(dict(row_id=rid,config=c))
        spec['rows'].append(dict(row_id=rid,method=c['arm'],purpose='fixed_factorial_source_and_full_test',gpu=None,
            config_ref=ref,resolved_config_ref=c['output_root']+'/resolved_config.json',data_overrides={},
            seeds=dict(model=c['model_seed'],split=392005,data=392005,augmentation=2027,support=None,evaluation=392005),
            seed_notes='data/split fixed392005; loader=modelseed; augmentation/receiver2027; evaluation reuses fixed observations, no resampling; support N/A',
            k=None,scenario=','.join(d.VIEWS),optimizer='AdamW+cosine',lr=.0002,epochs=200,fl_rounds=None,
            budget_ref='E200x50=10000updates;6300L/epoch;concat6300/epoch; extra readonlyL6300 E40-200 all arms',
            output_root=c['output_root'],prediction_root=base+'/'+rid+'/prediction',
            log_path=spec['execution']['remote_log_root']+'/source-'+rid+'.log',
            command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_receiver_residual.source --config '+d.PROJECT+'/releases/'+d.RELEASE+'/'+ref,
            expected_artifacts=spec['expected_artifacts']))
    d.write(d.ROOT/'experiments/cvs_receiver_residual/configs/experiment_spec.json',spec)
    d.write(d.ROOT/'experiments/cvs_receiver_residual/configs/launch_spec.json',launch)
    print(d.RUN)


if __name__=='__main__': main()
