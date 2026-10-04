import copy
from experiments.cvs_phase1_stack.design import *


def main():
    spec=copy.deepcopy(read(ROOT/'experiments/cvs_phase1_overlay/configs/experiment_spec.json'))
    spec.pop('registered_at',None)
    spec.update(run_id=RUN,display_name='reference_response原Phase1剩余机制：R2至R6共136行',
        description='用户授权一次发布剩余全部实验；依源域冻结顺序推进两阶段/伪标签、DG、开放世界特征、DAOT×RC4、完整重组及删组消融。全部136行源训练完成后统一测试。',
        authorization='用户2026-10-04：发布剩余的所有实验。',status='PLANNED',rows=[],
        tags=['cvs','reference_response','phase1','mechanism','sequential_source_selection','scratch','benchmark_informed'],
        parent_run_ids=[PARENT_RUN],replaces_run_id=None)
    spec['code'].update(checkout=str(ROOT),cwd=PROJECT+'/releases/'+RELEASE)
    spec['checkpoint'].update(initialization='scratch',sources=[],provenance_verdict='SCRATCH_NO_INHERITANCE',
        selection_rule='Own E200. Parent mechanism features only, never weights. Each stage fixed4seed mean(.5V+.5worstRX), exactties published arm order; no query until all stages frozen.')
    spec['execution'].update(launch_owner=OWNER,remote_run_root=BASE,remote_log_root=PROJECT+'/logs/'+RUN,
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_command='python -m experiments.cvs_phase1_stack.publish --output local_artifacts/'+RELEASE,
        gpu_policy='8 concurrent, <=2 total jobs/GPU, >=12GB free; sequential dependency queue; preserve others')
    spec['data']['leo_config_ref']='native practical_adapter and frozen matched recipe; supervised concat unless row ablates it; separate DAOT/MUSE views as declared'
    spec['metrics_plan'].update(primary='Accuracy/Macro-F1/worstRX mean and sampleSD; fixed stage controls and pairedseed differences; R5 interaction; complete row/view/RX/TX confusion and resource costs',
        scorer_ref='comparison_suite.score:p1 after136 predictions; cvs_phase1_stack.analyze independent bincount recount',prediction_ref=BASE+'/<row>/prediction/phase1_predictions.npz')
    spec['test_completion_plan'].update(rows=136,weight_selection='own_E200',all_source_stages_frozen_before_query=True)
    spec['expected_artifacts']=['initial_smoke.pt','initialization.json','source_contract.json','resolved_config.json','resolved_native_args.json','step_metrics.jsonl.gz','epoch_metrics.jsonl','epoch_metrics.csv','metrics_epoch.jsonl','final_ssdg.pth','phase1_resource_summary.json','source/completion.json','r2_source_frozen.json','r3_source_frozen.json','r4_source_frozen.json','r5_source_frozen.json','r6_source_frozen.json','all_sources_frozen.json','phase1_predictions.npz','phase1_complete.json','phase1_scored_results.json','independent_recount.json','paired_results.json','compact_summary.json','analysis.md','completion.json']
    spec['notes']=[
        'R1 source-only freeze selected leo before its query access; only parent_source_freeze consumed; historical/current target scores never input to remaining design or selection.',
        'R2 4arms, R3 7arms, R4 8arms, R5 4arms, R6 11arms ×4modelseeds =136 scratch runs. EachE200; U-length matched222steps/epoch, Lbatch128drop_lastTrue, Ubatch256drop_lastFalse. Planned44400updates, actualsuccessful updates logged.',
        'Exactsourcephysical L/U/V6300/56700/27000. All U labels and TXmetadata removed at datasetedge; legacy path has-1 label sentinel and unavailable pseudo truth diagnostic. No targetloader.',
        'Native AdamW original schedule/clipping retained; fullFP32 TF32False, sample_rate25MHz. R2 separates all-label200 vs130+70 vsEMA vsPL; R5 uses common originalMUSEM3 bridge and FastTrust schedule becauseRC4 requiresM3.',
        'Reference identity177025 plus native domain/adv and optional MUSE/DAOT heads; full actualtrainable counts runtime. Native domain retained in all arms, disabled losses explicitzero. Replaced reference identity does not share originalSinc/hf with native nuisance branch.',
        'All lambda_* defaults explicitly zero except declared component weights, CE and applicable supervisedLEO/PL weights. Originalmechanism warmups retained; PAIC adaptiveguard off throughout for controlled comparison. Existing supervision strongviews schedule retained and reported.',
        'R3 domain includes domainCE+GRL; orth/cons/groupCE/Fishr each compared after domain coupling. R4 sixsingle originalconstraints and allcombined. R5 DAOT×RC4 with common MUSE bridge.',
        'R6 full is fixeddeclared allmechanism stack; native_full replaces reference identity by originalnative identity with sametraining. selected is own scratch rebuild ofR5 source winner. no_leo disables supervisedconcat only, DAOT/MUSE channels remain; no_dg removes labeled DG group only. no_pseudo removes legacyPL and MUSE/RC4 dependencies; DAOT labeled remains. These are explicitly dependent grouped ablations, not isolated removed-loss claims.',
        'Every stage writes resolved config for allrows before launch; downstream parent_features selected automatically only from same-stage complete4seedsource means. Ties published arm order. All136 fixedcomparisons later tested irrespective of source rank.',
        'Scratch EMA originates only from ownstudent; no imported state. E200 final weights only. Fullnative metrics and compressed fullstep logs, compact JSONL/CSV with nonfinite missingvaluesnull.',
        'Clean168000 and one pairedsatellite168000 observations with disjointthreepracticalscene strata; immutableexistingVALIDATED_ONCEcapsule. Allpredictions beforetruth. No Phase2/K/newclass/H claims.',
        'Retain technicalfailures/negative results; noautomatic retry, no low-scorestop, no interference with healthy tasks. Nonfinite loss/gradient fails immediately; logged_steps and successful optimizer_steps must both equal44400 before freeze. Failure stops laterqueue after activejobsfinish; completion requiresindependent scoring/recount.',
        'R2 bridge includes originalRF augmentation and native schedules/domainbranch; EMA arm maintains teacher state but student remains inference/selection model; no isolatedEMA inference benefit claim.'
    ]
    policies={}
    for r in rows():
        rid=r['row_id'];stage=r['stage'];folder=BASE+'/'+rid
        parent='R1 source_matrix_frozen.json:leo' if stage=='r2' else BASE+'/'+list(STAGES)[list(STAGES).index(stage)-1]+'_source_frozen.json:features'
        template=dict(**r,initialization='scratch',parent_features_rule=parent,feature_delta_function='experiments.cvs_phase1_stack.design:features',
            resolved_source_config=BASE+'/configs/source-'+rid+'.json',prediction_config=BASE+'/configs/predict-'+rid+'.json')
        ref='experiments/cvs_phase1_stack/configs/'+rid+'.json';write(ROOT/ref,template)
        spec['rows'].append(dict(row_id=rid,method=stage+'-'+r['arm'],purpose='source_selected_mechanism_comparison',gpu=None,
            config_ref=ref,resolved_config_ref=folder+'/source/resolved_config.json',data_overrides={},
            seeds=dict(model=r['model_seed'],split=392005,data=None,augmentation=r['model_seed'],support=None,evaluation=None),
            seed_notes='Model/loader/nativeaugmentation derivedfrommodelseed; physicalsplitfixed; receiverhardwareseed2027; immutableevaluation.',
            k=None,scenario=','.join(['clean']+SCENES),optimizer='native AdamW; MUSE FastTrust when enabled',lr=.0002,epochs=200,fl_rounds=None,
            budget_ref='E200x222=44400plannedupdates; sameUlengthbudget forcontrols; actualupdates/multiviewexposure logged',
            output_root=folder+'/source',prediction_root=folder+'/prediction',log_path=PROJECT+'/logs/'+RUN+'/source-'+rid+'.log',
            command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_phase1_stack.source --config '+template['resolved_source_config'],
            expected_artifacts=spec['expected_artifacts']))
    spec['stage_policies']=dict(arms=STAGES,parent_selection='4seedmean(.5V+.5worstRX),exacttiesarmorder',test_barrier='all136sourcesfrozen',unknown_parent_parameters='deterministic substitution of previous source freeze before next stage launch')
    write(ROOT/'experiments/cvs_phase1_stack/configs/experiment_spec.json',spec)
    write(ROOT/'experiments/cvs_phase1_stack/configs/launch_spec.json',dict(run_id=RUN,owner=OWNER,stages=STAGES,rows=rows(),runtime_root=BASE))
    print(RUN)


if __name__=='__main__':main()
