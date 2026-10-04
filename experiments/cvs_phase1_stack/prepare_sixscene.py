import copy
from pathlib import Path
from experiments.cvs_phase1_stack import design as d
from experiments.cvs_phase1_stack import sixscene_after as s


def main():
    spec=copy.deepcopy(d.read(d.ROOT/'experiments/cvs_phase1_stack/configs/experiment_spec.json'))
    spec.pop('registered_at',None)
    spec.update(run_id=s.RUN,group_id='cvs-reference-phase1-stack-sixscene-evaluation',display_name='reference_response剩余136行：clean及完整六residual环境自动测试',
        description='等待父136行全部训练、源域冻结及既有测试收尾后，自动评估全部固定E200模型的clean与六个完整residual星地视图；每视图168000物理query。',
        authorization='用户2026-10-04：不要忘了跑完自动进行测试集测试包括clean和六个residual星地环境。',
        parent_run_ids=[d.RUN],status='PLANNED',rows=[],tags=['cvs','reference_response','phase1','sixscene','residual','evaluation_only','auto_after_parent'])
    spec['code'].update(commit='evaluation release_commit.txt fixes actual Git commit; worker code '+s.WORKER_COMMIT,cwd=s.WORKER_ROOT.as_posix())
    spec['execution'].update(remote_run_root=s.BASE.as_posix(),remote_log_root=s.PROJECT+'/logs/'+s.RUN,local_artifact_root='automation_reports/CV-SincNet/'+s.RUN,
        launch_owner='codex/root/reference-stack-sixscene-20261004',launch_command='python -m experiments.cvs_phase1_stack.publish_sixscene --output local_artifacts/'+s.RELEASE,
        gpu_policy='wait uses noGPU; inference16 concurrent, <=2/GPU and >=12GB free; no training intervention')
    spec['data'].update(representation='existing VALIDATED_ONCE residual/post_sync/noeq,25MHz,2x256; eachview168000 exactpairedphysicalIDs',
        dataset=s.VIEWS_ROOT.as_posix(),version='phase1_full_six_fixed_20261002_v1',physical_ids_ref=(s.VIEWS_ROOT/'index.npz').as_posix(),
        validation_ref=(s.VIEWS_ROOT/'manifest.json').as_posix(),roles=dict(query_per_view=168000,views=7,source_training='none; checkpoint contract matches parent'),
        leo_config_ref=(s.VIEWS_ROOT/'manifest.json').as_posix(),support_query_ref='query only; no support/adaptation',train_ratio=None)
    spec['checkpoint'].update(initialization='evaluation_only_load_own_parent_E200',sources=[dict(parent_run_id=d.RUN,source_root=d.BASE,ancestor='each own scratch student; no inherited checkpoint')],
        provenance_verdict='DEFERRED_TO_PARENT_COMPLETION_CHECK_BEFORE_ANY_QUERY',selection_rule='all136 fixedrows ownE200; allsource choices frozen beforequery; no testselection')
    spec['metrics_plan'].update(primary='Accuracy/Macro-F1, fourseed mean/sampleSD, worstRX, fixedpairedseed contrasts, allRX/TX confusion matrices',
        scorer_ref='sixscene_after.py --mode score: all136x7 predictions preflight beforetruth; secondimplementation bincount',
        prediction_ref=s.BASE.as_posix()+'/<row>/prediction/predictions.npz')
    spec['test_completion_plan'].update(views=list(s.VIEWS),rows=136,query_count=168000,capsule=s.VIEWS_ROOT.as_posix(),physical_id_index=(s.VIEWS_ROOT/'index.npz').as_posix(),
        truth=d.TRUTH,weight_selection='all136 ownE200',all_source_stages_frozen_before_query=True,scene_partition='eachscene full168000 sameIDs; sixviews are not disjointstrata',
        wait_for_parent_completion=d.BASE+'/completion.json',prediction_count=136*7*168000)
    spec['expected_artifacts']=['dispatcher.json','state.json','launch.json','<row>/prediction/provenance.json','<row>/prediction/resolved_config.json','<row>/prediction/predictions.npz','<row>/prediction/complete.json','scores.json','scores.csv','summary.json','summary.csv','analysis.md','scoring_complete.json','completion.json']
    spec['stage_policies']=dict(parent='all136 parent ANALYZED and SOURCE_FROZEN',selection='none; everyfixedrow',target_feedback=False)
    spec['notes']=['Evaluation only; does not stop/restart/hotmodify healthy training or either existing dispatcher.',
        'Seven complete views: clean,suburban high/mid/low andurban high/mid/low; residual/post_sync/noeq. Reuse existing VALIDATED_ONCE observations without regeneration or repeatedbuilder validation.',
        'All136 rows use source-validated ownE200 scratch model; source_role IDs, initialization ancestry, args, finalcheckpoint and globalfreeze rechecked beforequery. Query readonly, noadaptation ortraining.',
        'Separate scoring subprocess accessestruth only after everyprediction fixed; independent confusion/accuracy/F1 recount, retainall negative results; nofeedback to model/seed/structure/selection/rerun.',
        'Fullview counts differ from original 3scene partitions. Historicalbenchmark exposed; do notclaim freshblindtest. Phase2/K/newclasses/H are N/A.',
        'Wait process checks every60seconds withoutGPU; failedparent stops evaluation with retainedfailureevidence. Noautomatic retries; at most16 predictors and2/GPU.',
        'Existing threepartition test stays intact; this companion executes afterward and is required finalclean+sixscene delivery. Inference timing and CUDAallocatedpeak measured; unmeasuredcostsN/A.']
    for row in d.rows():
        rid=row['row_id'];spec['rows'].append(dict(row_id=rid,method=row['stage']+'-'+row['arm'],purpose='frozen_full_sixscene_test',gpu=None,
            config_ref='experiments/cvs_phase1_stack/sixscene_after.py#RUN,VIEWS,original.rows',resolved_config_ref=(s.BASE/rid/'prediction/resolved_config.json').as_posix(),
            checkpoint_ref=d.BASE+'/'+rid+'/source/final_ssdg.pth',data_overrides={},seeds=dict(model=row['model_seed'],split=392005,data=None,augmentation=None,support=None,evaluation=392005),
            seed_notes='Existing fixedchannel realization evaluation392005,receiver2027; model is parentseed; no stochastic adaptation.',
            k=None,scenario=','.join(s.VIEWS),optimizer=None,lr=None,epochs=None,fl_rounds=None,budget_ref='7x168000 predictions, batch256, no training',
            output_root=(s.BASE/rid/'prediction').as_posix(),log_path=s.PROJECT+'/logs/'+s.RUN+'/'+rid+'.log',
            command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u '+s.PROJECT+'/releases/'+s.RELEASE+'/sixscene_after.py --mode predict --row '+rid,
            expected_artifacts=spec['expected_artifacts']))
    d.write(d.ROOT/'experiments/cvs_phase1_stack/sixscene_configs/experiment_spec.json',spec)
    print(s.RUN)


if __name__=='__main__':main()
