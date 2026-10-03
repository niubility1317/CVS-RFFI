"""Register the already specified all-eight clean test matrix."""
import copy,json,subprocess
from pathlib import Path
from experiments.cvs_reference_residual_clean.contracts import *
ROOT=Path(__file__).resolve().parents[2]
def main():
    folder=ROOT/'experiments/cvs_reference_residual_clean/configs';folder.mkdir(exist_ok=False)
    parent=read(ROOT/'automation_reports/CV-SincNet'/SOURCE_RUN/'experiment.json')
    spec=copy.deepcopy(parent)
    spec.update(run_id=RUN,group_id='cvs-reference-residual-fixed-ce-test',display_name='CVS参考约束残差：全部8份E200权重clean测试',
        description='固定标量/逐频×四seed全部测试，源模型先冻结，预测全部完成后独立连接truth；不回流调参。',
        status='PLANNED',rows=[],parent_run_ids=[SOURCE_RUN],tags=['cvs','clean_only','reference_residual','fixed_comparison','independent_test'])
    spec.pop('outcome',None);spec.pop('registered_at',None)
    spec['code'].update(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),checkout=str(ROOT),cwd=PROJECT+'/releases/'+RELEASE)
    spec['checkpoint'].update(initialization='frozen E200 inference only',sources=[SOURCE_ROOT+'/'+rid+'/source/last.pt' for rid in expected_rows()],
        source_release_commit=SOURCE_COMMIT,selection_rule='Both preregistered comparison arms,all4seeds;fixed E200;no elimination or target feedback')
    spec['execution']=dict(host='N607',launch_owner=OWNER,remote_run_root=RUNTIME,remote_log_root=LOGS,local_artifact_root='local_artifacts/'+RELEASE,
        launch_command='python -m experiments.cvs_reference_residual_clean.publish --output local_artifacts/'+RELEASE,
        gpu_policy='Existing free capacity;max2processes/GPU;no healthy task interference',stop_rule='Failure preserves artifacts; no automatic rerun or process termination')
    spec['data']['runtime_dataset']=CAPSULE+'/clean.npy'
    spec['data']['query_physical_id_index']=CAPSULE+'/index.npz';spec['data']['truth_scorer_only']=TRUTH
    spec['metrics_plan']=dict(primary='Test accuracy,macroF1,macroaccuracy;64ALL/RX records;48TX records;4seed mean/SD;paired inverse-scalar',
        resources='Per-row measured GPU/time/memory;frozen source resource profiles referenced',
        independent='All8 predictions complete before truth;independent NumPy confusion/accuracy/F1 recount after scoring')
    spec['notes']=['No training,adaptation or query fitting. Only explicitly recorded source execution commit and original physical contract allowed.',
        'Registered upstream before source completion;all8fixedE200 rows tested,not selected by source or target performance.',
        'Clean168000samephysicalIDs;6classes7RX;no satellite,support,newclasses,truth in predictor,extra views or batch-state adaptation.',
        'Historical benchmark exposed. Complete results report architecture outcome;not a new blind confirmation;no feedback into next model or selective rerun.']
    spec['expected_artifacts']=['8 clean_predictions.npz','8 clean_complete.json','8 provenance.json','8 resolved_config.json','clean_scored_results.json','clean_summary.json','scoring_clean_complete.json','independent_recount.json']
    launch=make_spec();write(folder/'launch_spec.json',launch)
    for row in launch['rows']:
        rid=row['row_id'];write(folder/(rid+'.json'),prediction_config(rid))
        spec['rows'].append(dict(row_id=rid,method=row['variant'],purpose='fixed_clean_test',gpu=None,config_ref='experiments/cvs_reference_residual_clean/configs/'+rid+'.json',
            resolved_config_ref=row['output_root']+'/resolved_config.json',data_overrides={},seeds=dict(model=row['model_seed'],split=392005,data=None,augmentation=None,support=None,evaluation=None),
            seed_notes='Frozen source model seed only;deterministic clean input;no augmentation or support',k=None,scenario='clean',optimizer=None,lr=None,epochs=None,fl_rounds=None,
            budget_ref='168000samephysicalquery,one deterministic view per model',output_root=row['output_root'],log_path=LOGS+'/'+rid+'.log',
            command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_reference_residual_clean.predict --config '+row['config'],expected_artifacts=spec['expected_artifacts'][:4]))
    spec['test_completion_plan']['output_root']=RUNTIME
    write(folder/'experiment_spec.json',spec);print(RUN)
if __name__=='__main__':main()
