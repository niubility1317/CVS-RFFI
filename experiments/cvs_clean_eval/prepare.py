"""Register the four already fixed clean baselines; no CVS candidate selection."""
import copy
import json
from pathlib import Path
from experiments.cvs_clean_eval.contracts import BASELINES,SEEDS

ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261001-phase1-clean-baselines-manysig-m16-r01'
RELEASE='cvs_clean_baseline_eval_20261001_r01'
PARENT='20261001-phase1-cvs-clean-architecture-manysig-m32-r01'
SOURCE_ROOT=PROJECT+'/runs/'+PARENT
CAPSULE=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/capsule'
TRUTH=PROJECT+'/runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    prefix='experiments/cvs_clean_eval/configs/'
    remote=PROJECT+'/releases/'+RELEASE+'/'
    selection=remote+prefix+'fixed_baselines.json'
    write(ROOT/prefix/'fixed_baselines.json',dict(scope='baseline_only',status='FIXED_BASELINES_FROZEN',
        test_variants=list(BASELINES),model_seeds=sorted(SEEDS),target_access=False,target_score_used=False,
        source_matrix_ref=remote+'experiments/cvs_clean_design/configs/launch_spec.json',
        authorization='2026-10-01用户要求进行基准的测试集测试，给出报告。固定四基准立即测试；不等待新CVS候选，不允许结果回流候选选择。'))
    spec=copy.deepcopy(json.loads((ROOT/'experiments/cvs_clean_design/configs/experiment_spec.json').read_text(encoding='utf-8')))
    runtime=dict(run_id=RUN,launch_owner='codex/root/clean-baselines-test-20261001',runtime_root=PROJECT+'/runs/'+RUN,
        log_root=PROJECT+'/logs/'+RUN,selection_file=selection,p1_truth=TRUTH,rows=[])
    spec.update(run_id=RUN,group_id='manysig-clean-fixed-baselines-ce',display_name='无增强四基准：固定E200、4seed、clean测试集独立评分',
        description='原CVS身份骨干、CVCNN、普通1D CNN、轻量1D ResNet；相同源训练与物理测试划分，纯CE无增强；16固定权重先完成预测，再独立连接truth。',
        authorization='2026-10-01用户：进行基准的测试集测试，给出报告。既有四基准立即测试，CVS研发健康训练继续。',
        parent_run_ids=[PARENT],status='PLANNED',stage='Phase1-clean-test',rows=[],
        tags=['baseline','clean_only','ce_only','no_augmentation','truth_last','four_seeds'])
    spec['code'].update(checkout=str(ROOT),cwd=remote.rstrip('/'))
    spec['data'].update(validation_ref=CAPSULE+'/manifest.json',support_query_ref=CAPSULE+'/index.npz',leo_config_ref=None)
    spec['permissions'].update(regime='frozen_phase1_clean',query_use='per-packet inference only;all registered classes;all16 prediction files fixed before separate truth-last scorer',
        claim_scope='Fixed research clean benchmark;historically exposed target,not blind;no target-driven CVS selection/tuning/retraining')
    spec['checkpoint'].update(initialization='fixed source E200',sources=[SOURCE_ROOT+'/'+v+'-s'+str(s)+'/source/last.pt' for s in sorted(SEEDS) for v in BASELINES],
        provenance_verdict='Recheck actual payload against full physical contract and scratch ancestry before each prediction',selection_rule='All16 fixed E200, no target checkpoint or architecture selection')
    spec['execution'].update(launch_owner=runtime['launch_owner'],remote_run_root=runtime['runtime_root'],remote_log_root=runtime['log_root'],
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,launch_command='python -m experiments.cvs_clean_eval.publish --output local_artifacts/cvs_clean_baseline_eval_20261001_r01',
        gpu_policy='One evaluator per idle GPU with12GB free;includes other owners;no training or healthy process modification')
    spec['expected_artifacts']=['pipeline_state.json','each prediction/provenance.json','each prediction/resolved_config.json','each prediction/clean_predictions.npz','each prediction/clean_complete.json','clean_scored_results.json','clean_scored_results.csv','clean_summary.json','clean_summary.csv','clean_paired.csv','scoring_clean_complete.json']
    spec['metrics_plan']=dict(metric_names=['accuracy','macro_f1','macro_accuracy','seed_sample_sd','per_receiver_accuracy','confusion','prediction_seconds','peak_cuda_allocated_bytes'],
        dimensions=['method','model_seed','receiver','clean'],prediction_ref=runtime['runtime_root']+'/*/prediction/clean_predictions.npz',
        scorer_ref='experiments/cvs_clean_eval/score.py',primary='clean test accuracy mean and sample SD over4 fixed model seeds;paired native-CVS minus each baseline',
        later_test='CVS candidate remains source-only;this baseline scoring cannot alter its registered selection rule')
    spec['notes']=['No new training;reuse actual source E200 scratch weights after full physical-contract and ancestry checks.',
        '6300L/56700Uunused/27000V;source RX1,3,4,6,8 days1,2,3;split392005;200x50 updates;CE only,no augmentation/domain/extra losses.',
        'Only clean.npy opened;no satellite.npy;all16 predictions fixed before independent scorer first truth read.',
        'No support adaptation/new-class registration in this Phase1 closed-set test;D92 three-stage metrics andK table not applicable.',
        'Native retains cosine scale30/dropout;common baselines retain Linear classifier. Compact6-block widths32/64/96 ResNet1D is not author ResNet18.',
        'Existing VALIDATED_ONCE capsule reused without data revalidation;historical target exposure disclosed;results cannot tune current CVS candidates.']
    for seed in sorted(SEEDS):
        for variant in BASELINES:
            rid=variant+'-s'+str(seed);out=runtime['runtime_root']+'/'+rid+'/prediction';ref=prefix+rid+'.json'
            cfg=dict(method='cvs_clean_eval',variant=variant,model_seed=seed,selection_file=selection,baseline_source_root=SOURCE_ROOT,
                source_output=SOURCE_ROOT+'/'+rid+'/source',output_root=out,source_contract=spec['data']['contract_ref'],
                p1_capsule=CAPSULE,device='cuda:0',views=['clean'])
            write(ROOT/ref,cfg)
            runtime['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,config=remote+ref,output_root=out))
            spec['rows'].append(dict(row_id=rid,method=variant,purpose='fixed_clean_baseline_test',gpu=None,config_ref=ref,
                resolved_config_ref=out+'/resolved_config.json',data_overrides={},
                seeds=dict(model=seed,split=392005,data=None,augmentation=None,support=None,evaluation=None),
                seed_notes='Model seed from source row;fixed physical IDs;deterministic clean inference;no augmentation/support/evaluation RNG',
                k=None,scenario='clean',optimizer=None,lr=None,epochs=None,fl_rounds=None,budget_ref='Frozen E200;no optimization;batch256 inference',
                output_root=out,log_path=runtime['log_root']+'/'+rid+'.log',command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_clean_eval.predict --config '+remote+ref,
                expected_artifacts=['provenance.json','resolved_config.json','clean_predictions.npz','clean_complete.json']))
    write(ROOT/prefix/'experiment_spec.json',spec);write(ROOT/prefix/'launch_spec.json',runtime)
    print(json.dumps(dict(run_id=RUN,rows=16)))


if __name__=='__main__':main()
