import copy,json
from pathlib import Path
from experiments.cvs_clean_eval.prepare import PROJECT,RUN as BASELINE_RUN,RELEASE as BASELINE_RELEASE,SOURCE_ROOT,CAPSULE,TRUTH,write
from experiments.cvs_clean_eval.contracts import BASELINES,SEEDS
ROOT=Path(__file__).resolve().parents[2]
RUN='20261001-phase1-cvs-selected-clean-manysig-m20-r01'
RELEASE='cvs_selected_clean_eval_20261001_r01'
SOURCE_RUN='20261001-phase1-cvs-residual-identity-manysig-m8-r01'
SOURCE_NEW=PROJECT+'/runs/'+SOURCE_RUN
SELECTION=SOURCE_NEW+'/research_selection.json'
VARIANT='residual_fusion'


def main():
    prefix='experiments/cvs_selected_clean/configs/';remote=PROJECT+'/releases/'+RELEASE+'/'
    old=json.loads((ROOT/'experiments/cvs_clean_eval/configs/launch_spec.json').read_text(encoding='utf-8'))
    spec=copy.deepcopy(json.loads((ROOT/'experiments/cvs_clean_eval/configs/experiment_spec.json').read_text(encoding='utf-8')))
    spec.update(run_id=RUN,group_id='cvs-clean-source-selected-confirmation',display_name='源域冻结的轻量残差CVS：4seed clean确认与16个固定基准预测复用',
        description='联合六个源域候选按预登记规则选择residual_fusion，实际新增4个clean预测；复用四固定基准16个完整预测，统一20行独立评分。',
        authorization='用户活动目标：纯CE无星地增强，仅clean测试，凭CVS神经网络设计提升性能同时轻量化；候选源域冻结已完成。',
        parent_run_ids=[SOURCE_RUN,BASELINE_RUN,'20261001-phase1-cvs-clean-architecture-manysig-m32-r01'],stage='Phase1-selected-CVS-clean-confirmation',status='PLANNED',
        tags=['cvs','clean_only','source_selected','ce_only','no_augmentation','lightweight','frozen_baseline_reuse'])
    runtime=dict(run_id=RUN,launch_owner='codex/root/cvs-source-selected-clean-20261001',runtime_root=PROJECT+'/runs/'+RUN,log_root=PROJECT+'/logs/'+RUN,
        selection_file=SELECTION,p1_truth=TRUTH,rows=[dict(r,reuse_from_run=BASELINE_RUN) for r in old['rows']])
    for row in spec['rows']:
        row.update(purpose='reuse_complete_fixed_baseline_prediction',reuse_from_run=BASELINE_RUN)
        row['command']='N/A:read frozen original prediction;never launch baseline again'
    for seed in sorted(SEEDS):
        rid=VARIANT+'-s'+str(seed);out=runtime['runtime_root']+'/'+rid+'/prediction';ref=prefix+rid+'.json'
        cfg=dict(method='cvs_clean_eval',variant=VARIANT,model_seed=seed,selection_file=SELECTION,baseline_source_root=SOURCE_ROOT,residual_source_root=SOURCE_NEW,
            source_output=SOURCE_NEW+'/'+rid+'/source',output_root=out,source_contract=spec['data']['contract_ref'],p1_capsule=CAPSULE,device='cuda:0',views=['clean'])
        write(ROOT/ref,cfg);runtime['rows'].append(dict(row_id=rid,variant=VARIANT,model_seed=seed,config=remote+ref,output_root=out))
        template=copy.deepcopy(spec['rows'][0]);template.pop('reuse_from_run');template.update(row_id=rid,method=VARIANT,purpose='single_source_selected_CVS_confirmation',config_ref=ref,
            resolved_config_ref=out+'/resolved_config.json',seeds=dict(model=seed,split=392005,data=None,augmentation=None,support=None,evaluation=None),output_root=out,
            log_path=runtime['log_root']+'/'+rid+'.log',command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_clean_eval.predict --config '+remote+ref)
        spec['rows'].append(template)
    spec['checkpoint']['sources'] += [SOURCE_NEW+'/'+VARIANT+'-s'+str(seed)+'/source/last.pt' for seed in sorted(SEEDS)]
    spec['checkpoint'].update(selection_rule='Fixed E200;singleCVS selected solely by registered six-candidate source rule;no candidate rerank/test feedback',provenance_verdict='Old16 frozen provenance reused;new4 actualscratch/fullphysicalcontract/payload checks before any query')
    spec['code'].update(cwd=remote.rstrip('/'),checkout=str(ROOT))
    spec['execution'].update(launch_owner=runtime['launch_owner'],remote_run_root=runtime['runtime_root'],remote_log_root=runtime['log_root'],local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_command='python -m experiments.cvs_selected_clean.publish --output local_artifacts/cvs_selected_clean_eval_20261001_r01')
    spec['metrics_plan'].update(primary='Four-seed clean accuracy and paired residual_fusion minusnative plus3 commonbaselines;parameter/MAC/time/state costs',
        prediction_ref='New4 '+runtime['runtime_root']+'/*/prediction/clean_predictions.npz;old16 original outputs reused',later_test='No target feedback,automatic retraining or unselected-candidate test')
    spec['notes'] += ['Only4 newprediction processes;original16 baseline predictions/configs/resolved/provenance remain unchanged and are separately validated before reuse.',
        'residual_fusion was automatically source-selected from all6 registered candidates;sourceV98.069444,worstsourceRX94.495370;164225params,9708836conv/linearMACs;no target metrics used.',
        'Original16 prediction files were already complete/scored;new4 predictions must all be fixed before new separate scorer connects truth;all20 use identical opaque IDs/classes.',
        'Parameter reduction57.0256percent vsnative total and48.3006percent vsCEactive;MAC reduction1.5574percent;do not claim proportional latency savings.',
        'Scope remains clean-only/closedset;no supportadaptation,LEOsatellite,unknown ornewclass claims.']
    write(ROOT/prefix/'launch_spec.json',runtime);write(ROOT/prefix/'experiment_spec.json',spec)
    print(json.dumps(dict(run_id=RUN,total_rows=20,new_predictions=4,reused_predictions=16)))


if __name__=='__main__':main()
