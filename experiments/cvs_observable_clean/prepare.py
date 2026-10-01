"""Register only the actual source-selected new CVS, plus twenty frozen controls."""
import argparse,copy,json
from pathlib import Path
from experiments.cvs_observable_identity.prepare import RUN as SOURCE_RUN,CONFIRM_RUN as RUN,PROJECT
from experiments.cvs_observable_identity.model import VARIANTS
from experiments.cvs_residual_identity.prepare import SEEDS,write
from experiments.cvs_clean_eval.prepare import CAPSULE
ROOT=Path(__file__).resolve().parents[2]
RELEASE='cvs_observable_clean_eval_20261002_r01'
OLD_RUN='20261001-phase1-cvs-selected-clean-manysig-m20-r01'
PREVIOUS_PHYSICAL_RUN='20261002-phase1-cvs-rf-operator-clean-manysig-m24-r01'
SOURCE=PROJECT+'/runs/'+SOURCE_RUN

def main(selection):
    if selection['status']!='SOURCE_SELECTION_FROZEN' or selection['target_access'] or selection['target_score_used'] or selection['selected_variant'] not in VARIANTS:
        raise ValueError('Source-only freeze required')
    variant=selection['selected_variant'];prefix='experiments/cvs_observable_clean/configs/';remote=PROJECT+'/releases/'+RELEASE+'/'
    old=json.loads((ROOT/'experiments/cvs_selected_clean/configs/launch_spec.json').read_text(encoding='utf-8'))
    spec=copy.deepcopy(json.loads((ROOT/'experiments/cvs_selected_clean/configs/experiment_spec.json').read_text(encoding='utf-8')))
    spec.update(run_id=RUN,group_id='cvs-clean-observable-confirmation',display_name='性能优先整体物理观测CVS：4seed clean确认与20个冻结控制预测复用',
        description='只测试本轮source-selected候选；新4clean预测+原20预测只读复用统一24行独立评分；固定原物理query/seed，不反馈调参。',
        authorization='2026-10-01用户要求按原目标继续改进并默认完成测试集测试。',status='PLANNED',parent_run_ids=[SOURCE_RUN,OLD_RUN,PREVIOUS_PHYSICAL_RUN],stage='Phase1-whole-observable-CVS-clean-confirmation',
        tags=['cvs','clean_only','ce_only','performance_priority','source_selected','frozen_prediction_reuse'])
    selection_path=remote+prefix+'frozen_selection.json'
    enriched=dict(selection,scope='observable_source',source_matrix_ref=PROJECT+'/releases/cvs_observable_identity_20261002_r01/experiments/cvs_observable_identity/configs/launch_spec.json')
    write(ROOT/prefix/'frozen_selection.json',enriched)
    runtime=dict(run_id=RUN,launch_owner='codex/root/cvs-observable-clean-20261002',runtime_root=PROJECT+'/runs/'+RUN,log_root=PROJECT+'/logs/'+RUN,
        selection_file=selection_path,p1_truth=old['p1_truth'],rows=[dict(r,reuse_from_run=r.get('reuse_from_run',OLD_RUN)) for r in old['rows']])
    for row in spec['rows']:
        row.update(purpose='reuse_complete_frozen_control_prediction',reuse_from_run=row.get('reuse_from_run',OLD_RUN),command='N/A:read original frozen artifact, no relaunch')
    for seed in SEEDS:
        rid=variant+'-s'+str(seed);out=runtime['runtime_root']+'/'+rid+'/prediction';ref=prefix+rid+'.json'
        cfg=dict(method='cvs_clean_eval',variant=variant,model_seed=seed,selection_file=selection_path,
            baseline_source_root=PROJECT+'/runs/20261001-phase1-cvs-clean-architecture-manysig-m32-r01',residual_source_root=PROJECT+'/runs/20261001-phase1-cvs-residual-identity-manysig-m8-r01',observable_source_root=SOURCE,
            source_output=SOURCE+'/'+rid+'/source',output_root=out,source_contract=spec['data']['contract_ref'],p1_capsule=CAPSULE,device='cuda:0',views=['clean'])
        write(ROOT/ref,cfg);runtime['rows'].append(dict(row_id=rid,variant=variant,model_seed=seed,config=remote+ref,output_root=out))
        template=copy.deepcopy(spec['rows'][0]);template.pop('reuse_from_run',None)
        template.update(row_id=rid,method=variant,purpose='single_source_selected_new_CVS_confirmation',config_ref=ref,resolved_config_ref=out+'/resolved_config.json',
            seeds=dict(model=seed,split=392005,data=None,augmentation=None,support=None,evaluation=None),output_root=out,log_path=runtime['log_root']+'/'+rid+'.log',
            command='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_clean_eval.predict --config '+remote+ref)
        spec['rows'].append(template)
    spec['checkpoint']['sources'] += [SOURCE+'/'+variant+'-s'+str(seed)+'/source/last.pt' for seed in SEEDS]
    spec['checkpoint'].update(selection_rule='FixedE200;two new source-only candidates;performance first: max four-seed sourceV/worstRX score;costs only if performance exactly tied;read-only20 oldpredictions;no test feedback',provenance_verdict='New4actualscratch/fullphysicalcontract/payload checked beforequery;old20 originalprovenance/complete/selection verified')
    spec['code'].update(checkout=str(ROOT),cwd=remote.rstrip('/'))
    spec['execution'].update(launch_owner=runtime['launch_owner'],remote_run_root=runtime['runtime_root'],remote_log_root=runtime['log_root'],local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_command='python -m experiments.cvs_observable_clean.publish --output local_artifacts/'+RELEASE)
    spec['metrics_plan'].update(primary='clean4seedpairedvsnative,previousresidualand3commonbaselines;allRX/TX/F1/CM/resources',prediction_ref='New4:'+runtime['runtime_root']+';old20 readonlyoriginalroots',later_test='No target feedback/reselection/retrain/unselectedcandidatequery')
    spec['notes']=['Supplemental report-only comparison to completed rf_gmp frozen metrics;no new prediction or source selection feedback','Only4newpredictions;old20immutablecontrolsreused;24samephysicalquery/class/seedrows independentlyscored','Scopecleanonly/closedset6classes;noLEO/support/newclass/unknownclaims','Repeatedhistoricallyexposedfixedbenchmark;notnewblindtest;allnegativegainsreported','Source-onlyselected '+variant+' before new cleanquery;do not retune from test result']
    write(ROOT/prefix/'launch_spec.json',runtime);write(ROOT/prefix/'experiment_spec.json',spec)
    print(json.dumps(dict(run_id=RUN,variant=variant,new_predictions=4,reused=20,total=24)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selection',required=True);a=p.parse_args();main(json.loads(Path(a.selection).read_text(encoding='utf-8')))
