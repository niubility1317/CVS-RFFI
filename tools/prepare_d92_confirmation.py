"""Freeze a paired complete confirmation matrix using source-only calibration."""
import copy
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
WS=Path('E:/type10-7')
RUN='20260928-phase2-d92-scv-confirmation-manytx-m4-r01'
DATA_RUN='20260928-phase2-d92-confirmation-data-manytx-r01'
PROXY='20260928-diagnostic-d92-registration-proxy-s2026092701-r01'
RELEASE='d92_scv_confirmation_20260928_r01'


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    proxy_path=WS/'automation_reports/CV-SincNet'/PROXY/'results/complete.json'
    proxy=json.loads(proxy_path.read_text(encoding='utf-8'))
    if proxy['status']!='SOURCE_PROXY_COMPLETE' or proxy['rows']!=2100 or proxy['target_access'] is not False:raise ValueError('Incomplete source calibration')
    project='/home/szu2070436088/2510044040/CV-SincNet';release=project+'/releases/'+RELEASE
    remote=project+'/runs/'+RUN;data_root=project+'/runs/'+DATA_RUN
    legacy=json.loads((WS/'code/snapshots/cvs_practical_baselines_20260927_wt/configs/phase2_practical_data_20260927.json').read_text(encoding='utf-8'))
    cfg=copy.deepcopy(legacy)
    cfg.update(output_root=data_root,target_receivers=['19-1','8-14','8-7'],excluded_recent_target_receivers=legacy['target_receivers'],
        data_code=project+'/releases/phase2_practical_data_20260927_r01/code')
    write(ROOT/'configs/d92_confirmation_data_20260928.json',cfg)
    candidate=dict(k1_fft_weight=proxy['selected_k1_fft_weight'],k1_protection=1.0,
        fft_grid=[0.0,0.5,1.0,4.0],ridge_grid=[0.1,1.0,10.0],protection_grid=[0.0,0.5,1.0])
    write(ROOT/'configs/d92_scv_frozen_20260928.json',candidate)
    spec=json.loads((ROOT/'configs/cvs_d92_matched_20260927.json').read_text(encoding='utf-8'))
    original_root=spec['execution']['remote_run_root']
    spec.update(run_id=RUN,group_id='d92-fixed-phase1-support-cv-upgrade',display_name='固定Phase1的D92-SCV新旧类完整配对确认',
        description='4个固定新模型seed、3个与本轮源/目标RX不重合的接收机；全K、全部新类规模及5支持seed，对原D92与冻结SCV做一次完整评分。',
        stage='Phase2-joint-confirmation',parent_run_ids=[PROXY,DATA_RUN],
        authorization='2026-09-28用户要求优化D92，允许大幅修改；固定Phase1，全面提升新旧类及各K的H。',
        notes=['All model seeds and all registered cells retained; no target score selection or selective rerun.',
            'RX19-1,8-14,8-7 are disjoint from current source and recent target RX; all-time historical exposure is not certified.',
            'All predictions must finish before independent scorer opens query truth; scores cannot tune this frozen version.',
            'Primary: H improves at every K, old AND new mean accuracy each no worse by more than1pp; report all K/new-count cells, seed consistency, F1, class floors and forgetting.',
            'No training: actual support-fit diagnostics retained; optimizer loss/LR/gradient N/A.'],
        confirmation=dict(capsule=data_root+'/capsule',truth=data_root+'/score_only/truth.json',
            data_config=release+'/configs/d92_confirmation_data_20260928.json',candidate_config=release+'/configs/d92_scv_frozen_20260928.json',
            native_code=spec['native_code'],source_contract=spec['source_contract'],source_proxy_result=str(proxy_path),
            candidate=candidate,model_rows=4,splits_per_model=900,predictions_total=7236))
    for key in ['source_run_root','source_contract','native_code','final_capsule','final_truth','phase2_capsule','phase2_truth','phase2_scorer','registered_at','status']:spec.pop(key,None)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    spec['code'].update(commit=head,checkout=str(ROOT),cwd=release,commit_note='Exact pushed release HEAD passed in launch and startup; config points to parent implementation commit.')
    spec['data'].update(dataset=cfg['manytx_path'],version='existing_N607_ManyTx_actual19RX',target_receivers=cfg['target_receivers'],
        source_receivers=cfg['source_receivers'],capsule_id=None,split_id='900 IDs determined once by registered builder',
        physical_ids_ref=data_root+'/builder_report.json',validation_ref=data_root+'/capsule/manifest.json',
        support_query_ref=data_root+'/capsule/splits',tx_sets_ref=release+'/configs/d92_confirmation_data_20260928.json')
    spec['data'].pop('phase1_final_seeds',None)
    spec['checkpoint']['sources']=spec['checkpoint']['sources'][1:]
    spec['checkpoint']['selection_rule']='All four fresh source model seeds, fixed epoch200; reuse corresponding immutable source-only ground. No target selection.'
    spec['execution'].update(remote_run_root=remote,remote_log_root=remote,local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_owner='codex/root/d92-upgrade-20260928',gpu_policy='Sequential GPU0 frozen feature extraction; four CPU lanes each2BLASthreads, no gradient training',
        launch_command='C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 tools/publish_d92_confirmation.py')
    spec['metrics_plan'].update(metric_names=['accuracy','old_accuracy','new_accuracy','harmonic_mean','macro_f1','old_macro_f1','new_macro_f1','old_floor','new_floor','forgetting'],
        scorer_ref=release+'/tools/score_d92_confirmation.py')
    spec['expected_artifacts']=['startup.json','state.json','complete.json','scores.json','each row/predictions_complete.json','each row/scv/predictions_complete.json']
    spec['rows']=spec['rows'][1:]
    for row in spec['rows']:
        row_id=row['row_id'];out=remote+'/'+row_id
        row.update(method='paired D92 P2-256-FULL and D92-SCV-v1',purpose='joint_confirmation_fixed_model_seed',output_root=out,
            ground_source=original_root+'/'+row_id+'/ground',log_path=out+'/worker.log',config_ref='configs/d92_confirmation_20260928.json',
            resolved_config_ref=remote+'/startup.json',gpu=0,
            budget_ref='900 paired splits plus9 DG; no target-based selection or rerun',
            command=f'/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python {release}/tools/run_d92_confirmation.py --spec {release}/configs/d92_confirmation_20260928.json --commit RELEASE_HEAD',
            expected_artifacts=['ground/manifest.json','received_features/features_complete.json','predictions_complete.json','scv/predictions_complete.json','scv/compact.jsonl'])
        row['seed_notes']='Source model seed as listed, source split392005; target split/data2026092705, augmentation2026092707, evaluation2026092706, receiver2027. Support seeds2026092711..15 are inner rows.'
    write(ROOT/'configs/d92_confirmation_20260928.json',spec)
    data=copy.deepcopy(spec);data.update(run_id=DATA_RUN,display_name='D92确认集的单次received数据构建',stage='Phase2-data-preparation',
        description='3个当前轮未使用RX，26TX各198独立物理记录，3场景support36/query30，900划分。',parent_run_ids=[])
    data.pop('confirmation')
    data['execution'].update(remote_run_root=data_root,remote_log_root=remote,local_artifact_root='automation_reports/CV-SincNet/'+DATA_RUN,
        gpu_policy='CPU builder called once by registered confirmation launch owner')
    data['checkpoint']=dict(initialization='none; data builder never loads model',sources=[],contract_check_ref=cfg['data_code'],provenance_verdict='NOT_APPLICABLE',selection_rule='none')
    data['rows']=[dict(row_id='received-data',method='fixed_received_builder',purpose='data_preparation',gpu=None,
        config_ref='configs/d92_confirmation_data_20260928.json',resolved_config_ref=data_root+'/builder_report.json',data_overrides={},
        seeds=dict(model=None,split=cfg['data_seed'],data=cfg['data_seed'],augmentation=cfg['augmentation_seed'],support=None,evaluation=cfg['evaluation_seed']),
        seed_notes='No model seed; support inner seeds listed in data.support_seeds; receiver_seed2027.',k=cfg['shots'],scenario=','.join(cfg['scenarios']),
        optimizer='none',lr=None,epochs=None,fl_rounds=None,budget_ref='15444 observations;900 splits',output_root=data_root,
        log_path=remote+'/build_data.log',command=f'/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python {release}/tools/build_d92_confirmation_data.py --config {release}/configs/d92_confirmation_data_20260928.json',
        expected_artifacts=['builder_report.json','capsule/manifest.json','capsule/received.npz','score_only/truth.json'])]
    data['expected_artifacts']=data['rows'][0]['expected_artifacts'];data['metrics_plan']=dict(metric_names=[],dimensions=[],prediction_ref=None,scorer_ref=None)
    write(ROOT/'configs/d92_confirmation_data_record_20260928.json',data)
    print(json.dumps(dict(run=RUN,data_run=DATA_RUN,candidate=candidate,splits=900,predictions=7236)))


if __name__=='__main__':main()
