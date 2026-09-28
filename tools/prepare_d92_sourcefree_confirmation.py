"""Preregister a fixed source-data-free candidate without opening any scores."""
import copy
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
RUN='20260928-phase2-d92-sfhead-confirmation-manytx-m4-r01'
DATA_RUN='20260928-phase2-d92-sfhead-data-manytx-r01'
RELEASE='d92_sfhead_confirmation_20260928_r01'

def write(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')

def main():
    project='/home/szu2070436088/2510044040/CV-SincNet'
    release=project+'/releases/'+RELEASE;remote=project+'/runs/'+RUN;data_root=project+'/runs/'+DATA_RUN
    cfg=json.loads((ROOT/'configs/d92_confirmation_data_20260928.json').read_text(encoding='utf-8'))
    excluded=cfg['excluded_recent_target_receivers']+cfg['target_receivers']
    cfg.update(output_root=data_root,target_receivers=['20-19'],excluded_recent_target_receivers=excluded,
        support_pool_size=30,query_size=30)
    # Availability metadata only: 184 records in the smallest of26 cells.
    # 3scenes*(30support+30query)=180 distinct records/class; unchanged class registry.
    write(ROOT/'configs/d92_sourcefree_data_20260928.json',cfg)
    candidate=dict(max_iter=300)
    write(ROOT/'configs/d92_sourcefree_frozen_20260928.json',candidate)
    spec=json.loads((ROOT/'configs/d92_confirmation_recovery_20260928.json').read_text(encoding='utf-8'))
    spec.update(run_id=RUN,group_id='d92-fixed-phase1-sourcefree-upgrade',display_name='D92无源数据辅助分类头完整配对确认',
        description='固定4个Phase1模型；仅合法support训练统一分类头，冻结teacher约束旧类；全K/新类规模/5supportseed，不读取source样本或历史评分。',
        stage='Phase2-joint-confirmation',parent_run_ids=[DATA_RUN],
        authorization='2026-09-28用户允许辅助训练并明确禁止源域数据；允许已有模型、原型及少量统计+合法support，无硬传输上限、报告字节数。',
        notes=['Current candidate uses only legal support and frozen teacher predictions on support; new ground statistics payload zero bytes.',
            'Fixed formula from independent development without target results; no hyperparameter grid or source calibration.',
            'RX20-19 selected by complete26TX availability, not model performance. Disjoint from current source and both recent target receiver sets; all-time non-exposure not certified.',
            'One remaining complete receiver gives limited cross-receiver evidence; do not replace the comprehensive goal with this limitation.',
            'All paired predictions before truth; no score feedback, selective rerun or tuning.',
            'Primary each K: H>baseline and new_accuracy>baseline; old_accuracy>=baseline-0.01. Report every cell and seed; failing any K means not comprehensive improvement.',
            'Detailed measured optimizer loss/weights/gradient/steps/termination/time in fit_trace+text, compact JSONL/CSV. No Phase1 changes.'])
    spec.pop('replaces_run_id',None)
    spec.pop('registered_at',None);spec.pop('status',None)
    spec['confirmation']=dict(capsule=data_root+'/capsule',truth=data_root+'/score_only/truth.json',
        data_config=release+'/configs/d92_sourcefree_data_20260928.json',candidate_config=release+'/configs/d92_sourcefree_frozen_20260928.json',
        native_code=spec['confirmation']['native_code'],source_contract=spec['confirmation']['source_contract'],
        old_classes=cfg['old_classes'],candidate=candidate,candidate_method='D92-SFHead-v1',candidate_folder='sfhead',
        candidate_predictor='predict_d92_sourcefree_head.py',candidate_mode='d92_sfhead_registration',
        model_rows=4,splits_per_model=300,expected_split_count=300,predictions_total=2412)
    spec['code'].update(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        checkout=str(ROOT),cwd=release,commit_note='Exact pushed release HEAD recorded in launch/startup; this field is preparation parent.')
    spec['data'].update(target_receivers=cfg['target_receivers'],capsule_id=None,split_id='300 IDs from one preregistered builder',
        physical_ids_ref=data_root+'/builder_report.json',validation_ref=data_root+'/capsule/manifest.json',
        support_query_ref=data_root+'/capsule/splits',tx_sets_ref=release+'/configs/d92_sourcefree_data_20260928.json')
    spec['permissions'].update(query_use='read-only per sample; no fit/selection; truth only independent scorer',
        claim_scope='Paired frozen candidate confirmation on one current-unexposed complete receiver; all-time historical non-exposure not certified.')
    spec['execution'].update(remote_run_root=remote,remote_log_root=remote,local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_owner='codex/root/d92-upgrade-20260928',gpu_policy='Sequential GPU0 frozen inference; four CPU lanes each2BLASthreads for support head optimization; no encoder training',
        launch_command='C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 tools/publish_d92_confirmation.py --spec configs/d92_sourcefree_confirmation_20260928.json --release '+RELEASE)
    spec['metrics_plan'].update(scorer_ref=release+'/tools/score_d92_confirmation.py',
        prediction_ref=remote+'/each row/{predictions.jsonl,sfhead/predictions.jsonl}')
    spec['expected_artifacts']=['startup.json','state.json','complete.json','scores.json','each row/sfhead/fit_trace.jsonl','each row/sfhead/compact.csv']
    for row in spec['rows']:
        out=remote+'/'+row['row_id']
        row.update(method='paired D92 P2-256-FULL and D92-SFHead-v1',output_root=out,
            log_path=out+'/sfhead.log',config_ref='configs/d92_sourcefree_confirmation_20260928.json',resolved_config_ref=remote+'/startup.json',
            optimizer='CPU float64 L-BFGS-B; fixed objective; max_iter300,gtol1e-6,ftol1e-12',lr=None,epochs=None,
            budget_ref='300pairedsplits+3DG permodel; optimizermaxiter300 perfit; no target-based selection',
            command=f'/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python {release}/tools/run_d92_confirmation.py --spec {release}/configs/d92_sourcefree_confirmation_20260928.json --commit RELEASE_HEAD',
            expected_artifacts=['ground/manifest.json','received_features/features_complete.json','predictions_complete.json','sfhead/predictions_complete.json','sfhead/compact.jsonl','sfhead/compact.csv','sfhead/fit_trace.jsonl'])
        row['seed_notes']='Frozen model seed as listed; source split392005 metadata only. Target split/data2026092705, augmentation2026092707, evaluation2026092706, receiver2027; support2026092711..15. Head optimization deterministic, no RNG.'
    write(ROOT/'configs/d92_sourcefree_confirmation_20260928.json',spec)
    data=copy.deepcopy(spec);data.update(run_id=DATA_RUN,display_name='D92无源辅助确认的独立received构建',stage='Phase2-data-preparation',
        description='RX20-19、26TX各180独立物理记录；3场景support30/query30，300划分。仅数据可用性选取，无模型评分。',parent_run_ids=[])
    data.pop('confirmation');data['execution'].update(remote_run_root=data_root,remote_log_root=remote,
        local_artifact_root='automation_reports/CV-SincNet/'+DATA_RUN,gpu_policy='CPU builder, no checkpoint/source access')
    data['checkpoint']=dict(initialization='none',sources=[],contract_check_ref=cfg['data_code'],provenance_verdict='NOT_APPLICABLE',selection_rule='none')
    data['rows']=[dict(row_id='received-data',method='fixed_received_builder',purpose='data_preparation',gpu=None,
        config_ref='configs/d92_sourcefree_data_20260928.json',resolved_config_ref=data_root+'/builder_report.json',data_overrides={},
        seeds=dict(model=None,split=cfg['data_seed'],data=cfg['data_seed'],augmentation=cfg['augmentation_seed'],support=None,evaluation=cfg['evaluation_seed']),
        seed_notes='No model; inner support seeds2026092711..15; receiver2027.',k=cfg['shots'],scenario=','.join(cfg['scenarios']),
        optimizer='none',lr=None,epochs=None,fl_rounds=None,budget_ref='4680 observations;300 splits',output_root=data_root,
        log_path=remote+'/build_data.log',command=f'/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python {release}/tools/build_d92_confirmation_data.py --config {release}/configs/d92_sourcefree_data_20260928.json',
        expected_artifacts=['builder_report.json','capsule/manifest.json','capsule/received.npz','score_only/truth.json'])]
    data['expected_artifacts']=data['rows'][0]['expected_artifacts'];data['metrics_plan']=dict(metric_names=[],dimensions=[],prediction_ref=None,scorer_ref=None)
    write(ROOT/'configs/d92_sourcefree_data_record_20260928.json',data)
    print(json.dumps(dict(run=RUN,data_run=DATA_RUN,candidate=candidate,split_count=300,total_predictions=2412)))

if __name__=='__main__':main()
