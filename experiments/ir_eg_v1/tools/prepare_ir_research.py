"""Prepare launch-disabled IR matrices and their single canonical run record."""
from pathlib import Path
import json
import sys
import subprocess

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from cvsrffi.xuc_fusion.ir_experiments import write_configs,BASE


def main():
    rows=write_configs(ROOT/'configs')
    record=json.loads((ROOT/'acceptance/registration_template.json').read_text(encoding='utf-8'))
    run='20260927-phase1-ir-eg-wisig-m3-r01'
    record.update(run_id=run,group_id='phase1-native-joint-implicit-response',
        display_name='IR-EG隐式域头响应与BR历史预测研究准备',
        description='IR-ref/IR-cached开发验收；SIM/EG/IR性能与G0/OR/encoder-off机制消融；BR-EG/BR-IR效率确认。未启动真实数据训练。',
        status='PREPARED_NOT_LAUNCHED',tags=['IR-EG','BR-IR-EG','DAOT','FastTrust','source-only'],
        comparison_group_id='native-joint-same-exposure-44400',
        authorization='Current user requested implementation against IR_EG_DESIGN_SPEC and IMPLEMENTATION_PLAN; remote training not authorized.')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    record['code'].update(commit=commit,base_commit=BASE,checkout=str(ROOT.parents[1]),environment='ssr-gpu; PyTorch 2.10.0+cu128',cwd=str(ROOT))
    record['data'].update(dataset='ManySig',representation='equalized',
        contract_ref='experiments/ir_eg_v1/docs/IR_EG_DESIGN_SPEC.md#11-不可改变的数据与监督边界',
        source_receivers=[1,3,4,6,8],target_receivers=[0,2,5,7,9,10,11],source_days=[1,2,3],target_days=[0,1,2,3],
        roles=dict(L_s={'count':6300,'fraction':.07},U_s={'count':56700,'fraction':.63,'tx_labels':'hidden'},V={'count':27000,'fraction':.30}),
        train_ratio=.1,leo_config_ref='experiments/ir_eg_v1/configs/core90_recipe_reference.json')
    record['permissions'].update(claim_scope='development correctness and bounded synthetic acceptance; no new RFFI performance claims')
    record['checkpoint'].update(initialization='scratch',sources=[],provenance_verdict='NO_INHERITED_WEIGHTS',
        selection_rule='source_V_only; freeze before independent prediction/scoring')
    record['execution'].update(host='local_development_only',launch_owner='current_root_agent',
        local_artifact_root='experiments/ir_eg_v1/acceptance',
        launch_command=None,stop_rule='No real training launched. Future safety cap is not convergence.')
    record['expected_artifacts']=['source-only checkpoint','resolved config','per-step telemetry','profiling','frozen prediction','independent scorer output']
    record['metrics_plan'].update(metric_names=['clean','LEO mean','Macro-F1','worst TX','worst RX','day0','RX-by-TX','wall seconds','peak bytes'],dimensions=['model_seed','method','scene','RX','TX','day'],
        prediction_ref='freeze model and candidate before independent target inference',scorer_ref='tools/score_native_joint_predictions.py (separate truth-last process)')
    record['rows']=[]
    for key,item in rows.items():
        config=f'experiments/ir_eg_v1/configs/ir_rows/{key}.json'
        record['rows'].append(dict(row_id=key,method=item['label'],purpose='source_screen_then_frozen_confirmation',
            status='PLANNED',config_ref=config,resolved_config_ref=None,data_overrides={},seeds=item['seeds'],seed_notes=item['seed_notes'],
            k=None,scenario='clean plus paired LEO views',optimizer='AdamW',lr=.0002,epochs=200,fl_rounds=None,
            budget_ref='experiments/ir_eg_v1/configs/ir_eg_v1.json',output_root=item['output'],log_path=item['output']+'/actions.jsonl',
            command=f'python code/scripts/train_response_games.py --config configs/ir_rows/{key}.json --output {item["output"]}',
            expected_artifacts=record['expected_artifacts']))
    record['notes']=[
        'All commands default to validate-only. No --execute, remote process, historical checkpoint, or target is used by this delivery.',
        'Physical ID/label-map/source manifest locators are unbound until a future authorized run; current expected counts are not fresh data validation.',
        'Support seed, Phase2 capsule/split and K are not applicable. Data version and complete continuation manifest remain unbound.',
        'Confirmation candidates cannot be selected from target scores. Every seed contrast and every attempted candidate/cost must be retained.',
        'Scientific convergence confirmation is a separate unfrozen protocol; E200/44400 only supplies budget-audit readings.'
    ]
    target=ROOT.parents[1]/'automation_reports/CV-SincNet'/run
    target.mkdir(parents=True,exist_ok=True)
    (target/'experiment.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    event=dict(status='PREPARED_NOT_LAUNCHED',timestamp='2026-09-27',note='Created code/configuration and bounded local synthetic acceptance; no real training execution',evidence='experiments/ir_eg_v1/docs/traceability.md')
    if not (target/'events.jsonl').exists():
        (target/'events.jsonl').write_text(json.dumps(event,ensure_ascii=False)+'\n',encoding='utf-8')
    print(target)


if __name__=='__main__':main()
