"""Register the fixed completed-row evaluation without altering training."""
import importlib.util,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
BASE=Path('E:/type10-7')
REPORT=BASE/'automation_reports/CV-SincNet/response_matrix_eval_20260917'
WT=BASE/'code/snapshots/native_dr_eg_prepare_20260914_wt'
ART=BASE/'local_artifacts/response_matrix_eval_20260917_r1'
PY='C:/Users/lh594/.conda/envs/ssr-gpu/python.exe'
spec=importlib.util.spec_from_file_location('registry',BASE/'tools/experiment_registry.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
inv=json.loads((REPORT/'inventory.json').read_text(encoding='utf-8'))
data=m.template();run=inv['state']['run_id']
command=f'{PY} -X utf8 {WT}/experiments/adv3b02_xuc/tools/evaluate_completed_response.py --output {ART} --inventory {REPORT}/inventory.json --reuse-base {BASE}/local_artifacts/response_matrix_eval_20260916_r1'
data.update(run_id=REPORT.name,group_id='phase1-daot-response-game-evaluation',display_name='响应博弈矩阵已完成33行固定E200测试评估',description='复用16行已固定评分，新增17行；只按完成状态纳入，不按目标成绩选择。完整36行状态保留。',kind='cvs',stage='Phase1 evaluation',authorization='用户于2026-09-17要求对完成的实验执行测试集评估并给出完整数据',parent_run_ids=[run],tags=['response','core90','evaluation'])
data['code']=dict(commit=subprocess.check_output(['git','-C',str(WT),'rev-parse','HEAD'],text=True).strip(),checkout=str(WT),cwd=str(BASE),environment=PY+'; ssr-gpu; local CUDA')
data['data'].update(dataset='WiSig CORE90',representation='equalized received IQ',contract_ref=str(ART/'source_contract.json'),physical_ids_ref=str(ART/'target_inputs/manifest.json'),label_map_ref=str(ART/'truth_sidecar.json'),source_receivers=[1,3,4,6,8],target_receivers=[0,2,5,7,9,10,11],source_days=[1,2,3],target_days=[0,1,2,3],roles=dict(L_s=6300,U_s=56700,V=27000),leo_config_ref=str(WT/'experiments/adv3b02_xuc/configs/core90_recipe_reference.json'))
data['permissions'].update(claim_scope='Previously exposed benchmark descriptive evaluation; no feedback to training or selection',query_use='frozen prediction then independent scorer')
data['checkpoint']=dict(initialization='evaluation_only',sources=[dict(run_id=run,root=f'/home/szu2070436088/2510044040/CV-SincNet/runs/{run}',origin='scratch; no resume/teacher/pretrained upstream; training EMA derived within row')],contract_check_ref=str(ART/'state.json'),provenance_verdict='Existing scratch source contract; evaluator checks every row role_ids, initialization and fixed 44400 steps before prediction',selection_rule='All training-complete rows at inventory time; fixed final E200')
data['execution'].update(host='local RTX5070Ti; N607 read-only download',launch_owner='/root',local_artifact_root=str(ART),launch_command=command,stop_rule='Stop on missing artifact, source-role mismatch, invalid checkpoint, inference/scorer error; preserve outputs; do not affect training')
data['expected_artifacts']=['state.json','per-row predictions.json','per-row score.json','results/summary.csv','results/all_groups.csv','results/complete_test_data.xlsx']
data['metrics_plan'].update(metric_names=['accuracy','macro_f1','confusion','weak RX/TX','paired difference','cross-seed mean and sample SD'],dimensions=['row','seed','RX','day','TX','scene'],scorer_ref=str(WT/'experiments/adv3b02_xuc/tools/score_native_joint_predictions.py'))
rows=[]
for rid,entry in inv['state']['rows'].items():
    if entry['status']!='TRAINING_COMPLETE':continue
    cfg=inv['workers'][rid]['resolved'];j=cfg['joint'];row=m.template()['rows'][0]
    row.update(row_id=rid,method=rid.rsplit('_s',1)[0],purpose='fixed completed checkpoint evaluation',config_ref=str(REPORT/'inventory.json')+'#/workers/'+rid+'/resolved',resolved_config_ref=str(ART/rid/'resolved_config.json'),seeds=dict(model=j['model_seed'],split=392005,data=392005,augmentation=392005,support=None,evaluation=392005),seed_notes='support not applicable; fixed source contract/split and augmentation seeds; model seed varies',scenario=['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak'],optimizer='AdamW (training)',lr=cfg['lr'],epochs=200,budget_ref='44400 accepted training steps; 168000 samples x 4 evaluation scenes',output_root=str(ART/rid),log_path=str(ART/rid/'predict.log'),command=command,expected_artifacts=['final_ssdg.pth','prediction/predictions.json','score.json'])
    rows.append(row)
data['rows']=rows;data['notes']=['No few-shot support or Phase2 capsule; null fields not applicable unless specified.','Source roles checked before prediction. Truth fetched after all new predictions fixed.','33 completed and 3 incomplete at frozen inventory; previous 16 scored rows reused.']
errors=m.validate_spec(data,True);assert not errors,errors
assert len(rows)==33 and not ART.exists()
(REPORT/'experiment.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(REPORT/'events.jsonl').write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),status='PLANNED',note='33 completed rows frozen; 16 reused, 17 new; no training launch',evidence=['inventory.json','experiment.json']))+'\n',encoding='utf-8')
(REPORT/'report.md').write_text('# 响应博弈矩阵测试评估\n\n已冻结33个完成行，复用16行，新增17行。配置见experiment.json，实时完成证据见inventory.json。评估使用固定E200；不反馈训练或选模。结果完成后写入results/report.md。\n',encoding='utf-8')
print(command)
