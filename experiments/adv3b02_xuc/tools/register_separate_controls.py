"""Register exact native and game-only rows before N607 launch."""
import importlib.util,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];WT=ROOT.parents[1];BASE=Path('E:/type10-7')
RUN='phase1_daot_rc4_pure_game_m3_20260917_r1'
REPORT=BASE/'automation_reports/CV-SincNet'/RUN
sys.path.insert(0,str(ROOT/'code'));sys.path.insert(0,str(ROOT/'code/scripts'))
from SSDG import train_ssdg as native
from cvsrffi.xuc_fusion.response_config import make_row
from cvsrffi.xuc_fusion.runtime import resolve_args
spec=importlib.util.spec_from_file_location('registry',BASE/'tools/experiment_registry.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
matrix=json.loads((ROOT/'configs/separate_controls/matrix.json').read_text());data=m.template()
project='/home/szu2070436088/2510044040/CV-SincNet';remote=project+'/runs/'+RUN
data.update(run_id=RUN,group_id='phase1-daot-response-game-separated-controls',display_name='原生DAOT＋RC4与仅动力博弈三seed对照',
    description='3行原生F0配方DAOT＋RC4；18行仅动力博弈SIM/EG/CF/TR/XT/DRIC。共同source物理角色；新18行与原联合18行同seed、同预算及响应参数，移除全部DAOT/RC4功能；原生配方作独立recipe比较。',
    kind='cvs',stage='Phase1',authorization='用户2026-09-17明确要求启动DAOT＋RC4三种子及仅动力博弈以上方法三种子进行对比',tags=['daot','rc4','response','core90','three_seed'])
data['code']=dict(commit='FIXED_AT_RELEASE_IN_LANDING_EVIDENCE',checkout=str(WT),cwd='immutable N607 release',environment='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python; local validation ssr-gpu')
data['data'].update(dataset='WiSig ManySig CORE90',representation='equalized=1; 2x256; normalized center crop',contract_ref=project+'/runs/phase1_adv3b02_xuc15_s392005_20260913_r1/source_contract.json',physical_ids_ref=remote+'/source_contract.json',label_map_ref=str(ROOT/'configs/core90_recipe_reference.json'),source_receivers=[1,3,4,6,8],target_receivers=[0,2,5,7,9,10,11],source_days=[1,2,3],target_days=[0,1,2,3],roles=dict(L_s=6300,U_s=56700,V=27000),leo_config_ref=str(ROOT/'configs/core90_recipe_reference.json'))
data['checkpoint'].update(initialization='scratch',sources=[],provenance_verdict='NO_EXTERNAL_CHECKPOINT_OR_TEACHER; every row exact physical-role check at startup',contract_check_ref=remote+'/<row>/source_contract.json',selection_rule='fixed final E200; no target feedback')
data['permissions'].update(query_use='none in training; frozen final predictions then independent scoring',claim_scope='previously exposed benchmark exploratory recipe and ablation comparison; no blind confirmation')
command='python tools/publish_response_release.py --separate-controls --run-id '+RUN+' --output '+str(REPORT/'landing')
data['execution'].update(host='N607',launch_owner='/root',remote_run_root=remote,remote_log_root=project+'/logs/'+RUN,local_artifact_root=str(REPORT),launch_command=command,stop_rule='A technical row failure holds new submissions and drains healthy rows; no performance stop; no automatic retrain or target selection')
data['expected_artifacts']=['effective_matrix.json','pipeline_state.json','per-row resolved_config.json','per-row source_contract.json','final_ssdg.pth','completion.json','full source metrics','frozen predictions and independent scores after training']
data['metrics_plan'].update(metric_names=['accuracy','Macro-F1','weak RX/TX','paired seed delta','mean and sample SD','cost'],dimensions=['row','seed','RX','day','TX','scene'],prediction_ref='fixed E200; same opaque target package, batch256, eval seed392005',scorer_ref=str(ROOT/'tools/score_native_joint_predictions.py'))
rows=[];checks=[]
recipe=json.loads((ROOT/'configs/core90_recipe_reference.json').read_text())
for item in matrix['rows']:
    row=item['row'];rid=row['id'];seed=row['joint']['model_seed'];path=ROOT/'configs/separate_controls'/(rid+'.json');doc=json.loads(path.read_text());native_row=item['family']=='native_baseline'
    if native_row:
        argv=[]
        for k,v in doc['options'].items():
            argv.append(k)
            if v is not None:argv.append(str(v).format(project_root=project))
        argv+=['--output_dir','VALIDATE_ONLY/'+rid]
        args=native.build_arg_parser().parse_args(argv);native._validate_a1_scratch_only(args);native._validate_daot_config(args)
        assert args.seed==seed and args.fasttrust_rc4 and args.use_adv3b02_daot_stn and not args.use_a1_r3
        assert args.a1_source_screen_only and args.muse_external_final_eval and not args.baseline_ckpt and not args.teacher_ckpt
        seeds=dict(model=seed,split=392005,data=seed,augmentation=seed,support=None,evaluation=392005)
        note='support不适用；连续物理ID划分无随机步骤，逐角色与392005契约精确比对。原生seed控制初始化/loader随机流/部分增强；sat_view_seed='+str(args.sat_view_seed)+'。不宣称与响应入口随机轨迹完全相同。'
        actual=vars(args);budget='E200 native full unlabeled epochs; AMP may skip nonfinite gradients; disclose accepted updates';method='DAOT+RC4 native F0 recipe; no added game solver'
    else:
        assert row==make_row(rid,row['response'],seed,pure_game=True)
        args=resolve_args(recipe,row,dataset='ManySig.pkl',output='VALIDATE_ONLY/'+rid)
        assert not args.joint['native_dr'];actual=vars(args)
        seeds=dict(model=seed,split=392005,data=392005,augmentation=392005,support=None,evaluation=392005)
        note='support不适用；仅model seed扫描，其他随机角色固定';budget='E200 x 222 accepted steps = 44400';method='pure '+row['response']['method']+'; DAOT/RC4 routes, calibration and auxiliary losses disabled; L/U adversarial heads retained'
    cfgout=REPORT/'resolved_configs'/(rid+'.json');cfgout.parent.mkdir(parents=True,exist_ok=True);cfgout.write_text(json.dumps(actual,indent=2,default=str)+'\n',encoding='utf-8')
    r=m.template()['rows'][0];r.update(row_id=rid,method=method,purpose=item['family'],config_ref=str(path),resolved_config_ref=remote+'/'+rid+'/resolved_config.json',seeds=seeds,seed_notes=note,optimizer='AdamW',lr=float(args.lr),epochs=200,budget_ref=budget,scenario=['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak'],output_root=remote+'/'+rid,log_path=project+'/logs/'+RUN+'/'+rid+'.train.log',command='effective command generated by dispatch_separate_controls.command; recorded in pipeline_state.json and row.launch.json',expected_artifacts=['final_ssdg.pth','completion.json','source_contract.json','initialization.json'])
    rows.append(r);checks.append(dict(row=rid,parser='PASS',scratch=True,seed=seed))
data['rows']=rows;data['notes']=['原生对照保留历史AMP和fasttrust调度；新18行FP32，其他预算/参数与本批联合响应行相同。','不重复训练已有联合18行；原生3行全部新scratch启动，历史392005不代替本轮模型。','最终比较需训练完成后执行固定模型truth-last评分；不得根据目标结果选择性重跑。','TX0–5；无Phase2 support/capsule；空字段不适用。']
assert not m.validate_spec(data,True)
REPORT.mkdir(parents=True,exist_ok=True)
(REPORT/'experiment.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(REPORT/'config_acceptance.json').write_text(json.dumps(dict(status='PASS',rows=checks),indent=2)+'\n',encoding='utf-8')
(REPORT/'events.jsonl').write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),status='PLANNED',note='21 rows authorized, all exact parsers validated',evidence=['experiment.json','config_acceptance.json']))+'\n',encoding='utf-8')
(REPORT/'report.md').write_text('# 原生DAOT＋RC4与仅动力博弈三seed对照\n\n本轮21行：原生DAOT＋RC4三seed；SIM/EG/CF-EG/TR-EG/XT-DANN/DRIC仅博弈各三seed。seed392005/392006/392007，全部从零训练E200。逐行配置、seed角色和输出见experiment.json。\n\n仅博弈路径完全跳过DAOT、RC4校准/路由/H/P及自监督和z_dom辅助损失，保留共同基础目标及L/U域对抗头监督，U到编码器对抗梯度仍为0。原旧R0关闭行仍有RC4辅助项，不能替代本轮纯博弈。新18行与已完成联合响应行按方法/seed配对，包含移除整套DAOT/RC4的效应。\n\n原生三行采用F0_FIXED_BASE训练配方，保留AMP和fasttrust调度，不附加响应求解器；源域连续物理ID逐角色核对同一契约。原生训练seed控制多个训练随机流，与新响应入口不同，故为配方比较，不冒充严格单因素消融。\n\nsource为RX1/3/4/6/8、day1/2/3，L/U/V=6300/56700/27000。训练不构造目标loader，所有checkpoint来源为空、EMA由本轮学生产生。最终E200冻结后按相同opaque测试包、batch256、eval seed392005做Clean及三LEO评分，单列day0跨天＋跨接收机结果；不据目标成绩改参或重跑。\n\n每GPU最多2个训练，不干预已有健康任务；技术失败暂停后续排队，已有健康子任务继续。尚未完成训练，不能提供新测试结论。\n',encoding='utf-8')
print(str(REPORT))
