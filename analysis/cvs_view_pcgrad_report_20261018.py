"""Build report materials only from completed source and independently recounted scores."""
import argparse,copy,json,math
from pathlib import Path
MODES=('ORIGINAL','MEASURE','PROJECT')
NAMES=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
EPOCH_FIELDS=('epoch','epochs','phase','lr','epoch_time_s','nonfinite_val_metric_count','train_loss','train_loss_labeled','train_loss_tx_labeled','train_loss_unlabeled','train_tx_acc','val_tx_acc','val_tx_correct','val_tx_total')
METRICS=('concat/As_packets','concat/L_forwards','concat/SupCon','concat/SupCon_weight','concat/common_CE','concat/common_KL','concat/total_L','train/grad_clip_active','train/grad_clip_limit','train/loss','train/loss_labeled','train/loss_tx_labeled','train/loss_unlabeled','train/pseudo_truth_available')
NN_KEYS=('student_train_forward_packets','student_eval_forward_packets','teacher_train_forward_packets','teacher_eval_forward_packets')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def render(source,score,output):
 if source['status']!='COMPLETE_SOURCE_LOGS_VERIFIED' or score['status']!='VERIFIED' or source['run_id']!=score['run_id'] or score['CM_recounts']!=378 or not source['complete']:raise ValueError('Complete matched source and fixed score recount required')
 out=Path(output);out.mkdir(parents=True,exist_ok=False)
 public=copy.deepcopy(source);public.pop('parent_training_result',None)
 for mode in MODES:
  row=public['rows'][mode];row['private_metric_keys_parsed']=len(row['numeric_totals']);row['numeric_totals']={k:row['numeric_totals'][k] for k in METRICS if k in row['numeric_totals']};row['private_epoch_scalar_keys_parsed']=len(row['epoch_curves'][0]);row['epoch_curves']=[{k:r[k] for k in EPOCH_FIELDS if k in r} for r in row['epoch_curves']];row['training_cost']=score['cost'][mode]
 parent=source['parent_training_result'];parentNN=sum(parent[k] for k in NN_KEYS);newNN={m:sum(score['cost'][m][k] for k in NN_KEYS)+score['cost'][m]['prediction_NN'] for m in MODES}
 cost=dict(parent_shared_training_NN=parentNN,parent_updates=4440,new_NN_by_model=newNN,source_plus_prediction_NN=sum(newNN.values()),shared_real_CP_smoke_NN=2,development_synthetic_F_NN=4608,development_synthetic_F_updates=9,development_extra_F_VJP=12,development_2D_toy_VJP=20,new_actual_NN=sum(newNN.values())+2+4608,actual_NN_with_parent=parentNN+sum(newNN.values())+2+4608,new_real_updates=6660,actual_real_updates_with_shared_parent=11100,extra_training_VJP=sum(score['cost'][m]['extra_supervised_VJP_calls'] for m in MODES),extra_training_L_backward_packet_derivatives=sum(score['cost'][m]['extra_supervised_VJP_calls']*256 for m in MODES),parent_model_epochs_shared=20,new_model_epochs=30,cumulative_epochs_each=30,failed_r01_real_updates=0,IQ_or_NN_added_by_analysis=0)
 report=copy.deepcopy(score);formal=report.pop('formal_metrics');report['source_evidence']=dict(status=source['status'],matched=source['matched_comparisons'],objective_max_error=max(source['rows'][m]['supervised_objective_recount_max_error'] for m in MODES),gradient={m:source['rows'][m]['view_gradient'] for m in MODES});report['actual_cost']=cost
 report['joint_target_status']=dict(low_urban_above_60=score['summary']['PROJECT']['practical_low_urban']['accuracy']>.60,suburban_gap_exploratory_at_most_2pp=all(x<=2 for x in score['suburban_gaps_pp']['PROJECT'].values()),clean_gain_vs_strict_matched_noAs_B00='UNPROVEN_NO_MATCHED_NO_AS_CONTROL_IN_THIS_MATRIX',IQ_Ag_proved=False,Ag_As_interaction='NOT_DEFINED',cross_seed_proof=False,joint_goal_complete=False)
 write(out/'source_analysis.json',public);write(out/'result.json',report);write(out/'formal_metrics.json',dict(run_id=source['run_id'],status='VERIFIED',CM_count=378,records_by_mode=formal))
 lines=['# E30 固定三行完整结果','',source['run_id'],'','状态：ANALYZED／VERIFIED；三行固定E30、七视图prediction及独立truth-last全部完成。','', '固定判定：'+score['decision']+'。只有一个模型seed，新增IQ A_g=0，完整联合目标未完成。','','|模式|clean %|high sub %|mid sub %|low sub %|high urban %|mid urban %|low urban %|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for m in MODES:lines.append('|'+m+'|'+'|'.join(f"{score['summary'][m][v]['accuracy']*100:.6f}" for v in NAMES)+'|')
 lines+=['','|候选相对|clean pp|mid urban pp|low urban pp|全部固定条件|','|---|---:|---:|---:|---|']
 for r in ('ORIGINAL','MEASURE'):
  p=score['paired']['PROJECT__'+r];lines.append('|'+r+'|'+'|'.join(f"{p[v]['delta_accuracy_pp']:+.6f}" for v in ('clean','practical_mid_urban','practical_low_urban'))+'|'+str(score['criteria'][r])+'|')
 lines+=['','|模式|high suburban距clean pp|mid suburban距clean pp|low suburban距clean pp|','|---|---:|---:|---:|']
 for m in MODES:lines.append('|'+m+'|'+'|'.join(f'{score["suburban_gaps_pp"][m][v]:.6f}' for v in NAMES[1:4])+'|')
 lines+=['','|模式|实际修正更新|额外VJP|As梯度移除RMS比例|ordinary梯度等效系数均值|','|---|---:|---:|---:|---:|']
 for m in MODES:
  g=source['rows'][m]['view_gradient'];mult=g['effective_ordinary_gradient_multiplier']['mean'];lines.append(f"|{m}|{g['actual_projection_updates']}|{g['extra_VJP_calls']}|{g['actual_projection_L2_RMS_fraction_of_As_gradient']:.6f}|{mult if mult is not None else '未测量'}|")
 lines+=['','标量复算说明执行与记录自洽，不重建全部参数梯度；ordinary系数变化及两视图范数不含SupCon/U、clip或AdamW，不能当作识别有效证据。MEASURE记录的coefficient是理论值，实际修正为0。','', '|模式|训练wall s|预测wall s|CPU评分wall s|新增NN包前向|','|---|---:|---:|---:|---:|']
 for m in MODES:
  c=score['cost'][m];lines.append(f"|{m}|{c['elapsed_seconds']:.3f}|{c['prediction_wall_seconds']:.3f}|{c['scorer_CPU_wall_seconds']:.3f}|{newNN[m]}|")
 lines+=['','总新增NN（含失败发布、开发及共享CP smoke）'+str(cost['new_actual_NN'])+'；父E20共享训练NN '+str(parentNN)+' 单列。新增真实更新6660，含父共享11100；新增模型轮30，含父唯一训练50模型轮，各模型累计E30。额外训练VJP '+str(cost['extra_training_VJP'])+'；开发fullF额外VJP12、二维toy20，fullF合成更新9单列。不同GPU／并发下的wall不作为严格速度归因。','','源数据：全部五RX、L6300/U56700隐藏TX/V27000，无训练域留出。完整2220step／10epoch／370pool／4440 L-U序列、原监督reference和unit记录，以及2220条视图梯度记录均完成复算。实际native args各857项，仅candidate_id/output_dir不同。源V与损失曲线仅作机制和健康解释。','','七视图共3528000决策、378混淆矩阵／63分区和21个配对比较独立复算；只读已评分聚合及correctness，不重新读取raw truth／IQ／checkpoint。所有RX/TX/day结果均保留，不删困难层。','','无严格匹配no-As B00、无新IQ A_g、无四因子交互或跨seed证据；不能把clean或urban局部收益解释为完整联合成功，也不根据这些目标结果调参、换父、扩seed或选择性重跑。','','完整逐层结果见stratified_results.md，混淆矩阵见formal_metrics.json；曲线与七视图见figures/。']
 (out/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 strata=['|比较|视图|维度|层|N|左accuracy %|右accuracy %|差 pp|左macroF1 %|右macroF1 %|','|---|---|---|---|---:|---:|---:|---:|---:|---:|']
 for r in score['strata']:strata.append('|'+str(r['comparison'])+'|'+r['view']+'|'+r['dimension']+'|'+r['stratum']+'|'+str(r['count'])+'|'+f"{100*r['left_accuracy']:.6f}|{100*r['right_accuracy']:.6f}|{r['delta_pp']:+.6f}|{100*r['left_macro_F1']:.6f}|{100*r['right_macro_F1']:.6f}|")
 (out/'stratified_results.md').write_text('\n'.join(strata)+'\n',encoding='utf8')
 return dict(status='REPORT_MATERIALS_WRITTEN',files=5,IQ=0,CP=0,NN=0,raw_truth_reads=0,updates=0,cost=cost)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--score',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(render(read(a.source),read(a.score),a.output),allow_nan=False))
