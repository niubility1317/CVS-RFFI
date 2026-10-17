"""Public mandatory-reference report;only independently verified small metadata."""
import argparse,copy,json
from pathlib import Path
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
MODES=('B00','ORIGINAL','MEASURE','PROJECT')
NN_KEYS=('student_train_forward_packets','student_eval_forward_packets','teacher_train_forward_packets','teacher_eval_forward_packets')
METRICS=('concat/As_packets','concat/L_forwards','concat/SupCon','concat/SupCon_weight','concat/common_CE','concat/common_KL','concat/total_L','train/loss','train/loss_labeled','train/loss_unlabeled','train/pseudo_truth_available')
EPOCH_FIELDS=('epoch','epochs','phase','lr','epoch_time_s','nonfinite_val_metric_count','train_loss','train_loss_labeled','train_loss_unlabeled','train_tx_acc','val_tx_acc','val_tx_correct','val_tx_total')
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def render(value,output):
 if value['status']!='VERIFIED' or value['CM_recounts']!=504 or value['partition_recounts']!=84 or value['paired_comparisons']!=21:raise ValueError('Complete independent matching recount required')
 out=Path(output);out.mkdir(parents=True,exist_ok=False);r=copy.deepcopy(value);src=r.pop('source');formal=r.pop('formal_metrics');parent=src.pop('parent_training_cost');row=src['rows']['B00'];row['numeric_totals']={k:v for k,v in row['numeric_totals'].items() if k in METRICS};row['epoch_curves']=[{k:v for k,v in e.items() if k in EPOCH_FIELDS} for e in row['epoch_curves']];row['training_cost']=r['cost']['B00']
 parentNN=sum(parent[k] for k in NN_KEYS);c=r['cost']['B00'];newNN=sum(c[k] for k in NN_KEYS)+c['prediction_NN'];cost=dict(own_NoAs_parent_E20_NN=parentNN,own_NoAs_parent_real_updates=4440,new_NoAs_training_plus_prediction_NN=newNN,shared_real_CP_smoke_NN=2,development_synthetic_NN=2048,development_synthetic_updates=4,development_extra_F_VJP=4,new_actual_NN=newNN+2+2048,actual_NN_with_own_parent=parentNN+newNN+2+2048,new_real_updates=2220,real_updates_with_own_parent=6660,extra_training_VJP=4440,extra_training_L_backward_packet_derivatives=1136640,parent_epochs=20,new_epochs=10,cumulative_epochs=30)
 r['actual_cost']=cost;r['source_evidence']=dict(status=src['status'],all_source_RX=row['source_RX'],unique_L=row['observed_unique_L'],unique_U=row['observed_unique_U'],U_truth_available_max=row['numeric_totals']['train/pseudo_truth_available']['maximum'],matching=src['matched_As_comparisons'],As_packets=0,Ag_packets=0)
 r['interpretation_by_As_mode']={}
 for m in MODES[1:]:
  p=r['paired'][m+'__B00'];r['interpretation_by_As_mode'][m]=dict(clean_net_positive=p['clean']['delta_accuracy_pp']>0,midurban_net_positive=p['practical_mid_urban']['delta_accuracy_pp']>0,lowurban_net_positive=p['practical_low_urban']['delta_accuracy_pp']>0,lowurban_above_60=r['summary'][m]['practical_low_urban']['accuracy']>.60,each_suburban_gap_at_most_2pp=all(g<=2 for g in r['suburban_gaps_pp'][m].values()),IQ_Ag_proved=False,cross_seed_proof=False,joint_goal_complete=False)
 write(out/'result.json',r);write(out/'source_analysis.json',src);write(out/'formal_metrics.json',dict(status='VERIFIED',run_id=value['run_id'],CM_count=126,records=formal['B00'],other_378CM_ref='../'+value['reference_for_run']+'/formal_metrics.json'))
 lines=['# 匹配 E30 No-As 参考完整结果','',value['run_id'],'','状态 ANALYZED／VERIFIED。单一固定模型 seed，无新 IQ A_g。参考在读取当前 As 三行正式目标成绩前登记；没有按成绩晋级、重跑或修改配方。','','|模式|clean %|high sub %|mid sub %|low sub %|high urban %|mid urban %|low urban %|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for m in MODES:lines.append('|'+m+'|'+'|'.join(f"{100*r['summary'][m][v]['accuracy']:.6f}" for v in VIEWS)+'|')
 lines+=['','|As 模式相对 B00|clean pp|mid urban pp|low urban pp|','|---|---:|---:|---:|']
 for m in MODES[1:]:lines.append('|'+m+'|'+'|'.join(f"{r['paired'][m+'__B00'][v]['delta_accuracy_pp']:+.6f}" for v in ('clean','practical_mid_urban','practical_low_urban'))+'|')
 lines+=['','|模式|high suburban距clean pp|mid suburban距clean pp|low suburban距clean pp|','|---|---:|---:|---:|']
 for m in MODES:lines.append('|'+m+'|'+'|'.join(f"{r['suburban_gaps_pp'][m][v]:.6f}" for v in VIEWS[1:4])+'|')
 lines+=['','数据与模型：全部五源 RX [1,3,4,6,8]、三日、L6300/U56700 隐藏 TX/V27000，split392005，无训练 RX 留出或 L_dev 划分；仅现有 256 点 IQ。F＋跨 RX 同 TX SupCon、SIG_TIME 三路/统一 Sinc/joint160/既有 cosine 头联合训练，NoExplicitCFO。','','匹配边界：No-As 用原 fullsource REPEAT E20 student＋自身 EMA；As 三行用原 CONCAT E20 student＋自身 EMA。两父模型按同一原始 scratch seed、完整源契约、预算及初始化规则训练，因增强处理而权重不同。均追加 E10、重置 AdamW/cos10、固定 final30，不声称精确恢复原优化轨迹。','','四次 L 前向、物理序列、隐藏 U 与教师规则保持；B00/MEASURE/PROJECT 每步额外两次 VJP，ORIGINAL 每步无额外 VJP，因此严格额外求导成本匹配以 MEASURE 为主要 As 控制。As 的传播处理成本属于处理差异。每个 RDrop 对内复用已生成 IQ；不声称不同模型全部浮点轨迹逐位一致。','','L 损失质量明确：B00 为 .5ordinaryRD＋.5同一ordinaryIQ的额外RD＋.1SupCon；As 为 .5ordinaryRD＋.5完整practical(raw)RD＋.1SupCon。旧普通增强保留，所以 B00 不是完全无增强。As 视图直接从原 IQ 生成，没有在旧 ordinary 后重复加完整信道。此比较检验该固定 .5/.5 配方；不能把其 clean 损失解释为所有星地增强的必然代价。','','完整源日志、370 个输入池、4440 条 L/U 序列和两父来源已只读核对；全部七视图完成后独立 truth-last。504 个混淆矩阵、84 个分区和21 个 As-vs-B00 配对比较复算，无原始 truth／IQ／checkpoint 读取或新增网络计算。全部 RX/TX/day 保留见 stratified_results.md。','','No-As 新训练和预测 NN '+str(newNN)+'，另列真实 CP smoke2、开发合成 NN2048／更新4／VJP4；自身父 E20 NN '+str(parentNN)+'。新增真实更新2220、额外训练VJP4440，含自身父累计6660更新/E30。原 As 三行及其共享父成本在原报告单列，不将父训练或开发成本计为0。','','已关闭的固定梯度规则保持关闭。这里没有新 IQ A_g、B10/B11 新增强模型、Ag×As 交互或跨 seed 证明；完整联合目标未完成。']
 (out/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 strata=['|比较|视图|维度|层|N|As accuracy %|B00 accuracy %|差 pp|As macroF1 %|B00 macroF1 %|','|---|---|---|---|---:|---:|---:|---:|---:|---:|']
 for x in r['strata']:strata.append('|'+x['comparison']+'|'+x['view']+'|'+x['dimension']+'|'+x['stratum']+'|'+str(x['count'])+'|'+f"{x['left_accuracy']*100:.6f}|{x['right_accuracy']*100:.6f}|{x['delta_accuracy_pp']:+.6f}|{x['left_macro_F1']*100:.6f}|{x['right_macro_F1']*100:.6f}|")
 (out/'stratified_results.md').write_text('\n'.join(strata)+'\n',encoding='utf8')
 return dict(status='REPORT_WRITTEN',files=5,IQ=0,CP=0,NN=0,raw_truth=0,updates=0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(render(json.loads(Path(a.input).read_text(encoding='utf8')),a.output)))
