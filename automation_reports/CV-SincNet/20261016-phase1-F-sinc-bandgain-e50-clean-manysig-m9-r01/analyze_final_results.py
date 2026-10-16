"""Create the F target report only from complete, independently recounted rows."""
import csv,json,statistics,sys
from pathlib import Path
P=Path(sys.argv[1]);data=json.loads((P/'final_collected.json').read_text(encoding='utf8'));audit=json.loads((P/'final_target_recount.json').read_text(encoding='utf8'));result=data['result']
if data['status']!='COLLECTED_COMPLETE' or audit['status']!='VERIFIED' or len(result['records'])!=162 or data['new_steps_recounted']!=66600 or data['new_U_packets']!=5783400:raise ValueError('Complete independent F target/source execution evidence required')
S=audit['summary'];C=audit['comparisons'];rows=[]
for arm in ('FULL','NORM','BAND'):
 for v in S[arm]['per_seed']:
  cost=next(r for r in result['costs'] if r['arm']==arm and r['model_seed']==v['model_seed']);rows.append(dict(v,source_seconds=cost['source_seconds'],new_epochs=cost['new_epochs'],new_steps=cost['new_steps'],historical_epochs_separate=cost['historical_epochs_separate'],network_parameters=900722,feature_parameters=232033,default_head_parameters=960))
with (P/'per_seed_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
pairs=[dict(comparison=name,**v) for name,value in C.items() for v in value['per_seed']]
with (P/'paired_differences.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
pct=lambda x:f'{100*x:.4f}'
improved=[arm for arm in ('NORM','BAND') if S[arm]['accuracy']['mean']>S['FULL']['accuracy']['mean']]
headline='本轮在固定三种子目标域 clean 测试中观察到平均准确率增益。' if improved else '本轮未获得相对 F 的平均准确率提升。'
lines=['# F 三路＋统一 Sinc＋RDrop：目标域 clean 完整结果','', '**'+headline+'**','', '本轮性能结论来自测试集目标域。FULL是已完成F的三seed匹配基线；NORM/BAND各新增三seed，从零50轮=33L+17U。固定seed2026092701/02/03、原fullsource角色、默认头随骨干联合训练，三个独立SIG_TIME描述子共同进入joint160。没有另训或替换头、模型投票、MEAN_VAR/COV，也没有源域性能晋级门槛。','', '|条件|seed01|seed02|seed03|目标准确率均值 ± 样本SD|Macro-F1均值 ± SD|最差RX均值|最差day均值|','|---|---|---|---|---|---|---|---|']
for arm in ('FULL','NORM','BAND'):
 s=S[arm];lines.append('|'+arm+'|'+'|'.join(pct(v['accuracy'])+'%' for v in s['per_seed'])+'|'+pct(s['accuracy']['mean'])+'% ± '+pct(s['accuracy']['sample_SD'])+' pp|'+pct(s['macro_f1']['mean'])+'% ± '+pct(s['macro_f1']['sample_SD'])+' pp|'+pct(s['worst_RX']['mean'])+'%|'+pct(s['worst_day']['mean'])+'%|')
lines+=['','|配对比较|seed01／02／03准确率差（百分点）|均值 ± 样本SD（百分点）|正差数|','|---|---|---|---|']
for name,v in C.items():lines.append('|'+name+'|'+' / '.join(f"{p['accuracy_difference_pp']:+.6f}" for p in v['per_seed'])+'|'+f"{v['accuracy_difference_pp']['mean']:+.6f} ± {v['accuracy_difference_pp']['sample_SD']:.6f}"+'|'+str(v['positive_seeds'])+'/3|')
for arm in ('NORM','BAND'):
 v=C[arm+'-FULL'];diff=v['accuracy_difference_pp']['mean'];mf=100*(S[arm]['macro_f1']['mean']-S['FULL']['macro_f1']['mean']);lines+=['',arm+'相对F的平均准确率差为'+f'{diff:+.6f}'+'个百分点，Macro-F1差为'+f'{mf:+.6f}'+'个百分点，'+str(v['positive_seeds'])+'/3准确率正差；'+('达到' if S[arm]['accuracy']['mean']>=.84 else '未达到')+'84%三种子均值门槛。三seed是描述性证据，不据此宣称统计显著或普遍收益。']
bn=C['BAND-NORM'];lines+=['','BAND−NORM的均值为'+f"{bn['accuracy_difference_pp']['mean']:+.6f}"+'个百分点，'+str(bn['positive_seeds'])+'/3正差。NORM匹配自身输入上的BAND反事实自适应位移范数，不匹配不同训练轨迹或含物理衰减的总移动；该比较区分频带重分配和全局缩幅，不能把一项单seed差写成稳定互补。','', 'FULL使用原统一物理Adam；BAND用w_i=max(初始有效带宽_i,Fs/K)/Fs逐通道缩放，Fs25MHz、K79；NORM用正全局倍率匹配自身步的BAND自适应L2。所有条件保留p=p0+25000q、初始q0、lr_q=alphaFs/25000、eps_q=epsilon25000/Fs、物理Hz全局clip5，以及一次未缩放的q-=alpha lambda(q+p0/s)。moment和EMA同坐标、实际叶版本缓存失效沿用F；只调整Sinc自适应增量，不改分类头。','', '原普通增强/LS0.01/RDrop alpha1，两个随机L前向复用同次实际增强IQ；旧batch_neighbor门控/ownEMA/cos、NoExplicitCFO和原source L6300/U56700隐藏/V27000、RX1/3/4/6/8、day1/2/3、原6TX及split392005均保留。每新模型固定11100步，完整重计66600步、5783400个隐藏U观察，无非有限跳步或U真值可用；三路局部投影与student/EMA Sinc、移除W后的joint160边界检查保留。','', '|条件|源阶段均值秒 ± 样本SD|新模型轮数|历史模型轮数单列|网络／特征／W参数|','|---|---|---|---|---|']
for arm in ('FULL','NORM','BAND'):
 rr=[r for r in rows if r['arm']==arm];seconds=[r['source_seconds'] for r in rr];lines.append('|'+arm+'|'+f'{statistics.mean(seconds):.2f} ± {statistics.stdev(seconds):.2f}'+'|'+str(sum(r['new_epochs'] for r in rr))+'|'+str(sum(r['historical_epochs_separate'] for r in rr))+'|900722 / 232033 / 960|')
lines+=['','新训练共6模型、300模型轮、66600更新；历史F的3模型/150轮/33300更新单列。每行168000个clean决策，9行1512000个逻辑决策，1008000个新决策；每行18个RX/day/TX等分层，合计162个混淆矩阵，均独立整数复算。配对使用完全相同物理ID的correctness，保留修复与新增错误。时间包括训练函数加载/训练/源验证，GPU并发和共享负载不同，不把墙钟差直接当成算法速度收益。','', '9行及所有三seed在任何新目标汇总之前已固定并提交；worker完成final50后自动clean预测，再由独立CUDA为空的CPU scorer连接truth。sourceV只作健康诊断，没有按target选择checkpoint、改参数、重排、停止或选择性重跑；所有负结果按原矩阵保留。旧单路source20结果独立保留，未替代本测试集结论。','', '本结果属于已暴露ManySig clean默认头测试；不证明新TX、Phase2或真实卫星/LEO部署收益。每种方法只有3个模型seed，样本SD用ddof=1，物理包数量不冒充模型seed数量。','', '逐seed明细见per_seed_metrics.csv；配对整数差见paired_differences.csv；完整目标分层及独立复算见final_collected.json、final_target_recount.json。执行提交bc9130137155ae46cf302795465757b2f23e9f59，源码与配置见experiment.json/report.md。']
(P/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf8');summary=dict(status='VERIFIED',run_id=result['run_id'],summary=S,comparisons=C,improved_vs_F=improved,new_steps_recounted=66600,new_U_packets=5783400,head_fit_updates=0,target_feedback=False)
(P/'analysis_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(status='VERIFIED',improved_vs_F=improved,target_accuracy={k:v['accuracy'] for k,v in S.items()})))
