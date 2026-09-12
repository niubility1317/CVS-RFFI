"""Generate complete descriptive tables from the read-only evidence snapshot."""
from pathlib import Path
import csv
import gzip
import datetime
import json
import math
import re
import statistics

P=Path(__file__).resolve().parent
D=json.loads((P/'complete_evidence.json').read_text(encoding='utf-8') if (P/'complete_evidence.json').exists()
             else gzip.decompress((P/'complete_evidence.json.gz').read_bytes()).decode('utf-8'))
SCENES=['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak']
GROUPS={'test_unseen_day_seen_rx':'未知日期/已见RX','test_seen_day_unseen_rx':'已见日期/未知RX','test_unseen_day_unseen_rx':'未知日期/未知RX'}
V=list(D['rows']); data={}; flat=[]; curves=[]
def table(headers,rows):
    return ['|'+'|'.join(headers)+'|','|'+'|'.join(['---']+['---:']*(len(headers)-1))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows]
def fmt(x): return 'N/A' if x is None else f'{x:.4f}'
def finitevals(ms,key): return [(m['epoch'],float(m[key])) for m in ms if isinstance(m.get(key),(int,float)) and math.isfinite(m[key])]
for v,row in D['rows'].items():
    j=row['json']; ms=row['metrics']; audit=row['audit']
    assert audit['status']=='VERIFIED',(v,audit)
    assert [m['epoch'] for m in ms]==list(range(1,201)),v
    assert [m['phase'] for m in ms]==['label']*130+['pseudo']*70,v
    scores=j['final_predictions/independent_scores.json']; act=j['cross_response_activation.json']; res=j['phase1_resource_summary.json']; terminal=j['phase1_terminal_status.json']
    assert scores['status']==terminal['status']=='COMPLETE' and terminal['exit_code']==0
    assert act['status']=='ACTIVE_VERIFIED' and not act['missing']
    assert scores['truth_last'] and scores['selection_source']=='final_only'
    assert all(m.get('train_pseudo_truth_available',0)==0 for m in ms)
    errors=[l for l in row['stdout'].splitlines() if re.search(r'Traceback|\b\w*Error:|CUDA out of memory|Killed',l)]
    warnings=[l for l in row['stdout'].splitlines() if 'Warning' in l or '[WARN' in l]
    groups={}; f1={}; worst={}
    for scene in SCENES:
        aggregate=scores['test'] if scene=='clean' else scores['sat_test_named'][scene]['aggregate']
        named=scores['named_test'] if scene=='clean' else scores['sat_test_named'][scene]['named']
        groups[scene]=dict(aggregate=aggregate,named=named)
        matrix=audit['confusion'][scene]; fs=[]
        for i in range(len(matrix)):
            denom=sum(matrix[i])+sum(r[i] for r in matrix)
            fs.append(200*matrix[i][i]/denom if denom else 0.)
        f1[scene]=sum(fs)/len(fs)
        worst[scene]=min((named[f'test_unseen_day_rx_{rx}']['tx_acc'],rx) for rx in (0,2,5,7,9,10,11))
        for category,items in [('aggregate',{'all':aggregate}),('named',named),('receiver_all_days',scores['per_receiver'][scene]),('class',scores['per_class'][scene])]:
            for name,value in items.items():
                flat.append(dict(variant=v,scene=scene,category=category,group=name,correct=value['tx_correct'],
                    total=value['tx_total'],incorrect=value['tx_total']-value['tx_correct'],accuracy_percent=value['tx_acc']))
    stats={}
    for key in sorted(set().union(*(m.keys() for m in ms))):
        vals=finitevals(ms,key)
        if not vals: continue
        stats[key]=dict(first=vals[0],last=vals[-1],min=min(vals,key=lambda p:p[1]),max=max(vals,key=lambda p:p[1]),
            mean=statistics.mean(x for _,x in vals),count=len(vals))
    for m in ms: curves.append(dict(variant=v,**m))
    counts=act['counts']; steps=counts['batches']/200
    skip_grad=sum(m.get('train_skipped_nonfinite_grad',0)*steps for m in ms)
    skip_loss=sum(m.get('train_skipped_nonfinite_loss',0)*steps for m in ms)
    data[v]=dict(groups=groups,macro_f1=f1,worst_strict_rx=worst,counts=counts,resource=res,
        activation_status=act['status'],maxima=act['maxima'],gradient_counter_semantics=act.get('gradient_counter_semantics'),
        source_split=terminal['source_split_receipt'],checkpoint_sha256=terminal['selected_checkpoint_sha256'],
        logs=dict(epochs=200,stdout_lines=len(row['stdout'].splitlines()),errors=errors,warnings=warnings,
            skipped_gradient_steps=round(skip_grad),skipped_loss_steps=round(skip_loss),numeric_series=stats),
        diagnostics=row['diagnostics'],config=act['config'],run=row['run'])
    assert abs(skip_grad-round(skip_grad))<1e-6 and abs(skip_loss-round(skip_loss))<1e-6
    assert counts['batches']-counts['successful_steps']==round(skip_grad+skip_loss),v
assert all(data[v]['source_split']==data['U0']['source_split'] for v in V)
def acc(v,scene,group=None):
    g=data[v]['groups'][scene]
    return (g['aggregate'] if group is None else g['named'][group])['tx_acc']
means={v:statistics.mean(acc(v,s) for s in SCENES[1:]) for v in V}
contrasts={name:{s:acc(a,s)-acc(b,s) for s in SCENES} for name,a,b in [
    ('U1-U0:完整块运行整体差异','U1','U0'),('U1-U1_mask_off:角色mask','U1','U1_mask_off'),
    ('U3-U1:决策校准','U3','U1'),('Ux-U1:身份交互','Ux','U1'),
    ('Ux_normalized-Ux:归一化交互','Ux_normalized','Ux'),('head_only-U1:隔离辅助头','head_only','U1'),
    ('permanent_detach-U3:附加域响应及配置整体','permanent_detach','U3')]}
for d in contrasts.values(): d['leo_mean']=statistics.mean(d[s] for s in SCENES[1:])
oldpath=P.parents[1]/'core90_cross_response_s392005_20260911_r1/completed_detail/all_completed_summary.json'
old=json.loads(oldpath.read_text(encoding='utf-8')) if oldpath.exists() else {}
prior={}
for v in V:
    if v in old:
        prior[v]={s:acc(v,s)-old[v]['groups'][s]['aggregate']['tx_acc'] for s in SCENES}
        prior[v]['leo_mean']=statistics.mean(prior[v][s] for s in SCENES[1:])
with (P/'scores_long.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
keys=['variant']+sorted(set().union(*(m.keys() for m in curves))-{'variant'})
with (P/'epochs_all.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(curves)
summary=dict(variants=data,leo_means=means,contrasts=contrasts,previous_version_deltas=prior,
    audit={v:r['audit'] for v,r in D['rows'].items()},total_score_groups=len(flat),total_prediction_rows=sum(r['audit']['prediction_rows'] for r in D['rows'].values()))
(P/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
stamp=datetime.datetime.fromtimestamp(D['time'],datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
lines=['# 最新CORE90交叉响应V2：8行完整测试报告','',f'数据读取开始时间：{stamp}。8个已启动对照均完成E200及四场景预测/独立评分。r1的7行与r2重发head_only分别保留，原r1 head_only初始化失败不计作一次完成实验。','',
    f'完整解析1600轮训练、全部stdout及诊断JSONL；逐条验证{summary["total_prediction_rows"]:,}条固定预测，再读truth复算{len(flat)}组计数，全部一致。每项每场景198000条、6类；场景共享原始物理记录，因此四场景不能当作四倍独立样本。',
    '', 'ManySig equalized=1；source RX1/3/4/6/8、day1/2/3，目标RX0/2/5/7/9/10/11；seed392005，L_s/U_s/V=6300/56700/27000。E200=label130+pseudo70、batch128、49步/轮，scratch、final_only，U_s标签隐藏；本次8行source_split完全相同。','',
    '## 四场景总体准确率','', '单位为%；LEO均值为三个等样本量场景的算术均值。总体含未知日期/已见RX，不全部是跨接收机结果，须结合后面的分组。','']
lines+=table(['实验','clean','晴空LEO','低仰角LEO','雨衰LEO','LEO均值','clean−U1','LEO−U1'],
    [[v,*[fmt(acc(v,s)) for s in SCENES],fmt(means[v]),f'{acc(v,"clean")-acc("U1","clean"):+.4f}',f'{means[v]-means["U1"]:+.4f}'] for v in V])
lines+=['','## 同版对照差值','','单位：百分点。U0/U1存在loader、块组织与前向差异，不能将差值归为单一采样因素。']
lines+=table(['对照',*SCENES,'LEO均值'],[[name,*[f'{d[s]:+.4f}' for s in SCENES+['leo_mean']]] for name,d in contrasts.items()])
lines+=['','## 与上一版同名实验比较','','复用已核验的V1固定报告；数据/seed相同，但V2含多处机制与执行变化、head_only辅助事务也不同，此处仅描述整版差异。不是独立确认集，也不据此调参或重跑。']
lines+=table(['行',*SCENES,'LEO均值变化'],[[v,*[f'{d[s]:+.4f}' for s in SCENES+['leo_mean']]] for v,d in prior.items()])
lines+=['','## 三种泛化分组','', '30000条未知日期/已见RX＋126000条已见日期/未知RX＋42000条未知日期/未知RX=198000条。命名别名不重复累加。']
for group,title in GROUPS.items():
    lines+=['',f'### {title}','']+table(['实验',*SCENES],[[v,*[fmt(acc(v,s,group)) for s in SCENES]] for v in V])
lines+=['','## Macro-F1与最弱跨接收机分组','','Macro-F1由全部6×6混淆矩阵计算；每类权重相同。']
lines+=table(['实验',*SCENES],[[v,*[fmt(data[v]['macro_f1'][s]) for s in SCENES]] for v in V])
lines+=['','未知日期/未知RX中最弱接收机：每个RX每场景6000条。']
lines+=table(['实验',*SCENES],[[v,*[f'{data[v]["worst_strict_rx"][s][0]:.4f}（RX{data[v]["worst_strict_rx"][s][1]}）' for s in SCENES]] for v in V])
lines+=['','## 全部未知接收机明细','']
for scene in SCENES:
    for prefix,title in [('test_rx_','已见日期，每RX18000条'),('test_unseen_day_rx_','未知日期，每RX6000条')]:
        lines+=['',f'### {scene}：{title}','']+table(['实验',*[f'RX{rx}' for rx in (0,2,5,7,9,10,11)]],
            [[v,*[fmt(acc(v,scene,prefix+str(rx))) for rx in (0,2,5,7,9,10,11)]] for v in V])
lines+=['','## 全部类别准确率','','TX0–5为数据内部索引，每类每场景33000条。']
for scene in SCENES:
    lines+=['',f'### {scene}','']+table(['实验',*[f'TX{i}' for i in range(6)]],
        [[v,*[fmt(D['rows'][v]['json']['final_predictions/independent_scores.json']['per_class'][scene][str(i)]['tx_acc']) for i in range(6)]] for v in V])
lines+=['','## 成本与完整训练稳定性','','时间是此次并发环境下的观测，不能当作隔离性能基准；allocated是进程PyTorch峰值，不是整卡占用。']
lines+=table(['行','墙钟小时','峰值allocated GiB','成功更新/9800','非有限梯度跳步','非有限loss跳步','最终训练准确率','最终源V准确率'],
    [[v,fmt(data[v]['resource']['wall_time_seconds']/3600),fmt(data[v]['resource']['peak_cuda_memory_allocated_bytes']/2**30),data[v]['counts']['successful_steps'],data[v]['logs']['skipped_gradient_steps'],data[v]['logs']['skipped_loss_steps'],fmt(D['rows'][v]['metrics'][-1].get('train_tx_acc')),fmt(D['rows'][v]['metrics'][-1].get('val_tx_acc'))] for v in V])
lines+=['','跳步数按每轮49步×记录的跳步率求和，并与累计batch−成功更新计数交叉核对。原始stdout的所有警告、错误及所有数值序列的首末值/极值/均值保存在summary.json，1600轮完整字段保存在epochs_all.csv。','', '## 机制实际激活','']
lines+=table(['行','有效块','响应头梯度步','域响应梯度步','身份响应梯度步','决策梯度审计步','交互梯度审计步'],
    [[v,*[data[v]['counts'][k] for k in ['effective_blocks','head_gradient_steps','domain_gradient_steps','joint_gradient_steps','decision_gradient_steps','cross_gradient_steps']]] for v in V])
lines+=['','V2的决策/交互梯度计数仅计成功的审计步，不等于每步参与次数；响应路由计数覆盖成功更新。8项均ACTIVE_VERIFIED。所有已选行的身份响应均按定义关闭，因此本批不能证明联合身份响应、三条件源门或可靠反馈的效果。U2/U4_additive/U4_bilinear/U5/U3_delta/U3_delta_pairs/U4_decomposed/U5_reliable未启动，不填0分。','',
    '## 结论与未解决差异','',
    'Ux_normalized相对Ux：clean+1.9217、LEO均值+2.9101个百分点；是本轮较清晰的描述性改善。但相对U0，clean+1.4793而LEO均值−0.8266个百分点，所以没有同时改善两者。U0在三个LEO总体场景均最高。',
    '', 'U3相对U1的clean−0.1460、LEO均值−0.2325个百分点，本轮不支持决策校准带来收益。U1相对U1_mask_off的clean−0.9621、LEO均值−0.0855个百分点，角色mask未呈现稳定优势。',
    '', '未知日期/未知RX中，Ux_normalized的clean为75.1571%，低于permanent_detach的75.1786%和U1_mask_off的75.1643%；三LEO各项仍低于U0。最弱RX仍是RX2：Ux_normalized的clean为60.75%，LEO晴空仅39.2333%。总体均值未消除薄弱接收机问题。',
    '', '正式head_only与U1没有逐值相同：E1 train_loss分别17.000693262839803与17.000999567460042，E1源V准确率分别27.081481%与26.792593%；最终clean相差+1.1551个百分点。首个已有epoch汇总即不同，但没有本次逐step全状态轨迹，不能定位首次分歧或直接归因为辅助头副作用。合成确定性验证不能替代真实E200一致性证明；此差值不能写作辅助头学习提高身份表征。',
    '', '同名U0相对上一版clean+1.5899、LEO均值+3.2128个百分点，而其交叉响应本来关闭。跨版本同seed对照也存在训练结果漂移，不能把所有差值归因于新增机制。数据同契约和seed相同不等于已证明完整训练轨迹一致。',
    '', '训练/验证曲线见[完整曲线图](training_curves.png)。E1记录的train_tx_acc=0保留原日志口径，不能据此推断全部训练样本识别错误；该数值为日志聚合字段而非本次独立重算。E80总loss抬升对应卫星CE开始计入：各行train_w_loss_sat_cls_labeled由E79的0变为E80的4.57–4.76。阶段切换为E131，曲线中的跳变不能直接视为数值失败。','',
    '## 证据文件','', '- [1184行评分明细](scores_long.csv)：正确数、错误数、样本数及准确率；含命名别名，禁止直接累加所有行。',
    '- [1600轮完整训练字段](epochs_all.csv)。','- [摘要、所有混淆矩阵与对照差值](summary.json)。','- [完整日志、配置、评分和核验快照（gzip）](complete_evidence.json.gz)。',
    '- 原始逐样本预测与checkpoint保留在N607各row；本次仅只读分析，没有重训、调参、重评分回流或停止任务。',
    '', '本报告为单seed描述性结果，不提供跨seed显著性结论。历史目标结果已有公开观察，不将该相同测试集称为新的独立确认集；不得用本表做候选选择后再声称未经反馈的确认性成功。','']
(P/'detailed_results.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(dict(overall={v:[acc(v,s) for s in SCENES]+[means[v]] for v in V},contrasts=contrasts,prior=prior,
    errors={v:data[v]['logs']['errors'] for v in V},cost={v:{'hours':data[v]['resource']['wall_time_seconds']/3600,'steps':data[v]['counts']['successful_steps']} for v in V}),ensure_ascii=False,indent=2))
