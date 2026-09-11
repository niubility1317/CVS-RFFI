import csv
import json
from pathlib import Path
P=Path(__file__).resolve().parent
V=('U0','U1','U2','Ux','head_only')
S=('clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak')
T=('clean','LEO晴空弱信道','LEO低仰角弱信道','LEO雨衰弱信道')
G={'test_unseen_day_seen_rx':'未知日期、已见RX','test_seen_day_unseen_rx':'已见日期、未知RX','test_unseen_day_unseen_rx':'未知日期、未知RX'}
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
audit={**read(P/'prediction_audit.json'),**read(P/'new_prediction_audit.json')}
scores={v:read(P/v/'independent_scores.json') for v in V}
acts={v:read(P/v/'cross_response_activation.json') for v in V}
resources={v:read(P/v/'phase1_resource_summary.json') for v in V}
agg=lambda v,s:scores[v]['test'] if s=='clean' else scores[v]['sat_test_named'][s]['aggregate']
named=lambda v,s:scores[v]['named_test'] if s=='clean' else scores[v]['sat_test_named'][s]['named']
mean=lambda v:sum(agg(v,s)['tx_acc'] for s in S[1:])/3
health={}
flat=[]
for v in V:
    assert audit[v]['status']=='VERIFIED' and audit[v]['compared_score_groups']==148
    m=[json.loads(l) for l in (P/v/'metrics_epoch.jsonl').read_text().splitlines() if l.strip()]
    log=(P/v/f'{v}.log').read_text(errors='replace')
    assert [x['epoch'] for x in m]==list(range(1,201))
    assert [x['phase'] for x in m]==['label']*130+['pseudo']*70
    assert all(x.get('train_pseudo_truth_available',0)==0 for x in m)
    errors=[l for l in log.splitlines() if any(k in l for k in ('Traceback','Error:','CUDA out of memory'))]
    assert not errors and acts[v]['status']=='ACTIVE_VERIFIED' and not acts[v]['missing']
    health[v]=dict(epochs=len(m),log_lines=len(log.splitlines()),final_source_val_accuracy=m[-1]['val_tx_acc'],
                  final_train_accuracy=m[-1]['train_tx_acc'],errors=errors,
                  successful_steps=acts[v]['counts']['successful_steps'],skipped_steps=9800-acts[v]['counts']['successful_steps'])
    for s in S:
        for cat,groups in [('aggregate',{'all':agg(v,s)}),('named',named(v,s)),('receiver_all_days',scores[v]['per_receiver'][s]),('class',scores[v]['per_class'][s])]:
            for key,r in groups.items():
                flat.append(dict(variant=v,scene=s,category=cat,group=key,correct=r['tx_correct'],total=r['tx_total'],accuracy_percent=r['tx_acc']))
with (P/'five_completed_scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
(P/'five_completed_health.json').write_text(json.dumps(health,indent=2)+'\n',encoding='utf-8')
lines=['# CORE90交叉响应：新增完成项及五项详细测试数据','','快照：2026-09-11，11:55–11:58（UTC+8）。新增完成U1和head_only，共5/10项完成。全为seed392005、scratch-only、E200、label130+pseudo70、final-only。', '',
       'U1为完整块采样/运行路径对照，附加响应、决策、交互损失关闭；head_only仅训练响应辅助头，响应梯度不进入域或身份骨干。U0、U2、Ux定义及前三项完整数据保留在completed_results.md。', '',
       '## 数据与复核','','每项每场景198000个独立物理记录、6类，四场景792000条预测。新增两项全量预测及U0参考已只读复核，每项148组总体/分组/RX/类别计数均与scorer一致；与此前三项合并覆盖5项3960000条不同变体预测。5项各完整解析E1–E200和stdout，无Traceback/OOM，伪标签阶段真标签可用标志均0。', '',
       '所有百分数均为准确率%；LEO均值为三个等样本量场景均值。三个主组分别30000/126000/42000条，不能把别名重复相加。', '',
       '## 五项总体结果','','|场景|U0|U1（新增）|U2|Ux|head_only（新增）|','|---|---:|---:|---:|---:|---:|']
for s,t in zip(S,T):lines.append('|'+t+'|'+'|'.join(f"{agg(v,s)['tx_acc']:.4f}" for v in V)+'|')
lines.append('|三LEO均值|'+'|'.join(f'{mean(v):.4f}' for v in V)+'|')
lines+=['','## 相对U1的变化（百分点）','','|场景|U2−U1|Ux−U1|head_only−U1|','|---|---:|---:|---:|']
for s,t in zip(S,T):lines.append('|'+t+'|'+'|'.join(f"{agg(v,s)['tx_acc']-agg('U1',s)['tx_acc']:+.4f}" for v in ('U2','Ux','head_only'))+'|')
lines.append('|三LEO均值|'+'|'.join(f"{mean(v)-mean('U1'):+.4f}" for v in ('U2','Ux','head_only'))+'|')
lines+=['','U1当前四场景总体均最高。此前U2/Ux相对U0的小幅改善，不能作为附加损失优于完整块对照的证据；相对U1，U2、Ux和head_only的四场景总体均下降。该单seed对照包含随机训练路径差异，不把U1−U0全部归因于单一采样因素；head_only与U2预测器容量也不同，不能把两者差值解释为唯一梯度因素。','','## 日期/RX分组','','|场景|主组|U0|U1|U2|Ux|head_only|','|---|---|---:|---:|---:|---:|---:|']
for s,t in zip(S,T):
    for key,g in G.items():lines.append(f'|{t}|{g}|'+'|'.join(f"{named(v,s)[key]['tx_acc']:.4f}" for v in V)+'|')
lines+=['','## 新增项的每RX细分','','列示未知RX，已见日期每RX18000条，未知日期每RX6000条。']
for s,t in zip(S,T):
    lines+=['',f'### {t}','','|RX|U0已见日期|U1已见日期|head_only已见日期|U0未知日期|U1未知日期|head_only未知日期|','|---:|---:|---:|---:|---:|---:|---:|']
    for rx in (0,2,5,7,9,10,11):
        values=[named(v,s)[f'{pre}{rx}']['tx_acc'] for pre in ('test_rx_','test_unseen_day_rx_') for v in ('U0','U1','head_only')]
        lines.append(f'|{rx}|'+'|'.join(f'{x:.4f}' for x in values)+'|')
lines+=['','## 新增项各类别准确率','','每类每场景33000条，TX为内部索引。']
for s,t in zip(S,T):
    lines+=['',f'### {t}','','|TX|U0|U1|head_only|','|---:|---:|---:|---:|']
    for tx in range(6):lines.append(f'|{tx}|'+'|'.join(f"{scores[v]['per_class'][s][str(tx)]['tx_acc']:.4f}" for v in ('U0','U1','head_only'))+'|')
lines+=['','## 新增项资源与机制','','|指标|U1|head_only|','|---|---:|---:|']
for title,values in [('墙钟时间（小时）',[resources[v]['wall_time_seconds']/3600 for v in ('U1','head_only')]),('峰值allocated（GiB）',[resources[v]['peak_cuda_memory_allocated_bytes']/2**30 for v in ('U1','head_only')]),('最终源验证准确率',[health[v]['final_source_val_accuracy'] for v in ('U1','head_only')])]:
    lines.append('|'+title+'|'+'|'.join(f'{x:.4f}' for x in values)+'|')
for title,k in [('成功更新','successful_steps'),('有效完整块','effective_blocks'),('响应头梯度步数','head_gradient_steps'),('响应域梯度步数','domain_gradient_steps'),('响应身份梯度步数','joint_gradient_steps')]:
    lines.append('|'+title+'|'+'|'.join(str(acts[v]['counts'][k]) for v in ('U1','head_only'))+'|')
lines+=['','U1成功9790/9800步，head_only成功9792/9800步；无非有限loss跳步，少量非有限梯度导致跳步。head_only响应头梯度实际非零，域/身份响应梯度为0，符合对照定义。成本受同机并发影响，不是隔离吞吐实验。','','## 剩余时间估计','']
eta=read(P/'remaining_eta_1158.json')
lines+=['估计时点11:57左右，以最近10轮平均耗时计算；全部剩余项处于pseudo阶段，最新轮正常更新。','','|变体|已完成epoch|最近秒/轮|剩余训练小时|','|---|---:|---:|---:|']
for r in eta['rows']:lines.append(f"|{r['variant']}|{r['epoch']}|{r['recent10_seconds_per_epoch']:.1f}|{r['remaining_training_hours']:.2f}|")
lines+=['','最慢项按最近10轮约再需1.6小时训练；按更长窗口约1.9小时。考虑最终四场景预测和独立评分，预计今天14:00–14:30全部闭合（北京时间），仍受资源竞争和末段耗时影响。','','## 证据与边界','','全部740行评分见five_completed_scores.csv，新增预测复核与混淆矩阵见new_prediction_audit.json，完整日志统计见five_completed_health.json。原始评分/激活/资源/日志在各变体目录。单seed不作统计显著性或晋级判断；U3和两个U4等尚未完成，不能判断双任务联合收益。结果不回流阈值、选模或重跑。','']
(P/'five_completed_results.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(dict(csv_rows=len(flat),means={v:mean(v) for v in V},health=health),ensure_ascii=False))
