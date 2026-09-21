"""Reproducible full-log and all-group descriptive report of the eight V2 rows."""
import csv
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
VS=('U0','U1','U1_mask_off','U3','Ux','Ux_normalized','head_only','permanent_detach')
SS=('clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak')
TITLES=dict(zip(SS,('clean','晴空LEO','低仰角LEO','雨衰LEO')))
RX=(0,2,5,7,9,10,11)
GROUPS={'test_unseen_day_seen_rx':'未知日期＋已见RX','test_seen_day_unseen_rx':'已见日期＋未知RX',
        'test_unseen_day_unseen_rx':'未知日期＋未知RX'}
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def savecsv(name,rows):
    with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
audit=read(ROOT/'prediction_audit.json')
data={}; flat=[]; days=[]; matrices=[]; curves=[]
for v in VS:
    p=ROOT/v
    score=read(p/'independent_scores.json'); act=read(p/'cross_response_activation.json')
    resource=read(p/'phase1_resource_summary.json'); terminal=read(p/'phase1_terminal_status.json')
    receipt=read(p/'phase1_training_completion_receipt.json')
    epochs=[json.loads(line) for line in (p/'metrics_epoch.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    stdout=(p/(v+'.log')).read_text(encoding='utf-8')
    assert [m['epoch'] for m in epochs]==list(range(1,201))
    assert [m['phase'] for m in epochs]==['label']*130+['pseudo']*70
    assert score['status']==terminal['status']=='COMPLETE'
    assert terminal['exit_code']==0 and receipt['phase1_training_complete']
    assert act['status']=='ACTIVE_VERIFIED' and not act['missing']
    assert score['truth_last'] and score['selection_source']=='final_only'
    assert audit[v]['status']=='VERIFIED' and audit[v]['checkpoint_exists']
    assert all(m.get('train_pseudo_truth_available',0)==0 for m in epochs)
    errors=[line for line in stdout.splitlines() if any(k in line for k in ('Traceback','Error:','CUDA out of memory'))]
    assert not errors, (v,errors)
    groups={}; derived={}
    for s in SS:
        aggregate=score['test'] if s=='clean' else score['sat_test_named'][s]['aggregate']
        named=score['named_test'] if s=='clean' else score['sat_test_named'][s]['named']
        groups[s]=dict(aggregate=aggregate,named=named)
        cm=audit[v]['confusion_matrices'][s]
        f1=[200*cm[i][i]/(sum(cm[i])+sum(row[i] for row in cm)) if sum(cm[i])+sum(row[i] for row in cm) else 0. for i in range(6)]
        derived[s]=dict(macro_f1=sum(f1)/6,
            worst_strict_receiver=min((named[f'test_unseen_day_rx_{rx}']['tx_acc'],rx) for rx in RX),
            worst_seen_receiver=min((named[f'test_rx_{rx}']['tx_acc'],rx) for rx in RX))
        for category,values in [('aggregate',{'all':aggregate}),('named',named),('receiver_all_days',score['per_receiver'][s]),('class',score['per_class'][s])]:
            for name,m in values.items(): flat.append(dict(variant=v,scene=s,category=category,group=name,
                correct=m['tx_correct'],wrong=m['tx_total']-m['tx_correct'],total=m['tx_total'],accuracy_percent=m['tx_acc']))
        for day,m in audit[v]['per_day'][s].items(): days.append(dict(variant=v,scene=s,day=day,correct=m['tx_correct'],total=m['tx_total'],accuracy_percent=m['tx_acc']))
        for y in range(6):
            for pred in range(6): matrices.append(dict(variant=v,scene=s,true_tx=y,predicted_tx=pred,count=cm[y][pred]))
    # Parse all numeric series; no selected-epoch sampling of losses or events.
    statistics={}
    keys=sorted({k for m in epochs for k,x in m.items() if isinstance(x,(int,float)) and not isinstance(x,bool)})
    for key in keys:
        series=[(m['epoch'],m[key]) for m in epochs if isinstance(m.get(key),(int,float)) and math.isfinite(m[key])]
        if series: statistics[key]=dict(records=len(series),first=series[0][1],last=series[-1][1],
            minimum=min(x for _,x in series),maximum=max(x for _,x in series),mean=sum(x for _,x in series)/len(series),
            first_nonzero_epoch=next((e for e,x in series if x!=0),None))
    for m in epochs:
        curves.append(dict(variant=v,epoch=m['epoch'],phase=m['phase'],epoch_seconds=m['epoch_time_s'],
            train_loss=m.get('train_loss'),train_tx_acc=m.get('train_tx_acc'),source_val_acc=m.get('val_tx_acc'),
            update_fraction=m['train_optimizer_step_applied'],skip_gradient_fraction=m.get('train_skipped_nonfinite_grad',0),
            skip_loss_fraction=m.get('train_skipped_nonfinite_loss',0)))
    skipped={k:sum(round(m.get(k,0)*49) for m in epochs) for k in ('train_skipped_nonfinite_grad','train_skipped_nonfinite_loss')}
    updates=sum(round(m['train_optimizer_step_applied']*49) for m in epochs)
    assert updates==act['counts']['successful_steps']
    phase={}
    for name in ('label','pseudo'):
        subset=[m for m in epochs if m['phase']==name]
        phase[name]=dict(epochs=len(subset),seconds=sum(m['epoch_time_s'] for m in subset),
            minimum_update_fraction=min(m['train_optimizer_step_applied'] for m in subset))
    data[v]=dict(score=score,groups=groups,derived=derived,resource=resource,activation=act,
        checkpoint_sha256=terminal['selected_checkpoint_sha256'],source_split=terminal['source_split_receipt'],
        parsed_epochs=200,parsed_stdout_lines=len(stdout.splitlines()),errors=errors,
        warnings=[line for line in stdout.splitlines() if 'Warning' in line or '[WARN' in line],
        full_numeric_series_statistics=statistics,phase_stats=phase,skipped_steps=skipped,
        optimizer_updates=updates,final_train_accuracy=epochs[-1].get('train_tx_acc'),
        final_source_val_accuracy=epochs[-1].get('val_tx_acc'),audit=audit[v])
assert all(data[v]['source_split']==data['U0']['source_split'] for v in VS)
savecsv('scores_long.csv',flat); savecsv('per_day.csv',days); savecsv('confusion_matrices.csv',matrices); savecsv('training_curves.csv',curves)
nonfinite_paths=[]
def strict_json(value,path='root'):
    if isinstance(value,dict): return {k:strict_json(v,path+'.'+str(k)) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [strict_json(v,path+'.'+str(i)) for i,v in enumerate(value)]
    if isinstance(value,float) and not math.isfinite(value):
        nonfinite_paths.append(path)
        return None
    return value
clean_data=strict_json(data)
(ROOT/'completed_summary.json').write_text(json.dumps(clean_data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
(ROOT/'unavailable_fields.json').write_text(json.dumps(dict(note='Raw nonfinite fields represented as null only in the derived strict JSON; originals preserved, no imputation',paths=nonfinite_paths),indent=2),encoding='utf-8')
acc=lambda v,s:data[v]['groups'][s]['aggregate']['tx_acc']
means={v:sum(acc(v,s) for s in SS[1:])/3 for v in VS}
contrasts={'U1−U0：组织/执行对照':('U1','U0'),'U1−U1_mask_off：MixStyle角色mask':('U1','U1_mask_off'),
    'U3−U1：决策校准':('U3','U1'),'Ux−U1：原始身份交互':('Ux','U1'),
    'Ux_normalized−Ux：交互归一化':('Ux_normalized','Ux'),
    'head_only−U1：辅助头隔离对照':('head_only','U1'),
    'permanent_detach−U3：含预测器差异的域响应组合':('permanent_detach','U3')}
cd={name:{**{s:acc(a,s)-acc(b,s) for s in SS},'leo_mean':means[a]-means[b]} for name,(a,b) in contrasts.items()}
(ROOT/'contrasts.json').write_text(json.dumps(cd,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# CORE90交叉响应V2：8个已完成实验的详细测试数据','',
    '快照：2026-09-12。8个已启动对照全部完成E200、final checkpoint、四场景固定预测和独立评分。r1采用7行，head_only采用兼容性修复后的r2；r1原head_only初始化失败记录保留，不混入完成数据。其余8个源参数未冻结的条件行没有运行。','',
    f'完整解析1600轮metrics、{sum(d["parsed_stdout_lines"] for d in data.values())}行stdout和{sum(a["diagnostics"]["all_lines_parsed"] for a in audit.values())}条诊断JSONL。全部{sum(a["prediction_rows"] for a in audit.values()):,}条预测逐条验证6类有限logits、唯一ID和四场景完整覆盖，随后读取固定truth，复算{sum(a["compared_score_groups"] for a in audit.values())}组scorer计数，均一致。','',
    '数据与上一批一致：ManySig、equalized1、source RX1/3/4/6/8和day1/2/3；目标RX0/2/5/7/9/10/11。L_s/U_s/V=6300/56700/27000，模型及data seed392005，scratch、final_only、label130+pseudo70。八行source split receipt、测试物理ID、分组、scene seed及truth映射一致。','',
    '每场景198000个测试记录、6类，四场景792000条/行。未知日期＋已见RX30000条；已见日期＋未知RX126000条；未知日期＋未知RX42000条。前一组是source接收机的日期泛化对照，不能称全部198000条均为未知RX。三个主组互斥；命名别名不得重复相加。四场景共享物理记录，场景数不是独立样本或训练seed数。','',
    '所有准确率和Macro-F1单位为%，对照差值单位为百分点；LEO均值为三种等样本量场景的算术均值。单seed仅作描述性比较，结果不回流调参、选模或选择性重跑。','',
    '## 四场景总体准确率','', '|变体|clean|晴空LEO|低仰角LEO|雨衰LEO|三LEO均值|clean−U1|LEO均值−U1|',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for v in VS: lines.append('|'+v+'|'+'|'.join(f'{acc(v,s):.4f}' for s in SS)+f'|{means[v]:.4f}|{acc(v,"clean")-acc("U1","clean"):+.4f}|{means[v]-means["U1"]:+.4f}|')
lines += ['',f'本次观察中clean最高为{max(VS,key=lambda v:acc(v,"clean"))}，三LEO均值最高为{max(VS,key=lambda v:means[v])}。这是表内描述，不构成候选晋级或后续选择性重训依据。','',
    'U0是普通训练参考；U1是V2完整块/角色轮换、不加任务；U1_mask_off保持块组织但使用原MixStyle mask；U3增加向量化决策校准；Ux/Ux_normalized是原始/归一化身份交互；head_only隔离辅助optimizer/scaler；permanent_detach含域响应与决策、身份响应永久detach。','',
    '## 总体正确数与错误数','', '|变体|场景|正确|错误|总数|','|---|---|---:|---:|---:|']
for v in VS:
    for s in SS:
        m=data[v]['groups'][s]['aggregate']; lines.append(f"|{v}|{TITLES[s]}|{m['tx_correct']}|{m['tx_total']-m['tx_correct']}|{m['tx_total']}|")
lines += ['', '## 三种泛化子集','']
for g,title in GROUPS.items():
    lines += [f'### {title}','', '|变体|样本数/场景|clean|晴空LEO|低仰角LEO|雨衰LEO|','|---|---:|---:|---:|---:|---:|']
    for v in VS: lines.append(f"|{v}|{data[v]['groups']['clean']['named'][g]['tx_total']}|"+'|'.join(f"{data[v]['groups'][s]['named'][g]['tx_acc']:.4f}" for s in SS)+'|')
    lines.append('')
lines += ['## 预登记对照差值','', '|对照|clean|晴空LEO|低仰角LEO|雨衰LEO|三LEO均值|','|---|---:|---:|---:|---:|---:|']
for name,d in cd.items(): lines.append('|'+name+'|'+'|'.join(f'{d[k]:+.4f}' for k in (*SS,'leo_mean'))+'|')
lines += ['', 'U1与U1_mask_off更直接隔离角色mask；U1−U0仍同时包含数据组织/执行差异。permanent_detach与U3还存在预测器配置差异，不是纯域响应单因素归因。缺少已资格化的联合身份响应行，不能从本矩阵推断完整联合方案的收益或双因素协同。head_only的长期差异也不能仅凭终点归因于辅助任务，需实际逐步轨迹证据。','',
    '本次U1相对U0的clean下降1.2409个百分点、三LEO均值下降2.4241个百分点；块组织对照没有稳定收益。U1相对U1_mask_off的clean下降0.9621个百分点，LEO均值下降0.0855个百分点，mask效应不一致。U3相对U1四场景均略降。','',
    'Ux_normalized相对Ux的clean提升1.9217个百分点、LEO均值提升2.9101个百分点；相对U0则clean提升1.4793个百分点，但LEO均值仍低0.8266个百分点。因此支持的表述是“本次归一化交互优于原始交互”，不能表述为“完整V2稳定超过基线”。','',
    '## Macro-F1','', '|变体|clean|晴空LEO|低仰角LEO|雨衰LEO|','|---|---:|---:|---:|---:|']
for v in VS: lines.append('|'+v+'|'+'|'.join(f"{data[v]['derived'][s]['macro_f1']:.4f}" for s in SS)+'|')
lines += ['', '## 未知RX详细准确率','', '已见日期：每RX18000条；未知日期：每RX6000条。']
for s in SS:
    for prefix,title in [('test_rx_','已见日期'),('test_unseen_day_rx_','未知日期')]:
        lines += ['',f'### {TITLES[s]}，{title}','', '|变体|'+'|'.join('RX'+str(r) for r in RX)+'|','|---|'+'---:|'*7]
        for v in VS: lines.append('|'+v+'|'+'|'.join(f"{data[v]['groups'][s]['named'][prefix+str(r)]['tx_acc']:.4f}" for r in RX)+'|')
lines += ['', '## 最低未知RX准确率','', '每格为准确率（RX索引）；不是未知类拒识指标。']
for key,title in [('worst_seen_receiver','已见日期'),('worst_strict_receiver','未知日期')]:
    lines += ['',f'### {title}','', '|变体|clean|晴空LEO|低仰角LEO|雨衰LEO|','|---|---:|---:|---:|---:|']
    for v in VS: lines.append('|'+v+'|'+'|'.join(f"{data[v]['derived'][s][key][0]:.4f}（RX{data[v]['derived'][s][key][1]}）" for s in SS)+'|')
lines += ['', '## 类别准确率','', 'TX0–5为内部类别索引，每类每场景33000条；不是外部设备名称。']
for s in SS:
    lines += ['',f'### {TITLES[s]}','', '|变体|TX0|TX1|TX2|TX3|TX4|TX5|','|---|---:|---:|---:|---:|---:|---:|']
    for v in VS: lines.append('|'+v+'|'+'|'.join(f"{data[v]['score']['per_class'][s][str(c)]['tx_acc']:.4f}" for c in range(6))+'|')
lines += ['', '## 成本、训练稳定性与实际激活','',
    '|变体|墙钟小时|峰值allocated GiB|成功更新/9800|非有限梯度跳步|非有限loss跳步|最终source V准确率|',
    '|---|---:|---:|---:|---:|---:|---:|']
for v,d in data.items():
    lines.append(f"|{v}|{d['resource']['wall_time_seconds']/3600:.4f}|{d['resource']['peak_cuda_memory_allocated_bytes']/2**30:.4f}|{d['optimizer_updates']}|{d['skipped_steps']['train_skipped_nonfinite_grad']}|{d['skipped_steps']['train_skipped_nonfinite_loss']}|{d['final_source_val_accuracy']:.4f}|")
lines += ['', '跳步次数由每轮49步及日志平均率恢复为整数，成功更新总数与activation累计计数一致。完整stdout无Traceback/Error/OOM；允许的AMP跳步不等同于整行失败。r1 head_only初始化异常另列，不被这条成功行统计掩盖。墙钟受并发/评估影响，不是隔离速度基准；显存为进程PyTorch allocated。','',
    '|变体|响应头梯度步|域响应梯度步|身份响应梯度步|决策审计梯度步|交互审计梯度步|有效块|',
    '|---|---:|---:|---:|---:|---:|---:|']
for v,d in data.items():
    c=d['activation']['counts']; lines.append('|'+v+'|'+'|'.join(str(c[k]) for k in ('head_gradient_steps','domain_gradient_steps','joint_gradient_steps','decision_gradient_steps','cross_gradient_steps','effective_blocks'))+'|')
lines += ['', 'V2决策/身份交互梯度计数只统计被诊断抽中的成功步，不能解释为只在这些步训练；响应路由计数覆盖实际成功更新。当前8行均不要求联合身份响应，因此identity response为0符合定义，但不证明尚未运行的源门联合行有效。所有完整数值曲线的起止/极值/均值/首次非零轮在completed_summary.json，原始1600轮metrics与stdout在各变体目录。','',
    '## 文件与复核入口','',
    '- [完整分组CSV](scores_long.csv)：1184行scorer原始正确/错误/总数/准确率；含命名别名，不可直接加总所有行。',
    '- [逐日期CSV](per_day.csv)：按day0–3重新汇总；日期间接收机组成不同，不能将日差全部归因于时间。',
    '- [完整混淆矩阵CSV](confusion_matrices.csv)：32个6×6矩阵，行是真值、列是预测。',
    '- [训练曲线CSV](training_curves.csv)：1600轮关键训练指标。',
    '- [完整结果JSON](completed_summary.json)与[预测复算证据](prediction_audit.json)。',
    '- [对照差值JSON](contrasts.json)。源代码audit_remote.py/collect.py/build_report.py可复查；大预测、truth及checkpoint留在N607原run，本地不复制权重。',
    '- [不可用字段清单](unavailable_fields.json)：原始非有限辅助诊断字段仅在派生JSON写为null，未插补数值；原始文件保留。所有用于测试计数的预测logits均已通过有限值检查。','',
    '仍未运行：U2、U4_additive、U4_bilinear、U5、U3_delta、U3_delta_pairs、U4_decomposed、U5_reliable。真实源参数未冻结；本报告不补造这些行的数据，也不将V1同名结果混入V2表。','']
(ROOT/'completed_results.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(dict(report=str(ROOT/'completed_results.md'),score_rows=len(flat),prediction_rows=sum(a['prediction_rows'] for a in audit.values()),
    overall={v:{**{s:acc(v,s) for s in SS},'leo_mean':means[v]} for v in VS},contrasts=cd),ensure_ascii=False,indent=2))
