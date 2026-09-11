"""Parse all collected text evidence and produce detailed source result tables."""
import csv,json,math,re
from pathlib import Path
from datetime import datetime,timezone,timedelta
BASE=Path('analysis/core90_anchored_results_20260912')
RUN='core90_anchored_s392005_20260911_r1'
R=BASE/'runs'/RUN
def readcsv(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def savecsv(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def table(headers,rows):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])
def pc(x):return f'{100*float(x):.4f}'
inv=json.loads((BASE/'inventory.json').read_text());expected=[x for x in inv if Path(x['path']).suffix in {'.json','.jsonl','.csv','.log','.md'}]
assert all((BASE/x['path']).stat().st_size==x['size'] for x in expected)
parsed={'json':0,'jsonl':0,'csv':0,'csv_rows':0,'log':0,'log_lines':0};warnings={};errors=[];histories=[];geoms=[]
for item in expected:
    p=BASE/item['path'];s=p.read_text(encoding='utf-8');assert '\ufffd' not in s
    if p.suffix=='.json':json.loads(s);parsed['json']+=1
    elif p.suffix=='.jsonl':
        for line in s.splitlines():json.loads(line);parsed['jsonl']+=1
    elif p.suffix=='.csv':
        rows=readcsv(p);parsed['csv']+=1;parsed['csv_rows']+=len(rows)
        for row in rows:
            for k,v in row.items():
                if v is not None and v.lower() in {'nan','inf','-inf','infinity'}:errors.append(dict(path=item['path'],field=k,value=v))
        if p.name=='head_fit_history.csv':
            assert [int(x['epoch']) for x in rows]==list(range(1,81))
            assert int(rows[-1]['optimizer_steps'])==2800
            histories.append(dict(path=p.relative_to(R).as_posix(),epochs=len(rows),first=rows[0],last=rows[-1],min_loss=min(float(x['loss']) for x in rows),seconds=sum(float(x['epoch_seconds']) for x in rows),all_rows=rows))
        if p.name=='geometry_diagnostics.csv':geoms.extend(dict(path=p.relative_to(R).as_posix(),**x) for x in rows)
    elif p.suffix=='.log':
        parsed['log']+=1;parsed['log_lines']+=len(s.splitlines())
        for line in s.splitlines():
            if re.search(r'Traceback|CUDA out of memory|(^|\s)Killed($|\s)|Error:|\b(?:NaN|Inf)\b',line,re.I):errors.append(dict(path=item['path'],line=line))
            if 'Warning:' in line:warnings[line]=warnings.get(line,0)+1
assert len(histories)==40
metrics=json.loads((BASE/'group_metrics.json').read_text());groups=metrics['records'];savecsv(BASE/'all_group_metrics.csv',groups)
for row in groups:
    assert row['correct']-row['baseline_correct']==row['rescue']-row['harm']
for name in {x['candidate'] for x in groups if x['role']=='source_V_calibration_descriptive'}:
    got=next(x for x in groups if x['candidate']==name and x['role']=='source_V_calibration_descriptive' and x['group']=='all')
    old=readcsv(R/'candidates'/name/'calibrate/selective_policy.csv')[0]
    assert abs(got['accuracy']-float(old['closed_accuracy']))<1e-10
    assert abs(got['coverage']-float(old['coverage']))<1e-10
    assert abs(got['risk']-float(old['risk']))<1e-10
audit=readcsv(R/'candidates/A6/fuse/fusion_audit.csv');fixed=[x for x in audit if x['group']=='all']
nested=readcsv(R/'candidates/A6/fuse/nested_source_metrics.csv')
for x in nested:
    y=next(r for r in groups if r['role']=='nested_head_only_OOF' and r['candidate']==x['candidate'] and r['group']==x['group'] and r['value']==x['value'])
    assert all(y[k]==int(x[k]) for k in ['rows','correct','baseline_correct','rescue','harm'])
summary=dict(full_parse=parsed,inventory_files=len(inv),collected_text_files=len(expected),histories=len(histories),epochs=sum(h['epochs'] for h in histories),optimizer_steps=sum(int(h['last']['optimizer_steps']) for h in histories),warnings=warnings,errors=errors,verification='VERIFIED' if not errors else 'REVIEW',V_recomputed_matches_saved_policy=True,nested_recomputed_matches_saved_counts=True,gate=metrics['gate'])
dump(BASE/'analysis_summary.json',summary)
dump(BASE/'fit_summary.json',[{k:v for k,v in h.items() if k!='all_rows'} for h in histories]);savecsv(BASE/'geometry_all.csv',geoms)
savecsv(BASE/'epoch_curves.csv',[dict(fit=h['path'],**row) for h in histories for row in h['all_rows']])
def get(name,role,group='all',value='all'):return next(x for x in groups if x['candidate']==name and x['role']==role and x['group']==group and x['value']==value)
oofnames=['H0','A2','A3','A4','C_angle','C_angle_keep','A5_inner_selected','A6']
def role(n):return 'nested_head_only_OOF' if n in {'A6','A5_inner_selected'} else 'head_only_OOF'
vnames=['A0','A1','A2','A3','A4','C_angle','C_angle_keep','A5-0','A5-25','A5-50','A5-75','A5-100','A6']
views=['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak']
sections=['# CORE90锚定几何实验详细结果（seed392005）','',
'结论：本轮14个候选的source拟合、校准及导出已完成，状态SOURCE_SYSTEM_FROZEN，远端进程已退出。A6嵌套OOF净收益为负，本seed不满足科学收益条件；三seed判定仍为PENDING_ALL_HEAD_SEEDS。本轮没有target测试，不能称为完整target泛化实验完成。','',
'## 范围与完成证据','',
f'Run：`{RUN}`。实际代码commit：`1afa43a75c76b728d4e6352ba8e44acae44f277b`。完成时间：'+datetime.fromtimestamp(json.loads((R/'status.json').read_text())['time'],timezone(timedelta(hours=8))).isoformat()+'。','',
f"已全量解析{parsed['log']}份日志、{parsed['log_lines']}行，{parsed['csv']}份CSV、{parsed['csv_rows']}条数据，{parsed['json']}份JSON与{parsed['jsonl']}条JSONL记录；40次拟合均完整E1–80，共3200个epoch、112000个optimizer step。错误扫描结果：{len(errors)}条；警告仅按原文保存在analysis_summary.json。没有重训、重校准、修改远端或访问target。",'',
'数据沿用ManySig equalized=1；source RX=1/3/4/6/8，day=1/2/3，六TX；L_s/U_s/V=6300/56700/27000。本轮头部只消费L_s，H0继承来源与原启动报告一致。OOF共12600个view行（6300个物理样本，各clean+一个LEO）；V共54000个view行（27000个物理样本，各clean+一个LEO）。因此all为50%clean+50%LEO，不是四场景简单平均；每个物理样本只分配一个LEO场景，三个LEO子集不是同一批物理样本的全场景重复测量。','',
'OOF仅隔离新头部的接收机训练数据，冻结H0已经见过全部source RX和L_s。它是head-only研发对照，不能解释为整个模型对未见接收机的独立泛化。V用于温度和拒识阈值校准，下列V数值是同一校准集上的描述性结果。','',
'## 完整头部OOF与嵌套OOF','',
'准确率单位为%；Δ单位为百分点。救回=H0错→候选对，损伤=H0对→候选错；效用=救回−2×损伤。','',
table(['候选','正确/12600','准确率','ΔH0','救回','损伤','净正确','效用'],[[n,f"{(x:=get(n,role(n)))['correct']}/12600",pc(x['accuracy']),f"{100*x['net_correct']/x['rows']:+.4f}",x['rescue'],x['harm'],x['net_correct'],x['rescue']-2*x['harm']] for n in oofnames]),'',
table(['候选']+views,[[n]+[pc(get(n,role(n),'view',v)['accuracy']) for v in views] for n in oofnames]),'',
'A5_inner_selected是在每个外层折中只用内层数据选固定动作，再对外层评估；A6也是外层预测。两者可作同row比较。最终门控在全OOF上拟合后的表现不替代这张嵌套表。','',
'## A5固定融合动作审计','',
table(['请求α','正确/12600','准确率','ΔH0','救回','损伤','效用'],[[x['raw_alpha'],x['correct'],pc(int(x['correct'])/int(x['rows'])),f"{100*int(x['net_correct'])/int(x['rows']):+.4f}",x['rescue'],x['harm'],x['utility']] for x in fixed]),'',
'固定非零动作存在微小正净正确数，满足A6触发条件，但所有非零动作的加权效用都为负；最终按该效用选择的固定动作是α=0。触发A6不代表A6产生收益。','',
'## A6按RX/TX/day详细分解','']
tx=['14-10','14-7','20-15','20-19','6-15','8-20']
for dim in ['RX','TX','day']:
    rows=[x for x in groups if x['candidate']=='A6' and x['role']=='nested_head_only_OOF' and x['group']==dim]
    sections += [table([dim,'行数','H0准确率','A6准确率','ΔH0','救回','损伤'],[[tx[int(x['value'])] if dim=='TX' else x['value'],x['rows'],pc(x['baseline_correct']/x['rows']),pc(x['accuracy']),f"{100*x['net_correct']/x['rows']:+.4f}",x['rescue'],x['harm']] for x in rows]),'']
sections += ['本seedA6净正确−14（−0.1111pp）；RX8净−13（−0.5159pp），TX14-7净−22（−1.0476pp），超过设计允许的−0.5pp最差分组降幅；clean不变。5个RX中4个净收益非负的条件也未满足（实际3个）。所以即使暂不考虑另两个seed缺失，本seed的收益与保底条件也未通过。A6相对内层选择的A5再少8个正确结果（−0.0635pp）。','',
'## V校准集：全部冻结候选','',
'下表拒识后准确率=1−risk，分母仅为接受的90%view；不等价于全样本准确率。其余10%被拒识，不能算作已识别正确。','',
table(['候选','全样本准确率','clean','clear','low_elev','rain','接受覆盖率','接受准确率','温度T'],[[n,pc((x:=get(n,'source_V_calibration_descriptive'))['accuracy'])]+[pc(get(n,'source_V_calibration_descriptive','view',v)['accuracy']) for v in views]+[pc(x['coverage']),pc(1-x['risk']),f"{json.loads((R/'candidates'/n/'calibrate/calibration_report.json').read_text())['temperature']:.4f}"] for n in vnames]),'',
'上述13个冻结系统在CPU上只做推理，逐候选重算accuracy/coverage/risk与已保存selective_policy.csv一致；没有再次fit温度或阈值。全399条分组统计及21组混淆矩阵保存在group_metrics.json/all_group_metrics.csv。','',
'## P1缺块结果（V校准集）','']
p1groups=readcsv(R/'candidates/P1/calibrate/mask_source_groups.csv');p1cal=readcsv(R/'candidates/P1/calibrate/mask_calibration.csv');p1=[]
for pat in ['full','missing_t','missing_f','missing_pa','all_missing']:
    g=[x for x in p1groups if x['pattern']==pat and x['group']=='RX'];a=[x for x in p1cal if x['pattern']==pat]
    count=sum(int(x['rows']) for x in g);acc=sum(float(x['closed_accuracy'])*int(x['rows']) for x in g)/count
    accepted=sum(int(x['accepted']) for x in a);err=sum(int(x['accepted_errors']) for x in a)
    p1.append([pat,pc(acc),pc(accepted/count),pc(1-err/accepted) if accepted else 'N/A',sum(int(x['defer']) for x in a),sum(int(x['model_mismatch_candidates']) for x in a)])
sections += [table(['pattern','全样本准确率','接受覆盖率','接受准确率','defer','mismatch候选'],p1),'',
'all_missing全部defer；其16.6667%原始top-class正确率来自无观测时的占位决策，不能解释为识别能力。P1的coverage包含有效性/缺块与校准规则，不能假设所有pattern都达到90%。质量bin的physical_count存在跨bin重复，不把它们相加当独立物理样本数。','',
'## 拟合终点与机制','',
table(['候选','E1总loss','E80总loss','E80 clean CE','E80 LEO CE','keep梯度','metric梯度'],[[h['path'].split('/')[1],f"{float(h['first']['loss']):.6f}",f"{float(h['last']['loss']):.6f}",f"{float(h['last']['ce_clean']):.6f}",f"{float(h['last']['ce_leo']):.6f}",f"{float(h['last']['grad_norm_keep']):.6g}",f"{float(h['last']['grad_norm_metric']):.6g}"] for h in histories if '/all_source/' in h['path']]),'',
'这些梯度证明keep/metric确实参与优化，不能替代组件收益。A3与A4、C_angle与C_angle_keep是同预算对照；详细差值见OOF/V表。validation_ce字段为空，不能声称按验证loss早停或选择最佳epoch；所有专家固定使用E80。完整曲线已输出epoch_curves.csv，40次拟合终点/min loss/耗时见fit_summary.json。','',
'## 资源测量','',
'原始profile为batch内独立阶段计时，不是端到端单样本延迟；GPU并非独占。头部与fusion/calibration在CPU，identity backbone在cuda:0，warmup=3、repeats=10。','',
table(['候选','backbone均值ms','head均值ms','fusion均值ms'],[[n]+[f"{1000*json.loads((R/'candidates'/n/'profile/runtime_profile.json').read_text())['stages'][k]['seconds_mean']:.4f}" for k in ['identity_backbone','head','fusion_calibration']] for n in ['A0','A4','C_angle','A6']]),'',
'## 未完成范围与科学判定','',
'本次启动授权只覆盖seed392005的source矩阵，现已完成。392006/392007未运行；没有target预测、独立target评分、unknown拒识或新类注册指标，不填0也不借用旧run结果。设计受旧target观察启发，重复使用旧target不能变成独立确认。本次未启动新实验或修改任何健康任务。','',
'## 可复核产物','',
'- `analysis_summary.json`：全量解析范围、错误/警告和重算一致性。\n- `all_group_metrics.csv`、`group_metrics.json`：OOF/嵌套OOF/V的全分组与混淆矩阵。\n- `epoch_curves.csv`、`fit_summary.json`、`geometry_all.csv`：全部epoch、拟合和几何记录。\n- `runs/`、`logs/`：按远端相对路径保存的原始文本；`inventory.json`含权重与其他产物的大小清单，权重保留远端。','']
text='\n'.join(sections);(BASE/'detailed_results.md').write_text(text,encoding='utf-8')
assert '\ufffd' not in (BASE/'detailed_results.md').read_text(encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False));print(table(['candidate','OOF accuracy','net'],[[n,pc(get(n,role(n))['accuracy']),get(n,role(n))['net_correct']] for n in oofnames]))
