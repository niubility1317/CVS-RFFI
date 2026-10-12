from pathlib import Path
import json,sys,shutil,statistics as st
R=Path('E:/type10-7/code/worktrees/report_physical_ssl_delivery_20261011');sys.path.insert(0,str(R))
from experiments.cvs_finite_action import design as d
p=R/'automation_reports/CV-SincNet'/d.RUN;v=d.read(p/'evidence/source_full_analysis.json')
assert v['status']=='COMPLETE' and len(v['rows'])==3
lines=['','## 随机残差分支：均值与方差分开解释','','MLP以联合均值/方差的高斯NLL训练，双线性以均值MSE拟合且使用pooled方差。比较归一化MSE和NLL回答不同问题，不能仅以其中一个宣布该分支有效或无效。','','|层|留出组|双线性NLL|MLP NLL|双线性方差MSE|MLP方差MSE|','|---|---|---:|---:|---:|---:|']
for level in d.LEVELS:
 for family,splits in [('heldTX',[f'TX{i}' for i in range(6)]),('heldDay',[f'day{i}' for i in (1,2,3)]),('heldQuality',[f'quality{i}' for i in range(3)]),('U',['U_coverage'])]:
  rec=[x for row in v['rows'] for x in row['result']['records'] if x['kind']=='noise' and x['level']==level and x['split'] in splits]
  a={m:{k:st.mean(x[k] for x in rec if x['method']==m) for k in ('empirical_gaussian_NLL','variance_MSE')} for m in ('bilinear_ridge','MLP')}
  lines.append('|'+level+'|'+family+'|'+'|'.join(f'{a[m][k]:.6g}' for k in ('empirical_gaussian_NLL','variance_MSE') for m in ('bilinear_ridge','MLP'))+'|')
lines+=['','NLL只在相同层/条件内比较。浅中MLP在NLL上有改善，均值MSE并无稳定增量；最终层的方差MSE明显增大，NLL降低不能证明方差已准确标定。联合NLL可用增加方差吸收均值模型偏差，因此不能直接把预测方差当身份门控置信度，也不能从干净表示恢复具体随机噪声。','', '下一步R需要固定均值后单独验证条件方差与覆盖校准；L/T可进入独立身份机制实验，但其响应预测及真实RX统计覆盖仍不能自动证明身份收益。三个作用的覆盖有重叠，46.68%与38.91%不能相加。']
with (p/'source_interpretation.md').open('a',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
shutil.copy2('E:/type10-7/.codex_tmp/collect_finite_action_complete.py',p/'evidence/collection_script.py')
shutil.copy2('E:/type10-7/.codex_tmp/finalize_finite_source_report.py',p/'evidence/interpretation_script.py')
trace=R/'analysis/identity_evidence_report_traceability.md';t=trace.read_text(encoding='utf-8');rows=[]
for line in t.splitlines():
 if line.startswith('|R09|'):line='|R09|9.4|固定编码器已知有限参数作用预测、留出TX/day/质量、强简单基线与真实RX残差覆盖|cvs_finite_action|verified|实际3行351fit／351540aux更新／7020完整epoch／1890评估／135覆盖；全源冻结及角色核验闭合|浅层L/T留出TX优于双线性；真实源RX覆盖46.68%/38.91%，不是因果恢复或身份增益|'
 if line.startswith('|R10|'):line='|R10|8.2、9.5|新增U同包操作响应/可辨识性与不确定性；前导需先确认结构|cvs_finite_action|verified|全U56700/seed已知响应核验，无TX/拟合；noise4实现均值/方差、NLL/MSE完整分报|R均值无稳定增量、最终方差未校准；前导子项N/A未实现，lag16仅proxy|'
 rows.append(line)
t='\n'.join(rows)+'\n\n有限作用源端最终收尾：3/3全量闭合、1474200源模型样本读出、身份/头更新0、target访问0。R09/R10验证的是完整实施和正负证据，不代表物理解耦有效。当前verified12/implemented0/deferred0/rejected0/blocked0；R10前导子项N/A保留。最高剩余科学风险是L/T可迁移作用知识能否在源身份训练中带来净收益并跨未知RX泛化，84%尚未达成。后续机制只能从源证据设计，不据既有target成绩调权重/挑老师/重跑。\n'
trace.write_text(t,encoding='utf-8')
nextp=R/'analysis/finite_action_next_stage.md';t=nextp.read_text(encoding='utf-8');lines=t.splitlines();lines[2]='状态：三种子源端诊断已经完整完成VERIFIED；351fit、351540辅助更新、7020完整epoch、1474200源读出。身份更新0，target访问0。执行commit `9471ce500c48d477b9c40708e0ef3ed9f1ad485b`，run `20261012-phase1-finite-action-transfer-manysig-m3-r01`。全量结果见该run的final_analysis.md/source_interpretation.md；本设计先于关系15评分固定，未按target挑参考。'
nextp.write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('source report and traceability verified12 updated; R caveats retained')
