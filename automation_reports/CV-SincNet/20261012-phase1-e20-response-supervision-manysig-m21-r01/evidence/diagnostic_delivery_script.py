from pathlib import Path
import json,sys,shutil,time
R=Path('E:/type10-7/code/worktrees/report_physical_ssl_delivery_20261011');sys.path.insert(0,str(R))
from experiments.cvs_e20_response import design as d
p=R/'automation_reports/CV-SincNet'/d.RUN;s=d.read(p/'experiment.json');assert s['status']=='ANALYZED'
paragraph='源端新增证据：RESP_LU的四作用均值MSE仅从0.240645降至0.238837，约下降0.75%；已记录的30个梯度审计点中，其加权extra/CE骨干梯度范数比均值为6.1599%。终点三seed余弦均值约−0.0006，但逐seed有正负，不能称全程正交。time_fuse/t1/t2权重相对父E20分别移动约4.79%/3.57%/4.24%，证明本批存在实际调优。固定D拟合误差、低Jacobian敏感性或对原坐标的锚定，都可能使响应目标下降很小；这些仍是源端待区分的假设，不能由本轮target分数断言。'
for name in ['result_analysis.md','final_analysis.md']:
 f=p/name;t=f.read_text(encoding='utf-8');needle='源证据说明：';i=t.index(needle);t=t[:i]+paragraph+'\n\n'+t[i:]
 t+='\n保存范围：每row完整source_contract.json含90000物理角色ID的重复表，保留N607原路径及本地完整副本，不重复纳入Git。登记的源契约引用和已验证来源保持不变；模型权重、逐样本预测与全步gzip日志同样保留原路径。Git交付完整小型成绩、epoch/CSV、来源与全部源分析。\n';f.write_text(t,encoding='utf-8')
s['progress']['endpoint_source_rows']=21;s['progress']['endpoint_identity_updates']=0;s['progress']['diagnostic_partial']='No training cosine or fine-module movement history; fixed endpoint only'
s['independent_review']['endpoint_analysis']='separate read-only source21; no new training/selection or target access'
d.write(p/'experiment.json',s)
h=d.read(p/'handoff.json');h['diagnostic_limitation']=s['progress']['diagnostic_partial'];h['endpoint_source_evidence']='evidence/source_endpoint_audit.json';d.write(p/'handoff.json',h)
with (p/'events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(dict(event='SOURCE_ENDPOINT_DIAGNOSTICS_VERIFIED',time=time.time(),rows=21,identity_updates=0,auxiliary_updates=0,target_access=False,training_history_reconstructed=False))+'\n')
shutil.copy2(__file__,p/'evidence/diagnostic_delivery_script.py')
t=R/'analysis/e20_response_traceability.md';text=t.read_text(encoding='utf-8').replace('|implemented|','|verified|').replace('implemented12不等于实际训练验证或84%性能完成','初步实现检查不等于实际训练验证或84%性能完成')
text+='\n最终追踪：verified11/partial_diagnostics1/deferred0/rejected0/blocked0。N09完整损失/访问/epoch/CSV/梯度范数和终点细模块/余弦已提供，训练中余弦及细模块历史缺失不能追补。所有21行完整测试有效；目标NOT_MET。源端响应MSE下降约0.75%、extra/CE范数比约6.16%，需要源可辨识性/读出Jacobian继续解释，而不是按target调λ或扩大幅度。\n';t.write_text(text,encoding='utf-8')
print(json.dumps(dict(status='VERIFIED',trace='verified11/partial1',goal='NOT_MET')))
