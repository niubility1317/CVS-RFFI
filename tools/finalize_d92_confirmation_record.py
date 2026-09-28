"""Independent arithmetic audit and durable record of the complete failed candidate."""
import json
import math
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
WS=Path('E:/type10-7')
RUN='20260928-phase2-d92-scv-confirmation-manytx-m4-r02'
REL=Path('automation_reports/CV-SincNet')/RUN
folder=WS/REL
data=json.loads((folder/'results/scores.json').read_text(encoding='utf-8'))
assert data['status']=='SCORED' and len(data['results'])==7236
for row in data['results']:
    cm=row['confusion'];count=len(cm)
    total=sum(sum(line) for line in cm);correct=sum(cm[i][i] for i in range(count))
    old_total=sum(sum(line) for line in cm[:6]);old=sum(cm[i][i] for i in range(6))/old_total
    new_total=total-old_total;new=(correct-old*old_total)/new_total if new_total else None
    h=2*old*new/(old+new) if new is not None and old+new else (0.0 if new is not None else None)
    f1=[]
    for i in range(count):
        denom=sum(cm[i])+sum(line[i] for line in cm)
        f1.append(2*cm[i][i]/denom if denom else 0.0)
    values={'accuracy':correct/total,'old_accuracy':old,'new_accuracy':new,'harmonic_mean':h,'macro_f1':sum(f1)/count}
    for key,value in values.items():
        assert row[key] is None if value is None else math.isclose(row[key],value,abs_tol=1e-12), (row['split_id'],key)
    assert row['query_count']==total
artifacts=dict(run_id=RUN,records=7236,arithmetic_audit='VERIFIED all confusion matrices: accuracy,old,new,H,macroF1 and query totals',
    raw_scores=dict(local=str(folder/'results/scores.json'),remote='/home/szu2070436088/2510044040/CV-SincNet/runs/'+RUN+'/scores.json',bytes=(folder/'results/scores.json').stat().st_size),
    full_predictions='Original remote each-row predictions.jsonl and scv/predictions.jsonl; never modified or selected',
    summary='results/summary/summary.json',compact_fit_logs='results/fit_logs/',candidate_promoted=False,goal_complete=False,
    next_scope='Await user answer on independent source auxiliary training; formal4 Phase1 models remain frozen')
(folder/'results/artifacts.json').write_text(json.dumps(artifacts,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
report=(folder/'report.md').read_text(encoding='utf-8')
report=report.replace('尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。',
    '完整7236条结果已评分和汇总。候选未达到新旧类全面提升目标，未晋级。详见[完整汇总](results/summary/report.md)和[产物定位](results/artifacts.json)。')
if '![新旧类与H]' not in report:
    report+='\n![新旧类与H](results/summary/old_new_h.png)\n\n![全部K和新类规模的变化](results/summary/delta_all_k_new.png)\n'
(folder/'report.md').write_text(report,encoding='utf-8')
summary=folder/'results/summary/report.md'
content=summary.read_text(encoding='utf-8')
if '![各K结果]' not in content:
    content+='\n![各K结果](old_new_h.png)\n\n![完整K×新类规模](delta_all_k_new.png)\n'
summary.write_text(content,encoding='utf-8')
for relative in ['report.md','results/artifacts.json','results/summary/report.md']:
    src=folder/relative;dst=ROOT/REL/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
shutil.copyfile(WS/'local_artifacts/d92_upgrade_20260928/confirmation_source_class_readback.json',ROOT/REL/'evidence/source_class_readback.json')
print(json.dumps(artifacts,ensure_ascii=False,indent=2))
