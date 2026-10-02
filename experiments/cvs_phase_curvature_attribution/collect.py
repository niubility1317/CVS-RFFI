"""Independent full-file recount of frozen source predictions; no target access."""
import argparse,csv,json,statistics as st
from pathlib import Path
from experiments.cvs_phase_curvature_attribution.publish import inspect,ssh,RUN,PROJECT
from experiments.cvs_phase_curvature_attribution.run import CONDITIONS

RECOUNT='''
import json
from pathlib import Path
root=Path(ROOT);state=json.loads((root/'pipeline_state.json').read_text());answer=[]
if state['status']!='COMPLETE' or len(state['rows'])!=8:raise ValueError('Incomplete attribution matrix')
for rid,row in state['rows'].items():
    p=Path(row['output_root']);done=json.loads((p/'completion.json').read_text());baseline=None;records=[]
    contract=json.loads((p/'source_loader/source_contract.json').read_text())
    for c in CONDITIONS:
        q=json.loads((p/(c+'_source_predictions.json')).read_text())
        if baseline is None:baseline=q
        if len(q['ids'])!=27000 or len(set(q['ids']))!=27000 or set(q['ids'])!=set(contract['role_ids']['V']):raise ValueError('Physical source V mismatch')
        if any(len(q[k])!=27000 for k in ('predictions','truth','receiver','correct')) or any(q[k]!=baseline[k] for k in ('ids','truth','receiver')):raise ValueError('Unpaired source condition')
        if any(type(x) is not int or not 0<=x<6 for x in q['truth']+q['predictions']):raise ValueError('Invalid source class')
        correct=[int(a==b) for a,b in zip(q['truth'],q['predictions'])]
        if correct!=q['correct']:raise ValueError('Correctness flags differ')
        expected=next(r for r in done['rows'] if r['condition']==c)
        accuracy=sum(correct)/27000
        good=sum(a and not b for a,b in zip(baseline['correct'],correct));bad=sum(not a and b for a,b in zip(baseline['correct'],correct));changed=sum(a!=b for a,b in zip(baseline['predictions'],q['predictions']))
        if abs(accuracy-expected['accuracy'])>1e-12 or good!=expected['all_on_correct_intervention_wrong'] or bad!=expected['all_on_wrong_intervention_correct'] or changed!=expected['prediction_changes']:raise ValueError('Paired recount differs')
        rx={str(rx):sum(c for c,r in zip(correct,q['receiver']) if r==rx)/sum(r==rx for r in q['receiver']) for rx in set(q['receiver'])}
        if rx!=expected['rx_accuracy']:raise ValueError('RX recount differs')
        rx_effects=[];confusions=[]
        for receiver in sorted(set(q['receiver'])):
            indices=[i for i,r in enumerate(q['receiver']) if r==receiver]
            helped=sum(baseline['correct'][i] and not correct[i] for i in indices)
            hurt=sum(not baseline['correct'][i] and correct[i] for i in indices)
            rx_effects.append(dict(receiver=receiver,count=len(indices),errors=sum(not correct[i] for i in indices),helped_by_all_on=helped,hurt_by_all_on=hurt,contribution_pp=100*(helped-hurt)/len(indices)))
            for truth in range(6):
                for predicted in range(6):
                    n=sum(q['truth'][i]==truth and q['predictions'][i]==predicted for i in indices)
                    confusions.append(dict(receiver=receiver,truth=truth,predicted=predicted,count=n))
        records.append(dict(condition=c,accuracy=accuracy,ce=expected['ce'],ce_source='recorded_full_V_cross_entropy_not_recomputed_from_argmax',contribution_pp=100*(good-bad)/27000,helped_by_all_on=good,hurt_by_all_on=bad,prediction_changes=changed,rx_accuracy=rx,rx_effects=rx_effects,confusions=confusions,seconds=expected['seconds']))
    answer.append(dict(row_id=rid,variant=done['variant'],model_seed=done['model_seed'],records=records,all_prediction_files_reparsed=True,source_V_ids_reconciled=True))
print(json.dumps(dict(status='VERIFIED',models=8,conditions=72,prediction_decisions=1944000,unique_source_V_physical_samples=27000,rows=answer,target_access=False)))
'''

def csvwrite(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));writer.writeheader();writer.writerows(rows)

def collect(root,output):
    live=inspect(output)
    if not live['pipeline'] or live['pipeline']['status']!='COMPLETE':return False
    if live['dispatcher_process'] or len(live['rows'])!=8 or any(r['process'] or not r['completion'] for r in live['rows']):raise ValueError('Not independently terminal')
    script=RECOUNT.replace('ROOT',repr(PROJECT+'/runs/'+RUN)).replace('CONDITIONS',repr(list(CONDITIONS)))
    data=json.loads(ssh(script));e=root/'automation_reports/CV-SincNet'/RUN/'evidence';e.mkdir(parents=True,exist_ok=True)
    for name,value in [('final_readback.json',live),('independent_recount.json',data)]:
        (e/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    records=[dict(row_id=r['row_id'],variant=r['variant'],seed=r['model_seed'],**{k:v for k,v in s.items() if k not in ('rx_accuracy','rx_effects','confusions')}) for r in data['rows'] for s in r['records']]
    for field,filename in [('rx_effects','receiver_attribution.csv'),('confusions','source_confusions.csv')]:
        csvwrite(e/filename,[dict(variant=r['variant'],seed=r['model_seed'],condition=s['condition'],**item) for r in data['rows'] for s in r['records'] for item in s[field]])
    summary=[]
    for v in sorted({r['variant'] for r in records}):
        for c in CONDITIONS:
            values=[r for r in records if r['variant']==v and r['condition']==c];assert len(values)==4
            summary.append(dict(variant=v,condition=c,contribution_pp_mean=st.mean(r['contribution_pp'] for r in values),contribution_pp_seed_sd=st.stdev(r['contribution_pp'] for r in values),helped_by_all_on=sum(r['helped_by_all_on'] for r in values),hurt_by_all_on=sum(r['hurt_by_all_on'] for r in values),prediction_changes=sum(r['prediction_changes'] for r in values),positive_seeds=sum(r['contribution_pp']>0 for r in values)))
    csvwrite(e/'per_seed_attribution.csv',records);csvwrite(e/'attribution_summary.csv',summary)
    text='# CVS相位曲率：冻结源域归因结果\n\n状态VERIFIED。全部8个冻结E200模型完成9条件，72份预测逐文件独立重算，共1944000个预测决定；物理样本始终是同27000条源V，不能当作1944000条独立样本。全开复现原源指标，每次关闭系数后恢复原完整状态。无训练、目标访问或源重选。\n\n'
    text+='贡献=全开准确率−关闭后的准确率，正值表示该修正有帮助，单位百分点。帮助/损害/翻转计数合计四模型，同一物理样本可被不同模型重复计数。\n\n|候选|关闭项|贡献均值±seed SD|全开帮助|全开损害|预测翻转|正贡献seed|\n|---|---|---:|---:|---:|---:|---:|\n'
    for r in summary:text+=f"|{r['variant']}|{r['condition']}|{r['contribution_pp_mean']:+.4f}±{r['contribution_pp_seed_sd']:.4f}|{r['helped_by_all_on']}|{r['hurt_by_all_on']}|{r['prediction_changes']}|{r['positive_seeds']}/4|\n"
    text+='\n这些是固定已训练权重下的局部因果干预，不等于从零重训移除该模块的效果。逐层贡献不可直接相加，层间有交互；源V是已见源RX的验证集，不能据此声称未见RX泛化或唯一TX硬件辨识。诊断不改变原源排名或条件clean矩阵。\n\n[逐seed完整72行](evidence/per_seed_attribution.csv) · [逐文件独立复算](evidence/independent_recount.json) · [进程终态和参数恢复](evidence/final_readback.json)。原始预测保留在N607登记路径。\n'
    (e.parent/'report.md').write_text(text,encoding='utf-8');print(json.dumps(dict(status='VERIFIED',summary=summary),ensure_ascii=False));return True

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();collect(a.root,a.output)
