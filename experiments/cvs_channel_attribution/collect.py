"""Reparse all source prediction/scalar artifacts independently of the runner."""
import argparse
import csv
import json
import statistics as st
from pathlib import Path
from experiments.cvs_channel_attribution.publish import inspect,ssh,RUN,PROJECT

RECOUNT=r'''
import json
from pathlib import Path
import numpy as np
root=Path(ROOT);state=json.loads((root/'pipeline_state.json').read_text())
if state['status']!='COMPLETE' or len(state['rows'])!=8:raise ValueError('Incomplete attribution matrix')
answer=[];decisions=0;condition_count=0
for rid,row in state['rows'].items():
    out=Path(row['output_root']);done=json.loads((out/'completion.json').read_text())
    contract=json.loads((out/'source_loader/source_contract.json').read_text())
    resolved=json.loads((out/'resolved_config.json').read_text())
    variant=done['variant'];expected=['all_on','auxiliary_off','u_off','v_off','g_identity']+(['d_off'] if variant=='channel_order' else [])
    if variant not in ('channel_dual','channel_order') or done['conditions']!=expected or done['target_access'] is not False or done['optimizer_updates']!=0 or done['source_selection_changed'] is not False or done['model_unchanged'] is not True:raise ValueError('Invalid frozen diagnostic')
    baseline=None;records=[]
    for condition in expected:
        with np.load(out/(condition+'_source_predictions.npz'),allow_pickle=False) as q:packet={k:q[k] for k in q.files}
        if set(packet)!={'ids','truth','receiver','predictions'} or any(len(v)!=27000 for v in packet.values()) or set(packet['ids'].tolist())!=set(contract['role_ids']['V']) or len(set(packet['ids'].tolist()))!=27000:raise ValueError('Source V role mismatch')
        if baseline is None:baseline=packet
        if any(not np.array_equal(packet[k],baseline[k]) for k in ('ids','truth','receiver')):raise ValueError('Unpaired physical observations')
        if set(packet['receiver'].tolist())!={1,3,4,6,8} or any(not np.issubdtype(packet[k].dtype,np.integer) or np.any((packet[k]<0)|(packet[k]>=6)) for k in ('truth','predictions')):raise ValueError('Invalid class or receiver')
        correct=packet['truth']==packet['predictions'];original=baseline['truth']==baseline['predictions']
        rx={str(r):float(correct[packet['receiver']==r].mean()) for r in (1,3,4,6,8)}
        got=dict(accuracy=float(correct.mean()),rx_accuracy=rx,worst_rx=min(rx.values()),prediction_changes=int((packet['predictions']!=baseline['predictions']).sum()),helped_by_all_on=int((original&~correct).sum()),hurt_by_all_on=int((~original&correct).sum()))
        stored=next(r for r in done['rows'] if r['condition']==condition)
        if any(got[k]!=stored[k] for k in got):raise ValueError('Independent paired recount differs')
        confusions={}
        for receiver in ('ALL',1,3,4,6,8):
            mask=np.ones(27000,dtype=bool) if receiver=='ALL' else packet['receiver']==receiver
            cm=np.bincount(6*packet['truth'][mask]+packet['predictions'][mask],minlength=36).reshape(6,6)
            if cm.sum()!=int(mask.sum()):raise ValueError('Confusion total differs')
            confusions[str(receiver)]=cm.tolist()
        got.update(condition=condition,contribution_pp=100*(float(original.mean())-float(correct.mean())),ce=stored['ce'],ce_scope='recorded full-V CE; not recomputed from argmax',confusions=confusions)
        records.append(got);decisions+=27000;condition_count+=1
    if records[0]['accuracy']!=resolved['source_record']['accuracy'] or records[0]['worst_rx']!=resolved['source_record']['worst_rx']:raise ValueError('Original all-on source metrics differ')
    with np.load(out/'source_scalar_diagnostics.npz',allow_pickle=False) as q:
        if not np.array_equal(q['ids'],baseline['ids']) or not np.array_equal(q['receiver'],baseline['receiver']):raise ValueError('Scalar source pairing differs')
        scalar={}
        if set(q.files)-{'ids','receiver'}!=set(done['scalar_summary']):raise ValueError('Scalar metric matrix differs')
        for name in set(q.files)-{'ids','receiver'}:
            values=q[name]
            if values.shape!=(27000,) or not np.isfinite(values).all():raise ValueError('Incomplete/nonfinite source scalar')
            stats=dict(mean=float(values.mean()),p10=float(np.quantile(values,.1)),median=float(np.median(values)),p90=float(np.quantile(values,.9)),maximum=float(values.max()))
            if stats!=done['scalar_summary'][name]:raise ValueError('Scalar summary differs')
            stats['rx_mean']={str(r):float(values[q['receiver']==r].mean()) for r in (1,3,4,6,8)};scalar[name]=stats
    answer.append(dict(row_id=rid,variant=variant,seed=done['model_seed'],records=records,scalar_summary=scalar))
if condition_count!=44 or decisions!=1188000:raise ValueError('Incomplete registered decisions')
if {(r['variant'],r['seed']) for r in answer}!={(v,s) for v in ('channel_dual','channel_order') for s in range(2026092701,2026092705)}:raise ValueError('Incomplete seed matrix')
print(json.dumps(dict(status='VERIFIED',models=8,conditions=44,prediction_decisions=decisions,unique_source_packets=27000,rows=answer,target_access=False,optimizer_updates=0,source_selection_changed=False)))
'''


def csvwrite(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));writer.writeheader();writer.writerows(rows)


def collect(root,output):
    live=inspect(output)
    if not live['pipeline'] or live['pipeline']['status']!='COMPLETE':return False
    if live['dispatcher_process'] or len(live['rows'])!=8 or any(r['process'] or not r['completion'] or r['state']['status']!='COMPLETE' for r in live['rows']):raise ValueError('Not independently terminal')
    script=RECOUNT.replace('ROOT',repr(PROJECT+'/runs/'+RUN))
    wrapper='import subprocess\nsubprocess.run(["/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python","-c",'+repr(script)+'],check=True)'
    data=json.loads(ssh(wrapper));e=root/'automation_reports/CV-SincNet'/RUN/'evidence';e.mkdir(parents=True,exist_ok=True)
    for name,value in [('final_readback.json',live),('independent_recount.json',data)]:
        (e/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    records=[dict(variant=r['variant'],seed=r['seed'],**{k:v for k,v in row.items() if k not in ('confusions','rx_accuracy')}) for r in data['rows'] for row in r['records']]
    scalar=[dict(variant=r['variant'],seed=r['seed'],metric=name,**{k:v for k,v in values.items() if k!='rx_mean'}) for r in data['rows'] for name,values in r['scalar_summary'].items()]
    summary=[]
    for variant in ('channel_dual','channel_order'):
        for condition in next(r['completion']['conditions'] for r in live['rows'] if r['completion']['variant']==variant):
            values=[r for r in records if r['variant']==variant and r['condition']==condition]
            summary.append(dict(variant=variant,condition=condition,contribution_pp=st.mean(r['contribution_pp'] for r in values),contribution_sd=st.stdev(r['contribution_pp'] for r in values),positive_seeds=sum(r['contribution_pp']>0 for r in values),helped=sum(r['helped_by_all_on'] for r in values),hurt=sum(r['hurt_by_all_on'] for r in values),prediction_changes=sum(r['prediction_changes'] for r in values)))
    csvwrite(e/'per_seed_attribution.csv',records);csvwrite(e/'scalar_summary.csv',scalar);csvwrite(e/'attribution_summary.csv',summary)
    text='# CVS补偿与顺序差：完整源V冻结归因\n\n状态VERIFIED。8个冻结E200模型完成44条件，共1188000个预测决定；每个模型始终使用同27000条源V，不能当作1188000个独立样本。全部预测、物理ID及每包标量已逐文件重新计算，全开复现原源V与最差RX。模型状态不变，零训练、零目标访问、不改变旧选择。\n\n'
    text+='贡献=全开准确率−干预后准确率。正值代表该结构在当前冻结权重下帮助分类；不是从零重训的因果效应。u_off/v_off只移除各自投影，仍保留其他分支；g_identity将补偿设为恒等，保留共享F及原读出权重。\n\n|模型|干预|贡献均值±SD/百分点|正贡献seed|全开帮助|全开损害|预测变化|\n|---|---|---:|---:|---:|---:|---:|\n'
    for r in summary:text+=f"|{r['variant']}|{r['condition']}|{r['contribution_pp']:+.5f}±{r['contribution_sd']:.5f}|{r['positive_seeds']}/4|{r['helped']}|{r['hurt']}|{r['prediction_changes']}|\n"
    text+='\n完整V标量为每包分别计算后汇总；下表对四模型的包均值取平均。投影相对量以同包原主干特征范数为分母；D时序相干比例是交叉项先平均后的模长和，除以先取模再平均的和，小值提示平均抵消，不等于身份信息丢失的证明。\n\n|模型|指标|四模型均值|\n|---|---|---:|\n'
    for variant in ('channel_dual','channel_order'):
        for metric in sorted({r['metric'] for r in scalar if r['variant']==variant}):
            value=st.mean(r['mean'] for r in scalar if r['variant']==variant and r['metric']==metric)
            text+=f'|{variant}|{metric}|{value:.8g}|\n'
    text+='\n这些干预只说明当前冻结网络在已见源RX上的依赖；移除分支可能偏离其训练分布，贡献不可相加。它们不能证明信道恢复、TX/RX分离、未见RX泛化或目标模块消融。D幅度、读出向量及投影分别测量，未使用额外损失或任何训练更新。\n\n[逐seed贡献](evidence/per_seed_attribution.csv) · [全部标量](evidence/scalar_summary.csv) · [独立复算与逐RX混淆矩阵](evidence/independent_recount.json) · [完整终态](evidence/final_readback.json)。原始每包证据保留于N607登记路径。\n'
    (e.parent/'report.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(status='VERIFIED',models=8,conditions=44,summary=summary),ensure_ascii=False));return True


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();collect(a.root,a.output)
