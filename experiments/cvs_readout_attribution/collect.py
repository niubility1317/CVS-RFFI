"""Independently recompute full source logits and observed feature factors."""
import argparse,csv,json,statistics as st
from pathlib import Path
from experiments.cvs_readout_attribution.publish import inspect,ssh,RUN,PROJECT,RELEASE

REMOTE=r'''
from pathlib import Path
import json,sys,numpy as np
root=Path(ROOT);release=Path(RELEASE_PATH)
sys.path.insert(0,str(release))
from attribution_statistics import analyze_arrays
from run import validate_config,CONDITIONS,VARIANTS
state=json.loads((root/'pipeline_state.json').read_text())
if state['status']!='COMPLETE' or len(state['rows'])!=8:raise ValueError('Incomplete diagnostic matrix')
rows=[]
for rid,row in state['rows'].items():
    out=Path(row['output_root']);done=json.loads((out/'completion.json').read_text());resolved=json.loads((out/'resolved_config.json').read_text())
    actual_config=json.loads((root/(rid+'.json')).read_text());validate_config(actual_config)
    if any(resolved.get(k)!=v for k,v in actual_config.items()):raise ValueError('Actual diagnostic config differs')
    if (done['status']!='SOURCE_ATTRIBUTION_COMPLETE' or done['conditions']!=list(CONDITIONS) or done['target_access'] is not False
            or done['optimizer_updates']!=0 or done['source_selection_changed'] is not False or done['model_unchanged'] is not True):raise ValueError('Invalid frozen source diagnostic')
    with np.load(out/'source_readout_geometry.npz',allow_pickle=False) as value:q={k:value[k] for k in value.files}
    contract=json.loads((out/'source_loader/source_contract.json').read_text())
    if set(q['ids'].tolist())!=set(contract['role_ids']['V']):raise ValueError('Physical V mismatch')
    result=analyze_arrays(q)
    for name in CONDITIONS:
        got=result['classification'][name];stored=next(r for r in done['rows'] if r['condition']==name)
        if got['accuracy']!=stored['accuracy'] or {rx:r['accuracy'] for rx,r in got['receiver'].items()}!=stored['rx_accuracy']:raise ValueError('Independent accuracy/RX recount differs')
        if abs(got['ce']-stored['ce'])>2e-6:raise ValueError('Independent float64 CE differs from source GPU sum')
        predicted=q[name].argmax(1);reference=q['all_on'].argmax(1);truth=q['truth'];correct=predicted==truth;allcorrect=reference==truth
        counters=dict(prediction_changes=int(np.count_nonzero(predicted!=reference)),helped_by_all_on=int((allcorrect&~correct).sum()),hurt_by_all_on=int((~allcorrect&correct).sum()))
        if any(stored[k]!=v for k,v in counters.items()):raise ValueError('Paired decisions differ')
    ref=resolved['source_record'];got=result['classification']['all_on']
    if got['accuracy']!=ref['accuracy'] or min(v['accuracy'] for v in got['receiver'].values())!=ref['worst_rx']:raise ValueError('Frozen E200 source reproduction differs')
    rows.append(dict(row_id=rid,variant=done['variant'],seed=done['model_seed'],analysis=result))
if {(r['variant'],r['seed']) for r in rows}!={(v,s) for v in VARIANTS for s in range(2026092701,2026092705)}:raise ValueError('Missing model/seed pair')
print(json.dumps(dict(status='VERIFIED',models=8,conditions=32,source_packets_per_model=27000,decisions=864000,
    rows=rows,target_access=False,model_fitting=False,source_selection_changed=False,all_on_reproduces_E200=True),allow_nan=False))
'''

def table(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)

def collect(root,output):
    live=inspect(output)
    if not live['pipeline'] or live['pipeline']['status']!='COMPLETE':return False
    if live['dispatcher_process'] or len(live['rows'])!=8 or any(r['process'] or r['state']['status']!='COMPLETE' for r in live['rows']):raise ValueError('Not independently terminal')
    script=REMOTE.replace('ROOT',repr(PROJECT+'/runs/'+RUN)).replace('RELEASE_PATH',repr(PROJECT+'/releases/'+RELEASE))
    wrapper='import subprocess\nsubprocess.run(["/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python","-c",'+repr(script)+'],check=True)'
    data=json.loads(ssh(wrapper));e=root/'automation_reports/CV-SincNet'/RUN/'evidence';e.mkdir(parents=True,exist_ok=True)
    for name,value in [('final_readback.json',live),('independent_recount.json',data)]:
        (e/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    classification=[];factors=[]
    for row in data['rows']:
        a=row['analysis'];base=dict(variant=row['variant'],seed=row['seed'])
        for condition,metrics in a['classification'].items():
            classification.append(dict(base,condition=condition,accuracy=metrics['accuracy'],ce=metrics['ce'],
                worst_rx=min(v['accuracy'] for v in metrics['receiver'].values()),
                all_on_minus_condition_pp=100*(a['classification']['all_on']['accuracy']-metrics['accuracy'])))
        for feature,record in a['factor_scatter'].items():
            for factor,fraction in record['fractions'].items():factors.append(dict(base,feature=feature,factor=factor,fraction=fraction))
    table(e/'per_seed_classification.csv',classification);table(e/'per_seed_factor_fractions.csv',factors)
    summary=[]
    for variant in ('readout_attention','readout_complex_attention'):
        for condition in ('all_on','time_off','behavior_off','both_off'):
            rows=[r for r in classification if r['variant']==variant and r['condition']==condition]
            summary.append(dict(variant=variant,condition=condition,**{k+'_mean':st.mean(r[k] for r in rows) for k in ('accuracy','ce','worst_rx','all_on_minus_condition_pp')},
                contribution_seed_sd=st.stdev(r['all_on_minus_condition_pp'] for r in rows),positive_seeds=sum(r['all_on_minus_condition_pp']>0 for r in rows)))
    table(e/'classification_summary.csv',summary)
    text='# CVS读出：完整源V分支归因与TX/RX/日期关联\n\n状态VERIFIED。8个冻结E200模型、32条件、864000个分类决定，始终只有同27000个源V物理样本。全部logit、CE、配对预测和特征关联独立复算；全开准确率及最差RX复现原E200。零训练、零目标访问、不改变源选择。\n\n'
    text+='time_off/behavior_off分别把对应读出恢复为已训练主干下的原skip；both_off同时关闭两者。保留其余训练权重，不能把关闭结果当作从零重训的shallow模型。贡献=全开准确率−干预后准确率。\n\n|结构|条件|准确率/%|最差RX/%|CE|贡献/百分点±seed SD|正贡献seed|\n|---|---|---:|---:|---:|---:|---:|\n'
    for r in summary:text+=f"|{r['variant']}|{r['condition']}|{100*r['accuracy_mean']:.4f}|{100*r['worst_rx_mean']:.4f}|{r['ce_mean']:.6f}|{r['all_on_minus_condition_pp_mean']:+.4f}±{r['contribution_seed_sd']:.4f}|{r['positive_seeds']}/4|\n"
    text+='\n特征统计覆盖全部源V。每个TX×RX×day单元300条，正交均值散度分解包含TX、RX、day主效应、三项二阶交互、一项三阶交互及cell内残差；每包特征仅保留在地面N607。各项是平衡观测设计上的描述量，不是互信息、因果贡献或物理信道标签，不能推出TX/RX解耦。范数比按每包先计算再汇总，零分母边界见独立分析口径。\n\n四模型seed共享数据划分，SD不是独立域置信区间。干预可能偏离训练分布，收益不可直接相加；准确率2×2交互及全部RX/TX/day单元保存在完整JSON。该源诊断不产生新候选或新的目标测试结果。\n\n[逐seed分类](evidence/per_seed_classification.csv) · [特征因素占比](evidence/per_seed_factor_fractions.csv) · [完整独立复算](evidence/independent_recount.json) · [终态](evidence/final_readback.json)。\n'
    (e.parent/'report.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(status='VERIFIED',models=8,conditions=32,summary=summary),ensure_ascii=False));return True

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();collect(a.root,a.output)
