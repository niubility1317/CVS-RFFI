"""Independently recount all frozen source counterfactual artifacts."""
import argparse
import csv
import inspect as source_inspect
import json
from pathlib import Path
import statistics as st

import numpy as np

CONDITIONS=('all_on','mean_L','g_identity')
VARIANTS=('frontfilter_static','frontfilter_dynamic')
SEEDS=tuple(range(2026092701,2026092705))
SOURCE_COMMIT='dd5518d1493f2f79fb4213e4731e9f88cb28a349'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def recount_row(out):
    out=Path(out);done=read(out/'completion.json');resolved=read(out/'resolved_config.json')
    contract=read(out/'source_loader/source_contract.json')
    original=read(Path(resolved['source_output'])/'source_contract.json')
    if contract!=original:raise ValueError('Diagnostic source contract differs from checkpoint')
    variant=done['variant'];seed=done['model_seed']
    if (variant not in VARIANTS or seed not in SEEDS or resolved.get('source_commit')!=SOURCE_COMMIT or
        resolved['variant']!=variant or resolved['model_seed']!=seed or
        done['conditions']!=list(CONDITIONS) or done.get('coefficient_role')!='L_s' or
        done.get('coefficient_count')!=6300 or done.get('target_access') is not False or
        done.get('optimizer_updates')!=0 or done.get('source_selection_changed') is not False or
        done.get('model_unchanged') is not True):raise ValueError('Invalid frozen source diagnostic')
    L_ids=set(contract['role_ids']['L_s']);V_ids=set(contract['role_ids']['V'])
    if len(L_ids)!=6300 or len(V_ids)!=27000 or L_ids&V_ids:raise ValueError('Invalid source physical roles')
    fixed=read(out/'mean_L_coefficients.json')
    with np.load(out/'source_L_coefficients.npz',allow_pickle=False) as z:
        if set(z.files)!={'ids','coefficients'}:raise ValueError('L coefficient artifact schema')
        ids=z['ids'];coeff=z['coefficients']
        if ids.shape!=(6300,) or len(set(ids.tolist()))!=6300 or set(ids.tolist())!=L_ids:
            raise ValueError('Coefficient summary did not use exact L_s')
        if coeff.shape!=(6300,2,4) or coeff.dtype.kind!='f' or not np.isfinite(coeff).all():
            raise ValueError('Invalid per-L coefficients')
        if np.any(np.hypot(coeff[:,0],coeff[:,1]).sum(1)>1+1e-5):raise ValueError('L coefficient bound differs')
        mean=coeff.astype(np.float64).mean(0).astype(np.float32)
        if (fixed.get('role')!='L_s' or fixed.get('count')!=6300 or fixed.get('used_labels') is not False or
            fixed.get('frozen_before_V') is not True or fixed.get('ids')!=ids.tolist() or
            not np.array_equal(np.asarray(fixed['mean'],dtype=np.float32),mean)):
            raise ValueError('Fixed coefficient differs from independent L-only mean')
    baseline=None;records=[]
    if len(done['rows'])!=3 or {r['condition'] for r in done['rows']}!=set(CONDITIONS):raise ValueError('Condition completion matrix differs')
    for condition in CONDITIONS:
        with np.load(out/(condition+'_source_predictions.npz'),allow_pickle=False) as z:
            packet={k:z[k] for k in z.files}
        fields={'ids','truth','receiver','day','predictions','logits'}
        if set(packet)!=fields:raise ValueError('Prediction schema differs')
        if (any(packet[k].shape!=(27000,) for k in fields-{'logits'}) or packet['logits'].shape!=(27000,6) or
            len(set(packet['ids'].tolist()))!=27000 or set(packet['ids'].tolist())!=V_ids):
            raise ValueError('Prediction physical V or shape differs')
        if baseline is None:baseline=packet
        if any(not np.array_equal(packet[k],baseline[k]) for k in ('ids','truth','receiver','day')):
            raise ValueError('Unpaired source observations')
        for key,allowed in [('truth',set(range(6))),('predictions',set(range(6))),('receiver',{1,3,4,6,8}),('day',{1,2,3})]:
            if not np.issubdtype(packet[key].dtype,np.integer) or not set(packet[key].tolist())<=allowed:
                raise ValueError('Invalid source class/RX/day labels')
        logits=packet['logits']
        if logits.dtype.kind!='f' or not np.isfinite(logits).all() or not np.array_equal(logits.argmax(1),packet['predictions']):
            raise ValueError('Stored logits and predictions differ')
        shifted=logits.astype(np.float64)-logits.astype(np.float64).max(1,keepdims=True)
        losses=np.log(np.exp(shifted).sum(1))-shifted[np.arange(27000),packet['truth']]
        correct=packet['truth']==packet['predictions'];original_correct=baseline['truth']==baseline['predictions']
        rx={str(r):float(correct[packet['receiver']==r].mean()) for r in (1,3,4,6,8)}
        got=dict(accuracy=float(correct.mean()),rx_accuracy=rx,worst_rx=min(rx.values()),
            prediction_changes=int((packet['predictions']!=baseline['predictions']).sum()),
            helped_by_all_on=int((original_correct&~correct).sum()),hurt_by_all_on=int((~original_correct&correct).sum()))
        stored=next(r for r in done['rows'] if r['condition']==condition)
        if any(got[k]!=stored[k] for k in got):raise ValueError('Independent paired recount differs')
        ce=float(losses.mean())
        if not np.isfinite(stored['ce']) or abs(ce-stored['ce'])>1e-6:raise ValueError('Independent CE differs')
        confusions={}
        for receiver in ('ALL',1,3,4,6,8):
            mask=np.ones(27000,dtype=bool) if receiver=='ALL' else packet['receiver']==receiver
            cm=np.bincount(6*packet['truth'][mask]+packet['predictions'][mask],minlength=36).reshape(6,6)
            expected=27000 if receiver=='ALL' else 5400
            if int(cm.sum())!=expected:raise ValueError('RX confusion denominator differs')
            confusions[str(receiver)]=cm.tolist()
        cells=[]
        for tx in range(6):
            for receiver in (1,3,4,6,8):
                for day in (1,2,3):
                    mask=(packet['truth']==tx)&(packet['receiver']==receiver)&(packet['day']==day)
                    if int(mask.sum())!=300:raise ValueError('Unbalanced TX/RX/day source cells')
                    cells.append(dict(tx=tx,receiver=receiver,day=day,count=300,accuracy=float(correct[mask].mean())))
        cm=np.asarray(confusions['ALL']);den=cm.sum(0)+cm.sum(1)
        f1=float(np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0).mean())
        got.update(condition=condition,contribution_pp=100*(float(original_correct.mean())-float(correct.mean())),
            ce=ce,recorded_ce=stored['ce'],ce_scope='Independently recomputed from saved FP32 logits in float64; stored CE checked within1e-6',
            macro_f1=f1,confusions=confusions,cells=cells)
        records.append(got)
    if (records[0]['accuracy']!=resolved['source_record']['accuracy'] or
        records[0]['worst_rx']!=resolved['source_record']['worst_rx']):raise ValueError('All-on does not reproduce source E200')
    with np.load(out/'source_scalar_diagnostics.npz',allow_pickle=False) as z:
        if any(not np.array_equal(z[k],baseline[k]) for k in ('ids','receiver','day')):raise ValueError('Scalar pairing differs')
        fields=set(z.files)-{'ids','receiver','day'}
        if fields!=set(done['scalar_summary']):raise ValueError('Scalar metric matrix differs')
        scalar={}
        for name in fields:
            values=z[name]
            if values.shape!=(27000,) or not np.isfinite(values).all():raise ValueError('Invalid scalar source measurements')
            stats=dict(mean=float(values.mean()),p10=float(np.quantile(values,.1)),median=float(np.median(values)),p90=float(np.quantile(values,.9)),maximum=float(values.max()))
            if stats!=done['scalar_summary'][name]:raise ValueError('Scalar summary differs')
            stats['rx_mean']={str(r):float(values[z['receiver']==r].mean()) for r in (1,3,4,6,8)}
            scalar[name]=stats
    return dict(row_id=variant+'-s'+str(seed),variant=variant,seed=seed,records=records,scalar_summary=scalar,
        coefficient_mean=mean.tolist(),coefficient_count=6300,coefficient_role='L_s',coefficient_recount=True)


def recount(root):
    root=Path(root);state=read(root/'pipeline_state.json')
    expected={(v,s) for v in VARIANTS for s in SEEDS}
    if state['status']!='COMPLETE' or len(state['rows'])!=8:raise ValueError('Incomplete diagnostic matrix')
    answer=[]
    for rid,row in state['rows'].items():
        if row['status']!='COMPLETE' or row['exit_code']!=0:raise ValueError('Nonterminal diagnostic worker')
        item=recount_row(row['output_root'])
        if rid!=item['row_id']:raise ValueError('Diagnostic row identity differs')
        answer.append(item)
    if {(r['variant'],r['seed']) for r in answer}!=expected:raise ValueError('Incomplete source model/seed matrix')
    return dict(status='VERIFIED',models=8,conditions=24,prediction_decisions=648000,
        coefficient_observations=50400,unique_L_packets=6300,unique_V_packets=27000,
        all_CE_recomputed=True,all_L_coefficients_recomputed=True,rows=answer,
        target_access=False,optimizer_updates=0,source_selection_changed=False)


def csvwrite(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));writer.writeheader();writer.writerows(rows)


def collect(root,output):
    from experiments.cvs_frontfilter_attribution.publish import inspect,ssh,RUN,PROJECT
    live=inspect(output)
    if not live['pipeline'] or live['pipeline']['status']!='COMPLETE':return False
    if live['dispatcher_process'] or len(live['rows'])!=8 or any(r['process'] or not r['completion'] or r['state']['status']!='COMPLETE' for r in live['rows']):
        raise ValueError('Not independently terminal')
    script='import json\nfrom pathlib import Path\nimport numpy as np\n'
    script+='CONDITIONS='+repr(CONDITIONS)+'\nVARIANTS='+repr(VARIANTS)+'\nSEEDS='+repr(SEEDS)+'\nSOURCE_COMMIT='+repr(SOURCE_COMMIT)+'\n'
    script+='\n'.join(source_inspect.getsource(f) for f in (read,recount_row,recount))
    script+='\nprint(json.dumps(recount('+repr(PROJECT+'/runs/'+RUN)+')))\n'
    compile(script,'independent_source_recount','exec')
    wrapper='import subprocess\nsubprocess.run(["/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python","-c",'+repr(script)+'],check=True)'
    data=json.loads(ssh(wrapper));e=root/'automation_reports/CV-SincNet'/RUN/'evidence';e.mkdir(parents=True,exist_ok=True)
    for name,value in [('final_readback.json',live),('independent_recount.json',data)]:
        (e/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    records=[dict(variant=r['variant'],seed=r['seed'],**{k:v for k,v in row.items() if k not in ('confusions','rx_accuracy','cells')}) for r in data['rows'] for row in r['records']]
    scalar=[dict(variant=r['variant'],seed=r['seed'],metric=name,**{k:v for k,v in values.items() if k!='rx_mean'}) for r in data['rows'] for name,values in r['scalar_summary'].items()]
    cells=[dict(variant=r['variant'],seed=r['seed'],condition=row['condition'],**c) for r in data['rows'] for row in r['records'] for c in row['cells']]
    summary=[]
    for variant in VARIANTS:
        for condition in CONDITIONS:
            values=[r for r in records if r['variant']==variant and r['condition']==condition]
            summary.append(dict(variant=variant,condition=condition,accuracy=st.mean(r['accuracy'] for r in values),ce=st.mean(r['ce'] for r in values),
                contribution_pp=st.mean(r['contribution_pp'] for r in values),contribution_sd=st.stdev(r['contribution_pp'] for r in values),
                positive_seeds=sum(r['contribution_pp']>0 for r in values),helped=sum(r['helped_by_all_on'] for r in values),
                hurt=sum(r['hurt_by_all_on'] for r in values),prediction_changes=sum(r['prediction_changes'] for r in values)))
    for name,rows in [('per_seed_attribution',records),('scalar_summary',scalar),('attribution_summary',summary),('source_cells',cells)]:csvwrite(e/(name+'.csv'),rows)
    text='# CVS全主干前置滤波：固定权重源域反事实\n\n状态VERIFIED。8个E200冻结模型、24个条件、648000次预测；每个条件使用相同27000条源V，不是648000个独立样本。每模型固定系数仅由6300条L_s包计算，读取V前固定；独立重新读取50400条系数、全部预测logits、物理ID和标量完成复算。全开复现原E200；模型状态未变，零优化器更新，不改变源选择、不访问目标。\n\n'
    text+='贡献=全开准确率−干预后准确率。mean_L替代逐包系数，保持学习到的滤波基和同一主干；g_identity让原IQ进入同一已训练主干。mean_L只是机制诊断，不是新增校准模型或候选。\n\n|结构|条件|源V/%|源V CE|全开贡献/百分点±配对SD|正贡献seed|全开帮助|全开损害|预测变化|\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in summary:text+=f"|{r['variant']}|{r['condition']}|{100*r['accuracy']:.5f}|{r['ce']:.6f}|{r['contribution_pp']:+.5f}±{r['contribution_sd']:.5f}|{r['positive_seeds']}/4|{r['helped']}|{r['hurt']}|{r['prediction_changes']}|\n"
    text+='\n静态mean_L作为数值对照，实际差异原样报告。四seed共享同一物理数据，SD不是置信区间；移除滤波可能偏离训练分布，不能等同从零重训的因果效应。L均值替代还包含L/V分布差异，不能把差值全部归结为逐包适应。统计依赖不能证明信道恢复、TX/RX分离或未见RX泛化。\n\n[逐seed](evidence/per_seed_attribution.csv) · [完整TX/RX/day单元](evidence/source_cells.csv) · [标量](evidence/scalar_summary.csv) · [独立复算](evidence/independent_recount.json) · [完整终态](evidence/final_readback.json)。原始NPZ与日志保留N607登记路径。\n'
    (e.parent/'report.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(status='VERIFIED',models=8,conditions=24,summary=summary),ensure_ascii=False));return True


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();collect(a.root,a.output)
