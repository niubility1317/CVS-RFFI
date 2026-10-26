"""Finalize all fifteen prespecified models from frozen independent scores."""
import argparse
import csv
import gzip
from itertools import product
import json
from pathlib import Path
import statistics
import zipfile

from tools.reports.pamf_nosig_ablation_progress import recompute,NAMES,ARMS,TASKS,READOUTS,save

TESTS=(*TASKS,'pooled')
TITLES=dict(cross_date='跨日测试',cross_rx='跨接收机测试',cross_both='跨日跨接收机测试',pooled='三类测试综合')
SEEDS=(2026092701,2026092702,2026092703)


def load(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)


def run(snapshot_path,metadata_path,output):
    output=Path(output);snapshot=load(snapshot_path);meta=load(metadata_path)
    status,summary,paired,check=recompute(snapshot)
    if (len(status)!=15 or any(not r['scored'] or r['epoch']!=50 for r in status)
        or check['partial_arms'] or check['source_models_training'] or snapshot['state']['status']!='ANALYZED'
        or snapshot['state']['counts']!={'COMPLETE':37} or snapshot['state']['failures']):
        raise ValueError('Incomplete prespecified fifteen-model matrix')
    verify=json.loads((output/'verification.json').read_text(encoding='utf-8'))
    fields=('arm','test','readout','view','dimension','stratum','metric')
    index=lambda rows:{tuple(r[k] for k in fields):r for r in rows if r['dimension']=='overall'}
    comparisons=0
    for local,remote in ((summary,meta['summary']),(paired,meta['paired'])):
        left,right=index(local),index(remote)
        if set(left)!=set(right):raise ValueError('Remote/local summary coverage differs')
        for key in left:
            if left[key]['n']!=3 or any(abs(float(left[key][k])-float(right[key][k]))>1e-10 for k in ('mean','sd')):
                raise ValueError(('Independent replay disagrees',key))
            comparisons+=1
    timing=meta['timing_rounds']
    expected=set(product(meta['timing_complete']['cases'],(1,8,32,64,128),range(5)))
    if len(timing)!=250 or {(r['case'],r['batch'],r['round']) for r in timing}!=expected:
        raise ValueError('Incomplete five-round timing')
    costs=meta['costs'];new=[r for r in costs if r['reused']=='False']
    if len(new)!=12 or sum(int(r['updates']) for r in new)!=133200 or sum(int(r['epochs']) for r in new)!=600:
        raise ValueError('New training budget differs')
    # Make each actual model seed inspectable; do not replace them with scene repeats.
    per_seed=[]
    for seed in SEEDS:
        subset=dict(snapshot,records=[r for r in snapshot['records'] if r['row']['model_seed']==seed])
        _,ss,_,_=recompute(subset)
        for r in ss:
            if r['dimension']=='overall':
                per_seed.append(dict(seed=seed,**{k:v for k,v in r.items() if k not in ('n','sd','seeds')},percent=r['mean']))
    save(output/'per_seed_overall.csv',per_seed)
    save(output/'costs.csv',costs)
    save(output/'PA_reference_interaction.csv',meta['interaction'])
    verify.update(status='VERIFIED_FINAL_COMPLETE',remote_summary_cells_compared=comparisons,
        all_15_rows_complete=True,model_seeds=list(SEEDS),new_epochs=600,new_updates=133200,
        new_worker_hours=sum(float(r['worker_seconds']) for r in new)/3600,timing_case_rounds=250,
        automatic_followup_required=False)
    (output/'verification.json').write_text(json.dumps(verify,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    overall=[r for r in summary if r['dimension']=='overall'];effects=[r for r in paired if r['dimension']=='overall']
    find=lambda rows,a,t,rd,v,m:next(r for r in rows if r['arm']==a and r['test']==t and r['readout']==rd and r['view']==v and r['metric']==m)
    compact=[]
    for rd,a in product(READOUTS,ARMS[1:]):
        r=dict(readout=rd,arm=a,name=NAMES[a])
        for t in TESTS:
            for v,key in [('clean','clean'),('six_practical_mean','six')]:
                e=find(effects,a,t,rd,v,'accuracy')
                r[t+'_'+key+'_delta']=e['mean'];r[t+'_'+key+'_delta_sd']=e['sd']
        compact.append(r)
    save(output/'accuracy_change_compact.csv',compact)
    lines=['# 监督去 SIG-TIME PAMF 五组三种子消融：最终结果','',
       '已核实全部完成（VERIFIED）：五组各 3 个模型种子，共 15 个模型单元。12 个新增模型均完成 E50／11100 更新；基底 3 个模型按预登记复用。三类测试、每类 clean 与六种 practical 场景、直接识别与接入同一 DURA-L 均已固定预测并独立 truth-last 评分。37 个阶段任务全部完成，无失败；250 个计时 case-round 已完成。',
       '核验时间：'+verify['observed_at']+'。','',
       '## 数据口径与比较方式','',
       '监督训练只使用固定 L_s=6300，V=27000；U IQ／TX 标签、EMA、伪标签均关闭。沿用原 CE、LS0、普通增强、practical 拼接 CE0.68 和原联合训练的 160 维六类分类头；每模型保持 20＋10＋20 三段轨迹。五组均删除整个 SIG-TIME 路径描述支路；本矩阵不包含局部顺序打乱。',
       '三个识别器种子为 2026092701／02／03，恢复器是同一冻结的 DURA-L。三种子统计不代表三次独立恢复器训练，也不构成新的独立盲测。',
       '跨日：已见源接收机×未见 D0，每景 30000 包；跨接收机：未见接收机×已见 D1–D3，每景 126000 包；双跨：未见接收机×D0，每景 42000 包。综合每景 198000 包，按物理样本数加权准确率，并合并混淆矩阵重算宏平均 F1。D0 早于训练日，不声明未来日期外推。',
       '所有表中绝对值均为三模型种子均值±样本标准差，准确率／F1 单位为 %；Δ 是同种子逐对相减后的均值±样本标准差，单位为百分点，正值表示优于去 SIG-TIME 基底。六景值是六种 practical 场景的等权均值，clean 单列。','',
       '## 三类测试及综合统计','']
    absolute=lambda r:f"{r['mean']:.2f}±{r['sd']:.2f}"
    delta=lambda r:f"{r['mean']:+.2f}±{r['sd']:.2f}"
    for t in TESTS:
        lines.extend(['### '+TITLES[t],''])
        for rd,title in [('raw','直接识别'),('DURA-L','接入 DURA 后识别')]:
            lines.extend(['**'+title+'**','',
              '| 结构 | clean 准确率 | Δ clean | 六景准确率 | Δ 六景 | 六景宏平均 F1 | Δ F1 |',
              '| --- | --- | --- | --- | --- | --- | --- |'])
            for a in ARMS:
                cells=[]
                for v,m in [('clean','accuracy'),('six_practical_mean','accuracy'),('six_practical_mean','macro_f1')]:
                    s=find(overall,a,t,rd,v,m);p=None if a=='BASE' else find(effects,a,t,rd,v,m)
                    cells.extend([absolute(s),'—' if p is None else delta(p)])
                lines.append('| '+NAMES[a]+' | '+' | '.join(cells)+' |')
            lines.append('')
    lines.extend(['## 结果解释','',
       '功放支路的删除带来本矩阵最明显的单支路损伤。直接识别时，三类综合 clean／六景分别下降 3.48±0.34／7.21±0.46 个百分点；DURA 后仍下降 3.51±0.38／2.98±1.31 个百分点。DURA 缓解了六景损伤，但没有消除功放支路的收益。',
       '删除参考响应支路后，综合 clean 分别下降 1.44±0.56（直接）和 1.39±0.50（DURA）个百分点；综合六景分别下降 0.27±0.23 和 0.75±0.41 个百分点。双跨 DURA 六景变化仅为 −0.08±0.70，不能把这个分群接近持平推广到全部测试。',
       '同时删除两支路的三种子结果现已补齐。三类综合直接识别 clean／六景下降 5.32±1.03／8.37±0.28；DURA 后下降 5.30±1.08／4.08±1.05 个百分点。本版取代此前仅两个种子的阶段性估计。',
       '参考模板相位打乱保留参数量。综合直接识别 clean／六景变化 −0.55±0.51／−0.36±0.08，DURA 后为 −0.51±0.50／−0.41±0.36 个百分点。双跨 DURA 六景为 −0.04±0.10，差异很小；跨日直接识别六景反而为 +0.24±0.18。它不支持“正确模板相位在所有群体和场景均不可替代”的强结论。',
       '删支同时改变容量与计算，不能把掉分单独解释为恢复真实硬件参数或严格物理因果证明。三种子结果不作强显著性声明；全部负结果、微小正结果和分群差异均保留。','',
       '## 参数、成本和核验','',
       '| 结构 | 识别器参数（含六类头） | DURA＋识别器参数 |','| --- | --- | --- |'])
    params=dict(BASE=177025,NO_PA=165985,NO_REFERENCE=164225,NO_PA_REFERENCE=153185,WRONG_REFERENCE=177025)
    for a in ARMS:lines.append(f'| {NAMES[a]} | {params[a]} | {params[a]+266951} |')
    lines.extend(['',f"新增训练总计 600 个逻辑 epoch／133200 次更新、{verify['new_worker_hours']:.3f} 个 worker 小时；不是串行墙钟时间。未新训恢复器。计时覆盖 5 结构×2 路径×5 batch×5 轮，共 250 case-round，源 IQ、独占 GPU，保留既定 eager 识别／优化图恢复识别路径，不能跨执行路径直接归因速度。",'',
      '已从冻结独立评分重算 17010 个混淆矩阵、核对 8505 行净纠错，并将 640 个绝对汇总单元和 512 个配对汇总单元与远端自动分析逐项比较，均一致。核对全部 15 个模型来源、E50／11100 预算及完整预测／评分标记；没有新读取测试真值、改动训练或据结果选择性重跑。','',
      '## 文件','',
      '- `summary_overall.csv`：三类测试及综合、clean／逐场景／六景、准确率／宏平均 F1 的完整三种子统计。',
      '- `paired_vs_baseline_overall.csv`：相同口径的同种子配对差及每种子差值。',
      '- `per_seed_overall.csv`：全部实际种子逐项指标。',
      '- `accuracy_change_compact.csv`：直接识别／DURA、四种测试口径的 clean 与六景配对变化。',
      '- `summary_all_strata.csv.gz`、`paired_all_strata.csv.gz`：逐 TX／RX／day 及其分组统计。',
      '- `PA_reference_interaction.csv`、`costs.csv`、`verification.json`：预登记四格交互、训练成本与数值核验。'])
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    package=output.parent/(output.name+'.zip')
    with zipfile.ZipFile(package,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file():archive.write(path,arcname=path.name)
    with zipfile.ZipFile(package) as archive:
        if archive.testzip() is not None or not {'report.md','summary_overall.csv','paired_vs_baseline_overall.csv','verification.json'}<=set(archive.namelist()):
            raise ValueError('Final package incomplete')
    print(json.dumps(dict(status='VERIFIED_FINAL_COMPLETE',verification=verify,report=str(output/'report.md'),package=str(package),compact=compact),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--metadata',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.snapshot,a.metadata,a.output)
