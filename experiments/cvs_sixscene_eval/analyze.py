"""Read-only collection and complete confusion-matrix based test report."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
from experiments.cvs_sixscene_eval.common import RUN,RELEASE,PROJECT,SCENES,VIEWS,SEEDS,read,write
from experiments.cvs_selected_concat.publish import ssh
from experiments.cvs_selected_concat.analyze import metrics

ROOT=Path(__file__).resolve().parents[2]
NAMES={'clean':'clean','practical_high':'郊区高仰角','practical_mid':'郊区中仰角','practical_low_suburban':'郊区低仰角','practical_high_urban':'城区高仰角','practical_mid_urban':'城区中仰角','practical_low_urban':'城区低仰角'}
def collect(folder):
    script='''import json,time
from pathlib import Path
p=Path(PROJECT);run=p/'runs'/RUN;release=p/'releases'/RELEASE
def read(path):return json.loads(path.read_text())
d=dict(read_at=time.time(),completion=read(run/'completion.json'),scoring=read(run/'scoring_complete.json'),scores=read(run/'scores.json'),manifest=read(run/'received_views/manifest.json'),submit=read(release/'submit.json'),dispatcher=read(run/'dispatcher.json'),launch=read(run/'launch.json'),rows=[])
for row in d['launch']['rows']:
 q=run/row['row_id']/'prediction';d['rows'].append(dict(**row,complete=read(q/'complete.json'),resolved=read(q/'resolved_config.json'),provenance=read(q/'provenance.json'),alive=(Path('/proc')/str(row['pid'])).exists()))
print(json.dumps(d))
'''
    for key,value in [('PROJECT',PROJECT),('RUN',RUN),('RELEASE',RELEASE)]:script=script.replace(key,repr(value))
    d=json.loads(ssh(script))
    if d['completion'].get('status')!='ANALYZED' or d['scoring'].get('status')!='SCORED_COMPLETE' or d['scoring'].get('metric_records')!=784 or len(d['rows'])!=8:raise ValueError('Remote results incomplete')
    if any(r['alive'] or r['complete']['truth_read'] or r['provenance']['status']!='VERIFIED' or r['resolved']['query_fit'] or r['resolved']['total_parameters']!=164225 for r in d['rows']):raise ValueError('Runtime/provenance audit failed')
    folder.mkdir(parents=True,exist_ok=True);write(folder/'readback.json',d);return d

def summarize(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['condition'],r['view'],r['dimension'],r['stratum'])].append(r)
    result=[]
    for (condition,view,dimension,stratum),group in groups.items():
        if {r['model_seed'] for r in group}!=set(SEEDS) or len(group)!=4:raise ValueError('Incomplete seed stratum')
        entry=dict(condition=condition,view=view,dimension=dimension,stratum=stratum,count=group[0]['query_count'])
        if any(r['query_count']!=entry['count'] for r in group):raise ValueError('Different seed populations')
        for metric in ('accuracy','macro_f1'):
            values=[100*r[metric] for r in group];entry[metric+'_mean']=statistics.mean(values);entry[metric+'_sd']=statistics.stdev(values)
        result.append(entry)
    return result

def csvwrite(path,rows):
    with path.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def analyze(folder,d):
    rows=d['scores']['results']
    if len(rows)!=784:raise ValueError('Missing score rows')
    for r in rows:
        actual=metrics(r['confusion'])
        if any(not math.isclose(actual[k],r[k],rel_tol=0,abs_tol=1e-13) for k in ('accuracy','macro_f1')) or actual['query_count']!=r['query_count']:raise ValueError('CM metric mismatch')
    for condition in ('clean_train','mid_low_aug'):
        for seed in SEEDS:
            for view in VIEWS:
                use=[r for r in rows if r['condition']==condition and r['model_seed']==seed and r['view']==view]
                allrow=next(r for r in use if r['dimension']=='overall');cm=allrow['confusion']
                if allrow['query_count']!=168000:raise ValueError('Full-view count mismatch')
                for dim,n in [('receiver',7),('transmitter',6)]:
                    sub=[r for r in use if r['dimension']==dim]
                    if len(sub)!=n or any(sum(r['confusion'][i][j] for r in sub)!=cm[i][j] for i in range(6) for j in range(6)):raise ValueError('Strata CM mismatch')
    summary=summarize(rows);index={(r['condition'],r['view'],r['dimension'],r['stratum']):r for r in summary}
    by={(r['condition'],r['model_seed'],r['view'],r['dimension'],r['stratum']):r for r in rows}
    paired=[]
    for r in summary:
        if r['condition']!='clean_train':continue
        entry={k:r[k] for k in ('view','dimension','stratum','count')}
        for metric in ('accuracy','macro_f1'):
            values=[100*(by[('mid_low_aug',s,r['view'],r['dimension'],r['stratum'])][metric]-by[('clean_train',s,r['view'],r['dimension'],r['stratum'])][metric]) for s in SEEDS]
            entry[metric+'_delta_mean_pp']=statistics.mean(values);entry[metric+'_delta_sd_pp']=statistics.stdev(values)
            for seed,value in zip(SEEDS,values):entry[metric+'_delta_s'+str(seed)+'_pp']=value
        paired.append(entry)
    csvwrite(folder/'test_summary.csv',summary);csvwrite(folder/'paired_differences.csv',paired)
    csvwrite(folder/'per_seed_metrics.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in rows])
    write(folder/'confusion_matrices.json',dict(results=rows))
    def fmt(r,metric='accuracy'):return f"{r[metric+'_mean']:.3f} ± {r[metric+'_sd']:.3f}"
    def get(c,v,dim='overall',s='ALL'):return index[(c,v,dim,s)]
    lines=['# residual_fusion 完整六环境测试结果','',
        '状态：**ANALYZED／VERIFIED**。纯 clean 训练和 mid／low urban 拼接增强训练各4个冻结模型；8行全部独立预测完成后才由单独 scorer 连接 truth。每个模型、每个视图168000条，相同物理query覆盖 clean＋六环境，共9408000次预测。全部784个总体／RX／TX混淆矩阵已独立重算，RX及TX矩阵分别求和均与总体一致。无缺失seed或环境，无训练、适应、重新选模或选择性重跑。','',
        '以下 accuracy 和 Macro-F1 为百分数，± 为4个模型种子的样本标准差；差值为同seed“增强训练−纯clean训练”的百分点均值。该标准差只反映模型种子变化，信道观测是固定一组，不表示信道随机性或接收机泛化的置信区间。','',
        '| 测试视图 | 仰角 | 单模型样本数 | 纯clean准确率 | 增强准确率 | 差值/百分点 | 纯clean Macro-F1 | 增强 Macro-F1 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for view in VIEWS:
        a,b=get('clean_train',view),get('mid_low_aug',view)
        angle='—' if view=='clean' else '45°至80°' if 'high' in view else '20°至45°' if 'mid' in view else '10°至30°'
        lines.append(f"| {NAMES[view]} / `{view}` | {angle} | {a['count']} | {fmt(a)} | {fmt(b)} | {b['accuracy_mean']-a['accuracy_mean']:+.3f} | {fmt(a,'macro_f1')} | {fmt(b,'macro_f1')} |")
    lines+=['','## 六环境平均与最差环境','']
    for condition,label in [('clean_train','纯clean训练'),('mid_low_aug','mid／low urban增强训练')]:
        vals=[statistics.mean(by[(condition,s,v,'overall','ALL')]['accuracy']*100 for v in SCENES) for s in SEEDS]
        fvals=[statistics.mean(by[(condition,s,v,'overall','ALL')]['macro_f1']*100 for v in SCENES) for s in SEEDS]
        worst=min(SCENES,key=lambda v:get(condition,v)['accuracy_mean'])
        lines.append(f"- {label}：六环境等权平均准确率 {statistics.mean(vals):.3f}% ± {statistics.stdev(vals):.3f}%；场景Macro-F1等权平均 {statistics.mean(fvals):.3f}% ± {statistics.stdev(fvals):.3f}%。最差环境为 {NAMES[worst]}，准确率 {fmt(get(condition,worst))}%。")
    lines+=['','六环境样本数相同，准确率等权平均等于合并计数后的准确率；这里的Macro-F1为场景F1平均，不称作跨场景合并Macro-F1。','',
        '## 每个种子的总体结果','','表中单元格为 accuracy／Macro-F1（%），4列模型seed均保留。','',
        '| 训练条件 | 测试视图 | '+ ' | '.join(str(s) for s in SEEDS)+' |','|---|---|'+'---:|'*4]
    for condition,label in [('clean_train','纯clean'),('mid_low_aug','增强')]:
        for view in VIEWS:
            values=[by[(condition,s,view,'overall','ALL')] for s in SEEDS]
            lines.append('| '+label+' | '+NAMES[view]+' | '+' | '.join(f"{100*r['accuracy']:.3f}／{100*r['macro_f1']:.3f}" for r in values)+' |')
    lines+=['','## 逐接收机结果','','每个RX在每个视图固定24000条、含全部6类；数字为四seed均值±标准差（%）。RX标识直接来自独立truth评分，未用于预测或生成信道。','']
    receivers=sorted({r['stratum'] for r in summary if r['dimension']=='receiver'})
    for view in VIEWS:
        lines+=['### '+NAMES[view],'','| RX | 纯clean accuracy | 增强 accuracy | 增强−纯clean/百分点 | 纯clean Macro-F1 | 增强 Macro-F1 |','|---|---:|---:|---:|---:|---:|']
        for rx in receivers:
            a,b=get('clean_train',view,'receiver',rx),get('mid_low_aug',view,'receiver',rx)
            if a['count']!=24000:raise ValueError('RX expected count mismatch')
            lines.append(f"| {rx} | {fmt(a)} | {fmt(b)} | {b['accuracy_mean']-a['accuracy_mean']:+.3f} | {fmt(a,'macro_f1')} | {fmt(b,'macro_f1')} |")
    lines+=['','## 逐发射机结果','','每类每视图28000条。每格为“纯clean训练／增强训练”的准确率均值（%）；全部逐seed结果及标准差见CSV，单类不解释6类Macro-F1。','',
        '| TX | '+' | '.join(NAMES[v] for v in VIEWS)+' |','|---|'+'---:|'*7]
    for tx in d['manifest']['classes']:
        values=[]
        for view in VIEWS:
            a,b=get('clean_train',view,'transmitter',tx),get('mid_low_aug',view,'transmitter',tx)
            if a['count']!=28000:raise ValueError('TX expected count mismatch')
            values.append(f"{a['accuracy_mean']:.3f}／{b['accuracy_mean']:.3f}")
        lines.append('| '+tx+' | '+' | '.join(values)+' |')
    lines+=['','## 信道观测及执行成本','','配置保留 residual／post_sync／noeq、25MHz、equalized1／中心256点／单位RMS。两组模型共享每个ID的六观测，evaluation_seed=392005、receiver_seed=2027、virtual_target_session0、namespace=phase1_full_six_fixed_20261002_v1。','',
        '| 环境 | locked | degraded | unlocked | 生成耗时/秒 |','|---|---:|---:|---:|---:|']
    for item in d['manifest']['evidence']:
        counts=item['locks']
        if sum(counts.values())!=168000:raise ValueError('Channel count mismatch')
        lines.append(f"| {NAMES[item['scene']]} | {counts.get('locked',0)} | {counts.get('degraded',0)} | {counts.get('unlocked',0)} | {item['seconds']:.3f} |")
    lines+=['','| 训练条件 | seed | GPU | 七视图推理/秒 | CUDA峰值分配/MiB |','|---|---:|---:|---:|---:|']
    for row in d['rows']:
        c=row['resolved'];done=row['complete']
        lines.append(f"| {c['condition']} | {c['model_seed']} | {row['gpu']} | {sum(done['inference_seconds'].values()):.3f} | {done['peak_cuda_allocated_bytes']/1024**2:.3f} |")
    lines+=['','模型164225参数，固定E200，域骨干关闭；本轮可训练参数为0，无SFT、support或新类。推理batch256，统一完整FP32，cuDNN／matmul TF32关闭。硬件N607 RTX3090，Torch2.1.0+cu121；8行并行运行，耗时包含数据读取、buffer复制、传输和前向，不是独占GPU的微基准；峰值为PyTorch allocated，不是整卡或主机峰值。CPU峰值、常驻状态／新增传输字节、逐day指标：N/A，本轮未测量或truth未预登记该维度。','',
        '## 解释边界','','训练增强只有 practical_mid（郊区中仰角）和 practical_low_urban（城区低仰角）；其余四个环境没有在该训练增强配方中使用。本表直接反映冻结模型测试表现；不同场景损失及接收机差异不能只用一个总体分数代替。全部负差值保留，测试不回流结构、超参数、epoch、seed排除或选择性重跑。','',
        '两组父模型均从零训练，匹配相同source物理L／U／V。原纯clean训练backend记录与本次增强训练显式FP32规定存在差别，且结构已依据此前公开benchmark指定。因此差值是固定权重的描述性配对比较，不单独确证增强因果，不宣称首次盲测／目标无接触选结构；本轮六环境是工程代理信道，不能单凭这些分数证明实测在轨泛化或真正完成RFF物理可辨识。','',
        '此前三环境分数分别来自不同样本子集；本轮每环境完整168000且使用新固定realization，不将其当作旧分数的同口径复算。','',
        '[测试预登记](experiment.json) · [完整运行与评分读回](evidence/final/readback.json) · [总体/RX/TX均值及标准差](test_summary.csv) · [784行逐seed指标](per_seed_metrics.csv) · [同seed配对差值](paired_differences.csv) · [完整混淆矩阵](confusion_matrices.json)。']
    (folder/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(folder/'analysis.json',dict(status='VERIFIED',all_cm_recomputed=784,rx_tx_sums_match=True,summary=summary,paired=paired,claim='DESCRIPTIVE_EXPOSED_BENCHMARK_FIXED_WEIGHTS_NO_TARGET_FEEDBACK'))
    print(json.dumps([r for r in summary if r['dimension']=='overall'],ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report-root',type=Path,default=ROOT/'automation_reports/CV-SincNet'/RUN);a=p.parse_args()
    d=collect(a.report_root/'evidence/final');analyze(a.report_root,d)
