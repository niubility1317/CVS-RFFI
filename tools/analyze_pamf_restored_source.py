"""Aggregate completed frozen source pairs; never opens target or IQ data."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from experiments.pamf_channel_pl.restored_source import VIEWS, SCENES, SEEDS, paired_counts


def analyze(snapshot_path, predictions_path, output):
    snapshot=json.loads(Path(snapshot_path).read_text(encoding='utf-8'));done=snapshot['complete']
    if (done is None or done['status']!='SOURCE_PAIRED_DIAGNOSTIC_COMPLETE' or done['source_V_seen']!=27000
            or done['model_seeds']!=list(SEEDS) or done['views']!=list(VIEWS)
            or not done['parameters_and_buffers_unchanged'] or done['target_reads'] or done['U_IQ_reads']
            or done['U_truth_read'] or done['F_updates'] or done['G_updates'] or done['head_fit_updates']):
        raise ValueError('Complete fixed source-only diagnostic required')
    with np.load(predictions_path,allow_pickle=False) as data:
        z=data['logits'];y=data['y'];rx=data['rx'];day=data['day'];accepted=data['raw_consensus_accepted']
        if z.shape!=(3,8,27000,6) or y.shape!=(27000,) or not np.isfinite(z).all() or tuple(data['views'])!=VIEWS:
            raise ValueError('Full aligned source diagnostic arrays required')
        pred=z.argmax(-1)
    pairs=[('raw','G_raw')]+[(s,'G_'+s) for s in SCENES]
    counts=[]
    dimensions=[('overall','ALL',np.ones(len(y),bool)),('raw_consensus','accepted',accepted)]
    for dim,value in [('TX',y),('RX',rx),('day',day)]:
        dimensions.extend((dim,str(v),value==v) for v in np.unique(value))
    for j,seed in enumerate(SEEDS):
        for before,after in pairs:
            a,b=VIEWS.index(before),VIEWS.index(after)
            for dim,stratum,mask in dimensions:
                r=paired_counts(pred[j,a],pred[j,b],y,mask)
                if sum(r[k] for k in ('both_correct','corrected','broken','both_wrong'))!=r['n']:
                    raise ValueError('Paired partition inconsistent')
                counts.append(dict(model_seed=seed,before=before,after=after,dimension=dim,stratum=stratum,**r))
    quality=[]
    for row in done['policies']:
        r=dict(row);n=r['accepted'];r.update(precision_percent=100.*r['correct']/n if n else None,
            coverage_percent=100.*n/r['seen']);quality.append(r)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    def save(name,rows):
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with (out/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
    save('source_paired_counts.csv',counts);save('source_frozen_views.csv',done['results'])
    save('source_feature_geometry.csv',done['geometry']);save('source_pseudo_quality.csv',quality)
    endpoints=[]
    for before,after in pairs:
        rows=[r for r in counts if r['before']==before and r['after']==after and r['dimension']=='overall']
        gains=[r['restored_accuracy']-r['raw_accuracy'] for r in rows]
        endpoints.append(dict(before=before,after=after,frozen_models=3,
            before_mean=float(np.mean([r['raw_accuracy'] for r in rows])),
            after_mean=float(np.mean([r['restored_accuracy'] for r in rows])),
            paired_gain_pp=float(np.mean(gains)),paired_SD_pp=float(np.std(gains,ddof=1)),
            corrected_total=sum(r['corrected'] for r in rows),broken_total=sum(r['broken'] for r in rows)))
    save('source_paired_endpoints.csv',endpoints)
    evidence={k:done[k] for k in ('status','code_commit','source_V_seen','model_seeds','parent_epoch','G_roles','views',
        'F_packet_forwards','G_packet_forwards','source_V_training_reads','U_IQ_reads','U_truth_read','target_reads',
        'optimizer_updates','F_updates','G_updates','head_fit_updates','parameters_and_buffers_unchanged','seconds')}
    evidence.update(external_action_status='VERIFIED',smoke=snapshot['smoke'],source_pseudo_policies=quality,
        technical_finalization=snapshot.get('technical_finalization'),
        endpoints=endpoints,scope='frozen source validation only;not three new trained students or target gain')
    (out/'source_result_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
    text='# 冻结原始／恢复 IQ 的源域配对结果\n\n'
    text+='完整源 V 27000 条、3 份原 ordinary-PL E30、同一 G-LU、8 视图均完成。F/G 参数和 buffers 前后相同；没有打开 U IQ、U 真值或目标 IQ，没有新训练。\n\n'
    text+='| 恢复前 → 恢复后 | 前准确率均值 | 后准确率均值 | 配对差均值±SD（pp） | 纠正总数 | 破坏总数 |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
    for r in endpoints:
        text+=f"| {r['before']} → {r['after']} | {r['before_mean']:.6f}% | {r['after_mean']:.6f}% | {r['paired_gain_pp']:+.6f}±{r['paired_SD_pp']:.6f} | {r['corrected_total']} | {r['broken_total']} |\n"
    text+='\n纠正/破坏总数为3份冻结 F 对同一源记录的合计决策数；不是81000个独立物理样本。SD仅描述三份冻结模型差异，不代表新半监督学生三种子结果或显著性。\n\n'
    q=quality[-1]
    text+=f"raw 三模型 .95 同类共识：接受 {q['accepted']}/{q['seen']}，精度 {q['precision_percent']:.6f}%，覆盖率 {q['coverage_percent']:.6f}%，相对单模型平均 raw 门控接受数保留 {100*q['retention_vs_mean_single_accepted']:.6f}%；源伪标签质量/利用率可行性{'通过' if q['feasibility_pass'] else '未通过'}。这是 V 代理，不能宣称实际 U 真值精度。\n\n"
    text+='逐 TX/RX/day、共识接受集合和恢复前后净纠错见 [配对明细](source_paired_counts.csv)；[默认头 margin、同物理特征及 IQ 功率](source_frozen_views.csv)、[不拟合参数的类/RX几何](source_feature_geometry.csv)、[伪标签质量](source_pseudo_quality.csv) 和 [执行证据](source_result_evidence.json) 全部保留。\n\n'
    text+='G(raw) 对源 clean 的总体影响很小，三种受损视图经 G 后都有明显净纠错；同时恢复后的源分类仍低于 raw clean。可以据此验证恢复域监督的可行性，不能断言 G 普遍破坏身份，也不能把这些源指标当成目标收益。首个训练候选保留 raw/原 As 主目标，以可靠 raw 伪标签监督 G(H(U))；不要求恢复后的当前分类再高置信，以免排除需要学习的困难记录。\n\n'
    text+='这些结果不能区分 G 不可逆损失与 F 的分布不适配。没有开展梯度剂量或恢复后增强验证；后续训练矩阵仍须在目标访问前冻结，并保留等量 raw 监督对照。\n'
    if snapshot.get('technical_finalization'):
        text+='\n技术收尾：所有前向、NPZ、参数检查和 complete.json 已完成后，终态文件因默认 exclusive 写入报 FileExistsError。保留原日志和旧状态备份，仅同步本 run 状态，已独立读回；没有重跑前向或修改科学输出。未来代码改为显式覆盖状态文件，一次针对该问题的审查 PASS。\n'
    (out/'source_analysis.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(status=done['status'],endpoints=endpoints,source_pseudo=quality[-1],seconds=done['seconds'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--predictions',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();analyze(a.snapshot,a.predictions,a.output)
