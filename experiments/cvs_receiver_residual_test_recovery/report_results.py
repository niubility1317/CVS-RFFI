"""Render complete descriptive evidence; no target-driven model selection."""
import csv
import json
import math
from pathlib import Path
from experiments.cvs_receiver_residual_test_recovery import design as d

ARMS=('baseline','displacement','contribution','combined')
SEEDS=(2026092701,2026092702,2026092703,2026092704)
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
LABELS=dict(baseline='基础 CE+星地拼接',displacement='接收机位移校正',contribution='分支贡献调节',combined='两者组合')
VIEW_LABELS=dict(clean='Clean',practical_high='高仰角',practical_mid='中仰角',practical_low_suburban='低仰角郊区',
    practical_high_urban='高仰角城市',practical_mid_urban='中仰角城市',practical_low_urban='低仰角城市')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def stat(xs):
    m=sum(xs)/len(xs);return m,math.sqrt(sum((x-m)**2 for x in xs)/(len(xs)-1))
def fmt(xs):
    m,s=stat(xs);return f'{100*m:.3f} ± {100*s:.3f}'
def table(headers,rows):
    return ['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows]+['']
def csvwrite(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    root=d.ROOT/'local_artifacts'/d.SCORE_RELEASE/'final_results'
    audit=read(d.ROOT/'local_artifacts'/d.RELEASE/'completed_training_audit.json')
    report=['# 接收机残差机制：完整冻结测试结果','',
        '状态：VERIFIED。正式结果为 R2：四种方法×四个模型 seed×七个完整视图；R1 保留为随机数混杂诊断。',
        '全部 32 个模型 E200/10000 updates 源训练完成。先冻结全部源模型，再固定全部32×7预测，最后独立连接truth评分。评分恢复复用原预测，未重新训练、重新预测、适应或用目标结果调参。','',
        '测试包含6个既有TX、7个未参与源训练的RX、4个日期，每个TX×RX×day为1000个物理样本，共168000个。七个视图共享这168000个物理ID，不是七个独立数据集。',
        '源RX编号1/3/4/6/8，测试RX编号0/2/5/7/9/10/11。测试RX物理名称依次为1-1、14-7、2-1、20-1、7-14、7-7、8-8。',
        'TX类别顺序：14-10、14-7、20-15、20-19、6-15、8-20。TX与RX即使字符串相同也属于不同角色。',
        'Clean加六个完整practical residual/post_sync/noeq视图，是已有暴露基准上的固定消融，不是独立LEO分层或真实卫星测试。',
        '本轮是Phase1冻结识别，未执行Phase2。K、仅旧类support适应前后、新增类数、新旧类准确率与H均为N/A，不能从本轮结果推出适应收益。','',
        '表中准确率和Macro-F1均为百分数；±为四seed样本标准差（ddof=1）。最差RX先逐seed取7个RX中的最低准确率，再对四seed统计。配对差值单位为百分点；四seed证据只作描述性比较，不声明统计显著性。','']
    datasets={}
    for run in reversed(d.SCORE_RUNS):
        base=root/run['run_id'];check=read(base/'local_independent_recount.json')
        if check['status']!='VERIFIED':raise ValueError('Unverified score report')
        primary=read(base/'scores.json')['results'];days=read(base/'day_scores.json')['results']
        summary=read(base/'summary.json')['results'];paired=read(base/'paired_results.json')['comparisons']
        overall={(r['arm'],r['model_seed'],r['view']):r for r in primary if r['dimension']=='overall'}
        lookup={(r['arm'],r['view']):r for r in summary}
        formal=run['package']=='cvs_receiver_residual_v2';name='R2正式四seed结果' if formal else 'R1随机混杂诊断结果'
        lines=['## '+name,'',f"训练run：`{run['parent_run_id']}`；固定预测run：`{run['prediction_run_id']}`；独立评分run：`{run['run_id']}`。",
            '','R2使用CPU私有generator初始化新增头，不重置CUDA dropout随机流。' if formal else 'R1初始化新增头时重置CUDA随机流，与baseline的dropout序列不同；不能把其差值归因于新增机制，也不与R2合并为八seed。','',
            '### 准确率','']
        lines+=table(['视图']+[LABELS[a] for a in ARMS],[[VIEW_LABELS[v]]+[f"{100*lookup[a,v]['accuracy_mean']:.3f} ± {100*lookup[a,v]['accuracy_sd']:.3f}" for a in ARMS] for v in VIEWS])
        lines+=['### Macro-F1','']+table(['视图']+[LABELS[a] for a in ARMS],[[VIEW_LABELS[v]]+[f"{100*lookup[a,v]['macro_f1_mean']:.3f} ± {100*lookup[a,v]['macro_f1_sd']:.3f}" for a in ARMS] for v in VIEWS])
        lines+=['### 最差接收机准确率','']+table(['视图']+[LABELS[a] for a in ARMS],[[VIEW_LABELS[v]]+[f"{100*lookup[a,v]['worst_rx_mean']:.3f} ± {100*lookup[a,v]['worst_rx_sd']:.3f}" for a in ARMS] for v in VIEWS])
        names=('displacement_vs_baseline','contribution_vs_baseline','combined_vs_baseline','interaction')
        plook={(r['contrast'],r['view']):r for r in paired}
        lines+=['### 同seed配对准确率差','',
            '交互项=组合−位移−贡献+基础；正值表示描述性的非加性互补，负值表示非加性损失。','']+table(['视图','位移−基础','贡献−基础','组合−基础','交互项'],
            [[VIEW_LABELS[v]]+[f"{plook[n,v]['mean_pp']:+.3f} ± {plook[n,v]['sd_pp']:.3f}" for n in names] for v in VIEWS])
        lines+=['### 全部逐seed准确率','',
            '每一行均包含168000个样本；seed只表示model/loader随机性，其他split/data/augmentation角色按原登记固定。','']
        lines+=table(['方法','视图']+[str(s) for s in SEEDS],[[LABELS[a],VIEW_LABELS[v]]+[f"{100*overall[a,s,v]['accuracy']:.3f}" for s in SEEDS] for a in ARMS for v in VIEWS])
        aggregate=[]
        for dimension,strata in [('receiver',list(d.HELDOUT_RX.values())),('transmitter',['14-10','14-7','20-15','20-19','6-15','8-20']),('day',list(d.TARGET_DAYS))]:
            selected=days if dimension=='day' else primary
            lines+=['### '+dict(receiver='全部RX分层准确率',transmitter='全部TX分层准确率',day='全部日期分层准确率')[dimension],'']
            if dimension=='transmitter':lines+=['单TX行的准确率等于该TX的召回率。原CSV中的Macro-F1仍按固定六类计算，不等于该TX的类别F1，不能将其直接与总体Macro-F1比较。','']
            table_rows=[]
            for v in VIEWS:
                for stratum in strata:
                    cells=[]
                    for a in ARMS:
                        rs=[r for r in selected if r['arm']==a and r['view']==v and r['dimension']==dimension and r['stratum']==stratum]
                        if len(rs)!=4:raise ValueError('Stratum seed coverage differs')
                        cells.append(fmt([r['accuracy'] for r in rs]))
                        m,s=stat([r['accuracy'] for r in rs]);fm,fs=stat([r['macro_f1'] for r in rs])
                        aggregate.append(dict(arm=a,view=v,dimension=dimension,stratum=stratum,seed_count=4,
                            query_count_per_seed=rs[0]['query_count'],accuracy_mean=m,accuracy_sd=s,macro_f1_mean=fm,macro_f1_sd=fs))
                    table_rows.append([VIEW_LABELS[v],stratum]+cells)
            lines+=table(['视图','分层']+[LABELS[a] for a in ARMS],table_rows)
        csvwrite(base/'strata_summary.csv',aggregate);write(base/'strata_summary.json',dict(results=aggregate))
        costs=read(base/'resources.json')['rows'];costrows=[];costexport=[]
        for a in ARMS:
            rs=[r for r in costs if r['arm']==a];ts=[r['prediction'] for r in rs]
            train=[r for r in audit['rows'] if r['parent_run_id']==run['parent_run_id'] and r['arm']==a]
            item=dict(arm=a,parameters=train[0]['parameters'],source_training_seconds_mean=stat([r['source_elapsed_seconds'] for r in train])[0],
                seven_view_inference_seconds_mean=stat([sum(t['inference_seconds'].values()) for t in ts])[0],
                peak_cuda_allocated_bytes_max=max(t['peak_cuda_allocated_bytes'] for t in ts),
                peak_cuda_reserved_bytes_max=max(t['peak_cuda_reserved_bytes'] for t in ts),peak_process_rss_bytes_max=max(t['peak_process_rss_bytes'] for t in ts))
            item['final_parameter_buffer_bytes_max']=max(r['source']['total_parameters']*4+r['source']['resident_buffer_bytes'] for r in rs)
            item['checkpoint_bytes_max']=max(r['source']['checkpoint_bytes'] for r in rs)
            if formal:
                diag=read(d.ROOT/'local_artifacts'/d.SCORE_RELEASE/'formal_training_diagnostics.json')
                item['actual_source_peak_cuda_allocated_bytes']=next(r['peak_cuda_allocated_bytes'] for r in diag['results'] if r['arm']==a)
            costexport.append(item);costrows.append([LABELS[a],item['parameters'],f"{item['source_training_seconds_mean']:.1f}",f"{item['seven_view_inference_seconds_mean']:.2f}",
                f"{item['peak_cuda_allocated_bytes_max']/1024**2:.1f}",f"{item['peak_cuda_reserved_bytes_max']/1024**2:.1f}",f"{item['peak_process_rss_bytes_max']/1024**2:.1f}"])
        lines+=['### 实测成本','',
            'N607 RTX3090，FP32，预测batch256，按每GPU最多4个总实验进程调度。源训练wall含本轮辅助项与质心扫描；共享GPU运行时间受并发影响，不能视为隔离吞吐基准。峰值取该方法四seed最大值。',
            '原resource_profile中的训练步是合成CE-only profile，不代表完整辅助训练成本；保留原始资料，不据此声称星载训练省算力。Phase2可训练参数/训练开销、星载常驻状态、实际新增传输字节数均N/A。','']
        lines+=table(['方法','参数','E200训练均值(s)','七视图推理均值(s)','CUDA allocated(MiB)','CUDA reserved(MiB)','进程RSS(MiB)'],costrows)
        csvwrite(base/'cost_summary.csv',costexport)
        lines+=['末期参数+buffer常驻大小包含质心buffer，排除optimizer、激活和数据加载；checkpoint文件包含完整保存状态，不能等同星载传输量。','']
        lines+=table(['方法','末期参数+buffer(B)','checkpoint(B)','实际源训练CUDA峰值(MiB)'],
            [[LABELS[r['arm']],r['final_parameter_buffer_bytes_max'],r['checkpoint_bytes_max'],
                f"{r['actual_source_peak_cuda_allocated_bytes']/1024**2:.2f}" if formal else 'N/A：此报告未另提取R1峰值'] for r in costexport])
        lines+=['### 产物完整性','',f"独立重算：{check['score_rows']}条主结果+{check['day_rows']}条日期结果。所有混淆矩阵准确率、Macro-F1、RX/TX/day分区合计、四seed均值与样本SD均VERIFIED。",
            '完整scores.json保留每行6×6混淆矩阵。scores.csv为1568行原始seed/RX/TX结果，day_scores.csv为448行原始seed/day结果，strata_summary.csv为全部分层四seed汇总，paired_results.json保留逐seed配对差。','']
        (base/'full_results.md').write_text('\n'.join(lines).rstrip()+'\n',encoding='utf-8');report+=lines
        datasets[run['run_id']]=dict(summary=summary,paired=paired,costs=costexport)
    diagnostics=read(d.ROOT/'local_artifacts'/d.SCORE_RELEASE/'formal_training_diagnostics.json')
    report+=['## 结果解释','',
        'R2组合机制在Clean上提高0.567个百分点，在六个信道视图的描述性均值上提高0.215个百分点；低仰角城市只有0.043个百分点。单独贡献调节在低仰角城市下降0.016个百分点。四seed存在负差值，不能称为稳定的大幅泛化突破。',
        '组合机制的最差RX均值在所有视图提高，但幅度约0.083至0.620个百分点。单独位移校正在高仰角、中仰角、低仰角郊区、高仰角城市和中仰角城市的最差RX均值下降，平均准确率的小幅增加不等于各RX都改善。',
        '基础模型从Clean的79.755%降至低仰角城市的48.261%，相差31.493个百分点；新增机制未解决该信道退化。该结果支持本轮新增残差约束收益有限，但不能证明所有DG机制无效，GRL与正交方法本轮未纳入比较。',
        '源训练最后40个epoch的实际辅助梯度非零。位移组的校正范数/基础特征范数约10.94%，组合约10.60%；贡献组的平均门偏移约6.90%，组合约6.59%。因此弱收益不能简单归因为模块未启用。',
        '源L上的跨TX接收机位移方向一致性较弱：位移组平均cosine约0.208，组合约0.202；留一TX共享位移与该TX实际质心位移的相对平方误差约0.957和0.973。这里测量的是共享位移监督假设的几何一致性，不是预测头的拟合误差。它提示“其他TX的共同接收机位移可迁移到当前TX”这一监督代理较弱；此解释是源日志支持的推断，未据此修改本轮模型或重跑。','',
        '## 失败记录与来源','',
        '原训练控制器因source_contract不含classes字段在任何目标访问前失败。新的独立预测修复逐项核对所有原角色字段，并单独核对固定类别顺序，未放宽权重来源。',
        '全部32×7预测完成后，旧scorer把物理RX名称与编号比较，因Truth coverage differs失败，未写出指标。新的CPU scorer按原rx_list物理映射评分；保留原失败与预测，不重新生成预测。',
        '评分release提交：927bc9eb799b225d7e40ec14e68a487babf7de6b。对应源权重提交R1：3712e8e4f004f21b8ffdc74ce0459ad8a2020f3d；R2：8a88b3a2e369644ea1d746c5e02006218ee00df6。',
        '科学解释只能使用R2固定消融的描述性证据。全部负结果保留，不据测试结果重新排源域选模顺序、修改模型或选择性重跑。','']
    out=d.ROOT/'automation_reports/CV-SincNet'/d.SCORE_RUNS[1]['run_id']
    (out/'complete_test_results.md').write_text('\n'.join(report).rstrip()+'\n',encoding='utf-8')
    write(root/'readout.json',datasets)
    print(json.dumps(dict(status='VERIFIED',report=str(out/'complete_test_results.md'),runs=list(datasets)),ensure_ascii=False))


if __name__=='__main__':main()
