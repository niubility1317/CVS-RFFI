"""Render complete captured scores, completion evidence, and transparent ETA estimates."""
import argparse,csv,datetime,json,math,statistics,tarfile
from pathlib import Path

RUN='a1_mechanism_periodic_s392005_20260910_r1'
ORDER=['B0_FIXED','B1_TAIL_LR','B2_RC4_WEIGHT','D0_NO_ORBIT','D1_THREE_VIEW','D2_PHYSICAL_ORBIT',
       'D3_TANGENT','F0_R3_REFERENCE','F1_R3_CONTINUOUS','F2_R3_IDENTITY','F3_R3_SWAP',
       'G0_EQUAL_BRANCH','G1_FISHER_GATE','E0_RESPONSE_ONLY','E1_RESPONSE_FUSED','E2_RESPONSE_PAIR']
SCENES=['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak','leo_mean']

def number(value): return '—' if value is None else f'{value:.2f}'
def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+
        ['|'+'|'.join(str(x) for x in row)+'|' for row in rows])
def score_table(rows):
    return table(['实验','轮次','clean%','clear%','low-elev%','rain%','LEO均值%'],
        [[r['row'],r['epoch']]+[number(r[k]) for k in SCENES] for r in rows])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--folder',type=Path,required=True);args=parser.parse_args()
    p=args.folder;a=json.loads((p/'analysis.json').read_text(encoding='utf-8'))
    captured=datetime.datetime.fromisoformat(a['inventory'][0]['captured_at']).astimezone(datetime.timezone(datetime.timedelta(hours=8)))
    scores=a['scores'];current=[s for s in scores if s['run']==RUN]
    statuses=a['states'][RUN]['rows'];eta=[]
    for name in ORDER:
        s=a['summaries'].get(RUN+'/'+name,{})
        completed=sorted(r['epoch'] for r in current if r['row']==name)
        e=s.get('latest_epoch',0);running=statuses[name]['status']=='RUNNING'
        estimate=None;low=None;high=None;eval_seconds=None;end=None
        if running:
            evaluation=s['periodic_median']
            if evaluation is None:
                evaluation=s['recent_source_eval_median']*(168000/27000)
            remaining_evaluations=13-len(completed)
            estimate=((200-e)*s['recent_base_median']+remaining_evaluations*evaluation)/3600
            # Explicit engineering uncertainty, not a statistical confidence interval.
            low=estimate*.8;high=estimate*1.25
            if name=='G1_FISHER_GATE':high=estimate*1.5
            eval_seconds=evaluation
            end=(captured+datetime.timedelta(hours=estimate)).strftime('%m-%d %H:%M')
        eta.append({'row':name,'status':statuses[name]['status'],'epoch':e,'target_epochs':','.join(map(str,completed)),
            'recent_epoch_minutes':s.get('recent_base_median',0)/60 if s else None,
            'remaining_hours_central':estimate,'remaining_hours_low':low,'remaining_hours_high':high,
            'finish_time_central_HKT':end,'test_seconds_per_checkpoint':eval_seconds,
            'test_cost_measured':bool(s.get('periodic_median')),
            'source_clean_pct':s.get('latest_source_clean'),'source_leo_mean_pct':s.get('latest_source_leo'),
            'peak_reported_MiB':s.get('peak_mb'),'nonfinite_skip_epochs':len(s.get('skipped_epochs',[]))})
    with (p/'current_status_eta.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(eta[0]));writer.writeheader();writer.writerows(eta)
    with tarfile.open(p/'extra_json.tar.gz') as archive:
        lora=json.load(archive.extractfile('runs/tweak_config2_portability_20260909_v5/official_config2_full/results.json'))
    (p/'lora_complete_results.json').write_text(json.dumps(lora,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lora_rows=[]
    for r in lora['single_domain_calibration_4x4']:
        assert abs(r['accuracy']-r['correct_decisions']/r['decisions'])<1e-7
        lora_rows.append({'calibration':r['calibration_configuration'],'test':r['test_configuration'],
            'accuracy_pct':r['accuracy']*100,'correct':r['correct_decisions'],'decisions':r['decisions']})
    for r in lora['multiple_configuration_calibration']:
        assert abs(r['accuracy']-r['correct_decisions']/r['decisions'])<1e-7
        lora_rows.append({'calibration':'multiple','test':r['test_configuration'],
            'accuracy_pct':r['accuracy']*100,'correct':r['correct_decisions'],'decisions':r['decisions']})
    with (p/'lora_all_20_tests.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(lora_rows[0]));writer.writeheader();writer.writerows(lora_rows)
    files=len(a['parsed_files']);record_count=sum(x['records'] for x in a['parsed_files'])
    current_summary=table(['实验','状态','完成轮次','近10轮中位耗时/分','剩余时间估计/小时','已测目标轮次'],[
        [r['row'],'运行中' if r['status']=='RUNNING' else '失败',r['epoch'],number(r['recent_epoch_minutes']),
         f"{r['remaining_hours_low']:.1f}–{r['remaining_hours_high']:.1f}" if r['remaining_hours_central'] is not None else '无完成时间',
         r['target_epochs'] or '尚无'] for r in eta])
    report=f'''# A1实验完整状态、预计时间与测试结果

数据快照：{captured.strftime('%Y-%m-%d %H:%M:%S')}，香港时间。此后运行可能继续推进，以下指标不混入后续轮次。

## 结论

本轮16组均已获得过启动机会；当前14组运行、2组失败、0组完成E200。9组已产生17份完整周期测试。旧任务自然完成后，先前排队的3组自动补位，不再需要停止旧任务来释放名额。

新完成：R3_B160已完成E160及E80至E160全部9次目标测试；R3_REFERENCE_CLEAN_CROSS_RX已完成E200源域训练，但没有目标测试；LoRa Config2复现实验已完成100轮及20项测试。原checkpoint的epoch、scratch参数、无历史baseline/teacher、文件存在均已独立核实，见completed_artifact_verification.json。

按当前速度，基础/DAOT/R3大部分还需约5–20小时；两个仍运行的响应分支约1–2天；G1_FISHER_GATE中心估计约6天、工程范围约5–9天，且尚未进入后期，误差最大。两项失败不会自动恢复，因此不能给“全部16组完成”的确定时间。

## 本轮逐组进度与ETA

{current_summary}

估算方法：最近10个完整epoch的“epoch_time_s减周期测试耗时”的中位数×剩余轮数，加剩余固定测试次数×实测测试耗时中位数。尚无目标测试的组，暂用最近source验证耗时按168000/27000样本比放大估计一次四场景测试成本。区间取中心估计的0.8–1.25倍；G1上界放宽至1.5倍，属于工程余量，不是统计置信区间。现有同卡并发已体现在实测时间中，后续课程、负载和故障仍可能改变耗时。CSV另列中心预计完成时刻。

G1最近每轮约37.8分钟，单纯剩余训练线性外推约118小时，周期评分另需时间；它是当前运行批次的主要时间瓶颈。E0/E2最近每轮约11.4/13.1分钟。单纯等待不会补齐失败的G0/E1。

## 本轮全部已完成目标测试

单位均为准确率百分比。clear、low-elev、rain分别对应leo_clear_weak、leo_low_elev_weak、leo_rain_weak。每份评分包含四场景各168000条预测，总计672000条；逐场景total、correct/total与accuracy已核对。LEO均值为三种LEO的算术均值。

{score_table(sorted(current,key=lambda r:(ORDER.index(r['row']),r['epoch'])))}

以上17份覆盖所有已完成且到达固定测试轮次的本轮实验，没有发现应有E80/E90等评分缺口。其余7组尚无目标测试，其中G0/E1已失败，其余尚未到E80。没有结果的项不以source准确率代替target准确率。

同轮观察：E90的B2相对B0，clean高1.59个百分点、LEO均值高0.40个百分点；D0移除轨道教师在E90的LEO均值低8.98个百分点，但到E110有所恢复。F3交换修正E80的LEO均值比F0低1.30个百分点。这些只是固定轮次描述，不能据此重排候选或决定组合；正式选择仍仅使用source V。不同轮次不混排为“最佳方法”。

## 新完成的R3_B160全部9次目标测试

{score_table([r for r in scores if r['row']=='R3_B160'])}

E160最终结果：clean78.58%，三种LEO为67.79%、66.11%、66.38%，LEO均值66.76%。该run为从零训练、固定160轮预算，累计epoch时间14.66小时；旧调度将目标评分放在独立进程中，因此此累计时间不包含其全部外部评分耗时。

R3_REFERENCE_CLEAN_CROSS_RX的E200源V结果为clean98.5741%、LEO均值93.3667%、最差LEO场景92.6222%；累计epoch时间16.22小时。它按source-only配置结束，target结果不存在。clean跨接收机损失与身份梯度在200轮均有记录，LEO配对计数始终为0，不能称为完整ECRS跨LEO验证。

## 其余已完成CVS结果

这些不是本次查询期间全部新完成的实验，列出是为了提供完整可查结果。

### R3_B120：全部7次测试

{score_table([r for r in scores if r['row']=='R3_B120'])}

### ECRS跨接收机旧矩阵：E200

{score_table([r for r in scores if r['run']=='a1_ecrs_cross_rx_s392005_20260909_r1'])}

这些X组采用当时的身份表示跨接收机机制，不能代称本轮固定物理响应ECRS实现；单seed结果不构成晋级证据。

### A1-Fast V2旧矩阵：E200

{score_table([r for r in scores if r['run']=='a1_fast_v2_s392005_20260909_r1'])}

### R3 scratch旧矩阵：E200

{score_table([r for r in scores if r['run']=='a1_r3_scratch_s392005_20260908_r1'])}

### 历史checkpoint初始化结果：只作历史附录

{score_table([r for r in scores if r['run']=='a1_fast_selected_adv3b02_s392005_20260908_r1'])}

这两组日志明确from_scratch=false，继承ADV3B02_CORE90_SOFT_E200。完整祖先与目标接触未在本次重新审计，保留PROTOCOL_VALIDITY_UNVERIFIED，不能与本轮scratch结果合并作为干净泛化证据。

## LoRa复现：完整20项测试

这是独立数据集上的Tweak Config2实验，与上面的WiSig/LEO结果不可横向比较。100轮已结束，checkpoint存在；状态为PAPER_METHOD_PARITY_WITH_UNPUBLISHED_DEFAULTS，表示仍包含论文未公开参数的披露默认值，并非全部论文细节已获证实。

{table(['校准配置','测试配置','准确率%','正确/决策数'],[[r['calibration'],r['test'],number(r['accuracy_pct']),str(r['correct'])+'/'+str(r['decisions'])] for r in lora_rows])}

## 失败、稳定性与数据解释

- E1_RESPONSE_FUSED：E40源clean98.14%、LEO68.36%；E41下降至96.70%/62.08%；E42降至29.14%/17.19%，训练loss升至79.93。E43第38个batch已有8个非有限batch，占21.05%，触发RC4_SYSTEMIC_NONFINITE_BATCH_GUARD。尚未到E80，因此没有target测试。
- G0_EQUAL_BRANCH：E1第8个batch累计8个非有限batch，占100%，触发同一保护；没有完成一个epoch，也没有target测试。
- 这证明存在真实训练数值失稳。保护器名称不等于根因是RC4；尚未通过故障复现定位唯一算子。E1的退化发生于E41课程切换后，时间相邻不足以证明该切换是唯一原因。需要在独立修复任务中定位，不能通过关闭保护器宣称解决。本次状态/结果查询没有停机或重启健康任务，也未重跑失败实验。
- 其他运行组也有少量非有限梯度跳步，详见current_status_eta.csv和完整曲线；不能把所有NaN遥测都当成数值故障，因为未启用诊断与未测试项本来就写为缺省值。
- 当前目标scorer提供总体和逐类结果，没有目标接收机分组表；最弱接收机指标不能从这些汇总值补造。各类完整结果单独导出1200行。

## 完整性、文件与复现

已全量读取{files}个证据文件，共解析{record_count}条文本/结构化记录，覆盖48条有epoch记录的训练轨迹。没有JSON/JSONL解析错误、CSV列宽异常、重复epoch或中间缺失epoch。G0无epoch记录单列为启动后数值失败。原始日志中的全部行均扫描了traceback、OOM、非有限保护和警告，未以末尾片段代替全量分析。

- all_target_scores.csv：全部50份CVS四场景评分，包括本轮17份、R3预算16份及历史最终17份。
- all_target_per_class.csv：50份评分×4场景×6类，共1200行。
- all_training_curves.csv：全部已捕获epoch的loss、source验证、耗时、学习率和非有限跳步记录。
- current_status_eta.csv：16组状态、时间估算、source指标、日志峰值显存与稳定性计数。
- lora_all_20_tests.csv及lora_complete_results.json：独立LoRa结果与100轮训练记录。
- analysis.json：文件清单、解析计数、完整曲线摘要、状态、错误定位及机制激活统计。
- evidence.tar.gz、extra_json.tar.gz：本次只读抓取的原始文本快照，未含数据集、checkpoint张量、目标truth或巨量预测明细。

周期target数据仅作用户授权的探索性观察，不回流训练、调参、选模或有机组合决策；重复轮次和场景不是独立样本。完整评分文件代表产物闭合，不自动代表独立确认或科学晋级。
'''
    (p/'report.md').write_text(report,encoding='utf-8')
    (p/'eta_summary.json').write_text(json.dumps(eta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(eta,ensure_ascii=False))

if __name__=='__main__':main()
