"""Render the evidence tables for the September 8 snapshot and completed target test."""
import csv
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
D=ROOT/'docs/experiments/ecrs_results_20260908'


def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])


def main():
    a=json.loads((D/'analysis.json').read_text(encoding='utf-8'))
    source=json.loads((D/'source_final_scores.json').read_text(encoding='utf-8'))
    rows=sorted(a['rows'],key=lambda r:['B0','B2-V1','B2','B3a','B3b','B3c','B4'].index(r['row']))
    source_curve=list(csv.DictReader((D/'source_validation.csv').open(encoding='utf-8')))
    lines=['# ECRS-V1R/V2完整训练快照与已完成组目标域测试（2026-09-08）','',
        '## 状态与分析边界','',
        '训练快照采集于2026-09-08 09:05左右（N607本地时间）；下载是逐文件进行，运行组文件存在秒级采集偏差。3/7组已完成200轮并闭合源域产物；另外4组仍在训练。此处曲线只覆盖下载快照，末尾的实时状态不回填到旧曲线。', '',
        table(['组','状态','完整epoch','源域最优epoch','最优LEO均值%','最近验证epoch/均值%','末轮训练loss','跳过batch'],[
            [r['row'],r['status'],r['epoch'],r['best_epoch'],f"{r['best_source_leo_mean']:.4f}",f"{r['latest_source_epoch']}/{r['latest_source_leo_mean']:.4f}",f"{r['latest_train_loss']:.4f}",r['skipped_batches']] for r in rows]),'',
        'B0为原基线；B2-V1为旧V1/R7非融合路径；B2为V1R；B3a为V2 legacy28旧参考+real8；B3b为legacy28估计参考+real8；B3c为compact8+real8；B4为compact8+complex24。B5/B6未启动。未完成组不能与E200组直接做最终排名。', '',
        '同为E60时的描述性源域对照（不是最终排名；同seed不自动保证初始化与优化轨迹完全相同）：','',
        table(['组','E60 clean raw%','E60 LEO raw均值%'],[
            [r['row'],f"{next(float(x['tx_acc']) for x in source_curve if x['row']==r['row'] and x['epoch']=='60' and x['path']=='raw' and x['scene']=='clean'):.4f}",
             f"{statistics.mean(float(x['tx_acc']) for x in source_curve if x['row']==r['row'] and x['epoch']=='60' and x['path']=='raw' and x['scene']!='clean'):.4f}"] for r in rows]),'',
        '## 数据、选择与复算','',
        'seed=392005；源接收机1/3/4/6/8，day1/2/3；源池90000，L=6300、U=56700、validation=27000，比例0.07/0.63/0.30。U池存在不表示U损失生效。训练阶段未使用目标域选checkpoint。B0冻结E199，B3c/B4冻结E195，以源域raw三种弱LEO平均准确率选择。', '',
        f"完整读取{len(a['files'])}个快照文件，CSV与JSONL逐epoch核对训练loss、准确率和时长；全部324000条源域最终预测逐条复算准确率、唯一ID以及receiver/day分组，与保存的统计一致。14个best/latest checkpoint的元数据见checkpoint_audit.json。没有用日志尾部代替全量分析。", '',
        '## 已完成组：源域完整验证','',
        table(['组/路径','clean%','clear%','low_elev%','rain%','LEO均值%'],[
            [row+'/'+p]+[f"{f['scenarios'][s]['paths'][p]['tx_acc']:.4f}" for s in ['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak']]+[f"{statistics.mean(sc['paths'][p]['tx_acc'] for s,sc in f['scenarios'].items() if s!='clean'):.4f}"]
            for row,f in source.items() for p in f['scenarios']['clean']['paths']]),'',
        'B4的源域raw LEO均值比B0高0.2198个百分点，B3c低0.2642个百分点。B4响应分支clean准确率68.44%，三种LEO约27%–29%；B3c响应分支更弱。响应分支学到了可测的分类信号，但远弱于raw，尚不能说明得到稳定、可迁移的发射机指纹。fusion关闭时fused/common_head等于raw，rescue/harm=0不是融合收益证据。', '',
        '## 完整曲线与机制激活','',
        '![训练曲线](ecrs_results_20260908/training_curves.png)','',
        '![加权损失分量](ecrs_results_20260908/loss_components.png)','',
        'B0/B3c/B4后期源域LEO曲线已接近平台，最优点接近训练末尾。E80附近总loss上跳与日志中的加权sat分类项由0切为约4同步，且源域准确率继续上升；不能仅凭跨阶段总loss上跳判断发散。B2/B2-V1尚处早期阶段。', '',
        table(['组','响应CE有效执行batch','成功更新batch','首次epoch','EMA更新','cross_rx/U/fused有效batch'],[
            [r['row'],r['activation'].get('resp_ce',{}).get('executed_valid_batches','无revision遥测'),r['activation'].get('resp_ce',{}).get('successful_valid_batches','—'),r['activation'].get('resp_ce',{}).get('first_epoch','—'),r['activation'].get('ema_updates','—'),'0/0/0' if r['activation'] else '无该遥测'] for r in rows]),'',
        'V2四组响应CE从E1开始真实执行并产生成功优化步骤；EMA更新不代表U配对训练。B2的响应CE按阶段从E91启用，快照尚未到达，当前0次是阶段状态。旧V1物理项缺少逐batch专用loss记录，本报告不能补造其实际激活量；其配置保存在checkpoint审计。', '',
        'E200逐batch遥测中，B3c响应CE平均加权loss=0.219855、响应分支平均梯度范数=0.477103；B4分别为0.196583和0.834564。两组末轮均49个成功步骤。该证据支持响应分支实际优化，不替代跨域机制有效性证明。','',
        '## 数值异常、失败与资源','',
        '7组均出现少量unsafe backward/step skipped警告；已完成组B0为15次，B3c/B4各12次。它们不是零异常运行，但完成组无训练Traceback且正常完成。运行组stdout可能比最后完整epoch多1次警告：例如B2为7条、epoch累计6次，B2-V1为10条、epoch累计9次，属于采集中epoch边界偏差，不能抹去或混为失败次数。未因低性能停止或干预健康训练。', '',
        'V2此前r1在真实训练前的兼容性检查失败：PyTorch2.1不接受该Tensor.all的tuple维度写法。原始日志保留；修复两处降维写法后r2才开始训练，不能把失败r1计为完成实验。', '',
        table(['组','累计训练小时','累计epoch小时（含验证）','最近10轮平均秒','按当前速度剩余小时'],[
            [r['row'],f"{r['train_hours']:.3f}",f"{r['epoch_hours']:.3f}",f"{r['recent_epoch_seconds_mean']:.1f}",f"{r['eta_hours_at_recent_speed']:.1f}"] for r in rows]),'',
        '以上是各进程累计时间而非独占GPU计费时长。ETA使用最近10轮平均值，包含周期验证；B3a/B3b约5–8小时，B2/B2-V1按当前阶段约30–40小时，后续阶段和共享GPU负载可能扩大到约1.5–2天。最终源预测与测试另计。', '']
    targetfile=D/'target_test_summary.json'
    if targetfile.exists():
        target=json.loads(targetfile.read_text())
        assert target['status']=='TARGET_TEST_COMPLETE'
        lines[2:2]=['已完成的B0/B3c/B4已完成冻结目标测试：raw三种LEO均值分别为52.8706%、52.7808%、51.9599%。B3c相对B0低0.0899个百分点，B4低0.9107个百分点。B4源域的小幅优势未延续到目标域；当前保持NO_PROMOTION_TO_DEFAULT。其余4组仍在训练。','']
        target_rows=[]
        for row, result in target['rows'].items():
            for scene, score in result['scenarios'].items():
                assert score['count']==168000
                for path, metric in score['paths'].items():
                    matrix=metric['confusion']
                    assert sum(map(sum,matrix))==metric['count']==168000
                    assert sum(matrix[i][i] for i in range(6))==metric['correct']
                    assert abs(metric['accuracy']-100*metric['correct']/metric['count'])<1e-10
                    for group,cells in metric['groups'].items():
                        assert sum(v['count'] for v in cells.values())==168000
                        assert sum(v['correct'] for v in cells.values())==metric['correct']
                        for name,cell in cells.items():
                            target_rows.append({'row':row,'scene':scene,'path':path,'group':group,'group_value':name,**cell})
        with (D/'target_groups.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(target_rows[0])); writer.writeheader(); writer.writerows(target_rows)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        scenes=['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak']
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for row,result in target['rows'].items():
            color={'B0':'#287bb5','B3c':'#e68613','B4':'#29964a'}[row]
            for path in ('raw','response'):
                if path not in result['scenarios']['clean']['paths']: continue
                axes[0].plot(range(4),[result['scenarios'][s]['paths'][path]['accuracy'] for s in scenes],marker='o',color=color,linestyle='-' if path=='raw' else '--',label=row+'/'+path)
            axes[1].plot(range(4),[result['scenarios'][s]['paths']['raw']['unseen_day0_accuracy'] for s in scenes],marker='o',color=color,label=row+'/raw')
        for ax,title in zip(axes,['Frozen target accuracy (%)','Unseen day 0 raw accuracy (%)']):
            ax.set_title(title);ax.set_xticks(range(4),['Clean','Clear','Low elev.','Rain']);ax.set_ylim(0,100);ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(D/'target_accuracy.png',dpi=170);plt.close(fig)
        lines+=['## 已完成组：冻结目标域测试','',
                '状态：TARGET_TEST_COMPLETE / VERIFIED。使用未参与源训练的7个接收机0/2/5/7/9/10/11、day0/1/2/3、全部6个已注册TX；每场景168000条，共3×4×168000=2016000条预测。day0进一步构成未见日期子集。全部12个无truth预测文件完成后，独立scorer才连接truth；目标结果不回流checkpoint、head或超参数选择。', '',
                table(['组/路径','clean%','clear%','low_elev%','rain%','LEO均值%'],[
                    [row+'/'+p]+[f"{f['scenarios'][s]['paths'][p]['accuracy']:.4f}" for s in ['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak']]+[f"{statistics.mean(sc['paths'][p]['accuracy'] for s,sc in f['scenarios'].items() if s!='clean'):.4f}"]
                    for row,f in target['rows'].items() for p in f['scenarios']['clean']['paths']]),'',
                table(['组/场景','raw正确/总数','最差RX%','未见day0%','融合rescue/harm'],[
                    [row+'/'+s,f"{sc['paths']['raw']['correct']}/{sc['count']}",f"{sc['paths']['raw']['receiver_floor']:.4f}",f"{sc['paths']['raw']['unseen_day0_accuracy']:.4f}",f"{sc['rescue']}/{sc['harm']}"]
                    for row,f in target['rows'].items() for s,sc in f['scenarios'].items()]),'',
                '完整receiver/day/receiver×day/TX分组和6×6混淆矩阵位于target_test_summary.json；这是Phase1闭集冻结测试，不是Stage2未知类识别结果。', '',
                '目标clean相对B0：B3c提高0.1875个百分点，B4提高0.2452个百分点；但B3c的最差接收机clean从B0的72.0417%降至67.9708%。总体准确率和分组下限必须同时看。B4在三个目标LEO场景均低于B0，响应路径LEO均值24.4480%也低于B3c的26.5528%。这些结果不支持“复杂锚点带来跨域增益”；当前证据也不足以把下降唯一归因于某一个机制。', '',
                table(['目标RX','B0 clean%','B3c clean%','B4 clean%','B0 LEO%','B3c LEO%','B4 LEO%'],[
                    [rx]+[f"{target['rows'][row]['scenarios']['clean']['paths']['raw']['groups']['receiver'][rx]['accuracy']:.4f}" for row in ['B0','B3c','B4']]+[
                        f"{statistics.mean(sc['paths']['raw']['groups']['receiver'][rx]['accuracy'] for s,sc in target['rows'][row]['scenarios'].items() if s!='clean'):.4f}" for row in ['B0','B3c','B4']]
                    for rx in ['0','2','5','7','9','10','11']]),'',
                '![目标域准确率](ecrs_results_20260908/target_accuracy.png)','',
                table(['组','四场景推理秒数','CUDA峰值allocated MiB','CUDA峰值reserved MiB'],[
                    [row,f"{sum(x['elapsed_s'] for x in r['scenarios'].values()):.2f}",f"{r['peak_allocated_bytes']/2**20:.1f}",f"{r['peak_reserved_bytes']/2**20:.1f}"] for row,r in target['resources'].items()]),'']
    else:
        lines+=['## 目标域测试','', 'RUNNING：B0/B3c/B4已冻结checkpoint并启动四场景完整测试；未闭合前不发布目标分数。','']
    livefile=D/'live_readback.json'
    if livefile.exists():
        live=json.loads(livefile.read_text())
        lines+=['## 独立实时读回','',f"读回时间：{live['time']}；测试进程存活={live['test_pid_alive']}；目标汇总产物存在={live['target_summary_exists']}。",'',
            table(['组','完整epoch','进程存活','源域完成产物','GPU'],[[r['row'],r['epoch'],r['alive'],r['source_complete'],r['gpu']] for r in live['training']]),'',
            table(['GPU','利用率%','显存MiB','总显存MiB'],[line.split(', ') for line in live['gpu'].strip().splitlines()]),'',
            f"GPU compute唯一PID数={len({line.split(',')[0] for line in live['compute'].strip().splitlines()})}；包括其他实验、调度器上下文和本次纯推理，不能直接当作训练实验数。",'']
    lines+=['## 科学结论与交付边界','',
        '保持NO_PROMOTION_TO_DEFAULT：单seed、小幅raw差异不足以推广；响应支路跨域稳定性、参考估计质量、TX/RX信息分离、view一致性和可控融合收益仍未形成完整证据。其余4组未完成，尚不能闭合V1/V1R/V2桥接矩阵。', '',
        '复现脚本：code/scripts/analyze_ecrs_snapshot_20260908.py（完整解析、源预测复算、曲线）；code/scripts/eval_ecrs_completed_target_20260908.py（冻结推理与独立评分）；本报告表格由render_ecrs_report_20260908.py生成。目标测试发布代码提交9e801b182af8200c6b29155688a47a20ad7b86b5。', '',
        'Git附带完整epoch/分场景验证/损失/激活CSV、源最终分组统计、checkpoint元数据、曲线和目标测试统计。原始训练日志与源预测完整压缩包保存在本机E:/type10-7/local_artifacts/ecrs_analysis_20260908_0904/complete_logs.zip（42788798字节），未把.pth权重、数据集或大体积逐样本目标logits加入Git。N607原始run/log目录完整保留。', '',
        '训练run：phase1_ecrs_v1r_firstbatch_s392005_e200_20260908_r1；phase1_ecrs_v2_bridge_s392005_e200_20260908_r2。目标测试run：phase1_ecrs_completed_target_test_20260908_r1。', '']
    (ROOT/'docs/experiments/ECRS_FULL_RESULTS_20260908.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    main()
