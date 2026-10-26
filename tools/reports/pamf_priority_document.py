"""Chinese IEEE narrative for priority comparisons with visible missing data and shared targets."""
import json,sys
from pathlib import Path
ROOT=Path('E:/type10-7');sys.path.insert(0,str(ROOT))
from tools.reports.pamf_priority_data import OUT,OLD,P,LABELS,VIEW,SHORT,Data,rows

def main():
    d=Data();lines=[]
    payload=json.loads((P/'weekly_workbook_data.json').read_text(encoding='utf-8'))
    specs={s['slug']:s for s in payload['sheets'] if s.get('priority')};record={r['id']:r for r in payload['records']}
    def p(s=''):lines.extend([s,''])
    def table(title,head,body):
        p('表：'+title);p('\n'.join(['|'+'|'.join(head)+'|','|'+'|'.join('---' for _ in head)+'|']+['|'+'|'.join(map(str,r))+'|' for r in body]))
    def fmt(r):return '缺项' if r is None else f"{r['mean']:.2f}±{r['sd']:.2f}"
    def fig(slug,caption):
        p('!['+caption+'](figures/'+slug+'.png)');p(caption)
        sp=specs[slug];charts=sp['jointCharts']
        p('新增独立双跨测试：仅取目标接收机 2021 年 3 月 1 日的记录，与原图分开统计；训练、模型和增强配方保持。')
        def val(id):return fmt(record[id]) if id else '缺项'
        if slug=='main_no_channel':
            head=['方法','监督 CLEAN','普通 PL CLEAN'];body=[[c]+[val(s['ids'][i]) for s in charts[0]['series']] for i,c in enumerate(charts[0]['categories'])]
        elif slug=='main_channel_on':
            head=['方法','监督 CLEAN','PL CLEAN','监督综合星地','PL 综合星地'];body=[[c]+[val(s['ids'][i]) for ch in charts for s in ch['series']] for i,c in enumerate(charts[0]['categories'])]
        elif slug=='DURA_generality':
            head=['方法','未接 DURA（%）','接入 DURA（%）','配对变化（百分点）'];gains=list(rows(OUT/'tables/DURA_generality_cross_both.csv'));body=[[r['method'],val(charts[0]['series'][0]['ids'][i]),val(charts[0]['series'][1]['ids'][i]),f"{float(r['delta_pp']):+.2f}±{float(r['delta_sd']):.2f}"] for i,r in enumerate(gains)]
        elif slug in ('seven_scenes','source_count_clean','source_count_practical'):
            ch=charts[0];head=['测试场景' if slug=='seven_scenes' else '源接收机数']+LABELS;body=[[c]+[val(s['ids'][i]) for s in ch['series']] for i,c in enumerate(ch['categories'])]
        else:
            head=['测试设置']+LABELS;body=[]
            for ch in charts:
                for ss in ch['series']:body.append([ch['title'].replace('双跨 · ','')+' / '+ss['name']]+[val(id) for id in ss['ids']])
        table('跨日且跨接收机：'+sp['title']+'，准确率均值±样本标准差（%）',head,body)
        from tools.reports.pamf_priority_data import write
        write(OUT/'tables/joint_figure_tables'/(slug+'.csv'),[dict(zip(head,row)) for row in body])
        sample_note='共同 6 目标，每景 36000 包；5 源复用同种子主模型。' if slug.startswith('source_count') else '每景 42000 包。'
        cap='跨日且跨接收机：'+sp['title']+'。未见目标接收机，D0（2021 年 3 月 1 日）；'+sample_note+'误差线为三模型种子样本标准差。'+('仅 CLEAN；PAMF 普通 PL 缺项。' if slug=='main_no_channel' else '')
        p('!['+cap+'](figures/'+sp['jointSlug']+'.png)');p(cap)
    p('# PAMF 与对比方法的重点实验报告')
    p('本版按用户指定的五类问题组织主结果：先配对比较监督与普通 PL 半监督，再报告启用星地增强的半监督泛化、七场景、1 至 6 个源接收机和 DURA 泛用性。PAMF 及 PAMF+DURA 以红、橙色突出；颜色不改变坐标、统计或缺项含义。其余既有图表完整保留在 Excel 的补充工作表。证据整理日期为 2026 年 10 月 26 日。')
    p('## 1  实验定义与展示范围')
    table('本版重点实验的固定条件',['问题','训练条件','主图内容'],[
      ['监督与半监督','无增强、启用增强分别比较','无增强仅 CLEAN；有增强分别给 CLEAN 与综合星地'],
      ['接收机划分、标注比例、源数量','普通 PL 半监督＋星地增强','每个设置内 CLEAN／综合星地双柱；均含 PAMF 与 PAMF+DURA'],
      ['七场景','普通 PL 半监督＋星地增强','CLEAN、高/中/低郊区、高/中/低城市'],
      ['1 至 6 源折线','普通 PL 半监督＋星地增强','固定共同 6 目标；CLEAN 和综合星地分图'],
      ['DURA 泛用性','普通 PL 半监督＋星地增强','五种识别器各自未接入／接入恢复器双柱']])
    p('主比较方法为 RIEI、DRIFT、IB-RFF、SDG、PAMF、PAMF+DURA。前五项直接识别；PAMF+DURA 的实际顺序是先 DURA 恢复、再由同一 PAMF 识别器分类，不计为另一次识别器训练。外部方法接 DURA 的结果单列泛用性，不替换其主比较成绩。所有方法均保留原生完整目标、预算与默认分类头，不声称等计算比较。')
    p('普通 PL 采用当前模型 eval/no_grad 硬标签，阈值 0.95、权重 1、整批无标签损失平均，无 EMA 教师或复杂筛选。PAMF 监督为 50 轮有标签训练，普通 PL 为 13 轮有标签＋37 轮联合训练，均为 11100 次更新。保留 PAMF 自身有标签目标与信道视图，不把监督版称为纯交叉熵。外部方法的三场景概率为 40%/35%/25%，PAMF 主配方为三场景均匀；原生目标和增强剂量差异均保留。')
    p('综合星地指先在每个模型种子内平均六种 practical 的准确率，再计算三个模型种子的均值和样本标准差；CLEAN 不参加该平均。柱图和折线误差线均为模型种子样本标准差，不是置信区间或显著性结论。每个重点实验均保留目标接收机在已见日期的跨接收机原图，并新增目标接收机未见日期的跨日且跨接收机独立图、表。两组的日期、样本与统计分别报告。源接收机跨日诊断、Macro-F1、十类规模与时延等仍保留在 Excel 补充页。')
    p('数据来自 WiSig 地面代理。训练/源验证日期为 2021 年 3 月 8、15、23 日，3 月 1 日为更早的未见日期。仿真星地环境不是实测在轨链路。训练、选模和调度均不使用本次目标成绩反馈。')
    c={r['protocol']:r for r in rows(OLD/'tables/data_contracts.csv')}
    table('主要数据设置；监督组不读取无标签训练池',['设置','类数','源/原目标接收机数','有标签训练','无标签池','源验证'],[[SHORT[p],r['classes'],str(len(r['source_Rx'].split(';')))+'/'+str(len(r['target_Rx'].split(';'))),r['L_s'],r['U_s'],r['V']] for p in ('F0','F1_rx_repartition_r01','F2_rx_repartition_r01','label01','label05','m1-natural','m2-natural','m3-natural','m4-natural','m6-natural') for r in [c[p]]])
    p('## 2  监督与普通 PL 的双柱比较')
    p('无增强主图保留五种识别方法。此前导出没有纳入的 PAMF 无增强监督结果现已从完整冻结包补回，仍保留其原有监督参照配方；当前普通 PL 的匹配无增强结果不存在，按用户要求明确留缺，不填零、不用旧复杂半监督或已退出方法替代。PAMF 无增强监督参照与有增强主配方还存在辅助槽差异，因此两图之间的分差不能全部解释为信道增强的因果效果。')
    table('无增强、标准主划分、目标跨接收机 CLEAN 准确率（%）',['识别方法','监督','普通 PL 半监督'],[[l,fmt(d.get('F0',l,'supervised','clean',False)),fmt(d.get('F0',l,'semi_supervised','clean',False))] for l in LABELS[:5]])
    fig('main_no_channel','无附加星地增强的监督／普通 PL 双柱比较，仅报告目标 CLEAN。PAMF 监督柱为既有无增强参照，普通 PL 缺项以文字标出，未填零值。')
    table('启用星地增强的标准主划分准确率（%）；两类测试各自配对',['方法','监督 CLEAN','PL CLEAN','监督综合星地','PL 综合星地'],[[l,fmt(d.get('F0',l,'supervised','clean')),fmt(d.get('F0',l,'semi_supervised','clean')),fmt(d.get('F0',l,'supervised')),fmt(d.get('F0',l))] for l in LABELS])
    fig('main_channel_on','启用星地增强后，监督和普通 PL 半监督按方法配对；(a) CLEAN，(b) 六种 practical 的平均准确率。两幅均列单 PAMF 与 PAMF+DURA。监督恢复器只用有标签源 IQ，半监督恢复器可用全部允许源训练 IQ。')
    a=d.get('F0','PAMF','supervised');b=d.get('F0','PAMF');r=d.get('F0','PAMF+DURA');best=max(d.get('F0',l)['mean'] for l in LABELS[:4])
    p(f"在原目标接收机已见日期的增强半监督主比较中，PAMF 的综合星地准确率为 {fmt(b)}%，高于四种直接识别外部方法中的最高均值 {best:.2f}%；接入 DURA 后为 {fmt(r)}%。普通 PL 相对 PAMF 监督的直接识别均值变化为 {b['mean']-a['mean']:+.2f} 个百分点，故该结果不支持把 PAMF 的全部优势归因于普通 PL。两种学习组的恢复器源 IQ 权限也不同，不能将完整系统的跨组差全部归因于识别器伪标签。")
    p('## 3  接收机划分、标注比例与源数量')
    p('本节全部使用普通 PL 半监督＋训练星地增强。每个数据设置均列六个主比较项，以 CLEAN／综合星地双柱展示；PAMF 不再只出现在标准划分图。三个接收机划分都为 5 源/7 目标，三套源集合两两仅交一个接收机；每个模型和恢复器均匹配该划分。标注比例为源训练池的 1%/5%/10%，对应 630/3150/6300 个有标签包。')
    fig('receiver_splits','三套接收机划分的增强半监督结果：每种方法的左、右柱分别为 CLEAN 与综合星地。PAMF 和 PAMF+DURA 在全部面板保留。')
    fig('label_fraction','1%、5%、10% 标注比例的增强半监督结果，以 CLEAN／综合星地双柱比较六个主方法项。源验证量固定；保留非单调变化。')
    fig('source_count_paired','1 至 6 个源接收机的增强半监督双柱图，每种方法均给 CLEAN 和综合星地。全部面板使用共同 6 个目标接收机；5 源沿用同种子主实验模型，并由冻结逐接收机计数汇总。')
    p('## 4  主实验的 CLEAN 与六种星地环境')
    table('增强半监督主实验逐景准确率（%）',['场景']+LABELS,[[VIEW[v]]+[fmt(d.get('F0',l,view=v)) for l in LABELS] for v in (*VIEW.keys(),) if v!='six_practical_mean'])
    fig('seven_scenes','标准主划分、普通 PL 半监督＋增强的七场景结果。四种外部方法与单 PAMF 直接识别，PAMF+DURA 为恢复后识别；逐景单列 CLEAN 及六种 practical，不能以综合均值掩盖单景下降。')
    p('## 5  1 至 6 个源接收机的折线')
    p('5 个源接收机直接使用主对比实验的同种子模型，而不是另训模型。主实验原为 7 个目标接收机；其余源数量设置为共同 6 目标。为避免折线混入目标集合变化，本版从 5 源模型已经冻结的逐接收机统计中取共同 6 目标，按正确包数/总包数汇总：每个场景 108000 包。主表原 7 目标、每景 126000 包的成绩保持不变，也在 Excel 中保留。此操作仅重汇总冻结计数，没有重新预测或读取逐样本真值。')
    pids=['m1-natural','m2-natural','m3-natural','m4-natural','F0','m6-natural']
    for v,slug in [('clean','source_count_clean'),('six_practical_mean','source_count_practical')]:
        table('固定共同 6 目标的 '+VIEW[v]+' 准确率（%）',['源接收机数']+LABELS,[[str(i)]+[fmt(d.get(p,l,view=v,common6=True)) for l in LABELS] for i,p in enumerate(pids,1)])
        fig(slug,'完整 1、2、3、4、5、6 源接收机折线：'+VIEW[v]+'。全部为增强半监督，固定共同 6 目标；5 源为同种子主实验模型的冻结计数重汇总。误差线为三种子样本标准差，保留非单调变化。')
    p('源训练量随源接收机数量自然增长，因此折线同时反映接收机覆盖与训练记录量变化，不是固定样本预算下仅接收机多样性的因果效应。各设置仍使用原冻结测试的场景配置。')
    p('## 6  包含 PAMF 的 DURA 泛用性')
    gains=list(rows(OUT/'tables/DURA_generality_priority.csv'))
    table('标准主划分、增强半监督、综合星地的恢复前后配对',['识别方法','未接 DURA（%）','接入 DURA（%）','配对变化（百分点）'],[[r['method'],f"{float(r['before_mean']):.2f}",f"{float(r['after_mean']):.2f}",f"{float(r['delta_pp']):+.2f}±{float(r['delta_sd']):.2f}"] for r in gains])
    fig('DURA_generality','DURA 泛用性采用双柱，五种识别器均包括在内：RIEI、DRIFT、IB-RFF、SDG、PAMF。每种方法固定同一识别器、样本、场景和种子，只切换是否先恢复；柱顶标种子内配对百分点变化。全部为增强半监督、标准主划分的综合星地测试。')
    p('所有恢复后结果都使用匹配同一数据契约的恢复器。三个识别器种子共享一份匹配恢复器，不能称为三套独立完整系统。泛用性收益是识别器接入固定恢复网络的测试收益，不证明恢复器对每个方法、每个场景都提高。')
    p('## 7  保留资料与明确缺项')
    p('PDF 保留原 9 张重点图、19 个面板，并新增 9 张跨日且跨接收机独立图及对应结果表，新增图同样为 19 个面板。Excel 前 9 页对应重点实验，后 26 页保留此前补充资料；共 35 个工作表、98 个可编辑图表。31 个适用的识别性能实验页均有新增双跨图/表；模型覆盖、实验登记和两种学习条件的独占时延共 4 页不按目标日期划分，没有虚构对应指标。')
    p('当前普通 PL 配方的无增强 PAMF 仍是缺项，已获用户确认先交付已有证据。没有启动新增训练、预测、真值评分或 DURA 训练。当前方法自身消融的其他缺项仍按原报告记录，不把旧复杂半监督或已退出方法改名为普通 PL。旧报告与不可覆盖导出包均保留。')
    p('## 附录  冻结来源与复算')
    p('当前主方法来自既有 practical residual 完整对比包及 PAMF 监督/ordinary_PL 完成矩阵。新增双跨结果直接读取其中已完成的独立评分统计。无增强监督 PAMF 来自完整对比包的既有监督参照；其当前普通 PL 对照未完成。5 源曲线使用同种子主模型，从冻结接收机统计取预先指定的共同目标集合 [0,2,5,7,9,10]：原跨接收机每景 108000 包；新增双跨每景 36000 包。原 7 目标主实验的跨接收机 126000 包和双跨 42000 包均保留。没有根据目标成绩筛接收机或挑模型。')
    p('导出包包含逐图来源表、冻结种子统计、共同 6 目标正确/总包计数、IEEE 源码、PDF/SVG/PNG 和纯标准库复核器。先按种子汇总六景和配对差，再计算跨种子样本标准差；缺失结果保持缺失。正文采用识别器、恢复器和数据设置的可读名称，原始编号只在机器可读追溯数据中保留。')
    (OUT/'PAMF与四种对比方法实验报告.md').write_text('\n'.join(lines),encoding='utf-8')
    (OUT/'README.md').write_text('''# PAMF 重点对比实验（2026-10-26）

主报告按用户五类实验优先级组织。无增强只展示 CLEAN；其余重点图均为普通 PL 半监督＋增强，监督/半监督主对照除外。

源数量 1 至 6 完整，5 源复用同种子主实验模型，汇总共同 6 目标；原 7 目标主结果仍保留。无增强 PAMF 监督从既有完整包补入，当前普通 PL 缺项明确标出。没有新训练或新预测。

PDF 18 张重点图：原跨接收机 9 张与新增跨日且跨接收机 9 张，各有 19 个面板。Excel 35 个工作表、98 个可编辑图表：前 9 页主实验，后 26 页补充。31 个适用实验页新增独立双跨图/表；4 个模型数量、登记和时延页不适用。旧 26 份独立图保留在 figures/supplement_r20/。

双跨仅使用目标接收机的 2021 年 3 月 1 日；这是更早的未见日期，不声称未来日期外推。无增强仅展示 CLEAN，普通 PL 的无增强 PAMF 缺项保持。5 源复用同种子主模型，共同 6 目标的双跨每景 36000 包。没有新训练、预测或逐样本真值评分。

纯标准库复核：`python -X utf8 -S reproduce/verify_priority.py --root .`。该命令从冻结计数和逐种子输入复算，不依赖 Excel、绘图库、原始 IQ、模型权重或逐样本真值。重新排版 PDF/绘图另需 LaTeX、字体和 Matplotlib；工作簿构建另需原生成环境的 Artifact Tool。
''',encoding='utf-8')
    from tools.reports import pamf_pl_ieee
    pamf_pl_ieee.OUT=OUT;pamf_pl_ieee.main()
    # Replace the old abstract in the generated new document only.
    q=OUT/'ieee/main.tex';s=q.read_text(encoding='utf-8');a=s.index('\\begin{abstract}')+len('\\begin{abstract}');b=s.index('\\end{abstract}',a)
    s=s[:a]+'''\n本报告按五类重点问题整理PAMF与四种完整对比方法的冻结结果。每个重点实验保留原跨接收机图，并新增跨日且跨接收机独立图和结果表。后者仅使用目标接收机在2021年3月1日的记录，与训练/验证日期分离，也不与原测试人群混合。监督与普通PL采用双柱，无增强仅报告CLEAN，其余重点图固定增强半监督。所有适用图纳入PAMF及PAMF+DURA。1至6源数量完整，5源复用同种子主模型并汇总共同6个目标。普通PL的无增强PAMF缺项保留，原图表及补充资料不删除。本次无新增训练、预测或目标逐样本真值访问。\n'''+s[b:]
    s=s.replace('PAMF与对比方法的分组实验分析','PAMF与对比方法的重点实验分析').replace('2026年10月25日','2026年10月26日')
    q.write_text(s,encoding='utf-8')
    print('Priority narrative and IEEE source written',flush=True)
if __name__=='__main__':main()
