"""Create the Chinese results report from the reviewed snapshot and conclusions.

Run with Python containing reportlab and Pillow. Scientific figures are embedded
from the Python-generated previews; the standalone PDF/SVG retain vector marks.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from xml.sax.saxutils import escape
import json
import statistics as stats
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                               Flowable, PageBreak)
from PIL import Image as PILImage
from pypdf import PdfReader, PdfWriter, Transformation
from collect_completed_pamf import PROTOCOLS, SIX, VIEWS, read, write
from analyze_completed_pamf import one, rows


def pm(r,digits=2):
    return f'{float(r["mean"]):.{digits}f} ± {float(r["sd"]):.{digits}f}'


def build(root):
    data=read(root/'source_data/snapshot.json');t=data['tables'];f=read(root/'analysis/conclusions_data.json')
    common=dict(test='main',dimension='overall',stratum='ALL',metric='accuracy')
    blocks=[]
    def heading(text):blocks.append(dict(type='heading',text=text))
    def p(text):blocks.append(dict(type='p',text=text.replace('\u2212','-')))
    def table(headers,records,widths=None):blocks.append(dict(type='table',headers=headers,rows=records,widths=widths))
    def image(number,slug,width=165):blocks.append(dict(type='image',file=f'preview/Fig{number:02d}_{slug}.png',width=width))
    def page():blocks.append(dict(type='page'))
    def main(p='F0',r='native_cric',ro='DURA-LU',v=SIX):
        return one(t,'main',protocol=p,regime=r,readout=ro,view=v,**common)
    def gf(cell,v=SIX):return one(t,'gfactor',protocol_id='F0',cell=cell,test='main',view=v,metric='accuracy')
    blocks.append(dict(type='title',text='PAMF Phase1 实验结果与结论'))
    p('数据截至 2026 年 10 月 23 日。当前方法固定为 T1-E50（13L＋37U）识别器与 DURA v1.0 恢复器；本报告汇总近期已完成的有效实验，不改变默认方法。')
    heading('主要结论')
    p('最充分的正面证据来自 DURA 的信道鲁棒性增益；PAMF 完整系统也保持了对外部方法的优势。训练机制的结论应更克制：成对一致性 B 有较一致的条件收益，CRIC A、原生选择性半监督 C 以及它们与恢复器的协同，尚未显示稳定的普遍正贡献。')
    table(['F0 指标','结果'],[
        ['PAMF clean 准确率',pm(main(v='clean'))+'%'],
        ['PAMF 六景平均准确率',pm(main())+'%'],
        ['PAMF clean Macro-F1',pm(one(t,'main',protocol='F0',regime='native_cric',readout='DURA-LU',
            test='main',dimension='overall',stratum='ALL',metric='macro_f1',view='clean'))+'%'],
        ['PAMF 六景平均 Macro-F1',pm(one(t,'main',protocol='F0',regime='native_cric',readout='DURA-LU',
            test='main',dimension='overall',stratum='ALL',metric='macro_f1',view=SIX))+'%'],
        ['DURA 相对 raw 的六景增益',pm(f['final_F0_gain'])+' pp'],
        ['PAMF low urban 准确率',pm(main(v='practical_low_urban'))+'%']], [87,78])
    p('DURA-LU 在 17/17 个数据契约上提高了原生 T1 的六景均值，增益为 0.99 至 8.67 个百分点；但 5/17 个契约的 clean 均值变化非正，不能写成“所有条件无损”。')
    p(f'给外部方法同样接入 DURA 后，F0 最强的 SDG Full＋PL 六景均值为 {pm(f["external_F0_best_full"])}%，PAMF 为 {pm(main())}%，均值高 {f["native_vs_best_external_F0_GLU_pp"]:.2f} 个百分点。这是完整注册配方的比较，不是某一个训练组件的贡献。')
    p('low urban 的 70% 目标仍差 2.34 个百分点；其余五种信道中，mid urban 距 raw clean 约 10.00 个百分点，仍未满足 5 个百分点目标。实验完成不等于所有性能目标已经达成。')
    page();heading('1. 实验覆盖与统计口径')
    table(['面板','已完成内容','如何计数'],[
        ['主比较','17 契约 × 3 配方 × 3 F seed','153 个逻辑 F；含复用'],
        ['核心机制','A/B/C 八组合 × 3 F seed','24 个逻辑 F；12 个新 F'],
        ['细 F 诊断','21 设置 × 3 F seed','63 个修复后 F；另复用 T1'],
        ['严格监督结构','9 结构 × 3 F seed','24 个新 F＋3 个复用基线'],
        ['全 L 配方','F15/F20 × 3 F seed','6 条复用 F 的读出'],
        ['G 训练与消融','35 条新谱系；7 设置 × 3 学生','原始固定 G 与学生重复分列'],
        ['F/G 交叉','m3-natural：3 F × 3 G','9 格；对角为 3 个完整系统'],
        ['外部接入 DURA','516 个既有 F 的冻结读出','无新增 F/G 训练'],
        ['普通 PL × G 权限','17 契约 × 3 F × 4 格','204 格；51 条新增读出'],
        ['专用时延','统一 500＋细项 900＋结构 450','1,850 个 case-round，不是模型数']], [33,75,57])
    p('最终完整归档确认有效阶段全部结束：运行与排队均为 0。217 个排除历史阶段、旧输入绑定错误和已被 r02 替代的旧时延失败均不进入有效结论。不同面板会复用同一权重，逻辑模型数和成本不能直接跨面板相加。')
    p('除注明的配对 G 学生与 m3 对角完整系统外，均为三个 F seed 条件于同一固定 G 的结果。六景均值先在每个 seed 内平均，再计算三个 seed 的样本标准差。文中“±”和图中误差线均为样本标准差，不是置信区间或显著性检验。')
    p('F0 每景为 168,000 条物理观测、七个目标接收机；源数量曲线为固定六个目标接收机、每景 144,000 条观测。曲线仅使用 m＝1、2、3、4、6，不插入 F0 的五源结果。ManyRx 使用其原登记 10 类，其余契约为 6 类，不把异质契约混成一个总体平均。')
    p('G-L 与 G-LU 指恢复器及其祖先使用 L 或 L/U 源 IQ；G 不读取 TX 标签、训练好的 F 或目标数据。只有监督 F＋G-L 是完整 L-only 系统；普通 PL F＋G-L 仍通过 F 使用 U。所有测试预测先冻结，再独立 truth-last 评分。')
    page();heading('2. 当前最终 PAMF 的七景表现')
    scene_cn=['Clean','郊区 high','郊区 mid','郊区 low','Urban high','Urban mid','Urban low']
    table(['场景','raw T1 (%)','T1＋G-L (%)','PAMF：T1＋G-LU (%)'],[
        [label,pm(main(ro='raw',v=v)),pm(one(t,'core',bits='111',readout='DURA-L',view=v,**common)),pm(main(v=v))]
        for label,v in zip(scene_cn,VIEWS)], [34,43,43,45])
    p('六景平均由 '+pm(main(ro='raw'))+'% 提升至 '+pm(main())+'%，配对增益为 '+pm(f['final_F0_gain'])+' 个百分点。clean 的配对变化为 +0.0895 ± 0.0531 个百分点，保持较好；low urban 仍是最主要的性能缺口。')
    image(1,'final_channels')
    p('目标检查以同一原生 T1 的 raw clean 均值 84.3944% 为参照：郊区 high/mid/low 与 urban high 的差距均在 5 个百分点以内；urban mid 的差距为 9.9972 个百分点。不能使用旧 F50 或其他契约的分数替换当前最终 T1。')
    page();heading('3. 十七个数据契约：增益稳定，但原生配方并非处处最好')
    table(['契约','raw 六景 (%)','PAMF 六景 (%)','恢复增益 (pp)'],[
        [protocol,pm(main(p=protocol,ro='raw')),pm(main(p=protocol)),pm(one(t,'main_effects',protocol=protocol,
            effect='native_cric_G_minus_raw',readout='DURA-LU',view=SIX,**common))] for protocol in PROTOCOLS], [31,44,45,45])
    p('所有 17 个契约的六景均值增益为正；clean 的变化并非全部为正，m1-fixed 为 −0.6044 ± 0.2159 个百分点。原生 T1＋G-LU 相对监督控制在 3/17 个契约非正，相对普通 PL 在 5/17 个契约非正，不能概括为原生半监督配方全面占优。')
    p('自然预算与固定预算源数量曲线分开作图，保留非单调变化。m3-natural 的完整 3×3 F/G 交叉中，三组联合 F/G 种子对角结果为 clean 82.1000 ± 1.2192%、六景 72.8219 ± 1.0367%；九格不是九个独立完整系统，也不代表其他 16 契约的 G 方差。')
    p('对应图 2 至图 6 分别展示六景恢复增益、clean 变化和两组配方差；全部图表保留在单独的矢量合集。')
    page();heading('4. 外部方法接入同一恢复器后，PAMF 仍有完整系统优势')
    names={'riei':'RIEI','drift':'DRIFT','ibrff':'IB-RFF','sdg':'SDG'}
    ext_rows=[]
    for fam in ('riei','drift','ibrff','sdg'):
        for reg,label in [('supervised','Full'),('pl','Full＋PL')]:
            ext_rows.append([names[fam]+' '+label]+[pm(one(t,'external_full',protocol='F0',family=fam,
                regime=reg,variant='full',readout=ro,test='main',view=SIX,metric='accuracy')) for ro in ('raw','DURA-L','DURA-LU')])
    ext_rows.append(['PAMF 原生 T1',pm(main(ro='raw')),pm(one(t,'core',bits='111',readout='DURA-L',view=SIX,**common)),pm(main())])
    table(['F0 方法','raw 六景 (%)','G-L 六景 (%)','G-LU 六景 (%)'],ext_rows,[45,40,40,40])
    p('DURA-LU 对全部 86 个已登记 Full/Full＋PL 主任务条件的六景均值均有正增益；G-L 为 84/86。G-LU 比 G-L 更好的条件为 55/86，因此“使用更多无标签源 IQ”也不是普遍正收益。未登记的外部 PL 条件显示为 N/A，不填成 0，也不补造实验。')
    image(26,'external_F0_DURA-LU')
    p('公平的全 L 比较另列：F0 监督 PAMF＋G-L 为 '+pm(gf('Sup+G-L'))+'%，外部 Full＋G-L 中最高为 SDG 的 '+pm(one(t,'external_full',protocol='F0',family='sdg',regime='supervised',variant='full',readout='DURA-L',test='main',view=SIX,metric='accuracy'))+'%。原生 T1、普通 PL 控制和 F20 监督正则配方不能互相改标。')
    page();heading('5. 核心机制：支持 B 的条件收益，不支持稳定正协同')
    p('A 为同 TX 跨 RX 的 CRIC；B 为 ordinary/raw 成对随机预测的对称 KL；C 为 own EMA、原生门控、伪标签和无标签熵。八组合保留相同 CE 剂量和部署骨架，000 不是 F15 的独立 CE 配方。')
    table(['机制：完整 111 减移除项','raw clean (pp)','raw 六景 (pp)','G-LU 六景 (pp)'],[
        [label,pm(one(t,'core_effects',effect=e,readout='raw',view='clean',**common)),
         pm(one(t,'core_effects',effect=e,readout='raw',view=SIX,**common)),
         pm(one(t,'core_effects',effect=e,readout='DURA-LU',view=SIX,**common))]
        for label,e in [('A：CRIC','A_at_other_groups_on'),('B：成对一致性','B_at_other_groups_on'),('C：选择性半监督','C_at_other_groups_on')]], [48,39,39,39])
    p('B 的 raw clean 差在 3/3 seed 为正，接入 G-L/G-LU 后的六景差也在 3/3 为正；raw 六景差本身很小。A 与 C 在完整组合中未显示稳定正贡献。A 的边际 clean 效应与条件效应方向不同，说明机制依赖上下文，不能用某个正的边际均值代替完整组合的负差。')
    image(13,'core_contributions')
    p('匹配最终 T1 与修复 F16 的 CRIC×G 四格：G 在 T1 和 F16 中的六景增益分别为 +7.0455 ± 0.5949 和 +7.0675 ± 0.7913 个百分点；交互为 −0.0220 ± 1.2449 个百分点。当前证据不支持稳定正协同，标准差也不构成显著性检验。')
    page();heading('6. 二十一项细 F 诊断：完整保留正负结果')
    fine_cn=['浅层 CRIC','中层 CRIC','末层 CRIC','关闭成对 KL','单次随机前向','单个 signature 视图','去时间坐标',
        '去 signature 分支','去 PA 分支','去参考分支','student 作为 PL 源','去相邻批门控','去强视图门控','去 U 熵项',
        'CE＋默认增强配方','关闭 CRIC','As 视图换 raw','恢复 As KL','减少 raw CE 剂量','监督 T1 配方','普通 PL 代替原生 SSL']
    table(['设置','改变','raw clean (%)','raw 六景 (%)','G-LU 六景 (%)'],[
        [f'F{i:02d}',label,pm(one(t,'fine',arm=f'F{i:02d}',readout='raw',view='clean',**common)),
         pm(one(t,'fine',arm=f'F{i:02d}',readout='raw',view=SIX,**common)),
         pm(one(t,'fine',arm=f'F{i:02d}',readout='DURA-LU',view=SIX,**common))]
        for i,label in enumerate(fine_cn,1)], [12,54,33,33,33])
    p('F17 把 As 视图替换为 raw 后，raw 六景降至 41.5403 ± 4.1600%，但加 G 后恢复到 77.9813 ± 0.4494%。因此恢复器会遮蔽部分 raw 敏感性，不能只根据恢复后的接近分数判断训练结构没有作用。')
    p('F15 与最终 T1 的 raw 六景差约 24.25 个百分点，不能全部归于半监督：两者增强、损失和前向剂量不同。F05 改变随机前向与计算，F19 同时降低 CE 剂量和计算。F21 的普通 PL 接受率为 94.8482 ± 0.1073%；原生门控完整接受率在此次紧凑元数据中缺失，保持留空。')
    page();heading('7. 匹配监督结构：分支有作用，精细物理设计的必要性尚不充分')
    arch_names={'X3_CE':'完整 X3-CE','SIG_TIME':'单 signature 视图','NO_TIME':'去时间坐标','SHUFFLE_TIME':'打乱时间坐标',
        'NO_SIGNATURE':'去 signature','NO_PA':'去 PA','NO_REFERENCE':'去参考分支','WRONG_REFERENCE':'扰动参考','COV_X3':'协方差替代'}
    arch_order=list(arch_names)
    table(['严格 L-only 结构','raw clean (%)','raw 六景 (%)','G-L 六景 (%)'],[
        [arch_names[a]]+[pm(one(t,'physics',architecture=a,readout=ro,test='main',view=v,metric='accuracy'))
            for ro,v in [('raw','clean'),('raw',SIX),('DURA-L',SIX)]] for a in arch_order],[45,40,40,40])
    p('本面板是匹配 CE＋sat.68、LS0、无 U/EMA/CRIC/RDrop 的 50 轮结构消融。去 PA、signature 和参考分支分别使恢复后六景均值低约 2.79、2.17 和 1.48 个百分点；去 PA 对 raw 鲁棒性的影响更大。')
    p('单视图、去时间坐标、扰动参考与完整结构相近，打乱时间坐标在该面板的均值还更高；协方差替代也没有稳定劣势。因此现有证据支持保留有用的多分支表征，却不足以证明每一项精细物理设计都不可替代。此处不据目标结果重新选择默认版本。')
    image(19,'physics_GL_channels')
    p('源侧诊断只使用同一 720 条 L、90 个 TX/RX/day 层和 11 项固定干预，不拟合额外分类头。CFO 下的翻转和特征几何只说明敏感性与归纳偏置；模拟 PA 不保证身份保持，不能据此宣称完成真实硬件解耦或完全接收机不变。')
    page();heading('8. 无标签数据公平性：区分 F、G 与完整系统')
    table(['F0 普通 PL × G 四格','clean (%)','六景 (%)'],[[cell,pm(gf(cell,'clean')),pm(gf(cell))]
        for cell in ('Sup+G-L','PL+G-L','Sup+G-LU','PL+G-LU')],[65,50,50])
    p('F0 中，普通 PL 相对监督的六景差在 G-L、G-LU 下分别为 −0.1336 ± 0.5513 和 −0.1646 ± 0.1338 个百分点。给 G 增加 U 在这两种 F 配方下分别带来 +0.9401 ± 0.3669 和 +0.9091 ± 0.2949 个百分点；交互为 −0.0310 ± 0.4201。')
    p('跨 17 个契约，普通 PL 的均值提升在两种 G 条件下都是 13/17 为正；G 增加 U 的均值提升在监督 F 下为 13/17、PL F 下为 12/17。交互仅 5/17 为正，不支持“F 与 G 使用 U 必然正协同”。')
    image(30,'fairness_effects',155)
    p('另外，严格全 L 的 F15 CE＋G-L 六景为 74.4888 ± 1.1408%，F20 监督 T1＋G-L 为 76.8537 ± 0.5918%；两者前向剂量为每更新 384/768 个 L-packet，配方差值不能当作一个正则项的因果贡献。它们不是上表的匹配监督控制。')
    page();heading('9. 恢复器机制：完整恢复优于极简版本，初始化与蒸馏有条件收益')
    g_names={'raw':'Raw T1','default':'默认 DURA','coarse_only':'仅 coarse','prior_no_CG':'去 CG','no_image_action':'去 image action',
        'no_innovation_input':'去 residual input','copy_no_KD':'复制、不蒸馏','random_with_KD':'随机初始化＋蒸馏','random_no_KD':'随机、不蒸馏'}
    g_table=[]
    for arm,label in g_names.items():
        r=one(t,'G_summary',arm=arm,view='six_view_mean');c=one(t,'G_summary',arm=arm,view='clean')
        g_table.append([label,str(r['G_parameters']),f'{float(c["accuracy_mean"]):.2f} ± {float(c["accuracy_sd"]):.2f}',
            f'{float(r["accuracy_mean"]):.2f} ± {float(r["accuracy_sd"]):.2f}'])
    table(['G 设置','G 参数','clean (%)','六景 (%)'],g_table,[53,27,42,43])
    p('仅 coarse 比默认 DURA 低 5.54 个百分点；去 CG、去 residual input 分别低约 0.71、0.87 个百分点。完整物理求解与 prior 具有实际收益，不能只凭更小参数量判断压缩成功。')
    copy=one(t,'G_init_KD',contrast='copy_effect_without_KD',view='six_view_mean')
    kd=one(t,'G_init_KD',contrast='KD_effect_from_random',view='six_view_mean')
    p(f'在匹配的三对 F/G seed 下，无 KD 时复制初始化的六景效应为 +{float(copy["mean_pp"]):.2f} ± {float(copy["sample_sd_pp"]):.2f} 个百分点；随机初始化下加入 KD 为 +{float(kd["mean_pp"]):.2f} ± {float(kd["sample_sd_pp"]):.2f}。默认 copy＋KD 只有一个既有 G，不能补造完整三种子初始化×蒸馏交互。')
    image(23,'G_components',155)
    page();heading('10. 部署效率：参数小，但恢复带来计算成本')
    u={(r['case'],int(r['batch'])):r for r in t['unified_timing']}
    full=u['T4_graph_reuse_GF',1];b128=u['T4_graph_reuse_GF',128]
    table(['当前 PAMF 指标','数值'],[
        ['F / G / G＋F 参数','232,993 / 266,951 / 499,944'],
        ['统一 r02：B1 GPU-wall 中位数 / P95',f'{float(full["median_ms"]):.6f} / {float(full["P95_ms"]):.6f} ms'],
        ['统一 r02：含传输中位数',f'{float(full["H2D_D2H_median_ms"]):.6f} ms'],
        ['统一 r02：B128 吞吐量',f'{float(b128["throughput_IQ_s"])/1000:.3f} kIQ/s'],
        ['细机制队列：优化后 G＋F B1',f'{float(f["timing"]["fine_timing"]["median_ms"]):.6f} ms'],
        ['严格结构队列：X3-CE＋G-L B1',f'{float(f["timing"]["structure_timing"]["median_ms"]):.6f} ms']], [98,67])
    p('三组计时按独立时间窗口分别报告，不相互拼接排名。每组均为独占 RTX3090、FP32/TF32off、五轮计时；五轮不是五个模型种子。原生 G＋F 到 Graph＋reuse 的同权重加速约 8.1 倍，不能由更小参数量直接推导。')
    image(34,'execution_latency')
    p('统一面板中的外部 Full 是单独 F，PAMF 是完整 G＋F。PAMF 的 4.54 ms 高于这些单 F，B128 吞吐量也更低；该面板没有外部方法＋DURA 的同边界完整系统计时，不能声称在等恢复条件下速度领先。')
    p('A/B/C 属于训练机制，冻结部署没有额外 EMA、RDrop、CRIC 或 U 门控算子；“新增部署算子为 0”不等于训练成本为 0。显存记录含多模型和 Graph 池，不能改成单模型显存；参数字节数也不等于序列化文件体积。')
    page();heading('11. 成本、证据边界与论文结论')
    costs=f['costs']
    table(['训练范围','共享 worker 小时','成本含义'],[
        ['150 个新主 F',f'{costs["main_new_F_shared_worker_hours"]:.3f}','含 6 个技术替代控制'],
        ['153 个逻辑主 F',f'{costs["main_logical_F_shared_worker_hours"]:.3f}','含复用原 T1；与上一行重叠'],
        ['35 条新 G 谱系',f'{costs["G_35_lineage_shared_worker_hours"]:.3f}','280 个 E20 阶段；含交叉额外 G'],
        ['63 个修复后细 F',f'{costs["fine_63_new_F_shared_worker_hours"]:.3f}','其他面板复用其中权重'],
        ['12 个新核心 F',f'{costs["core_12_new_F_shared_worker_hours"]:.3f}','24 个逻辑格中的新增部分']], [55,39,71])
    p('这些是共享负载下记录的 worker elapsed，不是独占 GPU 小时或此次交付新增墙钟。复用来源、G 学生、原固定 G 历史、共享 IQ 缓存、预测与评分分列；不能把重叠逻辑成本相加成一个虚假的总成本。')
    heading('建议采用的论文结论')
    p('在 WiSig/ManySig/ManyRx 地面代理数据及预登记模拟信道压力下，PAMF 的主要可复现收益来自恢复器对传播扰动的补偿，以及识别器中部分物理感知分支和成对一致性带来的条件改善。DURA 对原生 T1 的六景平均准确率在全部 17 个数据契约上提高，对已有外部 Full/Full＋PL 条件也显示了广泛的恢复收益。给外部方法同样接入恢复器后，PAMF 在 F0 仍保持较高识别率，但付出额外恢复推理成本。')
    p('同时，原生选择性半监督、CRIC、时间坐标及部分精细结构的优势并不普遍，不能把完整配方的性能差解释为各个组件都有效，也未建立稳定的正协同或完整硬件解耦。low urban 与 mid urban 的目标缺口仍然存在。研究结果应保留负差、非单调性、不同数据权限与训练剂量，以及固定 G 下的种子方差边界；不将地面代理实验称为真实在轨验证。')
    heading('结果与复现文件')
    p('all_figures.pdf 为 40 页矢量图合集；png/ 与 tiff/ 为 4800 dpi 单图；svg/ 与 pdf/ 保留矢量元素和文字。captions_en.md 保存图注与统计定义，图内没有说明性副标题或图下说明。source_data/ 保存冻结汇总与逐种子新聚合，analysis/ 保存配对效应和结论数据；reproduce/ 保存仅做读取、统计与制图的 Python 源码。')
    table(['来源','本包关键文件'],[
        ['主比较、源数量、交叉','main.csv；main_effects.csv；crossed.csv'],
        ['核心机制、源几何','core.csv；core_effects.csv；core_geometry.csv'],
        ['修复后细 F、匹配交互','fine.csv；fine_effects.csv；cric_G.csv'],
        ['严格监督结构','physics_seed.csv；physics.csv；physics_interventions.csv'],
        ['全 L 与 G 消融','Lonly.csv；G_summary.csv；G_init_KD.csv'],
        ['外部 DURA 与公平性','external_full_seed.csv；external_full.csv；gfactor.csv'],
        ['效率','unified_timing.csv；fine_timing.csv；structure_timing.csv']], [52,113])
    return blocks


def render(root,blocks):
    fonts=Path('C:/Windows/Fonts')
    pdfmetrics.registerFont(TTFont('CN',str(fonts/'simsun.ttc'),subfontIndex=0))
    pdfmetrics.registerFont(TTFont('TNR',str(fonts/'times.ttf')))
    pdfmetrics.registerFont(TTFont('TNR-Bold',str(fonts/'timesbd.ttf')))
    pdfmetrics.registerFontFamily('CN',normal='CN',bold='CN',italic='CN',boldItalic='CN')
    pdfmetrics.registerFontFamily('TNR',normal='TNR',bold='TNR-Bold',italic='TNR',boldItalic='TNR-Bold')
    ink=colors.HexColor('#1D2939');muted=colors.HexColor('#5E6B7C');rule=colors.HexColor('#D9E0E7')
    body=ParagraphStyle('body',fontName='CN',fontSize=10.2,leading=16,spaceAfter=7,textColor=ink,wordWrap='CJK')
    heading=ParagraphStyle('heading',parent=body,fontSize=13.4,leading=19,spaceBefore=3,spaceAfter=10,keepWithNext=True)
    title=ParagraphStyle('title',parent=heading,fontSize=21,leading=29,spaceAfter=17)
    cell=ParagraphStyle('cell',parent=body,fontSize=8.7,leading=12.5,spaceAfter=0)
    numeric=ParagraphStyle('number',parent=cell,fontName='TNR',alignment=TA_CENTER)
    head=ParagraphStyle('head',parent=cell,fontSize=9.1,leading=13,textColor=colors.white)
    story=[];md=[];overlays=[]
    class VectorFigure(Flowable):
        def __init__(self,path,width,height):
            super().__init__();self.path=path;self.width=width;self.height=height
        def draw(self):
            x,y=self.canv.absolutePosition(0,0)
            overlays.append((self.canv.getPageNumber()-1,self.path,x,y,self.width,self.height))
    for b in blocks:
        kind=b['type']
        if kind in ('p','heading','title'):
            style=body if kind=='p' else heading if kind=='heading' else title
            story.append(Paragraph(escape(b['text']),style));md.append(('# ' if kind=='title' else '## ' if kind=='heading' else '')+b['text']+'\n')
        elif kind=='page':story.append(PageBreak())
        elif kind=='table':
            contents=[[Paragraph(escape(str(v)),head) for v in b['headers']]]
            for row in b['rows']:
                contents.append([Paragraph(escape(str(v)),numeric if i>0 and any(ch.isdigit() for ch in str(v))
                    and not any('\u4e00'<=ch<='\u9fff' for ch in str(v)) else cell) for i,v in enumerate(row)])
            widths=[x*mm for x in b['widths']] if b.get('widths') else None
            tab=Table(contents,colWidths=widths,repeatRows=1,hAlign='LEFT')
            tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),ink),('VALIGN',(0,0),(-1,-1),'TOP'),
                ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
                ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
                ('LINEBELOW',(0,0),(-1,0),.5,ink),('LINEBELOW',(0,1),(-1,-1),.3,rule),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F5F7FA')])]))
            story.extend([tab,Spacer(1,9)])
            md.extend(['| '+' | '.join(map(str,b['headers']))+' |','| '+' | '.join(['---']*len(b['headers']))+' |'])
            md += ['| '+' | '.join(map(str,r))+' |' for r in b['rows']];md.append('')
        elif kind=='image':
            path=root/b['file']
            with PILImage.open(path) as im:width,height=im.size
            w=b['width']*mm;h=w*height/width
            figure=VectorFigure(root/'pdf'/(path.stem+'.pdf'),w,h);figure.hAlign='CENTER'
            story.extend([figure,Spacer(1,6)])
            md.append(f'![科研图表]({path.as_posix()})\n')
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(rule);canvas.setLineWidth(.4)
        canvas.line(20*mm,16*mm,190*mm,16*mm)
        canvas.setFillColor(muted);canvas.setFont('TNR',8)
        canvas.drawString(20*mm,11.5*mm,'PAMF Phase1 | 2026-10-23')
        canvas.drawRightString(190*mm,11.5*mm,str(doc.page));canvas.restoreState()
    report=root/'PAMF_结果总结_20261023.pdf'
    layout=root/'analysis/report_layout.pdf'
    doc=SimpleDocTemplate(str(layout),pagesize=A4,rightMargin=20*mm,leftMargin=20*mm,
        topMargin=17*mm,bottomMargin=21*mm,title='PAMF Phase1 实验结果与结论',author='PAMF research results')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    # Merge the original scientific PDF pages into the reserved boxes. Text and
    # marks remain vectors rather than downsampling the report to preview DPI.
    with layout.open('rb') as stream:
        reader=PdfReader(stream);writer=PdfWriter()
        for page in reader.pages:writer.add_page(page)
        for page,path,x,y,w,h in overlays:
            vector=PdfReader(path).pages[0]
            transform=Transformation().scale(w/float(vector.mediabox.width),h/float(vector.mediabox.height)).translate(x,y)
            writer.pages[page].merge_transformed_page(vector,transform,over=False)
        writer.add_metadata({'/Title':'PAMF Phase1 实验结果与结论','/Author':'PAMF research results'})
        with report.open('wb') as out:writer.write(out)
    layout.unlink()
    (root/'PAMF_结果总结_20261023.md').write_text('\n'.join(md),encoding='utf-8')
    write(root/'analysis/report_content.json',blocks)
    print(json.dumps(dict(status='REPORT_CREATED',path=str(report)),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();render(a.root,build(a.root))
