"""A paper narrative organized first by learning regime, with paired restoration columns."""
import json,sys
from pathlib import Path
ROOT=Path('E:/type10-7');sys.path.insert(0,str(ROOT))
from tools.reports.pamf_reporting_model import *
def main():
    r=Results();md=[];figures=json.loads((OUT/'figures/manifest.json').read_text(encoding='utf-8'));fm={f['name']:f for f in figures}
    def p(s):md.extend([s,''])
    def h(s):md.extend(['## '+s,''])
    def table(caption,headers,data):
        p('表：'+caption);md.append('| '+' | '.join(map(str,headers))+' |');md.append('|'+'|'.join('---' for _ in headers)+'|');md.extend('| '+' | '.join(map(str,row))+' |' for row in data);md.append('')
    def fig(slug,caption=None):
        caption=caption or fm[slug]['caption'];md.extend(['!['+caption+'](figures/'+slug+'.png)','',caption,''])
    def fmt(x,n=2):return f"{float(x['mean']):.{n}f}±{float(x['sd']):.{n}f}"
    def get(p,m,cell,t='cross_rx',rd='raw',v='six_practical_mean',metric='accuracy'):return r.get(p,m,cell,t,rd,v,metric)
    p('# PAMF与对比方法：按监督条件分组的实验报告')
    p('本报告首先区分监督学习与普通PL半监督，再区分训练信道增强、推理恢复链和测试人群。各方法的训练条件在表题、图题和Excel工作表中明确标注；不把监督PAMF与半监督外部方法混成一张主比较图。证据截止2026年10月25日，本次没有新增训练、预测或真值评分。')
    h('1  实验分组、数据与统计规则')
    table('报告矩阵：学习方式是第一层分组',['学习方式','训练增强','报告测试','主方法与恢复器处理'],[
        ['监督','有信道增强','目标CLEAN＋六种practical；源RX未见日另列诊断','RIEI、DRIFT、IB-RFF、SDG、PAMF、DURA+PAMF；DURA只用有标签源IQ'],
        ['普通PL半监督','有信道增强','目标CLEAN＋六种practical；源RX未见日另列诊断','同六种主比较项；DURA可用全部允许源训练IQ'],
        ['监督','无信道增强','仅目标接收机CLEAN','四种外部方法单列；当前有增强PAMF不挪入本组'],
        ['普通PL半监督','无信道增强','仅目标接收机CLEAN','四种外部方法单列；不报告历史practical结果']])
    p('每个学习组均有165个独立主比较识别器：四种外部方法各33个，PAMF33个。PAMF和DURA+PAMF复用同一个识别器，故每组展示198个模型与推理链组合，两组共330个独立主比较识别器。含无增强控制时，全部594个逻辑模型由监督297个、半监督297个组成。')
    fig('coverage','每种学习条件下均有132个无增强外部模型、132个有增强外部模型和33个PAMF模型。该图只统计覆盖，不比较跨学习条件的混合性能。')
    p('数据为WiSig地面代理。源日期为2021年3月8、15、23日，3月1日为更早的未见日期，不能解释为未来日期预测。三类互斥人群为源接收机跨日诊断、目标接收机跨RX、目标接收机与日期双跨。标准划分每景分别为30000、126000、42000包；ManyRx分别为31279、92597、30424包，不等样本单元不强行等权成总准确率。')
    c=list(rows(OUT/'tables/data_contracts.csv'))
    table('源域划分与数据量；无标签池存在不等于监督组使用它',['数据设置','类数','源RX数','有标签训练','无标签池','源验证'],[[x['display'],x['classes'],len(x['source_Rx'].split(';')),x['L_s'],x['U_s'],x['V']] for x in c])
    p('监督识别器不读取无标签训练池。监督主系统的DURA也只使用有标签源集合的IQ；DURA不读取这些集合的发射机标签。半监督组的识别器按普通PL使用隐藏TX标签的源IQ，恢复器可使用全部允许源训练IQ。源验证不能更新模型状态；每行全部决策先冻结，再由独立CPU评分连接真值。')
    p('所有图均为竖向柱图。恢复网络实验采用双柱：同一方法未接入DURA与接入DURA，保持识别器、样本、场景与种子一致。变化值按种子内配对计算；误差线为三个模型种子的样本标准差，不是置信区间，也不自动表示显著性。三个识别器通常共用一份匹配恢复器，不能称为三套独立完整系统。')
    table('各方法的学习方式均显式区分',['方法','监督组','普通PL半监督组','原生训练预算'],[
        ['RIEI','完整原生目标；关闭U训练','相同目标＋普通PL','100轮，batch64，SGD，lr0.0001'],
        ['DRIFT','完整原生目标；关闭U训练','相同目标＋普通PL','100轮，batch256，Adam，lr0.0001'],
        ['IB-RFF','原两阶段与HSIC0.04；关闭U训练','相同两阶段＋普通PL','每阶段最多200轮，batch128；源NLL选模'],
        ['SDG','原双视图与对比目标；关闭U训练','相同目标＋普通PL','最多200轮，batch32；源NLL选模'],
        ['PAMF','50轮全部仅L，11100更新，无U加载器','50轮13L＋37联合L/U，11100更新','默认头联合训练；不另训分类头']])
    p('普通PL均采用当前模型eval/no_grad生成硬标签，阈值0.95、权重1、完整无标签批次平均，无EMA教师或复杂筛选。相同阈值不等于相同曝光剂量。外部有增强组保留0.68辅助分类槽，低郊区/中城市/低城市条件概率40%/35%/25%，施加概率依预算阶段为30%/60%/80%；无增强槽仍用原L。PAMF保留自身多视图有标签目标和三场景均匀抽取配方，监督版不是纯CE。')
    p('四种外部方法的主成绩默认直接识别。它们接入DURA后另列泛用性，不能把恢复后的成绩冒充识别方法本身。PAMF与DURA+PAMF在每个学习组均分列；监督组的完整系统不借用无标签源IQ恢复器。')
    for reg in REGIMES:
        title=RN[reg];cell,own,restore=settings(reg);tag='监督' if reg=='supervised' else '半监督';stem=reg
        h(title+'实验结果')
        p('本节所有方法均属于'+title+'。除单独标注的无增强CLEAN控制外，主比较均启用训练信道增强；RIEI、DRIFT、IB-RFF、SDG与PAMF直接识别，仅DURA+PAMF接入恢复器。')
        table(title+'组：标准划分主比较，准确率均值±标准差（%）',['测试人群/场景',*LABELS],[[TEST[t]+' / '+('CLEAN' if v=='clean' else '六景平均')]+[fmt(x) for x in r.main(reg,'F0',t,v)] for t in TASKS for v in ('clean','six_practical_mean')])
        fig(stem+'_standard',title+'组有增强主比较：标准划分、目标跨接收机六景准确率。六项均属同一学习条件，外部方法未接入DURA。')
        table(title+'组：11个设置的跨接收机六景准确率（%）',['数据设置',*LABELS],[[SHORT[p]]+[fmt(x) for x in r.main(reg,p)] for p in PIDS])
        p('接收机重划分保持5源、7目标与日期/样本预算，三套源集合两两仅交一个接收机。每套使用自身匹配数据契约的模型与恢复器，不从标准划分初始化；三套划分不合并成九种子重复。')
        fig(stem+'_receivers',title+'组：标准划分与两套接收机重划分的六景主结果。误差线为各设置内三个模型种子的标准差。')
        p('标注比例是相对源训练池的1%、5%、10%，对应630、3150、6300个有标签包，验证量均27000。源数量曲线只用1/2/3/4/6源并固定六目标，训练量也随源数改变；不插入标准5源/7目标点，不声称固定样本预算下纯接收机多样性的因果效果。')
        fig(stem+'_labels',title+'组：1%、5%、10%标注比例下的六景准确率，各组均报告全部六种主比较项。')
        fig(stem+'_sources',title+'组：固定六目标的自然源数量结果，保留非单调变化；所有柱均属本学习条件。')
        fig(stem+'_ten_class',title+'组：六类ManySig与十类ManyRx的六景Macro-F1。类别、接收机和样本量同时改变，只作规模条件对照。')
        off='A' if reg=='supervised' else 'B'
        table(title+'组无增强控制：仅目标CLEAN准确率（%）',['外部方法','目标跨RX CLEAN','目标双跨CLEAN'],[[NAMES[m],fmt(get('F0',m,off,v='clean')),fmt(get('F0',m,off,t='cross_both',v='clean'))] for m in METHODS])
        p('无增强页只展示同条件四种外部方法的CLEAN。当前PAMF主矩阵保留有标签信道视图，不搬入无增强组；缺少同条件项不补零、不用旧方法改名填齐。已冻结的无增强practical产物保留在历史包，本版不报告。')
        fig(stem+'_no_channel',title+'组无信道增强：只报告目标域CLEAN，跨RX与双跨分别成柱，均不经过DURA。')
        table(title+'组有增强逐景主结果：标准划分跨RX准确率（%）',['场景',*LABELS],[[VIEW[v]]+[fmt(x) for x in r.main(reg,'F0',view=v)] for v in SCENES])
        fig(stem+'_scenes',title+'组有信道增强：CLEAN与全部六种practical逐景结果。只有DURA+PAMF在所有场景均经过恢复器，CLEAN也不旁路。')
        p('本组DURA泛用性固定每个外部识别器，只比较接入恢复器前后。同一输入与种子配对，恢复器仅使用'+('有标签源训练集合IQ。' if reg=='supervised' else '全部允许源训练IQ。')+'结果不回流训练或参数选择。')
        for v in ('clean','six_practical_mean'):
            table(title+'组DURA泛用性：'+VIEW[v]+'，准确率%与配对百分点差',['外部方法','未接DURA','接入DURA','配对变化（pp）'],[[NAMES[m],fmt(get('F0',m,cell,v=v)),fmt(get('F0',m,cell,rd=restore,v=v)),fmt(r.gain(reg,'F0',m,view=v))] for m in METHODS])
        fig(stem+'_dura',title+'组DURA泛用性：标准划分、目标跨接收机六景准确率。每个方法左柱未接DURA、右柱接入DURA；柱顶为配对变化，保留负值。')
        fig(stem+'_dura_all_settings',title+'组：11个设置的目标跨接收机六景准确率，四种外部方法分别成面板；每一设置左柱未接DURA、右柱接入DURA。误差线为模型种子标准差，配对变化见数据表。')
        table(title+'组PAMF恢复对照：标准划分六景准确率（%）',['测试人群','未接DURA','接入DURA','配对变化（pp）'],[[TEST[t],fmt(get('F0',own,'native',t=t)),fmt(get('F0',own,'native',t=t,rd=restore)),fmt(r.gain(reg,'F0',own,t=t))] for t in TASKS])
        fig(stem+'_pamf_restoration',title+'组PAMF恢复对照：标准划分、三类互斥人群的六景准确率，固定识别器与种子；变化值按种子配对，不含CLEAN。')
        timing=[x for x in rows(OUT/'tables/PAMF_current_timing.csv') if x['case'].startswith(own+'_')]
        table(title+'识别器的实测时延：独占RTX3090、FP32',['执行路径','批量','中位ms','P95 ms','IQ/秒'],[['常规识别' if x['case'].endswith('_F') else 'DURA+PAMF优化图',x['batch'],f"{float(x['wall_median_ms']):.4f}",f"{float(x['wall_P95_ms']):.4f}",f"{float(x['throughput_IQ_s']):.1f}"] for x in timing])
        fig(stem+'_timing',title+'识别器的批量延迟竖柱图。两条路径分别为常规识别与DURA+PAMF优化图，5轮计时不是5个模型种子，不能把路径差当作恢复器算法加速。')
    h('成本、缺项与可声明范围')
    p('监督与普通PL半监督分别比较，不把两者完整系统的分差全部归因于伪标签，因为匹配DURA的无标签IQ权限也不同。本版不把仅PAMF的监督/半监督控制混进其他方法未标明学习方式的主图。半监督264个外部模型中61个实际伪标签接受数为0：RIEI56个、DRIFT5个，均保留原结果，不调整阈值重跑。')
    p('当前PAMF主矩阵66个识别器，其中6个标准划分模型精确复用、60个此前新训。去重后此前新主矩阵成本为3000个模型轮、666000次更新、82.0506共享worker小时。它不是本次报告新训练，也不是独占GPU小时或墙钟时间；不能把先前全套已退出方案的8670轮全部计到普通PL。DURA本次新训练和新缓存均为0，历史祖先成本及缺失计时不能补零。')
    p('PAMF普通PL下的三核心八组合还缺9个训练模型，完整结构层还缺24个；若重建七个有标签细项还需21个。恢复器组件和交叉种子可分别补21、6行冻结读出，不需要重训DURA。这些不是已发布补实验。旧复杂无标签机制与已退出外推方案仅保留历史，不能改名成当前普通PL自身消融。')
    p('所有practical结果都是物理启发仿真压力测试，不是实测在轨链路。恢复收益不能证明完全分离硬件身份与信道，也不保证每种方法、每种场景都提高。原生预算、优化目标和曝光不同，不按轮数或参数量推断效率。方法选择已有公开基准背景，不称独立盲选。')
    h('附录  数据来源与可读名称追溯')
    table('仅用于追溯的原始记录字段',['正文名称','原记录字段','含义'],[['识别器','F','原默认头随骨干联合训练'],['DURA恢复器','G','恢复接收IQ；不读取TX标签或识别器输出'],['监督组匹配恢复器','G-L / DURA-L','仅有标签源集合IQ，完整系统不读取U池'],['半监督组匹配恢复器','G-LU / DURA-LU','可用全部允许源训练IQ，U的TX标签仍隐藏'],['监督/半监督外部有增强组','C / D','两组性能分别成节、成图'],['监督/半监督外部无增强组','A / B','只报告目标域CLEAN'],['普通PL识别器','ordinary_PL / F21','当前半监督方案，标识不是方法版本名'],['历史教师/复杂门控配方','T1','不是当前普通PL，不改名混入']])
    p('科学源记录为20261023-phase1-practical-residual-full2x2-m600-r01及20261024-phase1-PAMF-REPL-completion-m201-r01。后者包含分别登记的监督与ordinary_PL行，历史run名称不决定每行实际方法。本报告只复用这些已冻结、独立评分统计，没有重新打开IQ、权重或逐样本目标真值。')
    p('PAMF60个新增主模型科学执行为dacafc8bfd7d3ef16e6f648cdc4491b65133ae8f；标准划分普通PL三模型精确复用原a462b7cf4c12a655e6c7a348ce2170a089f2d1cc。CPU评分恢复执行为a79bdf385489cb3822b37434a07e3c43e5ce839a，不是所有GPU训练的执行版本。')
    p('本版Excel按监督与半监督分别组织实验页，每页包含目的与设计、数据划分、竖柱图及双柱恢复对照、图表数据、完整结果。所有图保留PDF/SVG矢量和逐图数据CSV。新包不覆盖旧包，纯标准库核验器可从逐种子汇总复算均值与标准差。')
    text='\n'.join(md)+'\n';assert 'v2' not in text.lower();(OUT/'PAMF与四种对比方法实验报告.md').write_text(text,encoding='utf-8')
    (OUT/'README.md').write_text('''# 按学习方式分组的PAMF实验报告

先分监督/普通PL半监督，再分训练信道增强和推理链。每组主比较均为RIEI、DRIFT、IB-RFF、SDG、PAMF、DURA+PAMF；外部默认不接DURA。无增强只报目标CLEAN，有增强报CLEAN与六种practical。

PDF采用IEEE双栏。所有结果图均为竖柱，DURA实验为接入前后双柱；监督、半监督不混画。Excel每个实验页包含说明、对应数据划分、可编辑柱图、图表数据和完整结果；缺少同条件项不补零、不换配方填齐。

本版330个独立主比较识别器，每组165个；396个模型与推理链展示项，每组198个。全部594个逻辑模型仍不变，本次新增训练/预测/目标真值访问/DURA训练均为0。旧包保留。

标准库复核：`python -X utf8 -S reproduce/verify_report.py --root .`。IEEE源码在ieee/；逐图源数据在tables/figure_sources/。
''',encoding='utf-8')
    print(json.dumps(dict(status='REGIME_NARRATIVE_WRITTEN',figures=sum(x.startswith('![') for x in md)),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
