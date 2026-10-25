"""Readable Chinese report from the frozen ordinary-PL comparison projection."""
import csv,json,re,statistics,sys,shutil
from pathlib import Path
from html import escape
ROOT=Path('E:/type10-7');sys.path.insert(0,str(ROOT))
from tools.reports.pamf_pl_comparison import OUT,load,rows,METHODS,NAMES,PIDS,SHORT,TASKS,TN,SCENES,SN

def main():
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.enums import TA_CENTER
    from PIL import Image as PILImage
    pdfmetrics.registerFont(TTFont('Song','C:/Windows/Fonts/simsun.ttc',subfontIndex=0));pdfmetrics.registerFont(TTFont('Hei','C:/Windows/Fonts/simhei.ttf'));pdfmetrics.registerFont(TTFont('Times','C:/Windows/Fonts/times.ttf'))
    styles={
        'title':ParagraphStyle('title',fontName='Hei',fontSize=23,leading=32,textColor=colors.HexColor('#173C59'),spaceAfter=15),
        'sub':ParagraphStyle('sub',fontName='Song',fontSize=10,leading=16,textColor=colors.HexColor('#566777'),spaceAfter=12),
        'h':ParagraphStyle('h',fontName='Hei',fontSize=15,leading=22,spaceAfter=12,textColor=colors.HexColor('#173C59'),keepWithNext=True),
        'p':ParagraphStyle('p',fontName='Song',fontSize=10.2,leading=16.5,spaceAfter=8),
        'small':ParagraphStyle('small',fontName='Song',fontSize=8.5,leading=13.3,spaceAfter=7,textColor=colors.HexColor('#425566')),
        'cell':ParagraphStyle('cell',fontName='Song',fontSize=8.2,leading=12),
        'head':ParagraphStyle('head',fontName='Hei',fontSize=8.2,leading=12,textColor=colors.white),
        'caption':ParagraphStyle('caption',fontName='Song',fontSize=8.5,leading=13,spaceAfter=10,textColor=colors.HexColor('#425566'))}
    story=[];md=[];_,get=load();pairs=list(rows(OUT/'tables/paired_effects.csv'));factorial=list(rows(OUT/'tables/external_factorial.csv'));v=json.loads((OUT/'verification.json').read_text());contracts=list(rows(OUT/'tables/data_contracts.csv'));exposure=list(rows(OUT/'tables/PAMF_actual_exposures.csv'));ec=list(rows(OUT/'tables/external_training_and_exposures.csv'));timing=list(rows(OUT/'tables/PAMF_current_timing.csv'))
    def fmt(r,n=2):return f"{float(r['mean']):.{n}f}±{float(r['sd']):.{n}f}"
    def mean(r,n=4):return f"{float(r['mean']):.{n}f}"
    def pair(effect,p='F0',t='cross_rx',rd='DURA-LU',view='six_practical_mean',metric='accuracy'):return next(r for r in pairs if (r['protocol'],r['effect'],r['test'],r['readout'],r['view'],r['metric'])==(p,effect,t,rd,view,metric))
    def readable(text):
        text=str(text);text=re.sub(r'(?<=[\u3400-\u9fff])(?=[A-Za-z0-9])',' ',text)
        return re.sub(r'(?<=[A-Za-z0-9%])(?=[\u3400-\u9fff])',' ',text)
    def p(text,small=False):
        text=readable(text)
        visible=re.sub(r'[+−-]?\d+\.\d+%?',lambda m:'<nobr>'+m.group(0)+'</nobr>',escape(text))
        story.append(Paragraph(visible,styles['small' if small else 'p']));md.extend([text,''])
    def h(text):story.append(Paragraph(escape(text),styles['h']));md.extend(['## '+text,''])
    def page(title):
        if story:story.append(PageBreak())
        h(title)
    def table(headers,data,widths=None,caption=None):
        headers=[readable(x) for x in headers];data=[[readable(x) for x in r] for r in data]
        if caption:story.append(Paragraph(escape(caption),styles['caption']));md.extend([caption,''])
        width=507
        if widths is None:widths=[width/len(headers)]*len(headers)
        cells=[[Paragraph(escape(str(c)),styles['head']) for c in headers]]+[[Paragraph(escape(str(c)),styles['cell']) for c in r] for r in data]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#254C68')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F1F5F8')]),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#254C68')),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#AABBC8')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.extend([t,Spacer(1,11)])
        md.append('| '+' | '.join(str(x) for x in headers)+' |');md.append('|'+'|'.join('---' for _ in headers)+'|');md.extend('| '+' | '.join(str(x) for x in row)+' |' for row in data);md.append('')
    def fig(name,width=440,caption=None):
        path=OUT/'figures'/(name+'.png')
        with PILImage.open(path) as im:hh=width*im.height/im.width
        story.append(Image(str(path),width=width,height=hh));story.append(Spacer(1,6))
        if caption:story.append(Paragraph(escape(caption),styles['caption']))
        md.extend(['!['+(caption or name)+'](figures/'+name+'.png)','',caption or '',''])
    headers=['测试设置','RIEI','DRIFT','IB-RFF','SDG','PAMF监督','PAMF伪标签']
    widths=[99,68,68,68,68,68,68]
    # 1. Main answer, in current human-readable names.
    story.append(Paragraph('PAMF 与四种对比方法\n实验报告'.replace('\n','<br/>'),styles['title']));md.extend(['# PAMF 与四种对比方法实验报告',''])
    p('当前方案：PAMF 采用普通置信度伪标签；恢复部分沿用 DURA。证据截止：2026年10月25日（香港）。本报告是对已冻结结果的重新组织，本次新增训练、预测和真值评分均为0。',True)
    p('当前主比较保留 RIEI、DRIFT、IB-RFF、SDG 四种完整方法，以及 PAMF 的监督训练和普通伪标签训练。已退出的接收机外推伪标签方案不再作为当前 PAMF 主方法，也没有改名混入本报告。')
    p('标准接收机划分下，PAMF 经 DURA 恢复后的跨接收机六场景平均准确率为 '+fmt(get('F0','ordinary_PL'),4)+'%；相对同一设置下的 SDG，三次训练内配对均值差为 '+fmt(pair('PAMF_PL_minus_sdg_D'),4)+' 个百分点。')
    p('在11个数据设置中，PAMF 普通伪标签版的恢复后六场景均值均高于这里固定设置的四种外部方法。但这比较的是公开披露的完整训练配方，不表示各方法全部可能设置下的最优成绩，也不是显著性检验。')
    fig('F0_cross_rx_restored',width=435,caption='标准接收机划分：经过恢复后的六场景平均准确率。图中方法名不带版本后缀；误差线是三次识别器训练的样本标准差。')
    p('最重要的解释边界：PAMF 的监督版仍保留表示学习与多视图损失，不是纯交叉熵网络；伪标签版与监督版比较时，必须另外固定恢复器的数据权限，才能讨论伪标签本身。',True)
    # 2. Experimental module map and method differences.
    page('1  实验到底比较什么')
    p('IQ指接收信号的同相、正交采样。测试链条只有两种：接收IQ → 识别器 → 发射机类别；或接收IQ → DURA恢复器 → 识别器 → 发射机类别。前者在本报告叫“直接识别”，后者叫“恢复后识别”。')
    table(['模块','负责的任务','本次怎样处理'],[
        ['DURA恢复器','把当前接收IQ变为恢复IQ','权重冻结；训练不读取发射机标签、识别器权重/输出或目标域数据；本次不重训'],
        ['PAMF识别器','从IQ预测发射机类别','原骨干与原默认分类头联合训练；不另训分类头'],
        ['PAMF监督训练','只用有标签源训练数据学习身份表示','保留有标签跨接收机对比、配对一致性及原信道视图；不建立无标签训练加载器'],
        ['PAMF普通伪标签','在相同有标签配方上利用无标签源IQ','当前模型生成硬伪标签；置信度0.95、权重1；无教师模型和复杂筛选'],
        ['四种外部方法','比较完整识别方法的实测表现','保留原生损失和分类头；统一增加可开关的信道辅助分类槽与普通伪标签']],[92,156,259])
    p('普通伪标签的实际定义：先把当前模型切到评估模式，在不计算梯度的情况下生成预测；只接受最大类别概率至少0.95的样本。再回到训练模式计算硬标签交叉熵，未接受样本贡献0，损失仍除以整个无标签批次大小。')
    p('PAMF 的有标签批量为128，无标签批量为256；有标签端标签平滑为0.01，普通伪标签交叉熵不加标签平滑。每次更新保留6个有标签包前向，共768个有标签包评估。累计50轮、11100次更新，其中前13轮仅有标签，后37轮联合使用两类源训练数据；以20＋10＋20分段完成。')
    p('DURA 默认路径是双阶段展开恢复：两个学习先验调用、四次共轭梯度更新，固定FP32及既定优化图执行。它输出完整256点IQ，不因测试场景不同而绕过恢复或额外归一化。恢复器当前有266951个唯一参数。')
    # 3. Data.
    page('2  数据划分与测试任务')
    p('数据来自 WiSig 的 ManySig 与 ManyRx。除大规模十类数据外，其余设置均为6个发射机类别。源训练使用2021年3月8、15、23日；未见日期是更早的3月1日，不能称为未来日期预测。源和目标的物理记录保持原互斥划分。')
    table(['数据设置','源接收机数','有标签训练','无标签训练','源验证'],[[SHORT[r['protocol']],len(r['source_Rx'].split(';')),r['L_s'],r['U_s'],r['V']] for r in contracts],[143,68,96,100,100])
    table(['测试任务','接收机与日期组合','标准划分包数/视图'],[
        ['仅日期变化','源接收机，未见的3月1日',30000],['仅接收机变化','目标接收机，训练所用的三个日期',126000],['日期与接收机同时变化','目标接收机，未见的3月1日',42000]],[125,271,111])
    p('大规模十类数据实际分别为31279、92597、30424包/视图，单元样本数不等。1至6个源接收机的自然预算曲线固定同六个目标接收机；5源/7目标的标准划分不插作曲线的5源点。源数量改变时，有标签、无标签和验证数据量一起变化。',True)
    p('1%与5%表示相对于源训练池的标注比例；另有源验证标签，其数量独立披露。输入采用原均衡IQ、中央256点窗口、信道变换前单位RMS归一化。每条物理记录的所有衍生视图保持同一物理身份。',True)
    # 4. Methods and budgets.
    page('3  外部方法与训练配方')
    table(['方法','保留的完整目标','训练预算与冻结规则'],[
        ['RIEI','身份/接收机表征、分类及独立性相关目标；保留交替优化','100轮；批量64；SGD，学习率0.0001；最终轮冻结'],
        ['DRIFT','身份与接收机分支、对抗域目标、中心及分离相关目标','100轮；批量256；Adam，学习率0.0001；最终轮冻结'],
        ['IB-RFF','两阶段训练；第二阶段保留作者HSIC项，系数0.04','每阶段最多200轮；批量128；Adam，学习率0.001；源验证NLL选模，降学习率/早停耐心5/10'],
        ['SDG','原生双视图、身份对比和域对抗对比目标','最多200轮；批量32；Adam，学习率0.001；源验证NLL选择checkpoint'],
        ['PAMF','相同骨干/默认头及原有标签表示学习；按监督/普通伪标签区分','50轮，11100次更新；原优化器与分段日程；固定最终权重']],[63,220,224])
    p('这里比较的是经过输入长度、类别数及本项目数据接口适配的完整方法实现。RIEI使用登记的一维短输入stem；完整损失保留不等于逐字运行作者未适配脚本。各方法原生预算和优化步骤不同，不把相同轮数当作相同算力。')
    table(['外部训练设置','附加信道增强','普通伪标签','识别器使用的源训练数据'],[
        ['无信道＋监督','关闭','关闭','只有有标签训练集合'],['无信道＋伪标签','关闭','启用','有标签与无标签训练集合'],['有信道＋监督','启用','关闭','只有有标签训练集合'],['有信道＋伪标签（主比较）','启用','启用','有标签与无标签训练集合']],[151,85,85,186])
    p('四种设置都保留权重0.68的附加分类槽：无信道或未抽中变换时，槽中使用原始有标签IQ。因此“无信道”不是删掉辅助损失，也不删除SDG自己的原生增强。信道辅助视图从原有标签IQ生成。')
    p('外部增强使用真实 practical residual 路径：训练场景为低郊区/中城市/低城市，条件概率40%/35%/25%；按最大预算前20%、20%至45%、后55%采用30%/60%/80%的施加概率。IB-RFF各阶段重置预算进度。实际早停曝光量以记录为准。')
    p('PAMF 的原有标签信道视图在这三个环境中均匀抽取；普通伪标签与监督控制沿同一有标签配方。它不是外部0.68拼接配方的逐项相同复制。外部无标签/有标签批量比例为1，PAMF为2；阈值与损失定义相同，不代表所有曝光剂量相同。',True)
    # 5. Main F0 full table.
    page('4  标准划分：三种泛化任务的主结果')
    p('表中均为准确率%，三次不同模型初始化的均值±样本标准差。clean是未叠加新增信道的观测；“六景平均”先在同一次训练内平均六种信道，再汇总三次训练。外部主比较固定为有信道增强、有普通伪标签的完整训练，不按测试成绩选格。')
    for rd,title in [('raw','直接识别'),(None,'恢复后识别')]:
        table(headers,[[TN[t]+' / '+('clean' if view=='clean' else '六景平均')]+[fmt(get('F0',m,t,rd=rd,v=view)) for m in METHODS] for t in TASKS for view in ('clean','six_practical_mean')],widths,title)
    p('恢复后比较中，监督PAMF的恢复器只用有标签源集合的IQ；伪标签PAMF和四种外部主比较的恢复器可用有标签＋无标签源训练IQ。完整系统比较允许这个权限差别；要把变化解释为伪标签本身，需要第8节的固定恢复器对照。',True)
    # 6. Eleven settings.
    page('5  全部数据设置：跨接收机六景结果')
    for rd,title in [('raw','直接识别：准确率均值±标准差（%）'),(None,'恢复后识别：准确率均值±标准差（%）')]:
        table(headers,[[SHORT[p]]+[fmt(get(p,m,rd=rd)) for m in METHODS] for p in PIDS],widths,title)
    p('不同设置有不同类数、训练量和目标接收机集合。这里逐设置比较，不把十类与六类样本混成一个总准确率，也不对这些设置计算一个隐藏权重的“总冠军分数”。',True)
    # 7. Data-size curves.
    page('6  标注数量与源接收机数量')
    fig('labels',width=435,caption='1%、5%、10%标注条件；10%使用标准划分。其余数据角色保持登记约定。')
    fig('sources',width=435,caption='自然预算源数量：1、2、3、4、6个源接收机，固定六个目标接收机。横轴不是只改变多样性，也改变训练量。')
    p('PAMF伪标签版恢复后六景均值从1源的54.2681%到6源的75.7687%，但3源73.9371%与4源74.3923%接近。不能把非单调或小幅差异删去，也不能宣称固定标注预算下增加接收机必然提升。',True)
    # 8. Scenes and MacroF1.
    page('7  逐信道场景与大规模十类结果')
    table(['测试场景']+headers[1:],[[SN[s]]+[mean(get('F0',m,v=s),2) for m in METHODS] for s in SCENES],widths,'标准划分跨接收机、恢复后识别，准确率均值（%）；逐次训练与标准差保留于来源表。')
    fig('seven_scenes',width=430,caption='同一批冻结样本的逐景表现。高/中/低郊区与高/中/低城市均为物理启发仿真，不是实测卫星链路。')
    table(['十类数据端点']+headers[1:],[[route+' / '+viewlabel]+[fmt(get('ManyRx',m,rd=rd,v=view,metric='macro_f1')) for m in METHODS] for rd,route in [('raw','直接识别'),(None,'恢复后识别')] for view,viewlabel in [('clean','clean'),('six_practical_mean','六景平均')]],widths,'大规模十类数据的Macro-F1（%），始终覆盖全部10类。')
    # 9. Factorial.
    page('8  四种外部训练设置：不能只保留最高一格')
    names={'A':'无信道＋监督','B':'无信道＋伪标签','C':'有信道＋监督','D':'有信道＋伪标签'}
    for rd,title in [('raw','直接识别'),('matching','恢复后识别')]:
        table(['方法']+list(names.values()),[[NAMES[m]]+[fmt(get('F0',m,rd='raw' if rd=='raw' else ('DURA-L' if cell in ('A','C') else 'DURA-LU'),cell=cell)) for cell in names] for m in METHODS[:4]],[75,108,108,108,108],title+'：标准划分跨接收机六景准确率（%）')
    p('四格里，信道效果在“相同伪标签开关”下比较；伪标签效果优先看直接识别结果。恢复后跨伪标签开关时，恢复器的无标签IQ权限也一起改变，差值属于完整系统变化。')
    table(['方法','信道变化：无伪标签','信道变化：有伪标签','伪标签变化：无信道','伪标签变化：有信道'],[[NAMES[m]]+[fmt(next(r for r in factorial if r['method']==m and r['protocol']=='F0' and r['test']=='cross_rx' and r['route']=='raw' and r['view']=='six_practical_mean' and r['effect']==e)) for e in ('channel_no_PL','channel_with_PL','PL_no_channel','PL_with_channel')] for m in METHODS[:4]],[59,112,112,112,112],'直接识别的配对差（百分点）。所有负值保留，不把伪标签默认解释成增益。')
    p('完整四格共528个外部模型。主表采用其中132个“有信道＋伪标签”模型，目的是固定比较设置；这不等于每种外部方法都在该格取得其最高分。完整表中能看到某些方法加信道或伪标签后反而降低。',True)
    # 10. PAMF permissions/fairness.
    page('9  PAMF：伪标签收益与恢复收益要分开')
    table(['识别训练方式','直接识别','恢复器仅用有标签源IQ','恢复器使用全部允许源IQ'],[[NAMES[m]]+[fmt(get('F0',m,rd=rd)) for rd in ('raw','DURA-L','DURA-LU')] for m in ('supervised','ordinary_PL')],[141,92,137,137],'标准划分跨接收机六景准确率（%）')
    differences=[('直接识别：只增加普通伪标签',pair('PL_minus_Sup_fixed_route',rd='raw')),('同一“仅有标签IQ”恢复器：增加伪标签',pair('PL_minus_Sup_fixed_route',rd='DURA-L')),('同一“全部允许源IQ”恢复器：增加伪标签',pair('PL_minus_Sup_fixed_route',rd='DURA-LU')),('完整系统：伪标签识别器＋全部源IQ恢复器，对比监督系统',pair('complete_system_PL_minus_Sup',rd='PL_GLU_vs_Sup_GL'))]
    table(['比较口径','配对均值差±标准差（百分点）'],[[label,fmt(r,4)] for label,r in differences],[345,162])
    p('增加普通伪标签、固定恢复器：+0.1422 个百分点；直接识别：−0.2352 个百分点。')
    p('完整系统比较：+1.1563 个百分点。这里同时改变了恢复器训练数据范围，不能把全部差值归因于半监督。')
    p('在11个设置中，固定同一使用全部源训练IQ的恢复器，普通伪标签对监督识别器的六景均值在跨日为11个设置提高、跨接收机为8个提高、同时变化为9个提高。直接识别的跨接收机六景仅2个设置提高。选用普通伪标签作为暂定方法不等于已证明它在所有条件下都有效。')
    table(['识别方法','加入匹配恢复器的配对差（百分点）'],[[NAMES[m],fmt(pair('restoration_gain_'+m,rd='DURA-L' if m=='supervised' else 'DURA-LU'),4)] for m in METHODS],[210,297],'标准划分跨接收机六景：同一识别器恢复后减直接识别。')
    # 11. Exposures/cost.
    page('10  实际训练曝光与成本')
    er=[]
    for pid in PIDS:
        rr=[r for r in exposure if r['method']=='ordinary_PL' and r['protocol']==pid];seen=sum(int(r['U_seen']) for r in rr);accepted=sum(int(r['U_accepted']) for r in rr)
        er.append([SHORT[pid],min(int(r['U_seen']) for r in rr),max(int(r['U_seen']) for r in rr),f'{100*accepted/seen:.2f}%'])
    table(['数据设置','单次训练无标签曝光最小值','最大值','三次合计接受率'],er,[139,124,124,120])
    p('曝光指加载参与伪标签判定的次数，不是不同物理样本个数；同一物理样本可跨轮重复。接受率按三次训练的接受数/曝光数合计计算，未把低接受率隐藏或把未接受样本从损失分母去掉。监督控制的无标签曝光为0。',True)
    zero={m:sum(int(r['pl_seen'])>0 and int(r['pl_accepted'])==0 for r in ec if r['family']==m) for m in METHODS[:4]}
    p('外部264个启用伪标签的模型中，61个模型实际接受数为0：RIEI为'+str(zero['riei'])+'个，DRIFT为'+str(zero['drift'])+'个，IB-RFF与SDG为0。它们仍属于启用同一置信度规则的有效负结果，不允许事后降低阈值重跑。')
    table(['成本口径','核实数值与含义'],[
        ['当前PAMF两类主比较','66个识别器：11个设置×监督/伪标签×三次训练；其中6个标准划分模型精确复用，60个此前新训'],
        ['去重后的新PAMF主比较训练','3000个模型轮、666000次更新、'+f"{v['current_main_new_shared_worker_hours']:.4f}"+'共享worker小时；不是本次报告新增训练'],
        ['四种外部全部设置','528个完整模型；原生双阶段训练与早停按实际记录，不把模型数当作阶段数'],
        ['恢复器与缓存','本次新增训练0、新建恢复IQ缓存0；历史同契约恢复器祖先成本不能视为0'],
        ['没有混算的历史工作','此前全套201个PAMF模型的8670新模型轮和265.9564共享worker小时包含已退出方案；不能全记到当前普通伪标签主方法']],[155,352])
    p('共享worker小时来自各实际训练阶段的elapsed记录，已按唯一阶段去重。并发时它不是墙钟耗时或独占GPU小时。原标准划分一份恢复器的完整祖先计时未全部序列化，原缺失继续保留，不补零；技术失败的部分开销与成功模型成本分开。',True)
    # 12. Efficiency.
    page('11  现有工程效率证据')
    tr=[]
    for method,label in [('supervised','PAMF监督'),('ordinary_PL','PAMF伪标签')]:
        for suffix,route in [('F','直接识别／常规执行'),('GF','恢复后识别／优化图执行')]:
            for batch in ('1','128'):
                r=next(x for x in timing if x['case']==method+'_'+suffix and x['batch']==batch)
                tr.append([label,route,batch,f"{float(r['wall_median_ms']):.4f}",f"{float(r['wall_P95_ms']):.4f}",f"{float(r['throughput_IQ_s']):.1f}"])
    table(['方法','执行路径','批量','中位延迟ms','95%分位ms','IQ/秒'],tr,[84,143,35,80,80,85])
    p('测量使用独占RTX3090、FP32、五轮计时重复。单包普通PAMF恢复后识别的中位延迟为4.5657ms、95%分位为4.5854ms；含传输的CPU往返中位数为4.6222ms。五轮是计时重复，不是五个训练模型。')
    p('单包“恢复后”比这里“直接识别”快，是因为两行采用不同执行路径：直接识别为常规执行，完整恢复链使用已有优化图。不能据此声称增加恢复器本身带来算法加速。大批量128时，恢复后完整链为14.5251ms，直接识别为7.4950ms，显示恢复链的实际额外计算。')
    p('这些是当前PAMF完整模型的实测数据。旧外部模型时延面板有其自己的权重和执行背景，本报告不把它改标成当前正确信道四种训练设置下最终权重的统一实测结果。四种外部最终权重与PAMF同卡同路径的直接效率排名仍需单独核对或补测，不能靠参数量推定。')
    p('共驻留显存不等于单模型峰值显存；CUDA图缓冲区复用后的增量显存也不等于完整模型内存。缓存恢复IQ后的分类时间不能冒充在线“恢复＋识别”时间。',True)
    # 13. What is complete and what remains, without authorizing training.
    page('12  完成范围、论文安排与待补项目')
    table(['实验模块','当前证据','处理建议'],[
        ['六方法主比较','198模型，11设置与三次训练，三种泛化、clean与六景均已冻结评分','可直接写主文，不重训'],
        ['四种外部增强/伪标签四格','528模型、完整原生方法、所有正负结果','完整四格放附录；主表固定同一设置'],
        ['PAMF监督与普通伪标签','66模型，直接识别及两类恢复器均已读出','把识别器伪标签作用与恢复器无标签IQ作用分开'],
        ['PAMF完整模型时延','监督与普通伪标签100个“设置×重复”计时组合已测','可用；说明执行路径、硬件和内存口径'],
        ['PAMF结构与损失细项','监督无标签关闭格、严格交叉熵结构部分可沿用；原复杂方案的含无标签格不适用','单列自身实验报告；不得把旧方案改名'],
        ['恢复器组件组合/交叉种子','恢复器训练权重可复用；旧识别器上的读出不是当前普通伪标签读出','需要时只补冻结读出，不重训DURA']],[128,236,143])
    p('建议主文依次呈现：数据与方法预算表、标准划分三类泛化结果、完整11设置跨接收机表、逐信道图、标注与源数量曲线、PAMF固定恢复器公平性对照、实际效率与限制。完整四格、逐接收机/发射机/日期、完整混淆矩阵和负结果放补充材料。')
    p('当前不能声称：提出了一种新的半监督算法；普通伪标签在所有条件下必然有效；DURA完全分离硬件身份与信道；仿真等于真实在轨验证；三个共用恢复器的识别器重复等于三套独立系统；或者当前暂定方法是从未暴露目标结果的独立盲选。')
    p('本次按用户要求先交付主比较报告，未发起补实验。更完整的自身消融梳理中，普通伪标签下三核心八组合还缺9个训练模型；完整普通伪标签结构层还缺24个训练模型；七个有标签细项若全部重建还需21个。恢复器组件与交叉种子可分别补21、6行冻结预测评分，不需要新恢复器训练。这些是待讨论的最小缺项，不是已发布队列。',True)
    # 14. Readable terminology and evidence index, the sole location of legacy codes.
    page('附录  阅读名称与原始记录的对应')
    table(['正文统一称呼','原始记录中的代号','解释'],[
        ['识别器','F','预测发射机身份的网络，当前PAMF默认头联合训练'],['DURA恢复器','G','先处理接收IQ，再交给识别器'],['只用有标签源IQ的恢复器','G-L / DURA-L','集合有标签，但恢复器不读取发射机标签'],['使用全部允许源训练IQ的恢复器','G-LU / DURA-LU','使用有标签与无标签训练集合的IQ，仍不读TX标签'],['直接识别','raw','不经过恢复器'],['普通伪标签PAMF','ordinary_PL / F21','当前暂定半监督方法；F21是兼容实现编号，不是新的论文方法名'],['旧版带教师与复杂筛选的识别器','T1','历史方法；当前普通伪标签不继承其教师和筛选机制'],['已退出的外推伪标签方案','PAMF-REPL','历史对照，不能当作当前普通伪标签的消融'],['标准／接收机划分一／划分二','F0 / F1_rx_repartition_r01 / F2_rx_repartition_r01','三套接收机成员不同、数量相同的训练数据划分'],['四种外部训练设置','A / B / C / D','对应无信道监督、无信道伪标签、有信道监督、有信道伪标签']],[173,133,201])
    p('标准源接收机索引为1/3/4/6/8，接收机划分一为0/1/2/5/7，接收机划分二为0/3/9/10/11。三套源集合两两只交一个接收机；源日期保持相同。旧同名划分的不同成员不在本报告中冒充新划分。',True)
    p('源记录一：20261023-phase1-practical-residual-full2x2-m600-r01。本报告只取其中四种外部完整方法的528模型；旧PAMF配方没有混入当前主表。',True)
    p('源记录二：20261024-phase1-PAMF-REPL-completion-m201-r01。本报告只取其中监督/ordinary_PL主比较66模型，以及已测main_controls时延；同一历史run名不决定其每行实际方法。',True)
    p('冻结统计来源为已交付r17包，完整序列化CM与净纠错已独立复算。本报告再次核对20064行汇总、60192行总体逐次指标的三次覆盖，并从它们计算成对差；未重新打开IQ、checkpoint或逐样本目标真值。',True)
    p('表格文件：complete_summary.csv（全部四格及PAMF两类读出）；per_seed_overall.csv.gz（逐次指标）；paired_effects.csv（PAMF监督/伪标签与恢复差）；external_factorial.csv（四格主效应/交互）；data_contracts.csv（原始集合与包数）；实际曝光、唯一训练阶段成本和时延另列。所有原始代号仅用于附录和机器可读追溯。',True)
    p('外部方法训练/接口执行分别沿原发布记录。PAMF主比较60个新增模型的科学执行为dacafc8bfd7d3ef16e6f648cdc4491b65133ae8f；标准划分精确复用模型沿各自原执行记录，其中普通伪标签三次模型为a462b7cf4c12a655e6c7a348ce2170a089f2d1cc。独立CPU评分恢复执行为a79bdf385489cb3822b37434a07e3c43e5ce839a，不能把它标成所有模型的训练版本。',True)
    # Render and store the editable narrative.
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#C9D6DF'));canvas.line(44,805,551,805);canvas.setFont('Song',8);canvas.setFillColor(colors.HexColor('#627687'));canvas.drawString(44,814,'PAMF 与四种对比方法 · 已冻结结果分析');canvas.drawString(44,25,'2026-10-25 · 香港');canvas.drawRightString(551,25,str(doc.page))
    path=OUT/'PAMF与四种对比方法实验报告.pdf';doc=SimpleDocTemplate(str(path),pagesize=(595.28,841.89),leftMargin=44,rightMargin=44,topMargin=49,bottomMargin=43,title='PAMF与四种对比方法实验报告',author='Research experiment records');doc.build(story,onFirstPage=footer,onLaterPages=footer)
    text='\n'.join(md);assert 'v2' not in text.lower();(OUT/'PAMF与四种对比方法实验报告.md').write_text(text+'\n',encoding='utf-8')
    (OUT/'README.md').write_text('''# PAMF 与四种对比方法实验报告

先读同名 PDF 或 Markdown。当前 PAMF 使用普通伪标签，监督版为匹配对照；外部方法统一命名 RIEI、DRIFT、IB-RFF、SDG。正文以可读名称描述模块，原始代号仅用于附录和表格追溯。

本包从已冻结、已独立评分的统计构建，无新增训练、预测或真值访问。主比较198个模型；含四种外部全部训练设置时594个模型。全部60192行总体逐次指标和20064行汇总可检索。9份图含可编辑文字的PDF/SVG及PNG预览，各图源CSV同名。

`tables/` 的数值口径见报告附录。成对差先按相同训练种子计算，再汇总三次训练；六场景先在一次训练内平均。三次识别器通常共用一份恢复器，不是三套独立完整系统。

本包为新的r18报告，旧r17和更早结果原样保留。当前报告不把已退出方案的组件消融改名为普通伪标签，也没有自动发布待补实验。更广的自身实验梳理单独继续。
''',encoding='utf-8')
    print(json.dumps(dict(status='REPORT_WRITTEN',PDF=str(path),figures=9),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
