"""Detailed readable report for classification-only supervised PAMF branches."""
import json,re,statistics,sys
from html import escape
from pathlib import Path
ROOT=Path('E:/type10-7');sys.path.insert(0,str(ROOT))
from tools.reports.pamf_branch_ablation import OUT,ARMS,NAMES,EN,PANELS,TASKS,TN,SCENES,SN,load,rows,write
TITLE='PAMF网络分支消融实验报告'

def main():
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from PIL import Image as PILImage
    pdfmetrics.registerFont(TTFont('Song','C:/Windows/Fonts/simsun.ttc',subfontIndex=0))
    pdfmetrics.registerFont(TTFont('Hei','C:/Windows/Fonts/simhei.ttf'))
    styles={
        'title':ParagraphStyle('title',fontName='Hei',fontSize=23,leading=32,textColor=colors.HexColor('#173C59'),spaceAfter=15),
        'h':ParagraphStyle('h',fontName='Hei',fontSize=15,leading=22,spaceAfter=12,textColor=colors.HexColor('#173C59'),keepWithNext=True),
        'p':ParagraphStyle('p',fontName='Song',fontSize=10.1,leading=16.3,spaceAfter=8),
        'small':ParagraphStyle('small',fontName='Song',fontSize=8.4,leading=13.1,spaceAfter=7,textColor=colors.HexColor('#425566')),
        'cell':ParagraphStyle('cell',fontName='Song',fontSize=8,leading=11.6),
        'head':ParagraphStyle('head',fontName='Hei',fontSize=8,leading=11.6,textColor=colors.white),
        'caption':ParagraphStyle('caption',fontName='Song',fontSize=8.4,leading=13,spaceAfter=9,textColor=colors.HexColor('#425566'))}
    story=[];md=[];_,get=load();panel=PANELS[0]
    paired=list(rows(OUT/'tables/full_minus_variant.csv'));timings=list(rows(OUT/'tables/timing.csv'));params=list(rows(OUT/'tables/parameters.csv'));cost=list(rows(OUT/'tables/source_cost.csv'));geom=list(rows(OUT/'tables/source_geometry.csv'));interventions=list(rows(OUT/'tables/source_interventions.csv'))
    def readable(x):
        s=str(x);s=re.sub(r'(?<=[\u3400-\u9fff])(?=[A-Za-z0-9])',' ',s)
        return re.sub(r'(?<=[A-Za-z0-9%])(?=[\u3400-\u9fff])',' ',s)
    def p(text,small=False):
        text=readable(text);story.append(Paragraph(escape(text),styles['small' if small else 'p']));md.extend([text,''])
    def page(text):
        if story:story.append(PageBreak())
        text=readable(text);story.append(Paragraph(escape(text),styles['h']));md.extend(['## '+text,''])
    def table(headers,data,widths=None,caption=None):
        headers=[readable(x) for x in headers];data=[[readable(x) for x in r] for r in data]
        if caption:p(caption,True)
        widths=widths or [507/len(headers)]*len(headers)
        cells=[[Paragraph(escape(c),styles['head']) for c in headers]]+[[Paragraph(escape(str(c)),styles['cell']) for c in r] for r in data]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#254C68')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F1F5F8')]),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#AABBC8')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4.5),('BOTTOMPADDING',(0,0),(-1,-1),4.5)]))
        story.extend([t,Spacer(1,9)]);md.append('| '+' | '.join(headers)+' |');md.append('|'+'|'.join('---' for _ in headers)+'|');md.extend('| '+' | '.join(map(str,r))+' |' for r in data);md.append('')
    def fig(name,width=440,caption=''):
        path=OUT/'figures'/(name+'.png')
        with PILImage.open(path) as im:height=width*im.height/im.width
        story.append(Image(str(path),width=width,height=height));story.append(Spacer(1,5))
        if caption:p(caption,True)
        md.extend(['!['+(caption or name)+'](figures/'+name+'.png)',''])
    def stat(r,n=2):return f"{float(r['mean']):.{n}f}±{float(r['sd']):.{n}f}"
    def result(a,t='cross_rx',rd='raw',view='six_practical_mean',metric='accuracy'):return get(panel,a,t=t,rd=rd,view=view,metric=metric)
    def pair(a,t='cross_rx',rd='raw',view='six_practical_mean',metric='accuracy'):return next(r for r in paired if (r['arm'],r['test'],r['readout'],r['view'],r['metric'])==(a,t,rd,view,metric))
    def timing(a,route='direct',batch=1):return next(r for r in timings if (r['arm'],r['route'],r['batch'])==(a,route,str(batch)))
    def number(r,key,n=3):return f"{float(r[key]):.{n}f}"
    main_table=[]
    for a in ARMS:
        main_table.append(dict(结构=NAMES[a],直接clean=stat(result(a,view='clean'),4),恢复后clean=stat(result(a,rd='DURA-L',view='clean'),4),直接六景=stat(result(a),4),恢复后六景=stat(result(a,rd='DURA-L'),4)))
    write(OUT/'tables/阅读表_跨接收机准确率.csv',main_table)
    # Cover: actual conclusion first.
    story.append(Paragraph('PAMF 网络分支消融<br/>实验报告',styles['title']));md.extend(['# PAMF 网络分支消融实验报告',''])
    p('仅分类损失的监督训练 | 9 种结构 × 3 次独立训练 | 已冻结评分 | 2026年10月25日',True)
    p('功放相关支路是本面板中影响最大的被移除支路。跨接收机、六种信道平均准确率由完整网络的'+stat(result('full'),4)+'%下降至'+stat(result('NO_PA'),4)+'%；同一训练种子内配对后，下降'+stat(pair('NO_PA'),4)+'个百分点。')
    p('参考响应支路有较稳定的小幅收益。路径描述、显式时间坐标和三视图的收益依赖测试条件；固定打乱局部顺序的对照在部分条件下反而更好。因此，现有证据支持保留功放相关支路，但不支持把所有设计都写成普遍、独立且必需的性能来源。')
    fig('supervised_results',435,'图1  标准划分的跨接收机结果。点为三个模型均值，误差线为样本标准差；六景先在每个种子内等权平均。')
    p('本报告只展示这一监督结构面板及其匹配的恢复后读出。共27个独立识别器；接入DURA后的列复用相同识别器和同一个恢复器，不是另27个训练模型。本次新增训练、预测、目标真值访问和恢复器训练均为0。',True)
    # 1. Architecture.
    page('1  网络分支具体指什么')
    p('PAMF从256点复数IQ中提取身份表示，输出既有默认分类头的六类预测。共享骨干的时频表示、功放相关信息、局部路径描述和公共参考响应共同形成部署特征；本报告的“分支”指实际前向计算结构。')
    fig('network_branches',492,'图2  分支依赖示意。共享骨干身份特征始终保留；示意图不表示各支路等权相加，也不表示存在三套独立骨干。')
    table(['信息来源','实际作用与尺寸','本次控制'],[
        ['功放相关支路','包含功放相关计算块和统计投影，参与既有身份表示','删除相关计算块和投影，其他分支继续训练'],
        ['局部路径描述','同一中间特征序列上学习3个投影；4个局部窗口；每视图112维，拼接336维后映射到160维','单视图、去显式时间、固定顺序打乱、整路删除、矩统计替代'],
        ['公共参考响应','由固定公共参考模板计算响应描述，经29→64→160投影、归一化和增益后融合','删除响应整路；或只打乱模板频谱相位']],[100,258,149])
    p('三个“视图”是同一共享中间特征的三个可学习投影，不是三个独立模型或三模型集成。所有结构继续使用原默认分类头联合训练，融合后的部署特征均保持160维。',True)
    # 2. Definitions with limits.
    page('2  九种结构设置及解释边界')
    descriptions={
        'full':('全部既有支路保留；三视图路径描述、显式时间坐标和真实公共参考模板。','条件对照的基准，不预设它必然最优。'),
        'SIG_TIME':('局部路径描述由三个投影视图改为一个；融合投影从336→160改为112→160。','参数减少36992，不能只把差异归因于视图数。'),
        'NO_TIME':('路径描述去掉显式时间坐标，保留六个投影通道；每窗口21维结果补7个零。','没有删除时域骨干，也没有消除全部顺序信息。'),
        'SHUFFLE_TIME':('每个8位置窗口保留首尾，固定打乱中间6个位置，再建立时间坐标。','固定同一置换贯穿训练和测试；不是逐次随机打乱原IQ。'),
        'NO_SIGNATURE':('整条路径描述输出置零，删除相应可学习池化结构。','共享骨干时频信息仍在；参数同时减少55968。'),
        'NO_PA':('删除功放相关计算块及其统计投影，原位置返回零表示。','结构变化而非仅关闭一个损失；参数减少11040。'),
        'NO_REFERENCE':('删除参考响应、响应投影及增益，保留其余身份特征。','参数减少12800，不能单独证明参考信息的物理可辨识性。'),
        'WRONG_REFERENCE':('固定打乱公共模板频谱的相位，保留幅度谱和全部可学习投影维度。','网络可适应固定错误模板；不是每次输入不可学习的随机破坏。'),
        'COV_X3':('每窗口使用六维均值及21项上三角协方差，固定27→28维映射；仍为三视图。','学习参数数相同，描述的内在秩和信息类型仍不同。')}
    table(['结构','实际改动','解读时必须保留的边界'],[[NAMES[a],*descriptions[a]] for a in ARMS],[121,207,179])
    p('保留时间坐标的路径描述：6个投影通道加1个时间坐标，得到7项一阶增量和21项反对称二阶面积，共28项；4个窗口合为112维。去掉时间坐标后维数先变为6＋15＝21，再补零保持接口宽度。',True)
    p('本面板没有删除基础Sinc滤波器、整个时域骨干或整个频域骨干；不能把它称为对全部骨干模块的穷尽消融。',True)
    # 3. Training and legal target populations.
    page('3  训练、数据和统计口径')
    table(['项目','本面板固定设置'],[
        ['目标','普通输入分类交叉熵＋0.68×信道辅助槽分类交叉熵；标签平滑为0'],
        ['关闭的训练分支','不使用无标签训练数据、伪标签、教师、跨接收机对比损失、配对KL或独立分类头拟合'],
        ['训练预算','每模型50轮、11100次更新；20＋10＋20分段，均沿自身模型续训'],
        ['实际计算曝光','每次更新128个有标签物理包、384次包前向；每模型4262400次包前向'],
        ['重复训练','3个预先固定模型种子；27模型均从各自匹配结构起始并完成全部预算'],
        ['训练信道','真实ResidualChannel的残余信道、同步后、不均衡设置；仅源训练IQ生成'],
        ['信道混合','低郊区0.40、中城市0.35、低城市0.25；预算进度＜0.20／＜0.45／其后对应施加概率0.30／0.60／0.80'],
        ['恢复后列','固定DURA，仅用有标签源训练集合的IQ学习，不读取发射机标签；27模型共用同一恢复器']],[110,397])
    p('“仅分类损失”不等于无输入增强。原普通扰动和上述信道辅助槽保留。384次包前向包括额外拼接计算中的普通输入部分；本次没有把它删掉。损失使用首个普通输出和辅助槽输出，共256个包logit，而不是对384个输出都另加交叉熵。辅助槽每模型曝光1420800次，不等于同样多次都实际施加了信道变换。',True)
    table(['数据群体','接收机／日期关系','每景包数'],[
        ['仅日期变化','源接收机，未参与训练的较早日期','30000'],
        ['仅接收机变化','目标接收机，训练中已见日期','126000'],
        ['两者同时变化','目标接收机，未参与训练的较早日期','42000']],[110,317,80])
    p('标准划分共6类；有标签源训练6300包、源验证27000包。训练日期为三天，另一更早日期只作冻结测试，不称未来日期预测。三群互斥，分别报告，不混算一个总分。每群均覆盖clean和六种信道场景。',True)
    p('每个模型全部声明视图预测冻结后才由独立CPU连接真值评分。表中均为三种子均值±样本标准差；差值先在同一种子内相减，再对三个差值统计。没有把各包或六个场景当成独立模型重复，也不据此报告显著性。',True)
    # 4. Main detailed metrics.
    page('4  跨接收机：完整性能表')
    for metric,title in [('accuracy','准确率（%）'),('macro_f1','宏平均F1（%）')]:
        table(['网络结构','直接clean','恢复后clean','直接六景','恢复后六景'],[[NAMES[a],stat(result(a,rd='raw',view='clean',metric=metric)),stat(result(a,rd='DURA-L',view='clean',metric=metric)),stat(result(a,metric=metric)),stat(result(a,rd='DURA-L',metric=metric))] for a in ARMS],[143,91,91,91,91],title+'；三次独立训练，均值±样本标准差')
    p('六景均值不包括clean。宏平均F1先在每个模型、每个场景按全部六类计算，再在六景等权平均；它不是把三次训练的混淆矩阵合并后重新计算的F1。完整四位精度与逐种子值在附带表格中保留。',True)
    p('完整网络在clean上的恢复前后均值相同，仅表示本组总体计数保持；其他模型仍可能有小幅clean损伤。恢复后的clean、六景指标均单列，不能只报告有利的信道增益。',True)
    # 5. Paired primary differences.
    page('5  同种子配对：完整网络减结构变体')
    table(['相对完整网络的变体','直接clean','直接六景','恢复后六景'],[[NAMES[a],stat(pair(a,view='clean')),stat(pair(a)),stat(pair(a,rd='DURA-L'))] for a in ARMS[1:]],[164,114,114,115],'单位：百分点。正值表示完整网络更好，负值表示变体更好；±为配对差的样本标准差。')
    fig('supervised_paired',435,'图3  配对差。误差线对应三个配对差，不是两组标准差相加。零线两侧的结果完整保留。')
    p('完整网络在直接六景下，相对去功放、去参考响应、单视图和去路径描述的三个配对差均为正。恢复后，去路径描述和单视图两组只有1个种子仍由完整网络占优；固定打乱顺序的三个种子则全部优于完整网络。',True)
    # 6. Other generalization populations.
    page('6  日期变化与双重变化：不能外推一个结论')
    for metric,title in [('accuracy','六景平均准确率（%）'),('macro_f1','六景宏平均F1（%）')]:
        table(['网络结构','日期／直接','日期／恢复','双变／直接','双变／恢复'],[[NAMES[a],*[stat(result(a,t=t,rd=rd,metric=metric)) for t,rd in [('cross_date','raw'),('cross_date','DURA-L'),('cross_both','raw'),('cross_both','DURA-L')]]] for a in ARMS],[143,91,91,91,91],title+'；“双变”指接收机与日期同时变化')
    p('去除路径描述在跨接收机直接六景中降低0.88个百分点，但在接收机与日期同时变化的直接六景中反而高于完整网络约0.85个百分点。三视图、时间坐标、路径描述不能写成对所有泛化任务一致有效。',True)
    p('功放相关支路在本面板三种群体的直接六景均有正向作用；效应大小随群体和恢复路径变化。所有群体的clean、逐场景、逐接收机、逐发射机和逐日期指标均保留在数据包中。',True)
    # 7. Detailed environments, mean-only grid keeps the page readable.
    page('7  跨接收机：逐场景准确率')
    scene_labels=['clean','高郊区','中郊区','低郊区','高城市','中城市','低城市']
    for rd,title in [('raw','直接识别'),('DURA-L','同一DURA恢复后识别')]:
        table(['网络结构',*scene_labels],[[NAMES[a],*[f"{float(result(a,rd=rd,view=v)['mean']):.2f}" for v in SCENES]] for a in ARMS],[133,*([53.43]*7)],title+'：单位%，为三次训练均值；逐格标准差和种子值在CSV中。')
    p('“高／中／低”沿原信道场景命名表示对应质量档。前3种是郊区场景，后3种是城市场景；报告不把仿真信道结果写成真实在轨或实际现场验证。',True)
    p('不同场景可以出现相反变化。正文中的“六景平均”仅是固定等权摘要，不能用它掩盖某个具体环境的下降；全部原始精度、零差值与负差值均保留。',True)
    # 8. Scene-conditioned paired figures.
    page('8  哪些场景受结构改动影响')
    fig('scene_effects_direct',435,'图4  直接识别：完整网络减变体的逐场景差值，单位为百分点。')
    fig('scene_effects_restored',435,'图5  恢复后识别：相同定义。两图各自使用对称色标，比较数值时请读格内数字。')
    p('这些是“保留其余结构并按相同分类配方重新训练”条件下的差异，既包含结构容量改变，也包含优化轨迹改变。各格不是相互独立、可相加的分支因果贡献；本面板没有多分支全因子交互设计。',True)
    # 9. Parameters and latency.
    page('9  参数量与实测时延')
    table(['网络结构','识别器参数','参数减少','直接B1中位数','恢复B1中位数'],[[NAMES[a],next(r['loaded_parameters'] for r in params if r['arm']==a),f"{float(next(r['reduction_percent'] for r in params if r['arm']==a)):.2f}%",number(timing(a),'wall_median_ms'),number(timing(a,'restored'),'wall_median_ms')] for a in ARMS],[143,91,75,99,99],'参数统计含既有默认分类头；时延单位ms。B1表示单包批次。')
    fig('supervised_latency',410,'图6  时延误差线是五轮计时中位数的样本标准差。直接识别为常规执行；恢复后整链为优化图执行，二者不是相同执行基准。')
    p('RTX3090独占、FP32、关闭TF32；使用同一个训练种子的源IQ，避免计时访问目标。9结构×2路径×5批量×5轮＝450个计时组合。五轮是计时重复，不是五个训练种子；不得据本表声称“加恢复器会让算法变快”。',True)
    # 10. Throughput and cost.
    page('10  尾延迟、吞吐和训练成本')
    table(['网络结构','直接B1 P95','恢复B1 P95','直接B128吞吐','恢复B128吞吐'],[[NAMES[a],number(timing(a),'wall_P95_ms'),number(timing(a,'restored'),'wall_P95_ms'),number(timing(a,batch=128),'throughput_IQ_s',0),number(timing(a,'restored',128),'throughput_IQ_s',0)] for a in ARMS],[143,91,91,91,91],'P95单位ms；吞吐单位包/秒。中位数与P95从五轮全部重复合并计算。')
    cost_rows=[]
    for a in ARMS:
        vals=[float(r['cumulative_shared_worker_hours'])*60 for r in cost if r['arm']==a]
        cost_rows.append([NAMES[a],f'{statistics.mean(vals):.2f}±{statistics.stdev(vals):.2f}',f'{sum(vals)/60:.3f}'])
    table(['网络结构','每模型分段训练耗时（分钟）','三模型累计worker小时'],cost_rows,[180,190,137])
    total=sum(float(r['cumulative_shared_worker_hours']) for r in cost)
    p(f'共81个唯一训练阶段、1350个模型轮、299700次更新，记录的共享负载worker时间累计为{total:.3f}小时。每个阶段只计一次，含该阶段源验证和保存等开销；这是各worker训练分段耗时之和，不是作业排队到完成的日历时间，也不是独占硬件上的训练速度排名。',True)
    p('本表不重复测量纯DURA。恢复器的历史训练和共享IQ缓存不计入“识别器分支训练成本”；因此该数值不是端到端系统的全部研发成本。共驻留模型／图的显存基线与增量不能当成单模型峰值显存；本报告不提供未经验证的单模型显存或MACs排名。',True)
    # 11. Source diagnostic, explicit limits.
    page('11  源域表征诊断能说明什么')
    p('冻结模型在同一组720条有标签源包、90个发射机／接收机／日期单元上计算余弦距离，不拟合新的诊断分类头。比较“相同发射机、不同接收机”和“不同发射机、相同接收机”的距离，仅描述该源样本上的表示几何。')
    fig('source_geometry',430,'图7  三次模型训练的源域距离均值±样本标准差。两类距离的差并不等于目标域识别准确率。')
    selected=['flat_gain_0.5','flat_phase_pi2','relative_CFO_100000','FIR_delay3_echo02','synthetic_PA_alpha_0.1']
    labels=['幅度缩小到0.5','整体相位旋转90度','相对频偏＋100kHz','延迟3点、回波0.2','合成功放扰动0.1']
    def iv(a,name,key):
        vals=[float(r[key]) for r in interventions if r['method']=='CE_'+a and r['intervention']==name]
        return statistics.mean(vals)
    table(['源IQ干预','完整：特征距离','去功放：特征距离','完整：预测改变'],[[label,f"{iv('full',name,'feature_cosine_distance'):.4f}",f"{iv('NO_PA',name,'feature_cosine_distance'):.4f}",f"{100*iv('full',name,'default_head_flip_fraction'):.2f}%"] for name,label in zip(selected,labels)],[180,109,109,109],'示例干预；三模型均值。全部11种干预、每个模型的结果保留在CSV中。')
    p('去路径描述的源域距离间隔较大，并不使它在跨接收机目标域必然更好。合成功放干预也未被证明保持真实硬件身份，因此不能把敏感性写成已分离或准确测得了发射机功放参数；参考响应同样可能包含接收机、信道和均衡器影响。',True)
    # 12. Reusable scientific discussion.
    page('12  可用于论文的结论与边界')
    table(['模块','本面板支持的结论','不宜写成的结论'],[
        ['功放相关支路','跨接收机直接六景配对收益7.21±0.70个百分点，恢复后仍有3.61±0.84个百分点；三个种子方向均一致','已辨识或完全分离真实功放硬件参数；所有数据划分下收益相同'],
        ['参考响应支路','跨接收机直接六景收益0.58±0.04个百分点，恢复后0.74±0.35；真实模板优于固定相位扰乱模板','参考响应只有发射机信息；固定模板对照完全等容量等信息'],
        ['三视图路径描述','相对单视图，直接六景收益0.85±0.20个百分点；收益未普遍覆盖clean和恢复后设置','三个模型集成；三视图在任何环境必胜'],
        ['路径描述整路','直接跨接收机六景收益0.88±0.73个百分点；恢复后约0.16±0.72，且双变任务可反向','该支路提供普遍稳定、与其他支路可加的增益'],
        ['时间坐标／局部顺序','显式时间收益较小且不稳定；固定打乱局部顺序在跨接收机clean和恢复后六景更优','原始局部顺序必不可少；去时间坐标等于删除时域网络'],
        ['矩统计替代','跨接收机六景与路径描述完整网络的均值接近，差异随种子和路径改变','完整路径描述已在同容量、同信息条件下全面胜过矩统计']],[98,227,182])
    p('建议论文表题：PAMF网络结构在仅分类损失监督训练下的消融。表注明确：固定数据划分、50轮／11100次更新、三次模型训练、信道辅助交叉熵权重0.68、无无标签训练，差值先按种子配对；恢复后结果共用既有DURA。')
    p('可直接采用的结果段：在仅使用分类损失的监督控制中，移除功放相关支路使跨接收机六种信道平均准确率下降7.21±0.70个百分点，接入固定恢复器后仍下降3.61±0.84个百分点。移除参考响应支路的对应下降为0.58±0.04和0.74±0.35个百分点。相比之下，路径描述和显式时间坐标的效果依赖测试群体及推理路径，固定打乱局部顺序的对照在部分设置下优于完整网络。这说明功放相关表示对该控制配方的重要性较高，而现有证据不足以证明所有时序设计均是必需组件。')
    p('这些结论只对应本报告监督分类配方和标准六类划分，不自动推广到其他标签比例、接收机划分或其他训练目标。三个种子不足以支持普遍显著性声明。模型框架和公开基准存在历史选择背景，本面板是冻结后的描述性验证，不声称独立未见基准上的盲选结果；新目标结果未用于重选架构或选择性重跑。',True)
    # 13. Traceability appendix, only location of machine codes in narrative.
    page('附录  数据文件与原始记录对应')
    table(['阅读名称','原始代号'],[[NAMES[a],'CE_'+a] for a in ARMS],[285,222])
    p('本文“直接识别”对应raw；“恢复后识别”对应DURA-L，即只用有标签源集合IQ训练的既有DURA。标准接收机划分原始标识为F0。源接收机索引1/3/4/6/8，目标接收机索引0/2/5/7/9/10/11；训练日期索引1/2/3，未见日期索引0。',True)
    p('科学执行记录：20261024-phase1-PAMF-REPL-completion-m201-r01中的strict_structure组。原始run名称仅用于定位；本报告严格选择CE_开头的27个模型。模型执行提交dacafc8bfd7d3ef16e6f648cdc4491b65133ae8f；独立CPU评分入口恢复提交a79bdf385489cb3822b37434a07e3c43e5ce839a。报告生成提交不冒充模型执行提交。',True)
    table(['文件','内容'],[
        ['summary_overall.csv / per_seed_overall.csv','864行三种子汇总／2592行总体逐种子指标'],
        ['full_minus_variant.csv','完整网络减8个变体；三群、两路径、七景及六景均值、两种指标'],
        ['all_strata_per_seed.csv.gz','69984行原冻结指标；包括接收机、发射机、日期及组合'],
        ['per_stratum_summary.csv.gz / per_stratum_full_minus_variant.csv.gz','23328行分层汇总／20736行分层配对差'],
        ['net_corrections.csv.gz','恢复导致的错变对、对变错与净纠错计数'],
        ['parameters.csv / timing.csv / timing_samples.jsonl.gz','加载参数数；90个计时设置；450组原始计时重复'],
        ['source_cost.csv / source_stage_cost.csv / training_epochs.csv','27模型、81唯一阶段及1350轮训练事实'],
        ['source_geometry.csv / source_interventions.csv','源域表征和11种固定输入干预'],
        ['figures/','7幅图的PDF/SVG矢量版与PNG预览，同名CSV记录图中数值']],[226,281])
    p('本次交付包r20修正了r19源域诊断图重复写入造成的误差线问题，目标识别、配对差和时延统计保持原结果；r19保留为历史。解包后可用纯Python标准库运行reproduce/verify_report.py --root .，从逐种子指标重算汇总和配对差，并从原计时数组复核中位数、P95和吞吐；无需IQ、权重、训练环境或目标真值。本次没有重新做目标评分。',True)
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#C9D6DF'));canvas.line(44,805,551,805);canvas.setFont('Song',8);canvas.setFillColor(colors.HexColor('#627687'));canvas.drawString(44,814,'PAMF 网络分支消融 · 仅分类损失的监督训练');canvas.drawString(44,25,'2026-10-25 · 香港');canvas.drawRightString(551,25,str(doc.page))
    pdf=OUT/(TITLE+'.pdf')
    doc=SimpleDocTemplate(str(pdf),pagesize=(595.28,841.89),leftMargin=44,rightMargin=44,topMargin=49,bottomMargin=43,title=TITLE,author='Frozen experiment records')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    text='\n'.join(md).rstrip()+'\n';assert 'v2' not in text.lower();(OUT/(TITLE+'.md')).write_text(text,encoding='utf-8')
    (OUT/'README.md').write_text('''# PAMF 网络分支消融：仅分类损失的监督训练

先读同名 PDF 或 Markdown。9 种结构、3 个训练种子，共 27 个识别器。主报告仅展示直接识别和使用有标签源集合 IQ 的既有 DURA 恢复后识别。

本报告新增训练、预测、目标真值访问和恢复器训练均为 0。它不能自动代表其他训练目标或其他数据划分。所有负值、零差值和非单调结果保留。

`figures/` 含 7 幅矢量图及 PNG，`tables/` 含总体／逐种子／逐接收机／逐发射机／逐日期指标、配对差、净纠错、参数、时延、实际训练曝光和去重阶段成本。原始文件代号仅供追溯。

解包后纯标准库复算：

```text
python -X utf8 -S reproduce/verify_report.py --root .
```

复算不访问 IQ、checkpoint 或逐包目标真值，也不重新评分。新包 r20 修正了 r19 源域诊断图在重复生成时累加输入行、影响标准差的问题；主性能、配对差和时延未变。r19 保留历史，不覆盖任何旧包。
''',encoding='utf-8')
    print(json.dumps(dict(status='REPORT_WRITTEN',PDF=str(pdf),figures=7),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
