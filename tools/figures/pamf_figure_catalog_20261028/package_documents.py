"""Build vector atlases and separate explanations; no explanation text on figures."""
import argparse,csv,io,json,math,shutil,zipfile
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,LongTable,TableStyle,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader,PdfWriter,PageObject,Transformation

pdfmetrics.registerFont(TTFont('CJK','C:/Windows/Fonts/simhei.ttf'))
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='CN',fontName='CJK',fontSize=10,leading=16,spaceAfter=7,wordWrap='CJK'))
styles.add(ParagraphStyle(name='CNTitle',fontName='CJK',fontSize=18,leading=25,spaceAfter=16))
styles.add(ParagraphStyle(name='CNH1',fontName='CJK',fontSize=14,leading=20,spaceBefore=15,spaceAfter=8,keepWithNext=True))
styles.add(ParagraphStyle(name='CNH2',fontName='CJK',fontSize=11,leading=17,spaceBefore=12,spaceAfter=7,keepWithNext=True))
styles.add(ParagraphStyle(name='CNSmall',fontName='CJK',fontSize=8,leading=12,spaceAfter=4,wordWrap='CJK'))
def para(text,style='CN'):return Paragraph(escape(str(text).replace('\u2212','-')).replace('\n','<br/>'),styles[style])
def footer(c,d):
 c.setFont('CJK',8);c.setFillColor(colors.HexColor('#6B7782'));c.drawString(42,23,'PAMF 图表说明 | 2026-10-28');c.drawRightString(A4[0]-42,23,str(d.page))

def add_table(flow,md,headers,records,widths):
 rows=[[para(x,'CNSmall') for x in headers]]+[[para(x if x is not None else '未记录','CNSmall') for x in row] for row in records]
 tab=LongTable(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
 tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8EDF2')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#718293'))]))
 flow.append(tab);md.extend(['| '+' | '.join(headers)+' |','|'+'---|'*len(headers)])
 md.extend('| '+' | '.join(str(x) if x is not None else '未记录' for x in row)+' |' for row in records);md.append('')

def add_context(root,figs,flow,md):
 ctx=json.loads((root/'explanation_context.json').read_text(encoding='utf-8'))
 def heading(x):flow.append(para(x,'CNH1'));md.extend(['','## '+x,''])
 def note(x):flow.append(para(x));md.extend([x,''])
 heading('实验设置与数据组成')
 note('输入沿用 equalized1、中心 256 个采样点、FP32 I/Q 与单位 RMS 预处理，采样率 25 MHz。六类主设置为 ManySig；十类为 ManyRx。L 为可用身份标签的源数据，U 为隐藏 TX 标签的源数据，V 仅作源验证。RX 数字是零起始数据索引；发射机类别映射见 context_data/class_mapping.csv。')
 records=[]
 for r in ctx['current_settings']:
  source=r['source_receivers'] if r['classes']=='6' else '偶数索引 0 至 30（16 源）'
  records.append([r['name'],source,r['L'],r['U'],r['V']])
 add_table(flow,md,['当前设置','源 RX 索引','L','U','V'],records,[162,150,63,63,63])
 note('除单列历史日期诊断外，源日期为 D1、D2、D3。六类 Main、接收机重划分和标注比例设置，每景跨日 / 跨 RX / 双跨分别为 30000 / 126000 / 42000；目标两类合并 168000。十类对应为 31279 / 92597 / 30424。所有设置的目标 RX 完整列表保存在 context_data/data_settings.csv，不以相同显示名替代数据契约。')
 note('源数量比较统一目标 RX=[0,2,5,7,9,10]；各景跨 RX=108000、双跨=36000。5 源点从 Main 已冻结分层混淆矩阵精确汇总，没有新训练。1 至 6 源的自然规模同时改变 L/U/V 数量，不能解释为固定总数据量下只改变源多样性。旧 T1 的 fixed 与 natural 两组另列。1%、5%、10% 是源训练池的标注比例，V 固定；10% 精确复用 Main。')
 note('主要模型预算：网络架构层 E200，八个模型种子；当前 PAMF 主方法和结构层 E50，通常三种子。完整 PAMF 的实际日程为 20+10+20=50 轮、11100 更新：前 13 轮仅 L，随后 7 轮普通 PL，后 30 轮完整 SSL；后 37 轮均联合 L/U。L 批量 128，U 批量 256。监督版与普通 PL 保留同一有标签骨架，普通 PL 阈值 0.95、权重 1。外部方法保留其原生学习目标和原登记预算，不能因图上并列就声称所有训练成本相同。')
 note('外部方法的登记预算为 RIEI / DRIFT 各 E100，IB-RFF / SDG 各 E200；监督和普通 PL 臂各自保持其原生日程。相关记录见 context_data/training_budget_records.csv。这里的轮数不能替代按方法不同执行步骤计算的更新数或实测成本。')
 note('完整 PAMF 在 t2 局部特征窗口内使用固定排列 [0,5,3,6,2,1,4,7]，训练和推理一致；原 IQ 不作全局打乱。三个 SIG-TIME 视图独立学习同一份中层特征的描述。六类三视图 / 单视图 / 去 SIG-TIME 识别器分别为 232993 / 196001 / 177025 参数；完整 DURA 为 266951 参数。')
 note('Tri-SIG-TIME SSL 使用学生伪标签、跨 RX 身份保护配对、浅中层风格扰动、梯度约束、融合特征对比与接收机特征外推。完整阶段 U 权重 6，没有 EMA 教师。其与普通 PL 的组合差还包含时序重排，因此主比较不能单独证明某一个 SSL 子机制的作用。自然序 REPL 的内部消融仅为该历史版本的证据。')
 note('纯 CE 结构层仅真实标签 CE，U、KL 和身份对比关闭；其预登记有标签信道 CE 槽权重 0.68 保留。PAMF-Sup 则保留跨 RX 身份对比、ordinary/raw 成对 KL、As 真实标签 CE、LS0.01，As 独立 KL 关闭。这是两种不同监督条件。结构层的 Fixed permutation 与完整 SSL 层的 w/o permutation 方向相反，逐图分别注明。')
 heading('历史代号与复用边界')
 note('T1 是历史识别器配方，不是数据划分：E30 再续训 20 轮到 E50，累计 13L+37L/U，去 As 独立 KL，但保留当时自身 EMA、筛选门控和 U 熵。F0 是 Main split 的旧名；F1/F2 是各历史矩阵内部划分编号；F01 至 F21 是 T1 细项消融编号。图中已用语义名称替换，旧码映射见 context_data/historical_alias_codebook.csv。')
 note('历史 T1 有 17 个数据契约。下表的 L/U/V 与目标包数来自冻结实验登记，仅说明这些历史行的组成，不能用作当前重划分的别名。每行准确率仍使用该行实际评分人群；chrono 是日期诊断，不并入主泛化结论。')
 rec=[]
 names={'F0':'Main split','F1':'Legacy RX split A','F2':'Legacy RX split B','ManyRx':'10-class','chrono':'Chrono diagnostic','label01':'1% labels','label05':'5% labels'}
 for r in ctx['historical_contracts']:
  source=r['source_receivers'];source='偶数索引 0 至 30' if len(source)>55 else source
  rec.append([names.get(r['legacy_protocol_id'],r['legacy_protocol_id']),source,' / '.join(str(r[k]) for k in ['L','U','V']),r['target_count']])
 add_table(flow,md,['历史设置','源 RX 索引','L / U / V','登记目标包数'],rec,[153,135,140,73])
 note('DURA 复用须对应相同源 RX、日期和 L/U 权限。Main 的 5 源 G 不用 6 源 G 代替。监督+G-L 与 PL+G-LU 的直接差包含恢复器权限变化；只有固定同一 G 时才可讨论识别器用 U 的净差。同排列监督/完整 SSL 的四格、自然序监督/PL 的四格分别呈现。DURA 合同核对表随包提供。')
 heading('指标、作图数据与筛选方式')
 note('准确率、Macro-F1、逐类 precision/recall/F1 均按百分数显示。混淆矩阵行为真实类、列为预测类，按真实类归一化后以模型种子汇总。六景平均是场景等权而非重新合并包数的平均。配对差先对相同种子相减，再求均值和样本 SD；误差线不等于统计显著性。')
 note('正负差用绿色 / 暖红色区分，形状或纹理同时区分条件；颜色不单独承载实验身份。CSV 保存未四舍五入的作图数值。seed_n 是实际重复数：通常为模型种子数，计时图则为计时轮数，单次固定参数计数不冒充重复实验。')
 note('逐图 _plot.csv 保存实际画出的每格数值，_seeds.csv 保存可恢复的逐点值。部分历史汇总只保存顺序值，未附种子 ID，此时 model_seed 留空、replicate_index 表示原顺序；不能据此重建跨实验配对。历史汇总缺少逐点值时，只提供其实际均值和 SD。单视图无增强补充只有 clean；不把未测 practical 填为零。')
 note('当前主方法全集包含 11 个设置和完整三种子读出；专门的视图对比为八种子 raw 面板。不能把主矩阵三种子 DURA 扩写成八种子。所有并列图是候选表达，不是预先选定投稿主图；可按 Fig 编号筛选，第一版 46 张编号保持。')
 from collections import Counter
 groups=Counter(f['group'] for f in figs)
 note('候选图覆盖：'+'；'.join(k+' '+str(v)+' 张' for k,v in groups.items())+'。')
def build_atlas(root,figs,dest):
 writer=PdfWriter()
 for i,f in enumerate(figs,1):
  page=PdfReader(root/'pdf'/(f['id']+'.pdf')).pages[0];w=float(page.mediabox.width);h=float(page.mediabox.height)
  blank=PageObject.create_blank_page(width=w,height=h+24);blank.merge_transformed_page(page,Transformation().translate(0,24))
  buf=io.BytesIO();c=canvas.Canvas(buf,pagesize=(w,h+24));c.setFont('Helvetica',7);c.setFillColor(colors.HexColor('#606A74'));c.drawString(12,8,f'Page {i} | '+f['id']);c.save();buf.seek(0);blank.merge_page(PdfReader(buf).pages[0]);writer.add_page(blank);writer.add_outline_item(f"{i}. {f['title']}",i-1)
 writer.add_metadata({'/Title':'PAMF figure atlas','/Author':'PAMF research'});writer.write(dest)
 assert len(PdfReader(dest).pages)==len(figs)
def example(f):
 p=f['panels'][0];rows=p['rows'];chosen=[r for r in rows if str(r.get('category','')) in ['PAMF','PAMF+DURA','Full','Main split'] or str(r.get('series','')) in ['PAMF','PAMF+DURA']][:3]
 if not chosen and p['kind'] not in ['heatmap']:chosen=rows[:2]
 if not chosen:return '完整数值见本图的逐图 CSV；热图各格保留原始精度，不从图片读取四舍五入后的数值。'
 a=[]
 for r in chosen:
  unit='%' if p['unit']=='percent' else ' 个百分点' if p['unit']=='delta' else ' '+p['unit'];name=' / '.join(str(r[k]) for k in ['category','series'] if r.get(k));s=f"{float(r['estimate']):.4f}"+(f" ± {float(r['sd']):.4f}" if r.get('sd') is not None else '')+unit;a.append(name+'：'+s)
 return '图 a 的数值示例：'+'；'.join(a)+'。'
def write_explanation(root,figs,out,selected):
 title='PAMF 已选图表更新第一版：统一说明' if selected else 'PAMF 全部正式实验图表：统一说明'
 general=[
 '本说明与图集分开。图内仅保留坐标、图例、分面名称和必要数值；实验设置、结果解读、来源和限制集中在本文件。',
 f'本版包含 {len(figs)} 张图。图集每页一图，PDF 书签为中文标题；页面底部编号与逐图 CSV、SVG、PNG、PDF 文件名一致。已选主题的 Fig001 至 Fig046 在全集中保持相同编号。',
 '当前完整 PAMF = 三视图 SIG-TIME + 固定窗口时序重排 + Tri-SIG-TIME SSL。PAMF-Sup 保留完整有标签目标而关闭 U；PAMF-PL 仅增加阈值 0.95、权重 1 的普通伪标签。没有 PAMF-PL (Permuted) 实验。',
 '常见网络比较使用 E200 纯 CE、无 U、无普通/信道增强，八种子；E50 纯 CE 结构层保持其原信道 CE 槽，无 U；完整 SSL 结构层使用最终方法，三种子。三类训练设置分别解释。',
 '六景平均不含 clean：先在每个模型种子内对六个 practical 场景等权平均，再跨种子求均值和样本标准差（ddof=1）。误差线不是置信区间。存在逐点计时重复时，以该图专门说明的轮次为准。',
 '跨接收机为目标 RX 的 D1-D3；双跨为目标 RX 的 D0；跨日为源 RX 的 D0。D0 是更早未见日期，不声称未来日期预测。目标两类综合 = 跨接收机 + 双跨，三类综合另加跨日，二者不互换。',
 'DURA-L 只用合法 L IQ；DURA-LU 用合法 L+U IQ；均不读 TX 标签。普通监督系统优先配 G-L，半监督系统配 G-LU；固定 G 的权限分析另列。通常一个 G 在同一数据设置内跨 F 种子共享。',
 '所有图来自已完成并冻结的评分统计。没有新增模型训练、目标预测或逐包真值访问。历史 T1、自然序 REPL 和旧计时明确标为历史，不改名为当前完整 PAMF。探索研发候选的逐版搜索不纳入。',
 'source_data/*_plot.csv 是实际作图数值，含面板、类别、系列、均值、标准差和实际重复数；*_seeds.csv 保存图中逐种子点（适用时）。每图的数据文件独立对应，空缺保持未测或未记录，不补零。'
 ]
 flow=[para(title,'CNTitle')];md=['# '+title,''];flow.extend(para(x) for x in general);md+=general
 if not selected:add_context(root,figs,flow,md)
 md+=['','## 图表索引','']
 rows=[[para('图号 / 图集页','CNSmall'),para('实验图表','CNSmall'),para('分类','CNSmall')]]
 for i,f in enumerate(figs,1):rows.append([para(f"{f['id'].split('_')[0]} / {i}",'CNSmall'),para(f['title'],'CNSmall'),para(f['group'],'CNSmall')]);md.append(f"- {f['id']}（图集第 {i} 页）：{f['title']}")
 flow += [para('图表索引','CNH1')]
 tab=LongTable(rows,colWidths=[88,330,85],repeatRows=1,hAlign='LEFT');tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8EDF2')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#718293'))]));flow.append(tab)
 flow+=[PageBreak(),para('逐图实验设置与说明','CNH1')];md+=['','## 逐图说明','']
 for i,f in enumerate(figs,1):
  label=f['id'].split('_')[0]+' '+f['title'];flow.append(para(label,'CNH2'));md+=['','### '+label,'']
  text=[f'对应图集第 {i} 页；实验分类：{f["group"]}。',f['description']]
  if f.get('scope'):text.append('测试口径：'+f['scope']+'。')
  text.append(example(f))
  if f.get('finding'):text.append('结果解读：'+f['finding'])
  if f.get('caveat'):text.append('解释边界：'+f['caveat'])
  for j,p in enumerate(f['panels']):
   n=sorted(set(r.get('seed_n') for r in p['rows'] if r.get('seed_n') is not None));ns='、'.join(str(x) for x in n);text.append(f'面板 {chr(97+j)}：{p["title"] or "按坐标与图例分组"}；{len(p["rows"])} 条作图记录'+(f'；各格实际重复数 {ns}' if ns else '')+'。')
  text.append('对应数据：source_data/'+f['id']+'_plot.csv'+('；另有逐种子数据。' if (root/'source_data'/(f['id']+'_seeds.csv')).exists() else '。'))
  flow.extend(para(t,'CNSmall' if t.startswith('对应数据') else 'CN') for t in text);md+=text
 name='第一版图表说明' if selected else '全部图表说明'
 SimpleDocTemplate(str(out/(name+'.pdf')),pagesize=A4,leftMargin=43,rightMargin=43,topMargin=42,bottomMargin=40,title=title,author='PAMF research').build(flow,onFirstPage=footer,onLaterPages=footer)
 (out/(name+'.md')).write_text('\n\n'.join(md)+'\n',encoding='utf-8')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--selected',action='store_true');a=ap.parse_args();root=a.root;cat=json.loads((root/'catalog.json').read_text());figs=[f for f in cat['figures'] if not a.selected or f['selected']];out=root/'selected_v1' if a.selected else root
 out.mkdir(exist_ok=True)
 if a.selected:
  for sub in ['pdf','svg','png','source_data']:(out/sub).mkdir(exist_ok=True)
  for f in figs:
   for ext in ['pdf','svg','png']:shutil.copyfile(root/ext/(f['id']+'.'+ext),out/ext/(f['id']+'.'+ext))
   for p in (root/'source_data').glob(f['id']+'_*'):shutil.copyfile(p,out/'source_data'/p.name)
  (out/'catalog.json').write_text(json.dumps(dict(stage='selected_v1',figures=figs),ensure_ascii=False),encoding='utf-8')
 filename='已选图表_更新第一版.pdf' if a.selected else '全部正式实验_图表候选全集.pdf'
 build_atlas(root,figs,out/filename);write_explanation(root,figs,out,a.selected)
 with (out/'图表索引.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['figure_id','title','group','atlas_page','selected','plot_data','seed_data']);w.writeheader()
  for i,g in enumerate(figs,1):w.writerow(dict(figure_id=g['id'],title=g['title'],group=g['group'],atlas_page=i,selected=g['selected'],plot_data='source_data/'+g['id']+'_plot.csv',seed_data='source_data/'+g['id']+'_seeds.csv' if (root/'source_data'/(g['id']+'_seeds.csv')).exists() else ''))
 if a.selected:
  target=root.parent/'PAMF_已选图表_更新第一版_20261028.zip'
  with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
   for p in out.rglob('*'):
    if p.is_file():z.write(p,str(Path('PAMF_selected_v1')/p.relative_to(out)))
  print(json.dumps(dict(figures=len(figs),atlas=str(out/filename),explanation=str(out/'第一版图表说明.pdf'),archive=str(target),archive_MB=target.stat().st_size/1e6),ensure_ascii=False))
 else:print(json.dumps(dict(figures=len(figs),atlas=str(out/filename),explanation=str(out/'全部图表说明.pdf')),ensure_ascii=False))
if __name__=='__main__':main()
