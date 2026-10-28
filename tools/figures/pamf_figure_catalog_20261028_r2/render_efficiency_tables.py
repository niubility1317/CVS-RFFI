import argparse,json,csv,statistics,math
from pathlib import Path
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,LongTable,TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4,landscape
from xml.sax.saxutils import escape
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);T=ap.parse_args().root/'tables'
static=pd.read_csv(T/'network_efficiency_static.csv').to_dict('records')
deploy=pd.read_csv(T/'network_efficiency_deployment_summary.csv').to_dict('records')
raw=pd.read_csv(T/'network_efficiency_deployment_per_seed.csv').to_dict('records')
pamf=pd.read_csv(T/'PAMF_structure_timing.csv').to_dict('records')
pdfmetrics.registerFont(TTFont('CJK','C:/Windows/Fonts/simhei.ttf'));normal=ParagraphStyle('n',fontName='CJK',fontSize=9,leading=14,spaceAfter=7);small=ParagraphStyle('s',parent=normal,fontSize=8,leading=12,spaceAfter=0);head=ParagraphStyle('h',parent=normal,fontSize=13,leading=20,spaceAfter=12)
def p(s,style=normal):return Paragraph(escape(str(s)).replace('\n','<br/>'),style)
flow=[];md=['# 常用网络效率对比表','']
def text(s,heading=False):flow.append(p(s,head if heading else normal));md.extend([('## ' if heading else '')+s,''])
def table(headers,rows,widths):
 tab=LongTable([[p(x,small) for x in headers]]+[[p(x,small) for x in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
 tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8EDF2')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),8),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#8495A6'))]));flow.append(tab);flow.append(Spacer(1,12))
 md.extend(['| '+' | '.join(headers)+' |','|'+'---|'*len(headers)]);md.extend('| '+' | '.join(map(str,row))+' |' for row in rows);md.append('')
def pm(m,s,scale=1):return f'{m*scale:.3f} ± {s*scale:.3f}'
text('常用网络效率对比表',True)
text('选取六种常见骨干并加入 PAMF 三视图骨干。以下只使用既有冻结测量，不启动新的训练、目标预测或硬件计时。输入为六类识别、B×2×256 I/Q、FP32；2D 网络未纳入本次精简表。')
text('表 1  参数与已覆盖算术量',True)
table(['网络','参数（M）','Conv/矩阵 MACs（M）','Conv/矩阵 FLOPs（M）','FP32 参数（MiB）','FFT/IFFT 调用/包'],[[r['model'],f"{r['parameters_M']:.3f}",f"{r['conv_matrix_MACs_M']:.3f}",f"{r['conv_matrix_FLOPs_M']:.3f}",f"{r['FP32_parameter_MiB']:.3f}",r['FFT_IFFT_calls']] for r in static],[190,78,112,112,107,110])
text('FLOPs 采用 1 MAC=2 FLOPs。此列只统计现有计数器覆盖的卷积/矩阵运算，不能称为总 FLOPs；不含 FFT 复数算术、归一化、逐元素运算及部分功能式操作。PAMF 的 SIG-TIME 路径代数未被该 Conv/Linear 计数器完整计入。FFT/IFFT 另列调用次数，不能把次数当作 FLOPs。参数存储仅为参数数×4 字节，不是含缓存的部署显存。')
flow.append(PageBreak())
text('表 2  常见网络的独立部署测量（RTX 3090）',True)
table(['网络','B1 P50（ms）','B1 P95（ms）','B128 均值（ms/批）','B128 吞吐（千包/s）','B128 alloc 峰值（MiB）'],[[r['model'],pm(r['B1_P50_ms'],r['B1_P50_ms_SD']),pm(r['B1_P95_ms'],r['B1_P95_ms_SD']),pm(r['B128_mean_ms'],r['B128_mean_ms_SD']),pm(r['B128_throughput_kIQ_s'],r['B128_throughput_kIQ_s_SD']),f"{r['B128_peak_alloc_MiB']:.2f}"] for r in deploy],[163,108,108,114,116,100])
text('每网络 8 个模型种子；B1 每种子 300 次请求、B128 每种子 100 次请求，预热 30 次。表中 ± 为八个模型种子各自统计量的样本 SD，不是单请求 SD。吞吐先按每种子的批量/平均请求时延求得，再汇总；B128 时延为整批时延。alloc 峰值来自独立部署进程，未把 reserved 显存混入。')
flow.append(PageBreak())
text('表 3  PAMF 结构专用计时（RTX 3090）',True)
table(['结构','B1 GPU wall median（ms）','B1 P95（ms）','B128 GPU wall median（ms）','B128 吞吐（千包/s）','B128 分配增量（MiB）'],[[r['model'],pm(r['B1_wall_median_ms'],r['B1_round_median_sd']),f"{r['B1_wall_P95_ms']:.3f}",pm(r['B128_wall_median_ms'],r['B128_round_median_sd']),f"{r['B128_throughput_IQ_s']/1000:.3f}",f"{r['B128_incremental_peak_bytes']/2**20:.3f}"] for r in pamf],[192,114,92,124,100,87])
text('专用计时使用同一冻结纯 CE 识别器权重、F-only eager、PyTorch 2.1、FP32/TF32 关闭、2 个 CPU 线程，5 个计时轮次。± 为五轮中位数的 SD，不能称为五模型种子。固定重排的三视图结构与当前 PAMF 部署骨干相符，但权重身份仍是 CE 对照，不改名为新测完整 SSL。')
text('表 2 是独立部署请求测量，表 3 是专用 GPU wall 计时；两表测量边界与重复单位不同，不根据二者的绝对数值宣称网络间加速倍数。表 3 分配增量是在共驻留测量基线之上的增量，不能与表 2 独立进程峰值直接排名。以上均为识别器 F；不把 DURA+F 系统时延混作 PAMF 单网络时延。')
text('数据文件与使用方式',True)
text('network_efficiency_static.csv 保存参数与静态计数；network_efficiency_deployment_summary.csv 保存部署均值/SD；network_efficiency_deployment_per_seed.csv 保存 96 个网络×种子×批量单元；PAMF_structure_timing.csv 保存结构专用计时。论文如需一张完全同协议的速度排行榜，应另做同硬件、同软件、同输入传输边界的统一计时；本次不伪造这种一致性。')
(T/'常用网络效率对比表.md').write_text('\n'.join(md),encoding='utf-8')
SimpleDocTemplate(str(T/'常用网络效率对比表.pdf'),pagesize=landscape(A4),leftMargin=34,rightMargin=34,topMargin=30,bottomMargin=30,title='常用网络效率对比表').build(flow)
print(json.dumps(dict(static_models=len(static),baseline_models=len(deploy),deployment_cells=len(raw),PAMF_structures=len(pamf),pdf=str(T/'常用网络效率对比表.pdf')),ensure_ascii=False))
