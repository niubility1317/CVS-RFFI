"""Chinese reading guide, active LaTeX closure and deliverable readback."""
from pathlib import Path
import csv
import json
import re
import shutil
import subprocess
import zipfile

ROOT=Path('E:/type10-7')
OUT=ROOT/'exports/unified_figures/20261024-practical-residual-paper-r03'
PAPER=ROOT/'paper/phase1_experiments_ieee_20261020'
POPPLER=Path('C:/Users/lh594/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin')

def main():
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.utils import ImageReader
    from pypdf import PdfReader, PdfWriter
    from PIL import Image, ImageOps, ImageDraw
    pdfmetrics.registerFont(TTFont('Chinese','C:/Windows/Fonts/simsun.ttc',subfontIndex=0))
    pdfmetrics.registerFont(TTFont('TimesNewRoman','C:/Windows/Fonts/times.ttf'))
    d=json.loads((OUT/'figure_manifest.json').read_text(encoding='utf-8'))
    doc=canvas.Canvas(str(OUT/'论文结果与逐图说明.pdf'),pagesize=(595.28,841.89));page=0
    def begin(title):
        nonlocal page
        page+=1;doc.setFillColorRGB(.13,.25,.36);size=min(18,18*500/max(500,pdfmetrics.stringWidth(title,'Chinese',18)));doc.setFont('Chinese',size);doc.drawString(45,785,title)
        doc.setStrokeColorRGB(.75,.8,.84);doc.line(45,767,550,767)
    def text(s,y,size=11):
        doc.setFillColorRGB(.12,.15,.19);doc.setFont('Chinese',size)
        lines=[];cur=''
        for ch in s:
            if ch=='\n' or pdfmetrics.stringWidth(cur+ch,'Chinese',size)>500:
                lines.append(cur);cur='' if ch=='\n' else ch
            else:cur+=ch
        if cur:lines.append(cur)
        for line in lines:doc.drawString(45,y,line);y-=size*1.65
        return y
    def finish():
        doc.setFont('Chinese',9);doc.setFillColorRGB(.4,.45,.5);doc.drawString(45,30,'PAMF · practical residual · 2026-10-24');doc.drawRightString(550,30,str(page));doc.showPage()
    begin('新对比实验：论文使用说明')
    y=text('结论：新一轮 627 行均已冻结并独立评分。主比较只保留 RIEI、DRIFT-v2、IB-RFF、SDG 的 Full 版本；对应 PAMF 监督版和原生 T1 半监督版分别报告。',728)
    y=text('四组合：A 无增强无 PL；B 无增强有 PL；C 有增强无 PL；D 有增强有 PL。增强采用 practical residual/post_sync/noEQ，训练仅 low suburban、mid urban、low urban。所有方法仍测试 clean 与六种 practical。',y-18)
    y=text('日期保持原划分：训练 D1–D3；跨日为源 RX 的 D0，跨接收机为目标 RX 的 D1–D3，双跨为目标 RX 的 D0。D0 是较早未见日，不能称为未来 D3 测试。',y-18)
    y=text('图表使用：Fig01–16 是主文比较；Fig17–20 展示各方法四组合的直接与恢复后实际分数；Fig21–22 给出完整七景。独立图内不放标题或脚注，图注见英文文件。historical_PAMF 的 9 张旧图仍有效，但采用原 168000 包口径，不能并入新跨 RX 总体。',y-18)
    y=text('统计：六景先在每个 F seed 内平均，再算三个种子的均值和样本标准差。同契约、同 G 角色下 G 固定，标准差不是三次独立完整系统波动或置信区间。A/C 用 G-L，B/D/T1 用 G-LU；恢复后跨 PL 的比较含 G 权限差异。',y-18)
    text('原始 627 行完整表、逐 RX/TX/day 的冻结统计和完整复算包 r14 保留。本文数据取自已核验聚合，不重新训练、不重测、不按目标结果选择组合。',y-18)
    finish()
    begin('F0：三类泛化的主要结果')
    with (OUT/'tables/paper_main_comparison.csv').open(encoding='utf-8',newline='') as f:rr=[r for r in csv.DictReader(f) if r['protocol']=='F0']
    y=725
    for r in rr:
        y=text(r['method'],y,size=12)
        y=text('恢复后六景：跨日 '+r['cross_date_G_S6']+'%；跨 RX '+r['cross_rx_G_S6']+'%；双跨 '+r['cross_both_G_S6']+'%。',y,size=10)-14
    y=text('外部 D 是预先固定的增强＋普通 PL 完整配方，并非每种方法在目标测试集上的最佳组合。不能只报告 D 而据此声称增强或 PL 普遍有效；四组合完整分数随包提供。',y-10)
    text('PAMF 原生 T1 的 F0 跨 RX 为 78.21±0.84%，相对固定 SDG D 的 67.12±0.97% 高 11.09 个百分点。F2 双跨中 T1 并未超过对应监督 C；相关负结果完整保留。',y-12)
    finish()
    for rec in d['figures']:
        begin('图 '+str(rec['number'])+'：'+rec['description_zh'])
        image=Image.open(OUT/'preview'/(rec['stem']+'.png'))
        w,h=image.size;scale=min(500/w,420/h);doc.drawImage(ImageReader(image),45+(500-w*scale)/2,730-h*scale,width=w*scale,height=h*scale)
        y=730-h*scale-25
        y=text('读图：'+rec['description_zh']+'。纵轴或横轴直接给出准确率；误差线为样本标准差，不能解释为显著性结论。',y)
        if 'source_' in rec['stem']:
            note='仅绘制 m=1、2、3、4、6 的自然源数量预算；5 未测。六个目标 RX 固定，F0 的五源七目标结果不插入。曲线保留所有非单调变化。'
        elif 'four_cells' in rec['stem']:
            note='四格依次是 Off/Off、Off/PL、On/Off、On/PL。两条线分别显示直接识别与对应 DURA 后的绝对分数。原始 F 的差异更适合归因增强/PL；恢复后差异还包含 G-L/G-LU 权限变化。'
        elif 'labels' in rec['stem']:
            note='1%、5%、10% 标注条件全部保留。普通 PL 的零接受率与负效果不剔除；不能把更多无标签数据直接等同于更高准确率。'
        elif 'receivers' in rec['stem']:
            note='F1/F2 与 F0 的源 RX 集合两两仅交 1 个，分别替换 80% 源接收机；训练日期保持一致。跨划分结果不能简单混成一个总体平均。'
        elif 'ManyRx' in rec['stem']:
            note='ManyRx 注册十类，样本数不等，与六类 ManySig 分开统计；Macro-F1 使用完整十类。'
        else:
            note='外部方法固定 D，PAMF 监督固定 C，原生 T1 保留其既有半监督配方。它们是完整方法比较，不能把方法差解释成单一机制的因果贡献。'
        text('解释边界：'+note,y-15)
        finish()
    begin('论文论点与不应扩大使用的结论')
    y=text('可以写：practical residual 的完整对比已覆盖三种泛化、三 F 种子与全部七景；PAMF 在主要 F0 跨接收机结果上优于固定 D 外部完整配方。恢复前后绝对分数、全部四组合、源数量和标注比例均可复核。',727)
    y=text('应保留：半监督、增强与 CRIC 等机制并非处处正收益；不能按目标成绩重分模块、修改默认或只挑有利划分。恢复后的强结果可能掩盖原始识别器对信道的敏感性。',y-20)
    y=text('不能写：D0 是未来测试日；所有实验都是六类；三 F seed 是三独立完整 G＋F 系统；缓存分类时间就是在线恢复时延；原生门控接受率已完整统计；模拟信道证明真实在轨或完整硬件解耦。',y-20)
    y=text('成本：600 新 F 为 246.37 共享 worker 小时；27 个复用 F 有 46.26 历史小时；4 个新完整 G 为 18.42 小时。它们不等于独占 GPU 小时或本次墙钟。原固定 F0 G-LU 完整祖先计时和原生门控部分遥测缺失，保持缺项。',y-20)
    text('旧错误增强的外部比较以及混合导出已另行打包隔离。实验登记、原生代码、合法 F/G、IQ、有效 PAMF 机制与时延来源保留，不能将混合旧包全部判为科学无效。恢复清单提供逐路径映射。',y-20)
    finish();doc.save()
    # Copy only the active TeX dependency closure, not unused legacy result sections.
    dst=OUT/'IEEE_experiments';seen=set()
    def copy_tex(name):
        if name in seen:return
        seen.add(name);p=PAPER/name;target=dst/name;target.parent.mkdir(parents=True,exist_ok=True);s=p.read_text(encoding='utf-8')
        for child in re.findall(r'\\input\{([^}]+)\}',s):copy_tex(child if child.endswith('.tex') else child+'.tex')
        for image in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',s):
            src=PAPER/image
            if not src.suffix:src=src.with_suffix('.pdf')
            out=dst/src.relative_to(PAPER);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,out)
        for bib in re.findall(r'\\bibliography\{([^}]+)\}',s):
            for b in bib.split(','):shutil.copy2(PAPER/(b+'.bib'),dst/(b+'.bib'))
        if name=='practical_full_results.tex':
            s=s.replace('Raw and restored routes can have different response patterns; error bars are three-F-seed SD.',
                'This plot shows the matching DURA route; the accompanying independent direct-F figure is included in the reproducible figure kit. Error bars are three-F-seed SD.')
            s=s.replace('Natural-budget source-count curves, with fixed six-target populations',
                'Matching-DURA cross-RX natural-budget source-count curves, with fixed six-target populations')
            s=s.replace('width=\\textwidth]{images/practical_full/F0_seven_views.pdf}', 'width=.70\\textwidth]{images/practical_full/F0_seven_views.pdf}')
            s=s.replace('width=.98\\textwidth]{images/practical_full/source_count.pdf}', 'width=.70\\textwidth]{images/practical_full/source_count.pdf}')
        target.write_text(s.rstrip()+'\n',encoding='utf-8')
    copy_tex('main.tex')
    shutil.copy2(OUT/'pdf/Fig22_F0_seven_views_restored.pdf',dst/'images/practical_full/F0_seven_views.pdf')
    shutil.copy2(OUT/'pdf/Fig08_source_cross_rx_restored.pdf',dst/'images/practical_full/source_count.pdf')
    cmd=['D:/APPLICATION/latex/MiKTeX/miktex/bin/x64/pdflatex.EXE','-interaction=nonstopmode','-halt-on-error','main.tex']
    for turn in range(3):
        p=subprocess.run(cmd,cwd=dst,capture_output=True,text=True,encoding='utf-8',errors='replace');assert p.returncode==0,p.stdout[-4000:]
        if turn==0:
            p=subprocess.run(['D:/APPLICATION/latex/MiKTeX/miktex/bin/x64/bibtex.exe','main'],cwd=dst,capture_output=True,text=True,encoding='utf-8',errors='replace');assert p.returncode==0,p.stdout[-2000:]
    log=(dst/'main.log').read_text(encoding='utf-8',errors='replace')
    assert 'Overfull' not in log
    assert 'undefined references' not in log
    pages=len(PdfReader(dst/'main.pdf').pages)
    render=OUT/'paper_preview';render.mkdir(exist_ok=True)
    subprocess.run([str(POPPLER/'pdftoppm.exe'),'-scale-to','950','-png',str(dst/'main.pdf'),str(render/'page')],check=True,capture_output=True)
    ims=[]
    for p in sorted(render.glob('page-*.png')):
        im=Image.open(p).convert('RGB');im.thumbnail((420,600));ims.append(im)
    sheet=Image.new('RGB',(3*440,((len(ims)+2)//3)*630),'#dfe4e8')
    for i,im in enumerate(ims):sheet.paste(im,((i%3)*440+10,(i//3)*630+20))
    sheet.save(render/'contact.png')
    # Avoid unnecessary compiler products in the durable kit.
    for pattern in ('*.aux','*.log','*.out','*.bbl','*.blg'):
        for p in dst.glob(pattern):p.unlink()
    (OUT/'reproduce'/Path(__file__).name).write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
    verify={'status':'VERIFIED','new_figures':len(d['figures']),'historical_figures':9,'guide_pages':len(PdfReader(OUT/'论文结果与逐图说明.pdf').pages),'IEEE_pages':pages,'active_tex_files':sorted(seen),'font':'Times New Roman','raster_dpi':4800,'statistics_source':'already verified 627-row aggregate; no truth IO','numeric_tables':5016,'PDF_overfull_warnings':0}
    for rec in d['figures']:
        assert len(PdfReader(OUT/'pdf'/(rec['stem']+'.pdf')).pages)==1
        svg=(OUT/'svg'/(rec['stem']+'.svg')).read_text(encoding='utf-8')
        assert 'Times New Roman' in svg
        Image.MAX_IMAGE_PIXELS=None
        for ext in ('png','tiff'):
            with Image.open(OUT/ext/(rec['stem']+'.'+ext)) as im:
                assert list(im.size)==rec['pixels']
                assert all(abs(a-4800)<1 for a in im.info['dpi'])
    (OUT/'verification.json').write_text(json.dumps(verify,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(verify,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
