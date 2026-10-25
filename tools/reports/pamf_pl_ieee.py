"""Build a Chinese IEEEtran report from the verified, grouped narrative and vectors."""
import re,shutil,sys
from pathlib import Path
ROOT=Path('E:/type10-7');sys.path.insert(0,str(ROOT))
from tools.reports.pamf_pl_comparison import OUT

def tex(text):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}','±':r'\ensuremath{\pm}','−':'-','→':r'\ensuremath{\rightarrow}'}
    parts=[];at=0
    for m in re.finditer(r'[A-Za-z0-9_./-]{18,}',text):
        parts.append(''.join(mapping.get(c,c) for c in text[at:m.start()]));parts.append(r'\nolinkurl{'+m.group(0)+'}');at=m.end()
    parts.append(''.join(mapping.get(c,c) for c in text[at:]));return ''.join(parts)

def main():
    target=OUT/'ieee';target.mkdir(exist_ok=True)
    lines=(OUT/'PAMF与四种对比方法实验报告.md').read_text(encoding='utf-8').splitlines()
    preamble=r'''\documentclass[10pt,journal]{IEEEtran}
\usepackage{fontspec,xeCJK}
\setmainfont{Times New Roman}
\setsansfont{Arial}
\setmonofont{Consolas}
\setCJKmainfont{SimSun}[BoldFont=SimHei,ItalicFont=KaiTi,BoldItalicFont=SimHei]
\setCJKsansfont{SimHei}
\usepackage{amsmath,amssymb,graphicx,booktabs,array,tabularx,url,xurl,placeins,stfloats}
\urlstyle{same}
\usepackage[hidelinks]{hyperref}
\renewcommand{\abstractname}{摘要}
\renewcommand{\IEEEkeywordsname}{关键词}
\renewcommand{\tablename}{表}
\renewcommand{\figurename}{图}
\renewcommand{\refname}{参考记录}
\renewcommand{\arraystretch}{1.16}
\setlength{\tabcolsep}{3pt}
\setlength{\textfloatsep}{10pt plus 2pt minus 2pt}
\setlength{\dbltextfloatsep}{10pt plus 2pt minus 2pt}
\setlength{\floatsep}{8pt plus 2pt minus 1pt}
\renewcommand{\topfraction}{.92}
\renewcommand{\dbltopfraction}{.92}
\renewcommand{\textfraction}{.06}
\renewcommand{\floatpagefraction}{.8}
\renewcommand{\dblfloatpagefraction}{.8}
\setcounter{topnumber}{4}
\setcounter{dbltopnumber}{4}
\setcounter{totalnumber}{8}
\makeatletter
\setlength{\@fptop}{0pt}
\setlength{\@fpsep}{10pt}
\setlength{\@fpbot}{0pt plus 1fil}
\setlength{\@dblfptop}{0pt}
\setlength{\@dblfpsep}{10pt}
\setlength{\@dblfpbot}{0pt plus 1fil}
\makeatother
\clubpenalty=10000\widowpenalty=10000
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\begin{document}
\title{PAMF与对比方法的分组实验分析}
\author{实验结果报告\quad 2026年10月25日}
\maketitle
\begin{abstract}
本报告按监督与普通PL半监督分别分析PAMF和四种完整对比方法。每组均列RIEI、DRIFT、IB-RFF、SDG、PAMF及DURA+PAMF，前五项直接识别。外部方法接入DURA另作泛用性验证，并以同模型接入前后双柱表示准确率变化。无信道增强只报告目标CLEAN，有增强报告CLEAN及六种practical。全部594个逻辑模型中，监督与半监督各297个；每组主比较165个独立识别器，展示198个模型与推理链组合。两组的匹配恢复器使用各自允许的源IQ，不将跨组完整系统差全部归因于伪标签。本次没有新增训练、预测或目标真值访问。
\end{abstract}
\begin{IEEEkeywords}
射频指纹识别，跨接收机泛化，普通伪标签，信道增强，信号恢复，分组评估。
\end{IEEEkeywords}
'''
    body=[];i=0;section=0;pending=[];table_no=0;fig_no=0
    def paragraph(s):
        if s:body.append(tex(s)+'\n')
    def flush():
        if pending:paragraph(' '.join(pending));pending.clear()
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('# '):i+=1;continue
        if line.startswith('## '):
            flush();label=line[3:]
            # Do not let main-results floats leak into the following experiment class.
            if label in ('监督实验结果','普通PL半监督实验结果'):body.append(r'\clearpage')
            elif label.startswith(('附录','成本')):body.append(r'\FloatBarrier')
            if label.startswith('附录'):
                body.append(r'\appendices');body.append(r'\section{阅读名称与原始记录的对应}')
            elif '（续）' in label:body.append(r'\subsection{'+tex(label.split('  ',1)[-1])+'}')
            else:
                section+=1;body.append(r'\section{'+tex(re.sub(r'^\d+\s+','',label))+'}')
            i+=1;continue
        if line.startswith('!['):
            flush();m=re.fullmatch(r'!\[(.*)\]\((.*)\)',line);caption=m.group(1);stem=Path(m.group(2)).stem
            fig_no+=1
            wide=any(stem.endswith('_'+x) for x in ('receivers','labels','sources','ten_class','scenes','population','dura_all_settings'));env='figure*' if wide else 'figure';width=r'\textwidth' if wide else r'\columnwidth'
            body.extend([r'\begin{'+env+r'}[!tbp]',r'\centering',r'\includegraphics[width='+width+']{../figures/'+stem+'.pdf}',r'\caption{'+tex(caption)+'}',r'\label{fig:grouped-'+str(fig_no)+'}',r'\end{'+env+'}'])
            i+=1
            while i<len(lines) and not lines[i].strip():i+=1
            if i<len(lines) and lines[i].strip()==caption:i+=1
            continue
        if line.startswith('|'):
            # The immediately preceding standalone paragraph is the authored caption.
            caption=pending.pop()[2:] if pending and pending[-1].startswith('表：') else '分组实验结果'
            flush();raw=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                rr=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',x) for x in rr):raw.append(rr)
                i+=1
            heads=raw[0];data=raw[1:];n=len(heads);table_no+=1
            body.append('表~'+r'\ref{tab:grouped-'+str(table_no)+'}~'+tex(caption)+'。\n')
            wide=n>=5 or any(len(c)>22 for r in raw for c in r) or (n==4 and len(data)>3)
            if n<=3:wide=False
            env='table*' if wide else 'table';width=r'\textwidth' if wide else r'\columnwidth'
            body.extend([r'\begin{'+env+(r'}[!tbp]' if wide else r'}[!htbp]'),r'\caption{'+tex(caption)+'}',r'\label{tab:grouped-'+str(table_no)+'}',r'\centering',r'\small' if wide else r'\footnotesize'])
            if n>=5 and heads[0] in ('测试设置','测试场景','十类数据端点'):
                spec=r'>{\hsize=1.4\hsize}Y' + r'>{\hsize=.933333\hsize}Y'*(n-1) if n==7 else 'Y'*n
            else:spec='Y'*n
            body.extend([r'\begin{tabularx}{'+width+'}{'+spec+'}',r'\toprule',' & '.join(r'\textbf{'+tex(c)+'}' for c in heads)+r' \\',r'\midrule'])
            body.extend(' & '.join(tex(c) for c in r)+r' \\' for r in data)
            body.extend([r'\bottomrule',r'\end{tabularx}',r'\end{'+env+'}'])
            continue
        if line.startswith('```'):
            flush();i+=1
            while i<len(lines) and not lines[i].startswith('```'):paragraph(lines[i]);i+=1
            i+=1;continue
        pending.append(line)
        # Keep one paragraph buffered so a following table can consume its caption.
        if i+1<len(lines) and not lines[i+1].strip():
            j=i+2
            while j<len(lines) and not lines[j].strip():j+=1
            if j>=len(lines) or not lines[j].strip().startswith('|'):flush()
        i+=1
    flush();body.extend([r'\FloatBarrier',r'\end{document}'])
    path=target/'main.tex';path.write_text(preamble+'\n'.join(body)+'\n',encoding='utf-8')
    (target/'README.md').write_text('''# IEEE 双栏实验报告

`main.tex` 使用 IEEEtran journal、10 pt、双栏，英文为 Times New Roman，中文为宋体。图表使用独立矢量 PDF。本文为中文实验报告，未虚构作者单位、投稿状态或文献。

使用已安装 IEEEtran、xeCJK 的原生 XeLaTeX：

```text
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

完整结果在附录之前；无增强目标域 CLEAN、有增强 CLEAN＋practical、DURA 泛用性、PAMF 自身对照分别成节。机器可读表与标准库核验位于上一层目录。
''',encoding='utf-8')
    print('IEEE source written:',path,'tables=',table_no,'figures=',fig_no,flush=True)
if __name__=='__main__':main()
