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
本报告分析已冻结且经独立评分的PAMF与四种完整对比方法结果。主比较列为RIEI、DRIFT、IB-RFF、SDG、PAMF及DURA+PAMF；前五项直接识别，仅最后一项接入恢复器。其他方法接入DURA单独用于恢复网络泛用性验证。无信道增强训练只报告目标域CLEAN，有信道增强训练报告CLEAN及六种practical场景。标准划分跨接收机六景准确率，PAMF为71.0507\%，DURA+PAMF为78.5161\%；接入恢复器的种子内配对提升为7.4653个百分点。594个逻辑模型按训练与推理条件分组，主比较使用165个独立识别器，并展示198个模型与推理链组合。本次没有新增训练、预测或目标真值访问。
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
            if label.startswith('附录'):body.append(r'\FloatBarrier')
            if label.startswith('附录'):
                body.append(r'\appendices');body.append(r'\section{阅读名称与原始记录的对应}')
            elif '（续）' in label:body.append(r'\subsection{'+tex(label.split('  ',1)[-1])+'}')
            else:
                section+=1;body.append(r'\section{'+tex(re.sub(r'^\d+\s+','',label))+'}')
            i+=1;continue
        if line.startswith('!['):
            flush();m=re.fullmatch(r'!\[(.*)\]\((.*)\)',line);caption=m.group(1);stem=Path(m.group(2)).stem
            fig_no+=1
            body.extend([r'\begin{figure}[!t]',r'\centering',r'\includegraphics[width=\columnwidth]{../figures/'+stem+'.pdf}',r'\caption{'+tex(caption)+'}',r'\label{fig:grouped-'+str(fig_no)+'}',r'\end{figure}'])
            i+=1
            while i<len(lines) and not lines[i].strip():i+=1
            if i<len(lines) and lines[i].strip()==caption:i+=1
            continue
        if line.startswith('|'):
            # The immediately preceding standalone paragraph is the authored caption.
            defaults={1:'识别、恢复及训练模块的定义',2:'按科学问题与信道增强划分的报告范围',3:'11个数据设置的实际源域样本数',4:'三类互斥测试人群及标准划分样本数',5:'完整对比方法及其原生训练预算',6:'四种外部训练组合',20:'PAMF伪标签与监督的种子内配对差',21:'PAMF普通伪标签的实际无标签曝光与接受率',22:'训练阶段去重后的成本与计数边界',23:'当前PAMF推理路径的实测时延与吞吐',24:'当前实验模块的完成范围与缺项',25:'正文模块名称与历史记录代号对应'}
            caption=pending.pop() if 7<=table_no+1<=19 and pending else defaults.get(table_no+1,'分组实验结果')
            flush();raw=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                rr=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',x) for x in rr):raw.append(rr)
                i+=1
            heads=raw[0];data=raw[1:];n=len(heads);table_no+=1
            body.append('表~'+r'\ref{tab:grouped-'+str(table_no)+'}~'+tex(caption)+'。\n')
            wide=n>=5 or any(len(c)>22 for r in raw for c in r) or (n==4 and len(data)>3)
            if table_no>=20:wide=False
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
