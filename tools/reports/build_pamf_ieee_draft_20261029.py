"""Build aligned IEEE manuscripts and an evidence guide from frozen reports."""
import argparse,csv,html,json,re,shutil,statistics,subprocess,sys,zipfile
from pathlib import Path

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def esc(s):
    for x,y in [('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:s=s.replace(x,y)
    return s

def prose(s):
    # Citation/reference keys are identifiers, not inline mathematical variables.
    parts=re.split(r'(\\(?:cite|ref|eqref)\{[^}]*\})',s)
    for i in range(0,len(parts),2):
        part=re.sub(r'(?<!\\)%',r'\\%',parts[i])
        part=re.sub(r'10\^\((-?\d+)\)',lambda m:'$10^{'+m[1]+'}$',part)
        parts[i]=re.sub(r'(?<![\\\w])([A-Za-z]+_[A-Za-z0-9]+)',lambda m:'$'+m[1]+'$',part)
    return ''.join(parts)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--catalog',type=Path,required=True);ap.add_argument('--results',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--plot-python',required=True);ap.add_argument('--skip-figures',action='store_true');a=ap.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True);(out/'data').mkdir(exist_ok=True)
    manuscript=json.loads((a.source/'manuscript_blocks.json').read_text(encoding='utf-8'))
    if not a.skip_figures:subprocess.run([a.plot_python,str(Path(__file__).with_name('render_pamf_paper_figures_20261029.py')),'--catalog',str(a.catalog),'--results',str(a.results),'--output',str(out/'figures')],check=True)
    for fn in ['01_main_comparison.csv','02_augmentation_PL_factorial.csv','03_two_layer_structure_ablation.csv','04_nested_noSIG_CE.csv','06_view_comparison_8seed.csv','07_U_permission.csv','10_receiver_splits.csv','11_ten_class_scale.csv','14_paired_effects_summary.csv','14_paired_effects_per_seed.csv','15_architecture_E200_summary.csv','15_architecture_E200_per_seed.csv','17_training_U_exposure.csv','DURA_contract_audit.csv']:
        p=a.results/fn
        if p.exists():shutil.copy2(p,out/'data'/fn)
    effpath=a.catalog/'tables'/'network_efficiency_static.csv';eff=read(effpath);shutil.copy2(effpath,out/'data'/effpath.name)
    mainrows=read(a.results/'01_main_comparison.csv');learning=read(a.results/'07_U_permission.csv')
    def one(rows,**filters):
        found=[r for r in rows if all(r.get(k)==v for k,v in filters.items())];assert len(found)==1,(filters,len(found));return found[0]
    def value(r,metric='accuracy'):return f"{float(r[metric+'_mean_pct']):.2f} $\\pm$ {float(r[metric+'_sd_pp']):.2f}"
    def table(caption,label,head,rows,align=None,wide=True,note=''):
        env='table*' if wide else 'table'; align=align or ('l'*len(head))
        return '\\begin{'+env+'}[!t]\n\\caption{'+caption+'}\\label{tab:'+label+'}\n\\centering\\footnotesize\n\\begin{tabular}{'+align+'}\\toprule\n'+' & '.join(head)+r' \\ \midrule'+'\n'+'\n'.join(' & '.join(r)+r' \\' for r in rows)+'\n\\bottomrule\\end{tabular}\n'+('\\par\\smallskip\\begin{minipage}{\\textwidth}\\scriptsize '+note+'\\end{minipage}\n' if note else '')+'\\end{'+env+'}\n'
    def tables(lang):
        zh=lang=='zh'
        settings=[['Source RX','1, 3, 4, 6, 8','D1--D3'],['Target RX','0, 2, 5, 7, 9, 10, 11','D1--D3 / D0'],['Source L / U / V','6,300 / 56,700 / 27,000','Physical roles disjoint'],['Cross-RX / double holdout','126,000 / 42,000','Packets per scene'],['1\\% / 5\\% L','630 / 3,150','Fixed V; remaining train IQ is U'],['Ten-class source / target RX','16 / 16','Different packet populations']]
        ss=table('主要数据角色与评估群体。' if zh else 'Main data roles and evaluation populations.','settings',['Role','Setting','Meaning'],settings,'lll')
        rows=[]
        for m in ['RIEI + PL','DRIFT + PL','IB-RFF + PL','SDG + PL','PAMF']:
            for test in ['cross_rx','cross_both']:
                raw=one(mainrows,data_setting='Main split',method=m,test=test,readout='raw',scene='six_practical_mean',dimension='overall');clean=one(mainrows,data_setting='Main split',method=m,test=test,readout='raw',scene='clean',dimension='overall');g=one(mainrows,data_setting='Main split',method=m,test=test,readout='DURA-LU',scene='six_practical_mean',dimension='overall')
                rows.append([esc(m),'Cross-RX' if test=='cross_rx' else 'Double holdout',value(clean),value(raw),value(g),value(g,'macro_f1')])
        mm=table('Main split 主比较：均值与样本标准差，三个 F 种子。' if zh else 'Main-split comparison: mean and sample SD over three F seeds.','main',['Identifier','Test','Clean, raw','Six scenes, raw','Six scenes, G-LU','Macro-F1, G-LU'],rows,'llrrrr',note='All entries are percentages. G-LU is shared across F seeds within each data setting. External objectives use ordinary PL plus channel augmentation; PAMF uses complete Tri-SIG-TIME SSL.')
        rows=[]
        for m,order in [('PAMF-Sup','Natural'),('PAMF-PL','Natural'),('PAMF-Sup','Permuted'),('PAMF','Permuted')]:
            for test in ['cross_rx','cross_both']:
                rs=[one(learning,data_setting='Main split',method=m,variant='Full three-view',temporal_order=order,test=test,readout=g,scene='six_practical_mean',dimension='overall') for g in ['raw','DURA-L','DURA-LU']]
                rows.append([m,order,'Cross-RX' if test=='cross_rx' else 'Double holdout']+[value(r) for r in rs])
        ll=table('学习方式与 F/G 无标签权限：六景准确率。' if zh else 'Learning regimes and separate F/G unlabeled-data permissions: six-scene accuracy.','learning',['Identifier','Local order','Test','Direct','G-L','G-LU'],rows,'lllrrr',note='Sup retains labeled auxiliary objectives. Same-permutation Sup isolates the complete U package at fixed local order. Ordinary PL and complete PAMF differ in both U recipe and local order; no permuted-PL result is invented.')
        rows=[[esc(r['model']),f"{float(r['parameters_M']):.3f}",f"{float(r['conv_matrix_MACs_M']):.2f}",f"{float(r['conv_matrix_FLOPs_M']):.2f}",f"{float(r['FP32_parameter_MiB']):.2f}"] for r in eff]
        rows += [['DURA','0.267','--','--',f'{266951*4/2**20:.2f}'],['PAMF + DURA','0.500','--','--',f'{499944*4/2**20:.2f}']]
        ee=table('参数规模与部分运算计数；不包含统一可比的延迟排名。' if zh else 'Parameter size and partial operation counts; no unified latency ranking is inferred.','eff',['Model','Params (M)','Counted MACs (M)','Counted FLOPs (M)','FP32 weights (MiB)'],rows,'lrrrr',note='Counted operations cover convolution/matrix operators only. FFT, path algebra, normalization and other functional operations are incompletely covered. Historical latency boundaries differ; total FLOPs and current F+G latency remain unverified in this manuscript.')
        return dict(settings=ss,main=mm,learning=ll,eff=ee)
    refs=(a.source/'references.bib').read_text(encoding='utf-8')
    paired=['# PAMF 中英文逐段对照稿 / Aligned bilingual manuscript','','本文件段落与两个 IEEE PDF 共用同一来源；公式以 TeX 保留，PDF 中正常排版。','']
    for lang,folder in [('en','english'),('zh','chinese')]:
        p=out/folder;(p/'images').mkdir(parents=True,exist_ok=True)
        for f in (out/'figures').glob('*.pdf'):shutil.copy2(f,p/'images'/f.name)
        (p/'references.bib').write_text(refs,encoding='utf-8')
        header=(r'\RequirePackage[T1]{fontenc}'+'\n' if lang=='zh' else '')+r'''\documentclass[journal,10pt]{IEEEtran}
\usepackage{amsmath,amssymb,bm,graphicx,booktabs,array,cite,url,xurl,balance}
\interdisplaylinepenalty=2500
\raggedbottom
'''
        if lang=='zh':header+=r'''\usepackage{xeCJK}
\setCJKmainfont{SimSun}[AutoFakeBold=2,AutoFakeSlant=0.2,SmallCapsFont=SimSun]
\setCJKsansfont{Microsoft YaHei}
\setmainfont{Times New Roman}
\renewcommand{\abstractname}{摘要}
\renewcommand{\IEEEkeywordsname}{关键词}
'''
        header+=r'''\newcommand{\sg}{\operatorname{sg}}
\begin{document}
\title{'''+manuscript['title_'+lang]+r'''}
\author{Authors omitted in this working draft}
\markboth{PAMF Working Draft, October 2026}{PAMF}
\maketitle
\begin{abstract}
'''+prose(manuscript['abstract_'+lang])+r'''
\end{abstract}
\begin{IEEEkeywords}
Radio frequency fingerprinting, receiver generalization, semi-supervised learning, path descriptors, signal restoration.
\end{IEEEkeywords}
'''
        body=[];ts=tables(lang)
        for i,b in enumerate(manuscript['blocks']):
            typ=b['type']
            if typ in ['section','subsection']:
                if typ=='section' and b['en']=='Conclusion':body.append('\\clearpage\n\\balance\n')
                body.append('\\'+typ+'{'+b[lang]+'}\n')
            elif typ=='text':body.append(prose(b[lang])+'\n\n')
            elif typ=='equation':body.append(b['tex']+'\n')
            elif typ=='table':body.append(ts[b['key']])
            elif typ=='figure':body.append('\\begin{figure*}[!t]\n\\centering\\includegraphics[width=\\textwidth]{images/fig%02d.pdf}\n\\caption{%s}\\label{fig:%d}\n\\end{figure*}\n'%(b['number'],prose(b[lang]),b['number']))
        tex=header+''.join(body)+'\n\\bibliographystyle{IEEEtran}\n\\bibliography{references}\n\\end{document}\n'
        (p/'main.tex').write_text(tex,encoding='utf-8')
    for i,b in enumerate(manuscript['blocks']):
        if 'en' in b:paired+=['## '+str(i+1)+' · '+b['type'],'','**中文**',b['zh'],'','**English**',b['en'],'']
        elif b['type']=='equation':paired+=['```latex',b['tex'],'```','']
        elif b['type']=='table':paired+=['[Table: '+b['key']+' — identical data in both PDF versions]','']
    (out/'中英文逐段对照稿.md').write_text('\n'.join(paired),encoding='utf-8')
    guide=(a.source/'evidence_guide.md').read_text(encoding='utf-8');(out/'论文框架_图表取舍_结论证据与缺项.md').write_text(guide,encoding='utf-8')
    htmlbody=[]
    for line in guide.splitlines():
        if line.startswith('# '):htmlbody.append('<h1>'+html.escape(line[2:])+'</h1>')
        elif line.startswith('## '):htmlbody.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line.startswith('|'):
            if re.match(r'^\|[ :|-]+\|?$',line):continue
            htmlbody.append('<div class="row">'+''.join('<span>'+html.escape(c.strip())+'</span>' for c in line.strip('|').split('|'))+'</div>')
        elif line:htmlbody.append('<p>'+html.escape(line)+'</p>')
    (out/'论文框架与证据说明.html').write_text('<!doctype html><meta charset="utf-8"><title>PAMF paper evidence</title><style>body{max-width:1200px;margin:40px auto;font:16px/1.8 system-ui;color:#21313b}h1,h2{color:#16394b}.row{display:flex;border-bottom:1px solid #ddd}.row span{flex:1;padding:8px;font-size:13px;overflow-wrap:anywhere}p{white-space:pre-wrap}</style>'+''.join(htmlbody),encoding='utf-8')
    # A standalone, readable guide PDF is separate from the IEEE manuscripts.
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.utils import simpleSplit
    font='GuideCJK';pdfmetrics.registerFont(TTFont(font,'C:/Windows/Fonts/simsun.ttc',subfontIndex=0))
    cn=canvas.Canvas(str(out/'论文框架_图表取舍_结论证据与缺项.pdf'),pagesize=(595.28,841.89));y=795;page=1
    def newpage():
        nonlocal y,page
        cn.setFont(font,8);cn.drawRightString(550,28,str(page));cn.showPage();page+=1;y=795
    for line in guide.splitlines():
        if re.match(r'^\|[ :|-]+\|?$',line):continue
        size=15 if line.startswith('# ') else 11 if line.startswith('## ') else 9
        line=line.lstrip('# ').replace('**','')
        if line.startswith('|'):line=line.strip('|').replace('|','  /  ')
        for segment in line.split('\n'):
            # Character-aware wrapping handles Chinese and long evidence filenames.
            current=''
            for char in segment:
                if pdfmetrics.stringWidth(current+char,font,size)>505:
                    if y<55:newpage()
                    cn.setFont(font,size);cn.drawString(45,y,current);y-=size*1.5;current=char
                else:current+=char
            if current:
                if y<55:newpage()
                cn.setFont(font,size);cn.drawString(45,y,current);y-=size*1.5
        y-=5
    cn.setFont(font,8);cn.drawRightString(550,28,str(page));cn.save()
    # Every selected page is accounted for; the excluded external model detail pages are explicit.
    index=read(a.catalog/'图表索引.csv');mapping=[]
    main_map={1:'Fig.2',2:'Fig.2',6:'Table IV',23:'Supplement S3',24:'Supplement S3',25:'Fig.3 / Table III',26:'Fig.3 / Table III',27:'Fig.5',28:'Fig.5',29:'Fig.5',30:'Fig.5',31:'Fig.4',32:'Fig.4',33:'Fig.6',34:'Fig.6',35:'Fig.6',36:'Fig.6',37:'Fig.6 (same data)',38:'Fig.6 (same data)',39:'Fig.7 (alternative)',40:'Fig.7 (alternative)',41:'Fig.7 (alternative)',42:'Fig.7',43:'Fig.8',44:'Supplement S6',45:'Supplement S6',46:'Supplement S6',47:'Fig.8',48:'Supplement S6',49:'Supplement S6',50:'Supplement S6'}
    for r in index:
        n=int(r['figure_number']);decision='Excluded: external model detail page' if 11<=n<=22 else main_map.get(n,'Supplement: retained candidate')
        mapping.append(dict(selected_page=n,old_reference=r['old_reference'],figure_id=r['figure_id'],title=r['short_title'],paper_use=decision,plot_data=r['plot_data'],seed_data=r['seed_data']))
    with (out/'已选50页_正文补充映射.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(mapping[0]));w.writeheader();w.writerows(mapping)
    validation={'no_training':True,'no_prediction':True,'no_new_truth_scoring':True,'selected_pages':len(mapping),'excluded_external_detail_pages':12,'retained_candidates':38,'main_figures':8,'main_tables':4,'aligned_blocks':len(manuscript['blocks']),'same_equations_and_data':True}
    (out/'build_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8');shutil.copy2(a.source/'manuscript_blocks.json',out/'manuscript_blocks.json')
    print(json.dumps(validation))

if __name__=='__main__':main()
