"""Paper-only projection: complete comparators and matched PAMF Sup/T1 results.

Reads previously verified aggregate tables. No model, IQ, truth or scoring access.
"""
import argparse
import csv
import json
import math
from pathlib import Path

FAMILIES=('riei','drift','ibrff','sdg')
NAMES={'riei':'RIEI','drift':'DRIFT-v2','ibrff':'IB-RFF','sdg':'SDG'}
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban',
       'practical_high_urban','practical_mid_urban','practical_low_urban')
CONDITIONS=('F0','F1','F2','ManyRx','chrono','label01','label05')+tuple(
    'm%d-%s'%(m,b) for b in ('natural','fixed') for m in (1,2,3,4,6))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def write_csv(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def one(rows,**where):
    found=[r for r in rows if all(str(r[k])==str(v) for k,v in where.items())]
    if len(found)!=1:raise ValueError(('Not exactly one frozen aggregate',where,len(found)))
    r=found[0]
    if not all(math.isfinite(float(r[k])) for k in ('mean','sd')) or not 0<=float(r['mean'])<=100 or float(r['sd'])<0:
        raise ValueError('Invalid percentage summary')
    return r


def collect(project):
    project=Path(project);reports=project/'automation_reports/CV-SincNet'
    paths={
        'external':reports/'20261023-phase1-external-DURA-analysis-m516-r01/external_final_summary_overall.csv',
        'external_dates':project/'local_artifacts/final_delivery_20261023/external-final-py310/external_final_summary.csv',
        'main':reports/'20261021-phase1-T1-DURA-comparison-m153-r01/main_summary_overall.csv',
        'sup_G':reports/'20261023-phase1-PAMF-PL-Gfactor-analysis-m51-r01/gfactor_summary_overall.csv',
        'sup_G_dates':project/'local_artifacts/final_delivery_20261023/replay-py313/gfactor/gfactor_summary.csv'}
    src={k:read_csv(v) for k,v in paths.items()};rows=[]
    def put(r,protocol,method,setting,reader,source,regime,endpoint='main'):
        rows.append(dict(protocol=protocol,method=method,setting=setting,readout=reader,endpoint=endpoint,
            view=r['view'],metric=r['metric'],mean=r['mean'],sd=r['sd'],F_seeds=3,
            G_roles='none' if reader=='raw' else 'L' if setting=='supervised' else 'LU',
            source_table=paths[source].relative_to(project).as_posix(),source_regime=regime,
            comparator_variant='full' if method in NAMES.values() else 'PAMF'))
    for r in src['external']:
        if r['variant']!='full' or r['task']!='main':continue
        setting='supervised' if r['regime']=='supervised' else 'semi-supervised'
        matched='DURA-L' if setting=='supervised' else 'DURA-LU'
        if r['readout'] not in ('raw',matched):continue
        put(r,r['protocol'],NAMES[r['family']],setting,r['readout'],'external',r['regime'])
    for pid in CONDITIONS:
        for view in VIEWS+('six_practical_mean',):
            for metric in ('accuracy','macro_f1'):
                raw=one(src['main'],protocol=pid,regime='matched_sat_supervised',readout='raw',test='main',view=view,metric=metric)
                put(raw,pid,'PAMF-Sup','supervised','raw','main','matched_sat_supervised')
                g=one(src['sup_G'],protocol_id=pid,cell='Sup+G-L',test='main',view=view,metric=metric)
                put(g,pid,'PAMF-Sup','supervised','DURA-L','sup_G','Sup+G-L')
                for reader in ('raw','DURA-LU'):
                    r=one(src['main'],protocol=pid,regime='native_cric',readout=reader,test='main',view=view,metric=metric)
                    put(r,pid,'PAMF-T1','semi-supervised',reader,'main','native_cric')
    # Future endpoint: target RX on D3 only. Never substitute all-date chrono main.
    for family in FAMILIES:
        for metric in ('accuracy','macro_f1'):
            r=one(src['external_dates'],protocol='chrono',family=family,variant='full',regime='supervised',task='main',
                readout='DURA-L',view='six_practical_mean',dimension='date',stratum='2021_03_23',metric=metric)
            put(r,'chrono',NAMES[family],'supervised','DURA-L','external_dates','supervised','D3_target_RX')
    for metric in ('accuracy','macro_f1'):
        r=one(src['sup_G_dates'],protocol_id='chrono',cell='Sup+G-L',test='main',view='six_practical_mean',
            dimension='date',stratum='2021_03_23',metric=metric)
        put(r,'chrono','PAMF-Sup','supervised','DURA-L','sup_G_dates','Sup+G-L','D3_target_RX')
    keys=[tuple(r[k] for k in ('protocol','method','setting','readout','endpoint','view','metric')) for r in rows]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate paper result')
    for setting in ('supervised','semi-supervised'):
        for method in list(NAMES.values())+['PAMF-Sup' if setting=='supervised' else 'PAMF-T1']:
            for reader in ('raw','DURA-L' if setting=='supervised' else 'DURA-LU'):
                for v in VIEWS+('six_practical_mean',):
                    for metric in ('accuracy','macro_f1'):
                        one(rows,protocol='F0',method=method,setting=setting,readout=reader,endpoint='main',view=v,metric=metric)
    if any(r['G_roles']=='LU' for r in rows if r['setting']=='supervised'):
        raise ValueError('Supervised system must not use U through G')
    return rows


def pm(r):return r'\pmv{'+format(float(r['mean']),'.2f')+'}{'+format(float(r['sd']),'.2f')+'}'


def table(caption,label,headers,data,wide=True,groups=()):
    env='table*' if wide else 'table';width=r'\textwidth' if wide else r'\columnwidth'
    cols='l'+'r'*(len(headers)-1)
    out=[r'\begin{'+env+r'}[!t]',r'\centering\footnotesize',r'\caption{'+caption+'}',r'\label{'+label+'}',
        r'\begin{tabular*}{'+width+r'}{@{\extracolsep{\fill}}'+cols+'@{}}',r'\toprule',
        ' & '.join(headers)+r' \\',r'\midrule']
    for i,row in enumerate(data):
        if i in dict(groups):
            if i:out.append(r'\addlinespace[4pt]')
            out.append(r'\multicolumn{'+str(len(headers))+r'}{@{}l}{\textit{'+dict(groups)[i]+r'}} \\')
            out.append(r'\addlinespace[2pt]')
        out.append(' & '.join(row)+r' \\')
    out += [r'\bottomrule',r'\end{tabular*}',r'\end{'+env+'}']
    return '\n'.join(out)


def build(project,snapshot=None,figures=True):
    project=Path(project);paper=project/'paper/phase1_experiments_ieee_20261020'
    out=project/'local_artifacts/paper_full_methods_20261023_r13';out.mkdir(parents=True,exist_ok=True)
    (paper/'tables').mkdir(parents=True,exist_ok=True)
    frozen=json.loads(Path(snapshot).read_text(encoding='utf-8')) if snapshot else None
    if frozen and frozen.get('schema')!='pamf_paper_full_methods_v1':raise ValueError('Unknown paper snapshot')
    rows=frozen['comparisons'] if frozen else collect(project)
    write_csv(paper/'tables/full_methods_data.csv',rows)
    def get(method,setting,reader,view='six_practical_mean',metric='accuracy',pid='F0',endpoint='main'):
        return one(rows,protocol=pid,method=method,setting=setting,readout=reader,endpoint=endpoint,view=view,metric=metric)
    methods=list(NAMES.values());settings=('supervised','semi-supervised')
    data=[]
    for setting in settings:
        reader='DURA-L' if setting=='supervised' else 'DURA-LU'
        for method in methods+['PAMF-Sup' if setting=='supervised' else 'PAMF-T1']:
            label=r'\textbf{'+method+'}' if method.startswith('PAMF') else method
            data.append([label]+[pm(get(method,setting,r,v,m)) for r,v,m in
                (('raw','clean','accuracy'),('raw','six_practical_mean','accuracy'),(reader,'clean','accuracy'),
                 (reader,'six_practical_mean','accuracy'),(reader,'six_practical_mean','macro_f1'))])
    f0=table(r'F0 comparison of complete methods (\%). Raw applies F directly; restored applies the same-contract DURA before F. S6 is the within-seed six-scene mean. Entries are mean$\pm$sample SD over three F seeds.',
        'tab:full-f0',['Method','Raw clean','Raw S6','Restored clean','Restored S6','Restored S6 F1'],data,
        groups=((0,'Supervised F and G-L: labeled source IQ only'),(5,'Semi-supervised F and G-LU: unlabeled source IQ allowed')))
    scenes=[]
    for setting in settings:
        reader='DURA-L' if setting=='supervised' else 'DURA-LU'
        for method in methods+['PAMF-Sup' if setting=='supervised' else 'PAMF-T1']:
            label=r'\textbf{'+method+'}' if method.startswith('PAMF') else method
            scenes.append([label]+[pm(get(method,setting,reader,v)) for v in VIEWS])
    scene_table=table(r'Restored F0 accuracy on every declared condition (\%). S/U denote suburban/urban and H/M/L denote channel severity. Methods retain the same source and target roles within each setting.',
        'tab:full-scenes',['Method','Clean','S-H','S-M','S-L','U-H','U-M','U-L'],scenes,
        groups=((0,'Supervised: DURA-L'),(5,'Semi-supervised: DURA-LU')))
    general=[]
    for method in methods+['PAMF-Sup']:
        label=r'\textbf{'+method+'}' if method.startswith('PAMF') else method
        general.append([label]+[pm(get(method,'supervised','DURA-L',pid=pid)) for pid in ('F1','F2','ManyRx')]+
            [pm(get(method,'supervised','DURA-L',pid='chrono',endpoint='D3_target_RX'))])
    gt=table(r'Supervised generalization with G-L (S6 accuracy, \%). F1/F2 use alternate disjoint RX partitions. ManyRx has ten classes. Chronological evaluation uses target RX on D3 only (42000 packets), not the all-date aggregate.',
        'tab:full-generalization',['Method','F1','F2','ManyRx','Future D3'],general)
    label=[]
    for frac,pid in (('1','label01'),('5','label05'),('10','F0')):
        label.append([frac]+[pm(get(method,'supervised','DURA-L',pid=pid)) for method in ('IB-RFF','PAMF-Sup')])
    lt=table(r'Label efficiency under full L-only systems (S6 accuracy, \%). Only IB-RFF has registered external low-label models. All rows use the same seven target RX and separate source-validation labels.',
        'tab:full-labels',[r'L fraction (\%)','IB-RFF','PAMF-Sup'],label,wide=False)
    comparison=[r'\subsection{Comparison with Complete Methods}',
        r'\subsubsection{Supervised and Semi-Supervised Recognition}',
        r'Table~\ref{tab:full-f0} compares the four complete external methods with the corresponding PAMF recipe. The supervised block uses labeled-only F and same-contract G-L throughout. The semi-supervised block uses each external method with ordinary PL and PAMF-T1 with its native selective self-training, followed by G-LU. PAMF-Sup retains the registered supervised representation objectives and satellite augmentation but consumes no U or EMA; it is not the ordinary-PL control. PAMF-T1 is the fixed final method, not a relabeled PAMF-PL experiment.',f0,
        r'On F0, PAMF-Sup reaches '+pm(get('PAMF-Sup','supervised','DURA-L'))+r'\% S6 accuracy, while PAMF-T1 reaches '+pm(get('PAMF-T1','semi-supervised','DURA-LU'))+r'\%. Their respective raw S6 values are '+pm(get('PAMF-Sup','supervised','raw'))+r'\% and '+pm(get('PAMF-T1','semi-supervised','raw'))+r'\%. Table~\ref{tab:full-scenes} retains all seven conditions, including the difficult low-urban endpoint. These are complete-recipe comparisons with method-specific objectives and budgets; they do not isolate the effect of an individual SSL component.',scene_table,
        r'\begin{figure*}[!t]\centering\includegraphics[width=.94\textwidth]{images/full_methods/f0_complete.pdf}\caption{Complete external methods and the corresponding PAMF recipe on F0. Raw and restored S6 accuracy use the same frozen F in each comparison. Error bars are three-F-seed sample SD.}\label{fig:full-f0}\end{figure*}',
        r'Restoration does not uniformly improve clean observations or every external classifier. For example, F0 IB-RFF in the supervised setting has S6 accuracy '+pm(get('IB-RFF','supervised','DURA-L'))+r'\% with G-L, compared with '+pm(get('IB-RFF','supervised','raw'))+r'\% on raw input; RIEI changes from '+pm(get('RIEI','supervised','raw'))+r'\% to '+pm(get('RIEI','supervised','DURA-L'))+r'\%. All clean reductions and nonmonotonic results remain in the tables and curves. We do not choose the restoration route or bypass from the true test condition.',
        r'\subsubsection{Receiver, Date and Label Generalization}',
        r'Table~\ref{tab:full-generalization} retains the available complete supervised methods across receiver partitions, the larger TX/RX setting and the future-date endpoint. External semi-supervised models exist for F0 and the natural source-count suite; untrained external SSL cells are not invented for these other conditions. The low-label comparison in Table~\ref{tab:full-labels} includes only the registered IB-RFF and PAMF-Sup models.',gt,lt,
        r'\subsubsection{Source-Receiver Diversity}',
        r'Figure~\ref{fig:full-source} compares the five complete methods using exactly six target RX/144000 packets at every point. Natural budgets use $(|L|,|U|,|V|)=(1260m,11340m,5400m)$; the fixed-L/V supervised setting uses $(1080,12600m-1080,5400)$ with no U consumed. The source counts are $m=1,2,3,4,6$. F0 uses seven target RX/168000 packets and is not inserted as an $m=5$ point. A single source RX provides no cross-RX positive pair, so the corresponding PAMF contrast term has zero valid-anchor coverage. Natural SSL curves vary both receiver diversity and available U; they do not isolate receiver count alone.',
        r'\begin{figure*}[!t]\centering\includegraphics[width=\textwidth]{images/full_methods/source_complete.pdf}\caption{Source-count comparison using complete external methods and PAMF. Supervised panels use G-L; the semi-supervised panel uses G-LU. Each line retains its registered training budget, three F seeds, all negative effects and nonmonotonicity. The fixed-L/V panel controls annotation count.}\label{fig:full-source}\end{figure*}',
        r'The source curves provide a broader view than the F0 endpoint: increasing source RX does not guarantee monotonic gains, and PAMF is not assumed to dominate every low-source condition. The source-validation stopping rules remain fixed; test results never select an epoch, branch, seed or rerun.']
    (paper/'full_method_comparison.tex').write_text('\n\n'.join(comparison)+'\n',encoding='utf-8')
    # Focused PAMF mechanisms retain only the main groups and physical controls.
    reports=project/'automation_reports/CV-SincNet'
    core=frozen['core'] if frozen else read_csv(reports/'20261023-phase1-PAMF-core-factorial-recovery-r01/core_overall_summary.csv')
    structure=frozen['structure'] if frozen else read_csv(reports/'20261022-phase1-PAMF-supervised-physics-m24-r01/structure_summary_overall.csv')
    gr=frozen['restoration'] if frozen else read_csv(reports/'20261021-phase1-T1-DURA-G-ablation-m21-r01/G_ablation_summary.csv')
    primary=[]
    for bits,name in (('111','Full PAMF-T1'),('011','Without A: CRIC'),('101','Without B: consistency'),('110','Without C: selective SSL')):
        primary.append([name]+[pm(one(core,bits=bits,readout=reader,test='main',view=view,metric='accuracy')) for reader,view in
            (('raw','clean'),('raw','six_practical_mean'),('DURA-LU','clean'),('DURA-LU','six_practical_mean'))])
    ct=table(r'PAMF core-group ablation on F0 (accuracy, \%). All restored rows use fixed G-LU; C-off removes U from F only.',
        'tab:paper-core',['F training setting','Raw clean','Raw S6','DURA clean','DURA S6'],primary)
    arch=('X3_CE','SIG_TIME','NO_TIME','NO_SIGNATURE','NO_PA','NO_REFERENCE','COV_X3','SHUFFLE_TIME','WRONG_REFERENCE')
    alabel=('Full X3 backbone',r'Single SIG\_TIME','Without time coordinate','Without signature branch','Without PA branch','Without reference response','Covariance X3','Shuffled time','Scrambled reference')
    ad=[]
    for arm,name in zip(arch,alabel):
        ad.append([name]+[pm(one(structure,architecture=arm,readout=reader,test='main',view=view,metric=metric)) for reader,view,metric in
            (('raw','clean','accuracy'),('raw','six_practical_mean','accuracy'),('DURA-L','six_practical_mean','accuracy'),('DURA-L','six_practical_mean','macro_f1'))])
    at=table(r'Physical-structure ablation under matched labeled-only CE plus satellite augmentation: no CRIC/KL/EMA/U, 11100 updates (accuracy, \%). The full PAMF-Sup recipe is evaluated separately.',
        'tab:paper-structure',['PAMF backbone','Raw clean','Raw S6','G-L S6','G-L S6 F1'],ad)
    gs=[]
    for arm,name in (('default','Default DURA'),('coarse_only','Coarse only'),('prior_no_CG','Without CG'),('no_image_action','Without image action'),('no_innovation_input','Without residual input')):
        clean=one_g(gr,arm,'clean');six=one_g(gr,arm,'six_view_mean');low=one_g(gr,arm,'practical_low_urban')
        gs.append([name,format(int(six['G_parameters'])/1000,'.1f')]+[r'\pmv{'+format(float(r['accuracy_mean']),'.2f')+'}{'+format(float(r['accuracy_sd']),'.2f')+'}' for r in (clean,six,low)])
    gtable=table(r'DURA functional ablation (accuracy, \%). Default: three F seeds with one G; other rows: same-index F/G students. SD describes the respective seed design.',
        'tab:paper-G',['Restorer','Parameters (K)','Clean','S6','Low urban'],gs)
    mech=[r'\subsection{PAMF Mechanism Analysis}',
        r'\subsubsection{Three Core Training Mechanisms}',
        r'We group the identification objectives by function rather than observed gain. A is CRIC: shallow/intermediate same-TX, cross-physical-RX contrast. B is stochastic prediction consistency: ordinary128, ordinary64 and raw128 symmetric-KL terms switch together. C is native selective self-training: own EMA, confidence and adjacent/strong-view gates, pseudo-label CE and U entropy switch together. Table~\ref{tab:paper-core} reports the complete model and one removal per group. Each removal retains the supervised CE dose and update schedule; C-off performs no U loading or EMA work.',ct,
        r'B increases raw clean accuracy by $1.25\pm1.09$ pp and restored S6 by $1.01\pm0.80$ pp in the complete combination. A and C have negative mean conditional restored-S6 differences ($-0.41\pm0.38$ and $-0.21\pm0.58$ pp for full minus removal). The data therefore do not support a claim that all three groups yield stable positive gains in this fixed F0 setting. These results retain the designated default and are not used to regroup objectives or select reruns. The complete eight-combination diagnostic is retained in the experiment archive rather than expanded into the main comparison.',
        r'\subsubsection{Physical Structure and Inductive Bias}',
        r'Table~\ref{tab:paper-structure} isolates architecture under a common labeled-only objective. The single SIG\_TIME variant changes the number of learned views; branch deletions change capacity and computation. No-time, covariance, shuffled-time and scrambled-reference controls retain parameter count but change representation or input structure. They distinguish useful response/statistical features from a claim that a specific coordinate ordering is indispensable.',at,
        r'Removing PA, signature or reference-response features reduces G-L S6 accuracy relative to the full backbone by $2.79\pm0.86$, $2.17\pm0.11$ and $1.48\pm0.01$ pp. However, shuffled time exceeds the original ordering after restoration, and covariance X3 is slightly higher on average. These counterexamples are retained: physical structure supplies an inductive bias, not a guarantee that every chosen coordinate or prior is optimal.',
        r'The frozen source diagnostics use the same 720 labeled packets from 90 TX/RX/day strata and 11 fixed interventions. They access neither U nor target data and fit no probe. For full T1, same-TX cross-RX feature distance is $0.0618\pm0.0105$, compared with $0.0685\pm0.0118$ without A, while the separation margin does not increase. Lower within-identity distance alone does not establish complete receiver/channel disentanglement. Simulated PA interventions test sensitivity and are not assumed to preserve unknown hardware identity.',
        r'\subsubsection{Restoration Mechanisms}',
        r'Table~\ref{tab:paper-G} retains the coarse restoration, conjugate-gradient solver, conjugate-image action and residual-input controls that directly describe DURA. All use frozen T1 default heads, identical seven-view observations and source-only G training. Coarse restoration alone is insufficient; removing CG, image action or residual input also reduces the restored S6 mean. Initialization and distillation diagnostics remain in the archive and do not occupy a separate external-comparator analysis.',gtable]
    (paper/'pamf_focused_mechanisms.tex').write_text('\n\n'.join(mech)+'\n',encoding='utf-8')
    latency=frozen['latency'] if frozen else read_csv(reports/'20261021-phase1-T1-DURA-latency-r02/latency_summary.csv')
    eff=[]
    for case,name,route in (('riei_full','RIEI F','Eager'),('drift_full','DRIFT-v2 F','Eager'),('ibrff_full','IB-RFF F','Eager'),('sdg_full','SDG F','Eager'),('F_graph','PAMF-T1 F','CUDA Graph'),('T4_graph_reuse_G','DURA G','Graph + residual reuse'),('T4_graph_reuse_GF','PAMF-T1 F+G','Graph + residual reuse')):
        r=exact(latency,case=case,batch=1)
        eff.append([name,route]+[format(float(r[k]),'.3f') for k in ('median_ms','P95_ms','H2D_D2H_median_ms')])
    batch=[]
    for b in (1,8,32,64,128):
        r=exact(latency,case='T4_graph_reuse_GF',batch=b)
        batch.append([str(b),format(float(r['median_ms']),'.3f'),format(float(r['throughput_IQ_s']),'.1f')])
    et='\n'.join([
        r'\begin{table*}[!t]\centering\footnotesize',
        r'\caption{Inference on RTX3090 (ms, five rounds). (a) Batch-one GPU wall and CPU-return latency; external F is eager, online external G+F is unmeasured. (b) PAMF F+G throughput in 256-sample IQ windows/s.}',
        r'\label{tab:paper-latency}',
        r'\begin{minipage}[t]{.71\textwidth}\centering',
        r'\textit{(a) Batch-one latency}\\[4pt]',
        r'\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrrr@{}}\toprule',
        r'Model & Execution & Median & P95 & CPU return \\ \midrule',
        '\n'.join(' & '.join(row)+r' \\' for row in eff),
        r'\bottomrule\end{tabular*}\end{minipage}\hfill',
        r'\begin{minipage}[t]{.27\textwidth}\centering',
        r'\textit{(b) Batch throughput}\\[4pt]',
        r'\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}rrr@{}}\toprule',
        r'Batch & Median & IQ/s \\ \midrule',
        '\n'.join(' & '.join(row)+r' \\' for row in batch),
        r'\bottomrule\end{tabular*}\end{minipage}',
        r'\end{table*}'])
    efficiency=[r'\subsection{Computational Efficiency}',
        r'All inference measurements use frozen seed01 weights, FP32 with TF32 disabled, two CPU threads and exclusive RTX3090 access. Five measurement rounds describe timing variation, not five model seeds. Table~\ref{tab:paper-latency} includes complete external identifiers F, the proposed identifier and the restoration components. The external F-only measurements do not establish their online G+F latency; cached classification time is not substituted for it. The default PAMF F+G route has GPU-wall median $4.541$ms and P95 $4.725$ms; CPU class return has median $4.611$ms. The same weights take $37.190$ms on the native execution path, giving an $8.189\times$ execution-route improvement. This is not a claim that every external model received identical implementation optimization.',et,
        r'F and G contain 232993 and 266951 registered learned parameters (499944 total). Separate eager Conv/Linear profiles give 8.856M and 25.068M real MACs; these omit functional-head arithmetic, FFT complex operations, CG reductions and elementwise work, so they are not total FLOPs. Eager G uses 43 FFT/IFFT calls; the residual-reuse route uses 27. The larger training wrapper, optimizer and EMA are not counted as deployment parameters. Multi-model/Graph resident allocation is not reported as single-model VRAM.',
        r'Table~\ref{tab:paper-latency}(b) preserves the nonmonotonic batch latency. Amortized throughput does not establish continuous full-rate radio operation. Training-only CRIC, stochastic forwards, EMA and U gates add no deployment passes. Source-stage cost, G ancestry, cached inference and scoring costs are reported separately in the evidence archive.']
    (paper/'pamf_focused_efficiency.tex').write_text('\n\n'.join(efficiency)+'\n',encoding='utf-8')
    # A compact snapshot supports paper regeneration without the historical CE/pair panels.
    selected=dict(schema='pamf_paper_full_methods_v1',comparisons=rows,
        core=[r for r in core if r['bits'] in ('111','011','101','110') and r['readout'] in ('raw','DURA-LU') and r['test']=='main' and r['view'] in ('clean','six_practical_mean') and r['metric']=='accuracy'],
        structure=[r for r in structure if r['architecture'] in arch and r['test']=='main' and
            (r['readout'],r['view'],r['metric']) in (('raw','clean','accuracy'),('raw','six_practical_mean','accuracy'),('DURA-L','six_practical_mean','accuracy'),('DURA-L','six_practical_mean','macro_f1'))],
        restoration=[r for r in gr if r['arm'] in ('default','coarse_only','prior_no_CG','no_image_action','no_innovation_input') and r['view'] in ('clean','six_view_mean','practical_low_urban')],
        latency=[r for r in latency if (r['case'] in ('riei_full','drift_full','ibrff_full','sdg_full','F_graph','T4_graph_reuse_G','T4_graph_reuse_GF') and str(r['batch'])=='1') or (r['case']=='T4_graph_reuse_GF' and str(r['batch']) in ('8','32','64','128'))])
    (paper/'tables/full_methods_snapshot.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if figures:make_figures(rows,paper)
    proof=dict(status='PAPER_PROJECTION_VERIFIED',rows=len(rows),external_variants=['full'],PAMF_supervised='matched_sat_supervised + G-L',
        PAMF_semisupervised='native_cric T1 + G-LU',supervised_system_U_use=0,external_SSL_only_registered_conditions=True,
        chrono_future_endpoint='D3_target_RX /2021_03_23 only',new_training=0,new_prediction=0,new_scoring=0,
        original_verified_results_preserved=True,removed_from_active_manuscript=['External CE models','Full-minus-CE diagnostics','External pair/permission interactions','Old33/17 comparison','Fine21-item appendix'],
        retained='PAMF A/B/C core removals, physical architecture controls, DURA functional controls, all negative results within shown conditions')
    (out/'paper_projection_verification.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(proof));return proof


def one_g(rows,arm,view):
    found=[r for r in rows if r['arm']==arm and r['view']==view]
    if len(found)!=1:raise ValueError('G mechanism aggregate missing')
    return found[0]


def exact(rows,**where):
    found=[r for r in rows if all(str(r[k])==str(v) for k,v in where.items())]
    if len(found)!=1:raise ValueError(('Unique frozen table row absent',where))
    return found[0]


def make_figures(rows,paper):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    dest=paper/'images/full_methods';dest.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.savefig(dest/(name+'.pdf'),bbox_inches='tight');fig.savefig(dest/(name+'.png'),dpi=200,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.15,2.7),sharey=True)
    for ax,setting in zip(axes,('supervised','semi-supervised')):
        methods=list(NAMES.values())+['PAMF-Sup' if setting=='supervised' else 'PAMF-T1']
        for reader,offset,color in (('raw',-.12,'#999999'),('DURA-L' if setting=='supervised' else 'DURA-LU',.12,'#1676a6')):
            rr=[one(rows,protocol='F0',method=m,setting=setting,readout=reader,endpoint='main',view='six_practical_mean',metric='accuracy') for m in methods]
            ax.errorbar([i+offset for i in range(5)],[float(r['mean']) for r in rr],yerr=[float(r['sd']) for r in rr],fmt='o',capsize=2,color=color,label='Raw' if reader=='raw' else 'DURA-L' if setting=='supervised' else 'DURA-LU')
        ax.set_xticks(range(5),['RIEI','DRIFT','IB-RFF','SDG','PAMF']);ax.set_title('Supervised' if setting=='supervised' else 'Semi-supervised');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('S6 accuracy (%)');axes[0].legend(fontsize=7);axes[1].legend(fontsize=7);fig.tight_layout();save(fig,'f0_complete')
    fig,axes=plt.subplots(1,3,figsize=(7.15,2.6),sharex=True,sharey=True)
    colors=['#4477aa','#ee7733','#228833','#cc6677','#222222']
    for ax,(budget,setting,title) in zip(axes,(('natural','supervised','Natural / supervised'),('natural','semi-supervised','Natural / semi-supervised'),('fixed','supervised','Fixed L,V / supervised'))):
        reader='DURA-L' if setting=='supervised' else 'DURA-LU';methods=list(NAMES.values())+['PAMF-Sup' if setting=='supervised' else 'PAMF-T1']
        for method,color in zip(methods,colors):
            rr=[one(rows,protocol=f'm{m}-{budget}',method=method,setting=setting,readout=reader,endpoint='main',view='six_practical_mean',metric='accuracy') for m in (1,2,3,4,6)]
            ax.errorbar([1,2,3,4,6],[float(r['mean']) for r in rr],yerr=[float(r['sd']) for r in rr],marker='o',ms=3,capsize=2,color=color,label='PAMF' if method.startswith('PAMF') else method,lw=1.6 if method.startswith('PAMF') else 1)
        ax.set_xticks([1,2,3,4,6]);ax.set_title(title,fontsize=8);ax.set_xlabel('Source RX');ax.grid(alpha=.2)
    axes[0].set_ylabel('S6 accuracy (%)');handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=5,loc='upper center',bbox_to_anchor=(.53,1.05),frameon=False,fontsize=7)
    fig.tight_layout(rect=(0,0,1,.91));save(fig,'source_complete')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--frozen');p.add_argument('--no-figures',action='store_true')
    a=p.parse_args();build(a.project,a.frozen,not a.no_figures)
