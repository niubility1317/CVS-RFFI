"""Source-grounded IEEE experiment text, compact tables and Chinese figure guide."""
import argparse,csv,json,shutil,statistics,subprocess
from pathlib import Path
ROOT=Path('E:/type10-7');DATA=ROOT/'local_artifacts/pamf_repl_completion_20261024/final_analysis_20261025';OUT=ROOT/'exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01';PAPER=ROOT/'paper/phase1_experiments_ieee_20261020'
METHODS=('riei','drift','ibrff','sdg','PAMF-supervised','PAMF-REPL');NAMES=('RIEI','DRIFT-v2','IB-RFF','SDG','PAMF-Sup','PAMF-REPL');TASKS=('cross_date','cross_rx','cross_both')
def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main(data=DATA,out=OUT):
    data,out=Path(data),Path(out);paper=out/'IEEE_experiments';paper.mkdir(exist_ok=True)
    rows=read(data/'paper_all_recipes.csv');idx={(r['protocol'],r['method'],r['cell'],r['test'],r['readout'],r['view'],r['metric']):r for r in rows}
    own=read(data/'summary_overall.csv');oi={(r['protocol'],r['method'],r['test'],r['readout'],r['view'],r['metric']):r for r in own}
    paired=read(data/'paired_effects.csv');core=read(data/'core_effects.csv');timing=read(data/'timing_recomputed.csv');exposure=read(data/'training_exposures.csv')
    def get(pid,method,t='cross_rx',rd=None,v='six_practical_mean',m='accuracy'):
        return idx[pid,method,'native' if method.startswith('PAMF-') else 'D',t,rd or ('DURA-L' if method=='PAMF-supervised' else 'DURA-LU'),v,m]
    def fmt(r,latex=False):return ('\\pmv{%.2f}{%.2f}' if latex else '%.4f±%.4f')%(float(r['mean']),float(r['sd']))
    def effect(name,pid='F0',rd='DURA-LU',v='six_practical_mean',t='cross_rx',m='accuracy'):
        return next(r for r in paired if (r['protocol'],r['effect'],r['test'],r['readout'],r['view'],r['metric'])==(pid,name,t,rd,v,m))
    def ce(name,rd='raw',v='clean'):
        return next(r for r in core if (r['effect'],r['test'],r['readout'],r['view'],r['metric'])==(name,'cross_rx',rd,v,'accuracy'))
    def timecase(case,panel='REPL_structure',batch='1'):
        return next(r for r in timing if r['panel']==panel and r['case']==case and r['batch']==batch)
    manifest=json.loads((out/'figure_manifest.json').read_text());figs={r['stem'].split('_',1)[1]:r for r in manifest['figures']}
    def figure(slug,label,caption):return '\n\\begin{figure}[!t]\n\\centering\n\\includegraphics[width=\\columnwidth]{images/repl/'+figs[slug]['stem']+'.pdf}\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\end{figure}\n'
    setup=r'''% Current PAMF-REPL experiments. Historical T1 remains in prior immutable packages.
\section{Experimental Evaluation}
\label{sec:experiments}
\subsection{Data, Protocol and Complete Methods}
We evaluate closed-set radio-frequency fingerprinting on the ManySig and ManyRx subsets of WiSig~\cite{wisig}. Source and target share the registered transmitter (TX) classes. Training uses dates D1--D3 (March 8, 15 and 23, 2021); D0 (March 1) is the earlier unseen date. Three disjoint populations test a date shift (source RX on D0), a receiver shift (target RX on D1--D3), and both shifts (target RX on D0). The date result is not future-date extrapolation. ManySig has six TX classes; ManyRx retains ten classes and unequal or missing cells. We do not pool their accuracies.

\begin{table*}[!t]
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\caption{Main physical partitions. F1/F2 replace four of the five source RX in F0; each pair of source sets intersects in only one receiver. Counts are physical packets per view.}
\label{tab:repl-data}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcrrr@{}}\toprule
Contract & Source RX indices & TX classes & Date shift & RX shift & Both shifts\\\midrule
F0 & 1,3,4,6,8 & 6 & 30000 & 126000 & 42000\\
F1 & 0,1,2,5,7 & 6 & 30000 & 126000 & 42000\\
F2 & 0,3,9,10,11 & 6 & 30000 & 126000 & 42000\\
ManyRx & Even indices 0--30 & 10 & 31279 & 92597 & 30424\\\bottomrule
\end{tabular*}\end{table*}

ManySig source TX/RX/day cells contain 70 labeled (L), 630 TX-hidden unlabeled (U), and 300 validation (V) packets, giving 6300/56700/27000 for F0. The default L fraction is 10\% of the training pool, with additional validation labels disclosed separately. The 1\% and 5\% conditions use 630 and 3150 L packets and preserve V. ManyRx has 6533 L, 58828 U and 28023 V packets. Source-count experiments use natural budgets at $m=1,2,3,4,6$ and six fixed target RX; F0's five-source/seven-target result is not inserted into that curve. We do not introduce a fixed-L budget or a separate chronological training experiment.

Equalized IQ is centrally cropped to 256 samples and normalized to unit RMS before channel transformation, giving a $2\times256$ float32 input. Each observation retains its physical packet identity. We test clean and all six practical suburban/urban channel views from the same fixed received-IQ handles. These are synthetic residual-channel stress tests, not measurements of an operational satellite link. DURA processes clean as well as impaired inputs, without scene-aware bypass or additional output renormalization.

We compare the complete RIEI~\cite{riei}, DRIFT-v2~\cite{drift}, IB-RFF~\cite{ibrff} and SDG~\cite{sdg} methods. Their original identity head is trained jointly with the backbone; no post-hoc head is fitted. Each Full recipe is evaluated with channel augmentation and ordinary pseudo-labeling (PL) independently off/on. Cells A/B have channel off and C/D have channel on; A/C have PL off and B/D have PL on. The main comparison fixes D rather than selecting a target-optimal cell. All four cells remain in the supplement; external CE-only variants are excluded.

\begin{table*}[!t]\centering\footnotesize
\caption{Training budgets and source-only checkpoint rules. A PAMF logical epoch has 222 updates. The E20 prefix is shared only where the same-seed data, architecture and objective are identical.}
\label{tab:repl-budget}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lp{.34\textwidth}p{.43\textwidth}@{}}\toprule
Method & Budget and optimizer & Freeze / semi-supervised objective\\\midrule
RIEI Full & E100; batch64; SGD $10^{-4}$ & Final epoch; native three-step complete objective.\\
DRIFT-v2 Full & E100; batch256; Adam $10^{-4}$ & Final epoch; native adversarial, center and separation terms.\\
IB-RFF Full & Two stages, each at most E200; batch128; Adam $10^{-3}$ & Source-V minimum NLL; LR patience5, early-stop patience10.\\
SDG Full & E200; batch32; Adam $10^{-3}$ & Source-V minimum NLL; native two-view and contrastive objectives.\\
PAMF-Sup & E50/11100; L128; native AdamW and segment schedules & Fixed final; supervised representation terms, no U/EMA.\\
PAMF ordinary PL & E50/11100; L128/U256 & Current-model confidence0.95, weight1; no EMA.\\
PAMF-REPL & E20+10+20, cumulative E50/11100; L128/U256 & Same ordinary-PL E20 prefix; selected REPL objective during E21--50.\\\bottomrule
\end{tabular*}\end{table*}

External channel augmentation uses the actual practical ResidualChannel engine in residual/post-sync/no-equalization mode. Training samples low suburban, mid urban and low urban with probabilities .40/.35/.25, applying them with probabilities .30/.60/.80 over budget intervals $[0,.20)$, $[.20,.45)$ and $[.45,1)$. The schedule resets within each IB stage. A common .68 auxiliary TX-CE slot is retained in all four cells; unselected or channel-off samples use original L IQ. Native Full losses and internal augmentations remain active. External PL uses the current model at confidence .95 and weight1, whole-U-batch reduction and no EMA or complex gate. The independently completed external matrix and its source-only early-stopping behavior are reused without retraining.

The selected PAMF-REPL identifier keeps the original supervised CRIC, ordinary/raw paired symmetric KL, two As-label CE passes with As KL zero, label smoothing .01 and six L forwards (768 packet evaluations/update). Its original L As uses the three practical environments uniformly; it is not silently replaced by the external auxiliary-concatenation recipe. The matched Sup and ordinary-PL controls keep this same L branch. REPL uses current-model hard labels at confidence .95 and U weight6, six U student passes, shared cross-physical-RX style perturbations, complete-L gradient projection/capping, joint160 contrast and .25 receiver extrapolation through the existing head. There is no EMA, native neighbor/strong-view gate or new inference head. The strict-CE architecture layer instead uses the separately registered .68 channel slot and384 L forwards, with no U/EMA/CRIC/KL; it is not an exposure-matched substitute for the multi-view core control.

PAMF-REPL was selected earlier using the already public F0 clean criterion. We therefore report a post-selection benchmark comparison, not independent blind method selection. This rerun's matrices, parameters and final-epoch rules were fixed before its new target scoring; scores do not feed back into training, tuning, seed selection, queue order or selective reruns.

F denotes the identifier and G the DURA restorer. Sup systems use G-L; PL/REPL systems use G-LU. All G ancestors match the physical source contract and train from source IQ without TX labels, F weights/outputs or target data. G-L uses L IQ only; G-LU may also use TX-hidden U. We additionally freeze both G roles for each PAMF F to separate their permissions. Sup F plus G-L is a completely L-only system; PL F plus G-L still uses U through F. No DURA training or new restored-IQ cache is performed in this rerun.

All predictions for a row's three populations, seven views and declared routes freeze before a separate CPU scorer opens truth. Accuracy and Macro-F1 use all registered classes. S6 first averages six channel views within a seed, then reports the mean and sample SD over F seeds2026092701/02/03. Most three-F summaries condition on one fixed G; they are not three independent complete-system replicas or confidence intervals. The separate $3F\times3G$ study provides three diagonal complete-system pairs, not nine independent repetitions.

\input{pamf_repl_results}
'''
    table=[]
    for pid,pname in [('F0','F0'),('F1_rx_repartition_r01','F1'),('F2_rx_repartition_r01','F2'),('ManyRx','ManyRx')]:
        for task,label in zip(TASKS,('Date','RX','Both')):table.append(pname+' / '+label+' & '+' & '.join(fmt(get(pid,m,task),True) for m in METHODS)+r'\\')
    full_table=r'''\begin{table*}[!t]\centering\scriptsize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.18}
\caption{S6 accuracy (\%) under matching restoration: G-L for PAMF-Sup, G-LU for external Full+PL and PAMF-REPL. Complete recipes and G permissions differ; values are not single-component causal effects. Mean $\pm$ sample SD over three F seeds.}
\label{tab:repl-main}\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}\toprule
Partition / shift & RIEI & DRIFT-v2 & IB-RFF & SDG & PAMF-Sup & PAMF-REPL\\\midrule
'''+ '\n'.join(table)+r'\bottomrule\end{tabular*}\end{table*}'
    f0=get('F0','PAMF-REPL');gain=effect('REPL_minus_sdg_D');fixed_sup=effect('REPL_minus_supervised');fixed_pl=effect('REPL_minus_ordinary_PL');matched=effect('matched_REPL_minus_Sup',rd='matched_LU_vs_L')
    neg_sup=sum(float(effect('REPL_minus_supervised',p)['mean'])<=0 for p in ('F0','F1_rx_repartition_r01','F2_rx_repartition_r01','ManyRx','label01','label05','m1-natural','m2-natural','m3-natural','m4-natural','m6-natural'))
    neg_pl=sum(float(effect('REPL_minus_ordinary_PL',p)['mean'])<=0 for p in ('F0','F1_rx_repartition_r01','F2_rx_repartition_r01','ManyRx','label01','label05','m1-natural','m2-natural','m3-natural','m4-natural','m6-natural'))
    tex='\\subsection{Complete-Method Results and Generalization}\n'+full_table+'\n'
    tex+=f"Table~\\ref{{tab:repl-main}} reports the main physical partitions. On F0 cross-RX, PAMF-REPL plus DURA reaches {fmt(f0,True)}\\% S6 accuracy, compared with {fmt(get('F0','sdg'),True)}\\% for the fixed SDG D recipe. The within-seed difference is {fmt(gain,True)} percentage points (pp), positive for all three F seeds. The corresponding REPL raw clean and raw S6 values are {fmt(get('F0','PAMF-REPL',rd='raw',v='clean'),True)}\\% and {fmt(get('F0','PAMF-REPL',rd='raw'),True)}\\%; after restoration clean is {fmt(get('F0','PAMF-REPL',v='clean'),True)}\\%. We retain every scene rather than infer robustness from clean alone.\n\n"
    tex+=figure('F0_cross_rx_restored','fig:repl-main','Fixed complete recipes on F0 cross-RX. Error bars are three-F-seed sample SD. The external D cell is fixed in advance, not chosen by target rank.')
    tex+=f"The matched Sup/G-L system reaches {fmt(get('F0','PAMF-supervised'),True)}\\%, making the complete-system difference {fmt(matched,True)} pp. This comparison changes both F and G permissions. Holding G-LU fixed, REPL minus Sup is {fmt(fixed_sup,True)} pp and REPL minus ordinary PL is {fmt(fixed_pl,True)} pp. Their S6 means are nonpositive on {neg_sup}/11 and {neg_pl}/11 contracts, respectively. Thus the data support strong restored complete-system performance relative to these fixed external recipes, but not uniformly beneficial semi-supervision or an optimal D cell for every method.\n\n"
    tex+=figure('F0_seven_views_restored','fig:repl-scenes','All seven F0 cross-RX observations. Clean also passes through the restorer. Suburban (S) and urban (U) impairments are simulations; scene-specific weaknesses and clean changes remain visible.')
    tex+=f"REPL cross-RX S6 is {fmt(get('F1_rx_repartition_r01','PAMF-REPL'),True)}\\% on F1 and {fmt(get('F2_rx_repartition_r01','PAMF-REPL'),True)}\\% on F2. ManyRx cross-RX S6 Macro-F1 is {fmt(get('ManyRx','PAMF-REPL',m='macro_f1'),True)}\\%, using all ten classes. The label-fraction and natural source-count curves retain their negative and nonmonotonic differences. The source-count axis changes source data volume as well as RX diversity, so it does not isolate receiver count at a fixed labeling budget.\n\n"
    tex+=figure('source_cross_rx_restored','fig:repl-source','Natural-budget source-count study with six fixed target RX. Only measured m=1,2,3,4,6 points are shown; F0 is not inserted as a five-source point.')
    tex+='\\subsection{Functional Groups and REPL Mechanisms}\nThe core groups are defined by function: A is labeled-source cross-RX CRIC, B is labeled-source paired symmetric KL, and C is the complete REPL U objective, including its U consistency and contrastive terms. Main rows111/011/101/110 are accompanied by all eight combinations for marginal effects and interactions. C0 does not construct a U loader. The matched000 CE skeleton keeps768 L packet forwards/update.\n\n'
    crows=[]
    for method,label in [('REPL','111: Full'),('ABC011','011: no A'),('ABC101','101: no B'),('supervised','110: no C')]:
        crows.append(label+' & '+' & '.join(fmt(oi['F0',method,'cross_rx',rd,v,'accuracy'],True) for rd,v in [('raw','clean'),('raw','six_practical_mean'),('DURA-LU','six_practical_mean')])+r'\\')
    tex+=r'\begin{table}[!t]\centering\scriptsize\setlength{\tabcolsep}{3pt}\caption{Core removal results on F0 cross-RX (\%). G-LU still consumes U when F has C0.}\label{tab:repl-core}\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrr@{}}\toprule Row & Raw clean & Raw S6 & G-LU S6\\\midrule'+'\n'+'\n'.join(crows)+r'\bottomrule\end{tabular*}\end{table}'+'\n'
    tex+=f"For raw clean, full-minus-removal differences are {fmt(ce('full_minus_no_L_CRIC'),True)}, {fmt(ce('full_minus_no_L_consistency'),True)} and {fmt(ce('full_minus_no_U'),True)} pp for A/B/C, respectively. The marginal B/C effects across the full factorial are {fmt(ce('B'),True)} and {fmt(ce('C'),True)} pp, while BC is {fmt(ce('BC'),True)} pp. Marginal and full-conditional effects answer different questions; these interactions preclude an additive interpretation. SD with three seeds is not a significance test.\n\n"
    tex+=figure('core_conditional_effects','fig:repl-core','Full111 minus011/101/110. Raw and restored effects are separate; negative and seed-dependent contributions are retained.')
    tex+='The internal C ablations remove U style, gradient protection, joint-feature contrast or receiver extrapolation, each with the same E50 trajectory and head. The no-extrapolation row reuses the exact matching prior Joint160 model rather than retraining it. These are the four mechanisms of the selected REPL method; obsolete EMA/neighbor/strong-gate controls are not relabeled as REPL ablations. Full-minus-removal effects and all three seeds are supplied for every population/view/route.\n\n'
    tex+=figure('REPL_inner_mechanisms','fig:repl-inner','Actual REPL component removals on F0 cross-RX. The same within-seed full/removal pairing is used before computing three-seed SD.')
    tex+='Seven labeled-branch controls examine shallow-only, intermediate-only and joint-only CRIC, single stochastic forwarding, no As, As KL enabled and no raw CE. They remain diagnostic appendix rows. Single forwarding changes computation/stochastic exposure; no-raw-CE changes CE dose; the strict CE recipe changes the complete supervised objective. We do not attribute these complete-recipe differences to one isolated factor.\n\n'
    tex+='\\subsection{Physical Structure and Restorer Controls}\nThe nine architectures comprise full, single time route, no time, shuffled time, no signature, no PA, no reference, fixed wrong reference and COV-X3. Each is evaluated under both full REPL and strict supervised CE, three seeds per layer. All physically matched readers use raw/G-L/G-LU; primary matching is G-LU for REPL and G-L for strict CE. The two layers differ in supervised dose and losses and are not interchangeable controls.\n\n'
    tex+=figure('REPL_structure','fig:repl-structure','Architecture sensitivity under the complete REPL recipe. Matching restoration is G-LU; all raw and alternative-G-role results remain in the tables.')
    diag=next(r for r in read(data/'m3_crossed_diagonal.csv') if r['test']=='cross_rx' and r['view']=='six_practical_mean' and r['metric']=='accuracy')
    tex+=f"We reconnect seven already trained G controls to REPL for all three populations, without retraining G. Their paired F/G seeds differ from the fixed-G default summary, so the variation is not a pure-G estimate or a complete replicated copy-by-KD factorial. The three-source $3F\\times3G$ grid gives diagonal S6 {fmt(diag,True)}\\% on cross-RX; all nine cells are supplied, and same-F raw stratified CMs agree across the extra G readouts. Only the three diagonal pairs are three complete-system training repetitions.\n\n"
    tex+='Source-only diagnostics evaluate the identical720 L packets from90 TX/RX/day strata under11 interventions for111 F0 models. They read no U/target and fit no diagnostic head. Default-head flips and feature/reference distances quantify sensitivity; they do not prove complete hardware disentanglement, and simulated PA distortion is not assumed to preserve TX identity.\n\n'
    tex+='\\subsection{Efficiency and Accounting}\n'
    tr=[]
    for method,label,panel in [('supervised','Sup','main_controls'),('ordinary_PL','PL','main_controls'),('REPL','REPL','REPL_structure')]:
        for suffix in ('F','GF'):
            r=timecase(method+'_'+suffix,panel);tr.append(label+' '+suffix+' & '+f"{float(r['wall_median_ms']):.4f} & {float(r['wall_P95_ms']):.4f} & {float(r['CPU_roundtrip_median_ms']):.4f}"+r'\\')
    tex+=r'\begin{table}[!t]\centering\footnotesize\caption{Exclusive RTX3090 B1 timings (ms), pooled across five rounds. F is eager; GF uses the existing optimized graph engine. These are different execution paths, not an isolated speedup caused by G.}\label{tab:repl-latency}\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrr@{}}\toprule Case & Median & P95 & CPU round trip\\\midrule'+'\n'+'\n'.join(tr)+r'\bottomrule\end{tabular*}\end{table}'+'\n'
    r=timecase('REPL_GF');tex+=f"The updated optimized G+REPL path has B1 wall median {float(r['wall_median_ms']):.6f}ms, P95 {float(r['wall_P95_ms']):.6f}ms and CPU round-trip median {float(r['CPU_roundtrip_median_ms']):.6f}ms. Four exclusive panels cover1175 case-rounds: REPL/strict-CE nine-architecture readers, matching Sup/ordinary-PL controls and seven existing G combinations. Batch sizes are1/8/32/64/128; five rounds repeat timing, not training seeds. We retain329000 timed calls and395975 total inference invocations including warm-up, cold and transfer checks. Training-only REPL mechanisms add no inference-specific parameters. Loaded-module parameter counts and co-resident memory are retained with their measurement scope; they are not claims of active deployment MACs or isolated-model VRAM.\n\n"
    tex+=figure('measured_throughput','fig:repl-throughput','Measured exclusive-device throughput. The eager-F and optimized-GF paths are explicitly separated; shared-cache classifier timing is not substituted for online G+F timing.')
    tex+='The189 new F completions use84 unique shared E20 prefixes and8670 unique new model-epochs (1924740 updates), totaling265.956426 shared-worker elapsed hours. The201 logical F rows include12 frozen F0 models and reused prefixes; summing logical cumulative costs would double-count shared stages. Each source stage, actual U acceptance and L-forward dose is retained. All C0/strict rows have zero U exposure. DURA adds zero new training and zero new restored-IQ caches in this rerun; its original training ancestry and shared-cache production are historical costs, not free computation. Known matched G costs are listed separately; the original fixed F0 G-LU ancestor timing is incompletely serialized and remains missing rather than inferred as zero.\n\n'
    tex+='\\subsection{Evidence Boundaries}\nAll negative gains, zero external PL acceptance and nonmonotonic source/label curves remain available. The benchmark shares WiSig acquisition hardware; synthetic residual channels do not establish in-orbit validity. Post-selection F0 clean performance, conditional F-seed variation, distinct G permissions and exposure differences constrain causal claims. The supplementary package contains the full frozen CMs/net corrections, source/epoch counters, timing samples and a pure-standard-library replay; IQ, model weights, per-sample target truth and private controller state are excluded.\n'
    # Relevant fine/structure tables remain accessible without crowding the main comparison.
    tex+='\\appendices\n\\section{Complete PAMF Diagnostic Rows}\n'
    for group,title in [('fine_L','Labeled-branch diagnostic recipes'),('structure','REPL architecture layer'),('strict_structure','Strict-CE architecture layer')]:
        methods=list(dict.fromkeys(r['method'] for r in own if r['group']==group))
        if group in ('fine_L','structure'):methods=['REPL']+methods
        if group=='strict_structure':methods=['CE_full']+[m for m in methods if m!='CE_full']
        ar=[]
        for method in methods:
            rd='DURA-L' if group=='strict_structure' else 'DURA-LU';rr=[oi['F0',method,'cross_rx',route,view,'accuracy'] for route,view in [('raw','clean'),('raw','six_practical_mean'),(rd,'six_practical_mean')]]
            label=method
            for prefix in ('REPL_','CE_','L_'):label=label.removeprefix(prefix)
            label='Full' if label=='REPL' else label.replace('_',' ')
            ar.append(label+' & '+' & '.join(fmt(r,True) for r in rr)+r'\\')
        tex+=r'\par\medskip\noindent\begin{minipage}{\columnwidth}\centering\scriptsize\setlength{\tabcolsep}{2pt}\makeatletter\def\@captype{table}\makeatother\caption{'+title+r': F0 cross-RX accuracy (\%). Complete rows retain every seed; exposure differences are described in the main text.}\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrr@{}}\toprule Recipe & Raw clean & Raw S6 & G S6\\\midrule'+'\n'+'\n'.join(ar)+r'\bottomrule\end{tabular*}\end{minipage}\par\medskip'+'\n'
    tex+='The supplementary27 standalone figures additionally show the complete core, internal removals, source geometry, fairness four-cell contrasts, measured architecture timing and all external Full A/B/C/D cells. They retain the same physical partitions and metric definitions as the main text.\n'
    for old,new in {'batch64':'batch 64','batch256':'batch 256','batch128':'batch 128','batch32':'batch 32','patience5':'patience 5','patience10':'patience 10','confidence0.95':'confidence 0.95','weight1':'weight 1','weight6':'weight 6','and384':'and 384','seeds202609':'seeds 202609','rows111':'rows 111','matched000':'matched 000','keeps768':'keeps 768','Full111 minus011':'Full 111 minus 011','identical720':'identical 720','from90':'from 90','under11':'under 11','for111':'for 111','cover1175':'cover 1175','are1/':'are 1/','retain329000':'retain 329000','and395975':'and 395975','The189':'The 189','use84':'use 84','and8670':'and 8670','totaling265':'totaling 265','The201':'The 201','include12':'include 12','supplementary27':'supplementary 27'}.items():
        setup=setup.replace(old,new);tex=tex.replace(old,new)
    (paper/'current_experiments.tex').write_text(setup,encoding='utf-8');(paper/'pamf_repl_results.tex').write_text(tex,encoding='utf-8')
    for name in ('main.tex','references.bib','IEEEtran.bst'):
        src=PAPER/name
        if src.exists():shutil.copy2(src,paper/name)
    (paper/'images/repl').mkdir(parents=True,exist_ok=True)
    for rec in manifest['figures']:shutil.copy2(out/'pdf'/(rec['stem']+'.pdf'),paper/'images/repl'/(rec['stem']+'.pdf'))
    # Compact prose and table entry points for direct manuscript reuse.
    md=['# PAMF-REPL：论文结果与使用边界','', '当前结果来自完整201逻辑F及27额外G读出；四外部Full沿用已核验practical residual四组合。本次未重训DURA，旧T1不改名。','',
        '|方法|跨日 S6|跨接收机 S6|双跨 S6|','|---|---:|---:|---:|']
    for method,label in zip(METHODS,NAMES):md.append('|'+label+'|'+'|'.join(fmt(get('F0',method,t))+'%' for t in TASKS)+'|')
    md+=['',f"F0跨RX：REPL raw clean {fmt(get('F0','PAMF-REPL',rd='raw',v='clean'))}%；raw S6 {fmt(get('F0','PAMF-REPL',rd='raw'))}%；G后S6 {fmt(f0)}%，clean {fmt(get('F0','PAMF-REPL',v='clean'))}%。",'',
      f"相对固定SDG D的seed内配对差为{fmt(gain)}个百分点，3/3正。完整系统相对Sup/G-L差{fmt(matched)}个百分点；固定G-LU后相对Sup为{fmt(fixed_sup)}、相对普通PL为{fmt(fixed_pl)}个百分点，不能混称为同一增益。",'',
      f"11契约跨RX S6下，固定G-LU的REPL−Sup均值有{neg_sup}/11非正，REPL−普通PL有{neg_pl}/11非正；保留全部负值，不宣称半监督普遍有效。",'',
      '核心A=L CRIC，B=L配对KL，C=整个REPL U。完整条件移除与边际效应区分，SD不作为显著性检验。严格CE384 L前向与匹配核心768前向分层，不能孤立归因。','',
      'PAMF-REPL曾按已公开F0 clean标准选定；当前是如实披露的选择后比较，不宣称独立盲选。训练/权重冻结保持源侧，所有新目标结果不回流调参或选模。','',
      f"新的G+REPL B1 median={float(r['wall_median_ms']):.6f}ms，P95={float(r['wall_P95_ms']):.6f}ms；F eager和GF优化图执行分列。五轮不是五seed；共驻留显存不是单模型峰值。",'',
      '新增8670唯一模型轮/1924740更新、265.956426共享worker小时；DURA新训练0、新恢复缓存0，祖先历史成本并非0；原F0 G-LU部分祖先计时缺失明确保留。','',
      '三类泛化互斥：源RX D0、目标RX D1-D3、目标RX D0。D0是更早未见日。ManyRx十类，其他六类。源数量仅1/2/3/4/6，固定六目标，不插F0五源七目标。','',
      '输入包保留379239份序列化CM、247863净纠错、所有seed/分层、111×720 L源诊断及1175case-round。原始IQ/权重/逐样本目标真值/私有runtime不进入导出。','',
      '文件入口：IEEE_experiments/main.pdf 为实验部分；figure_captions_en.md 为逐图英文图注；tables 为来源表；source_data 为完整汇总与成本；复算命令见README。']
    (out/'论文结果与边界.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    (out/'manuscript_results_en.md').write_text('PAMF-REPL experimental results\n\n'+f'F0 cross-RX restored S6 accuracy is {fmt(f0)}%, with a paired gain of {fmt(gain)}pp over fixed SDG D. At fixed G-LU, REPL minus ordinary PL is {fmt(fixed_pl)}pp; the data do not support uniformly positive semi-supervised gains.\n\n'+
      'PAMF-REPL was selected using a previously public F0 clean criterion. These are post-selection benchmark results, not independent blind method selection. Training uses source data only; new target results do not retune the fixed matrix.\n\n'+
      'The complete experiment text, source-grounded tables and detailed limitations are provided in IEEE_experiments/current_experiments.tex and pamf_repl_results.tex. All27 English figure captions are in figure_captions_en.md.\n',encoding='utf-8')
    write(out/'tables/main_S6_comparison.csv',[dict(get(p,m,t),task=t) for p in ('F0','F1_rx_repartition_r01','F2_rx_repartition_r01','ManyRx') for t in TASKS for m in METHODS])
    (out/'tables/main_S6_comparison.tex').write_text(full_table+'\n',encoding='utf-8')
    guide(out,manifest,md)
    print(json.dumps(dict(status='PAPER_SOURCE_AND_GUIDE_WRITTEN',figures=len(manifest['figures']),paper=str(paper))),flush=True)

def guide(out,manifest,md):
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.utils import ImageReader
    from PIL import Image
    pdfmetrics.registerFont(TTFont('Chinese','C:/Windows/Fonts/simsun.ttc',subfontIndex=0));pdfmetrics.registerFont(TTFont('TNR','C:/Windows/Fonts/times.ttf'))
    c=canvas.Canvas(str(out/'论文结果与逐图说明.pdf'),pagesize=(595.28,841.89));page=0
    def start(title):
        nonlocal page
        page+=1;c.setFillColorRGB(.13,.25,.36);c.setFont('Chinese',16);c.drawString(40,790,title);c.setStrokeColorRGB(.75,.8,.84);c.line(40,773,555,773)
    def text(s,y,size=10):
        c.setFillColorRGB(.12,.15,.19);c.setFont('Chinese',size);cur='';lines=[]
        for ch in s:
            if ch=='\n' or pdfmetrics.stringWidth(cur+ch,'Chinese',size)>510:lines.append(cur);cur='' if ch=='\n' else ch
            else:cur+=ch
        if cur:lines.append(cur)
        for line in lines:c.drawString(40,y,line);y-=size*1.55
        return y
    def finish():
        c.setFont('TNR',8);c.setFillColorRGB(.4,.45,.5);c.drawString(40,28,'PAMF-REPL / 2026-10-25');c.drawRightString(555,28,str(page));c.showPage()
    start('PAMF-REPL：论文使用说明');y=747
    for line in md:
        if line.startswith('|') or not line or line.startswith('#'):continue
        height=(len(line)//45+2)*16
        if y-height<65:finish();start('结果与边界（续）');y=747
        y=text(line,y)-15
    finish()
    for i,rec in enumerate(manifest['figures'],1):
        start('图%d：%s'%(i,rec['stem'].split('_',1)[1].replace('_',' ')[:43]))
        image=Image.open(out/'preview'/(rec['stem']+'.png'));w,h=image.size;scale=min(510/w,370/h)
        c.drawImage(ImageReader(image),40+(510-w*scale)/2,746-h*scale,width=w*scale,height=h*scale)
        y=746-h*scale-24;y=text('用途：'+rec['claim'],y)-15
        y=text('图内只保留坐标、方法与实际数值，完整英文图注另见 figure_captions_en.md；源表与图同名。',y)-15
        note='误差为三个实际F种子的样本标准差；六景先seed内平均。通常固定一个同契约G，不是三个独立完整系统，SD不代表显著性。'
        if 'crossed' in rec['stem']:note='九格是同三F与三G的全部组合，每格没有重复SD；三对角才是三组完整训练种子，不能称九次独立重复。'
        elif 'latency' in rec['stem'] or 'throughput' in rec['stem']:note='五轮是计时重复，不是五模型seed。F为eager、GF为既定优化图执行，不能据差值孤立声称加G带来算法加速。共驻留显存与单模型显存分开。'
        elif 'source_physical' in rec['stem']:note='同720条L、90个TX/RX/day层、11干预，U/target/拟合探针0。仅为物理敏感性证据，不证明完整硬件解耦，不假设模拟PA保持身份。'
        elif 'four_cells' in rec['stem']:note='四格都是外部Full，A/C配G-L、B/D配G-LU；raw比较更直接隔离增强/PL，恢复后跨PL同时改变G权限。原已验证外部统计复用。'
        elif 'core_' in rec['stem']:note='A为L CRIC、B为L对称KL、C为整个REPL U。边际主效应、交互与完整条件移除不同；C0不打开Uloader，000不是384前向严格CE。'
        elif 'structure' in rec['stem']:note='完整REPL和严格CE两层分别比较。REPL 768 L前向/更新及U，strict CE 384 L前向且无U。跨层配方不同，不能孤立归因单一结构。'
        text('解释边界：'+note,y);finish()
    c.save()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=DATA);p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();main(a.data,a.output)
