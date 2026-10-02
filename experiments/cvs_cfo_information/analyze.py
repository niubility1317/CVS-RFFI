"""Reconcile complete source-only diagnostic and report no target conclusions."""
import argparse,csv,json,statistics,math
from pathlib import Path
import numpy as np
from experiments.cvs_cfo_information.audit import RUN,FEATURES,RX,RIDGE


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')


def check_cm(r,count,per_class):
    cm=np.asarray(r['confusion']);tp=np.diag(cm);den=cm.sum(0)+cm.sum(1)
    if cm.shape!=(6,6) or cm.dtype.kind not in 'iu' or np.any(cm<0) or r['count']!=count or cm.sum()!=count or not np.all(cm.sum(1)==per_class):raise ValueError('Incomplete source diagnostic confusion')
    f1=float(np.divide(2*tp,den,out=np.zeros(6),where=den>0).mean())
    if abs(float(tp.sum()/count)-r['accuracy'])>1e-12 or abs(f1-r['macro_f1'])>1e-12:raise ValueError('Source probe metrics differ from independent CM')


def analyze(root,artifact):
    d=read(artifact/'readback.json');done=d['completion'];s=d['statistics'];c=d['resolved']
    if d['process'] or done['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or done['counts']!={'L_s':6300,'V':27000} or done['groups']!=180 or done['probes']!=24:raise ValueError('Diagnostic not terminal and complete')
    if c['commit']!=d['submit']['commit'] or c['target_access'] or c['target_constructed'] or c['checkpoint_access'] or c['formal_CVS_training'] or c['probe_V_fit'] or c['probe_fit_role']!='L_s' or c['source_roles_exact_match'] is not True:raise ValueError('Resolved diagnostic provenance differs')
    if any(s[k] for k in ('target_access','checkpoint_access','V_fit','formal_CVS_training')):raise ValueError('Source-only permission changed')
    expected={(role,t,r,day) for role in ('L_s','V') for t in range(6) for r in RX for day in (1,2,3)}
    if len(s['groups'])!=180 or {(g['role'],g['tx'],g['receiver'],g['day']) for g in s['groups']}!=expected or any(g['count']!=(70 if g['role']=='L_s' else 300) for g in s['groups']):raise ValueError('Incomplete source cell coverage')
    wanted={(f,r) for f in FEATURES for r in (None,*RX)}
    if len(s['probes'])!=24 or {(p['feature'],p['held_source_rx']) for p in s['probes']}!=wanted or len(s['fitted_probes'])!=24:raise ValueError('Incomplete fixed diagnostic fits')
    coef={(p['feature'],p['held_source_rx']):p for p in s['fitted_probes']};dims=dict(cfo80=2,coherence80=1,cfo_coherence80=3,three_window=9)
    for p in s['probes']:
        held=p['held_source_rx'];n=27000 if held is None else 5400;fit_n=6300 if held is None else 5040
        if p['fit_count']!=fit_n or p['fit_role']!='L_s' or p['evaluation_role']!='V' or p['fit_receivers']!=(list(RX) if held is None else [r for r in RX if r!=held]):raise ValueError('Probe L_s/held-RX contract differs')
        check_cm(p,n,4500 if held is None else 900)
        for r in p['per_source_rx'].values():check_cm(r,5400,900)
        if set(p['per_source_rx'])!=({str(r) for r in RX} if held is None else {str(held)}):raise ValueError('Probe source V receiver coverage differs')
        state=coef[(p['feature'],held)];k=dims[p['feature']]
        if state['training_count']!=fit_n or state['fit_role']!='L_s' or state['V_fit'] or state['ridge']!=RIDGE or np.shape(state['weights'])!=(k,6) or len(state['mean'])!=k or len(state['scale'])!=k:raise ValueError('Probe fitted-state provenance differs')
        for name in ('mean','scale','weights','prior'):
            if not np.isfinite(np.asarray(state[name])).all():raise ValueError('Nonfinite fitted diagnostic state')
    log=d['log_text'];after=log.split('RESOLVED_CONFIG ',1)
    if len(after)!=2 or any(x in after[1] for x in ('Traceback (most recent call last)','FloatingPointError','Killed')) or len([v for v in log.splitlines() if v.startswith('PROBE ')])!=24 or len([v for v in log.splitlines() if v.startswith('ROLE ')])!=2:raise ValueError('Incomplete/error diagnostic full text')
    report=root/'automation_reports/CV-SincNet'/RUN;e=report/'evidence';e.mkdir(exist_ok=True)
    write(e/'final_readback.json',d);write(e/'information_statistics.json',s)
    flat=[{k:v for k,v in p.items() if k not in ('confusion','per_source_rx','fit_receivers')} for p in s['probes']]
    with (e/'all24_source_probes.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for p in flat for k in p}));w.writeheader();w.writerows(flat)
    for role,a in s['associations'].items():
        fractions=a['association_fraction']
        if any(v is not None and (not math.isfinite(v) or v<0 or v>1+1e-10) for v in fractions.values()) or abs(sum(v for v in fractions.values() if v is not None)-1)>1e-10:raise ValueError('Invalid circular variance accounting')
    summaries={}
    text='# CVS 相对频偏坐标：完整源域身份信息诊断\n\n'
    text+='VERIFIED / ANALYZED_SOURCE_ONLY。原 L_s6300/V27000 全量、180 个 TX×RX×day×role 单元、24 个预登记探针全部完成。仅L_s拟合标准化和六类ridge权重；V只读评分，U不迭代，无checkpoint、目标/query/目标truth、增强、神经网络训练或源候选重排。全部源角色与原物理契约一致。\n\n'
    text+='| 固定特征 | pooled 源 V（%） | 留一源RX V均值（%） | 五源RX间SD（百分点） | 留一RX逐组（%） |\n|---|---:|---:|---:|---|\n'
    for name in FEATURES:
        pooled=next(p for p in s['probes'] if p['feature']==name and p['held_source_rx'] is None)
        held=[next(p for p in s['probes'] if p['feature']==name and p['held_source_rx']==r)['accuracy'] for r in RX]
        summaries[name]=dict(pooled_accuracy=pooled['accuracy'],leave_source_rx_accuracy_mean=statistics.mean(held),leave_source_rx_accuracy_sd=statistics.stdev(held),leave_source_rx_accuracies=dict(zip(map(str,RX),held)))
        text+=f"| {name} | {100*pooled['accuracy']:.4f} | {100*statistics.mean(held):.4f} | {100*statistics.stdev(held):.4f} | "+', '.join(f'{r}:{100*a:.4f}' for r,a in zip(RX,held))+' |\n'
    text+='\n六类均衡机会水平16.6667%。以上是固定线性探针的源诊断，SD来自五个源RX而非随机seed。留一组仍属于合法源V，不能称target泛化测试；低探针准确率不证明该坐标完全没有身份信息，高准确率也不证明它是稳定TX硬件参数。全部24结果与F1/CM保留，不从V调lambda/窗口/特征。\n\n'
    text+='| 角色 | TX主效应 | RX主效应 | day主效应 | TX×RX交互 | 剩余 |\n|---|---:|---:|---:|---:|---:|\n'
    for role,a in s['associations'].items():text+='| '+role+' | '+' | '.join(f'{100*a["association_fraction"][k]:.4f}%' for k in ('TX','RX','day','TX_RX_interaction','remaining'))+' |\n'
    text+='\n分解的是cos(argC)、sin(argC)的平衡源样本平方变化；剩余包含其他交互、包间变化和噪声。只描述身份/接收机/日期标签的统计关联，不是TX/RX硬件因果拆分。频偏仅是模1.25MHz的接收相对坐标：相同观测不能唯一决定TX与RX各自频率。线性Hz均值/SD只作逐组遥测，不用于方差归因。\n\n'
    text+='上一轮全部路径同步的公共相位及±80kHz仿射性质已经核实，但没有源性能收益。本诊断能检验“被删除坐标有身份关联”的一部分解释，不能单独归因上一轮性能下降；结构/优化差异及剩余RX/信道响应仍须区分。后续真正RFF研发须同时保留可辨识TX差异、处理明确接收干扰并通过冻结后独立clean性能验证。\n\n'
    text+=f"CPU-only完整耗时{s['elapsed_seconds']:.3f} s，零optimizer步；本次无正式CVS模型或新测试，目标性能仍未完成。原CE/noaug/身份骨干/同划分/clean-only目标不变，后续正式候选继续从零训练并默认完成测试。未热改、停止或重启任何既有任务。\n\n"
    text+='[完整180单元、24混淆矩阵与拟合参数](evidence/information_statistics.json) · [24行CSV](evidence/all24_source_probes.csv) · [独立完成读回](evidence/final_readback.json) · [核对结论](evidence/analysis_validation.json) · [原源负结果](../20261002-phase1-cvs-synchronized-identity-manysig-m8-r01/report.md)。\n'
    (report/'report.md').write_text(text,encoding='utf-8')
    verdict=dict(status='VERIFIED',source_only=True,counts=s['counts'],groups=180,fits=24,all_source_probe_confusions_recomputed=True,
                 full_text_checked=True,V_fit=False,target_access=False,checkpoint_access=False,goal_achieved=False,source_probe_summaries=summaries,source_associations=s['associations'])
    write(e/'analysis_validation.json',verdict);write(e/'next_source_handoff.json',dict(verdict,run_id=RUN,source_commit=c['commit'],target_scores_included=False,
         next_action='Use source-only circular information and receiver association to distinguish retained identity information from receiver shortcuts before preregistering next scratch CE architecture. No targetfeedback or selective rerun.'))
    print(json.dumps(verdict,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();analyze(a.root,a.artifact)
