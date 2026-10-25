"""Prepare typed, human-readable research tables for the weekly Excel report."""
import json,sys
from pathlib import Path
ROOT=Path('E:/type10-7');sys.path.insert(0,str(ROOT))
from tools.reports.pamf_pl_comparison import OUT,load,rows,PIDS,SHORT,TN,SN,MAIN_PANELS
P=ROOT/'local_artifacts/pamf_pl_comparison_report_20261025'
N={'riei':'RIEI','drift':'DRIFT','ibrff':'IB-RFF','sdg':'SDG','ordinary_PL':'PAMF','supervised':'PAMF监督'}
V=dict(SN,six_practical_mean='六种practical平均');MET={'accuracy':'准确率','macro_f1':'Macro-F1'}
CELL={'A':'无增强＋监督','B':'无增强＋普通伪标签','C':'有增强＋监督','D':'有增强＋普通伪标签','native':'PAMF原生有标签配方'}
ROUTE={'raw':'直接识别','DURA-L':'DURA仅用有标签源IQ','DURA-LU':'DURA用全部允许源IQ'}
def main():
    _,get=load();spec=[]
    def add(name,headers,data,widths,note='',copy=None):spec.append(dict(name=name,headers=headers,data=data,widths=widths,note=note,copy=copy))
    source='来源：已冻结独立评分结果；外部Full四组合与PAMF监督/普通伪标签；截至2026-10-25。'
    main=list(rows(OUT/'tables/main_comparison.csv'))
    h=['数据设置','方法','测试人群','测试场景','指标','均值（%）','标准差（%）','均值±标准差','模型种子数']
    data=[[SHORT[r['protocol']],r['display_method'],TN[r['test']],V[r['view']],MET[r['metric']],float(r['mean']),float(r['sd']),None,3] for r in main]
    add('主比较_数值',h,data,[27,22,25,24,16,17,17,25,15],source,dict(target=8,mean=6,sd=7))
    ix={(r['protocol'],r['display_method'],r['test'],r['view'],r['metric']):i+5 for i,r in enumerate(main)}
    broad=[]
    for pid in PIDS:
        for view in ('clean','six_practical_mean'):
            broad.append([SHORT[pid],V[view]]+['@copy:主比较_数值:'+str(ix[pid,label,'cross_rx',view,'accuracy']) for _,_,label in MAIN_PANELS])
    add('主比较_11设置',['数据设置','目标域场景']+[p[2] for p in MAIN_PANELS],broad,[27,24,22,22,22,22,22,24],'跨接收机准确率（%），三模型种子的均值±标准差；前五项直接识别。')
    allsum=list(rows(OUT/'tables/complete_summary.csv'))
    def flatten(rs,pamf=False):return [[SHORT[r['protocol']],N[r['method']],('监督训练' if r['method']=='supervised' else '普通伪标签') if pamf else CELL[r['cell']],ROUTE[r['readout']],TN[r['test']],V[r['view']],MET[r['metric']],float(r['mean']),float(r['sd']),None] for r in rs]
    heads=['数据设置','方法','训练设置','推理链','测试人群','场景','指标','均值（%）','标准差（%）','均值±标准差']
    widths=[27,20,27,36,25,24,16,17,17,25]
    add('无增强_CLEAN',heads,flatten([r for r in allsum if r['cell'] in ('A','B') and r['readout']=='raw']),widths,'仅目标接收机CLEAN；无信道增强，不接DURA。',dict(target=10,mean=8,sd=9))
    add('有增强_直接识别',heads,flatten([r for r in allsum if r['cell'] in ('C','D') and r['readout']=='raw']),widths,'CLEAN与六种practical完整保留；监督与普通伪标签分行；不接DURA。',dict(target=10,mean=8,sd=9))
    gains=[]
    for r in rows(OUT/'tables/DURA_generality.csv'):
        before=get(r['protocol'],r['method'],r['test'],rd='raw',v=r['view'],metric=r['metric']);after=get(r['protocol'],r['method'],r['test'],rd='DURA-LU',v=r['view'],metric=r['metric'])
        gains.append([SHORT[r['protocol']],N[r['method']],TN[r['test']],V[r['view']],MET[r['metric']],float(before['mean']),float(after['mean']),None,float(r['sd']),int(r['positive_seeds']),3])
    add('DURA泛用性',['数据设置','识别方法','测试人群','场景','指标','直接识别（%）','接DURA后（%）','提升（百分点）','配对差标准差','提升为正的种子数','种子数'],gains,[27,20,25,24,16,22,22,22,22,25,12],'固定有增强＋普通伪标签的同一识别器；先按相同模型种子计算恢复后减恢复前。')
    add('DURA接入_逐景',heads,flatten([r for r in allsum if r['cell']!='native' and r['readout']!='raw']),widths,'单独的恢复器实验；外部方法主结果不采用本表。无增强仅目标CLEAN，有增强保留全部场景。',dict(target=10,mean=8,sd=9))
    add('PAMF自身对照',heads,flatten([r for r in allsum if r['cell']=='native'],True),widths,'监督与普通伪标签采用相同有标签配方；恢复器权限固定后再解释伪标签收益。',dict(target=10,mean=8,sd=9))
    grouped=[]
    for r in rows(OUT/'tables/experiment_groups.csv'):
        part='PAMF自身监督控制' if r['method']=='supervised' else 'PAMF及DURA+PAMF主比较' if r['method']=='ordinary_PL' else '外部主比较及有增强控制' if r['cell']=='D' else '外部有增强控制' if r['cell']=='C' else '外部无增强CLEAN控制'
        grouped.append([SHORT[r['protocol']],N[r['method']], '开启' if r['channel_augmentation']=='on' else '关闭','开启' if r['PL']=='on' else '关闭',r['seed'],'CLEAN＋六种practical' if r['channel_augmentation']=='on' else '仅目标域CLEAN',part,'同一识别器恢复前后对照另列'])
    add('实验分组',['数据设置','方法','信道增强','普通伪标签','模型种子','报告场景','实验部分','DURA说明'],grouped,[27,20,17,17,22,32,40,42],'594个已完成逻辑模型；不以推理链数量代替独立模型数量。')
    td=[]
    for r in rows(OUT/'tables/PAMF_current_timing.csv'):
        method='PAMF监督' if r['case'].startswith('supervised_') else 'PAMF';route='DURA＋识别器优化图' if r['case'].endswith('_GF') else '识别器常规执行'
        td.append([method,route,int(r['batch']),float(r['wall_median_ms']),float(r['wall_P95_ms']),float(r['throughput_IQ_s'])])
    add('时延效率',['识别器','执行路径','批量','中位延迟（ms）','95%分位延迟（ms）','吞吐（IQ/秒）'],td,[23,36,13,24,27,25],'独占RTX3090、FP32、5轮计时；路径不同，不以单包时延比较推断恢复器算法加速。')
    summary=[['RIEI','raw'],['DRIFT','raw'],['IB-RFF','raw'],['SDG','raw'],['PAMF','raw'],['DURA+PAMF','DURA-LU']]
    result=dict(sheets=spec,summary=[dict(method=l,clean=ix['F0',l,'cross_rx','clean','accuracy'],practical=ix['F0',l,'cross_rx','six_practical_mean','accuracy']) for l,_ in summary],output=str(OUT/'PAMF实验周报数据.xlsx'),source=source)
    (P/'weekly_workbook_data.json').write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8');print([(s['name'],len(s['data'])) for s in spec])
if __name__=='__main__':main()
