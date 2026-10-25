"""Prepare worksheet sections with learning regime as the first reporting dimension."""
import json,sys
from pathlib import Path
ROOT=Path('E:/type10-7');sys.path.insert(0,str(ROOT))
from tools.reports.pamf_reporting_model import *
P=ROOT/'local_artifacts/pamf_pl_comparison_report_20261025_r20'
HEAD=['数据设置','方法','学习方式','训练信道增强','推理链','测试人群','场景','指标','均值（%）','标准差（%）','均值±标准差','模型种子数']
WIDTH=[26,24,23,22,28,31,23,18,18,18,25,16]
LABELS=['RIEI','DRIFT','IB-RFF','SDG','PAMF','DURA+PAMF']
def row(r):
    reg=r.get('learning_regime',mode(r));rd='直接识别' if r['readout']=='raw' else '先DURA恢复后识别'
    return [SHORT[r['protocol']],r.get('display_method',NAMES.get(r['method'],'PAMF')),RN[reg],'关闭' if r['cell'] in ('A','B') else '开启',rd,TEST[r['test']],VIEW[r['view']],METRIC[r['metric']],float(r['mean']),float(r['sd']),None,3]
def main():
    result=Results();spec=[];by={};contracts={r['protocol']:r for r in rows(OUT/'tables/data_contracts.csv')}
    def add(name,reg,pids,data,question,design,slug,title,chartheaders,chartdata,unit='Accuracy (%)',headers=HEAD,widths=WIDTH,copy=True):
        s=dict(name=name,regime=reg,protocols=list(pids),headers=headers,data=data,widths=widths,question=question,design=design,copy=dict(target=11,mean=9,sd=10) if copy else None,chart=dict(slug=slug,title=title,type='bar',orientation='column',headers=chartheaders,data=chartdata,unit=unit,note='竖向柱图；图示均值，标准差与完整逐景数据见下方。'))
        spec.append(s);by[name]=s;return s
    # Build the two independent numerical sources before creating formula references.
    sourceNames={r:RN[r].replace('普通PL','')+'_完整数值' for r in REGIMES}
    for reg in REGIMES:
        rr=[r for r in rows(OUT/'tables/main_comparison.csv') if r['learning_regime']==reg];data=[row(r) for r in rr]
        add(sourceNames[reg],reg,PIDS,data,'查询本监督条件下各方法、场景与人群的数值','本页全部方法处于同一学习条件；两条PAMF推理链共享识别器，其他方法直接识别。',f'{reg}_population','本组：三类人群的六景结果',['测试人群']+LABELS,[],copy=True)
        by[sourceNames[reg]]['source_index']={(r['protocol'],r['display_method'],r['test'],r['view'],r['metric']):i+5 for i,r in enumerate(rr)}
    def mainref(reg,p,label,t='cross_rx',view='six_practical_mean',metric='accuracy',c='I'):
        s=by[sourceNames[reg]];return f"@num:{s['name']}:{s['source_index'][p,label,t,view,metric]}:{c}"
    def rows_main(reg,pids,views=('clean','six_practical_mean'),tasks=('cross_rx',),metrics=('accuracy',)):
        return [row(r) for p in pids for t in tasks for v in views for metric in metrics for r in result.main(reg,p,t,v,metric)]
    def series_main(reg,pids,metric='accuracy',view='six_practical_mean'):
        return [[SHORT[p]]+[mainref(reg,p,l,view=view,metric=metric) for l in LABELS] for p in pids]
    for reg in REGIMES:
        tag='监督' if reg=='supervised' else '半监督';cell,own,restore=settings(reg)
        name=f'{tag}_标准对比'
        add(name,reg,['F0'],rows_main(reg,['F0'],tasks=TASKS),f'{RN[reg]}条件下，六种推理系统如何比较','四种外部完整方法、PAMF与DURA+PAMF均使用有信道增强训练；外部方法不接DURA。',f'{reg}_standard',f'{tag}组：标准划分跨接收机六景',['方法','六景准确率'],[[l,mainref(reg,'F0',l)] for l in LABELS])
        for suffix,pids,question,design,slug in [
          ('接收机划分',PIDS[:3],'更换源与目标接收机成员后的组内泛化','保持5源、7目标、相同日期与样本预算；三套源集合两两只交一个接收机。','receivers'),
          ('标注比例',('label01','label05','F0'),'减少有标签训练记录对本组的影响','标注率为源训练池的1%/5%/10%，源验证量保持不变；监督组不读取剩余无标签训练记录。','labels'),
          ('源接收机数量',('m1-natural','m2-natural','m3-natural','m4-natural','m6-natural'),'增加源接收机及自然训练量对本组的影响','只比较1/2/3/4/6源，固定同六个目标接收机；样本预算随源数改变，不插入标准5源/7目标划分。','sources'),
          ('十类规模',('F0','ManyRx'),'十类不均衡数据下本组是否保持性能','六类ManySig与十类ManyRx的接收机、样本量也不同；这是规模条件对照，不是只改变类数的因果实验。','ten_class')]:
            metric='macro_f1' if slug=='ten_class' else 'accuracy';title=f'{tag}组：'+suffix+(' Macro-F1' if slug=='ten_class' else '六景准确率')
            ch=series_main(reg,pids,metric=metric)
            add(f'{tag}_{suffix}',reg,pids,rows_main(reg,pids,metrics=('accuracy','macro_f1') if slug=='ten_class' else ('accuracy',)),question,design,f'{reg}_{slug}',title,['数据设置']+LABELS,ch,unit='Macro-F1 (%)' if slug=='ten_class' else 'Accuracy (%)')
        noCell='A' if reg=='supervised' else 'B';off=[dict(r,learning_regime=reg,display_method=NAMES[r['method']]) for r in result.source if r['method'] in METHODS and r['cell']==noCell and r['readout']=='raw']
        offData=[row(r) for r in off]
        def offref(m,t):
            i=next(i for i,r in enumerate(off) if r['protocol']=='F0' and r['method']==m and r['test']==t and r['view']=='clean' and r['metric']=='accuracy');return f'@num:无增强_{tag}_CLEAN:{i+5}:I'
        add(f'无增强_{tag}_CLEAN',reg,PIDS,offData,'无附加信道增强时，本学习条件下的目标CLEAN结果','仅列四种外部方法的同条件冻结结果；当前PAMF主矩阵为有增强，不挪入本页，也不补零。',f'{reg}_no_channel',f'{tag}组无增强：仅目标CLEAN',['方法','跨接收机CLEAN','双跨CLEAN'],[[NAMES[m],offref(m,'cross_rx'),offref(m,'cross_both')] for m in METHODS])
        add(f'有增强_{tag}_全景',reg,PIDS,rows_main(reg,PIDS,views=(*SCENES,'six_practical_mean'),tasks=TASKS,metrics=('accuracy','macro_f1')),'本组在CLEAN与各practical场景下分别表现如何','训练启用信道增强；所有六种方法均属同一学习条件，只有DURA+PAMF接入恢复器。',f'{reg}_scenes',f'{tag}组有增强：逐场景主比较',['场景']+LABELS,[[VIEW[v]]+[mainref(reg,'F0',l,view=v) for l in LABELS] for v in SCENES])
        gains=[r for r in rows(OUT/'tables/DURA_generality.csv') if r['learning_regime']==reg]
        gainData=[[SHORT[r['protocol']],NAMES[r['method']],RN[reg],TEST[r['test']],VIEW[r['view']],METRIC[r['metric']],float(r['before_mean']),float(r['after_mean']),float(r['mean']),float(r['sd']),int(r['positive_seeds']),3] for r in gains]
        def gainref(m,c):
            i=next(i for i,r in enumerate(gains) if r['protocol']=='F0' and r['method']==m and r['test']=='cross_rx' and r['view']=='six_practical_mean' and r['metric']=='accuracy');return f'@num:DURA泛用性_{tag}:{i+5}:{c}'
        gs=add(f'DURA泛用性_{tag}',reg,PIDS,gainData,'DURA对本组不同外部识别器的恢复作用','固定同一个有增强识别器，只切换是否接入同契约DURA；不把这些恢复后成绩放进外部方法主列。',f'{reg}_dura',f'{tag}组：DURA泛用性',['方法','直接识别','接入DURA'],[[NAMES[m],gainref(m,'G'),gainref(m,'H')] for m in METHODS],headers=['数据设置','识别方法','学习方式','测试人群','场景','指标','直接识别（%）','接DURA后（%）','配对提升（百分点）','配对差标准差','提升为正种子数','种子数'],widths=[26,22,23,31,23,18,22,22,23,23,22,16],copy=False)
        gs['gain_formula']=dict(target=9,after=8,before=7)
        gs['chart']['delta']=True;gs['chart']['headers']=['方法','未接入DURA','接入DURA']
        gs['chart']['title']=f'{tag}组DURA：标准划分跨接收机六景准确率'
        gs['chart']['note']='标准划分、目标跨接收机测试；六种practical准确率均值，不含CLEAN。双柱为同一识别器恢复前后，标准差与逐景数据见下方。'
        pamfRows=[dict(r,display_method='PAMF',learning_regime=reg) for r in result.source if r['method']==own and r['readout'] in ('raw',restore)]
        ps=add(f'PAMF恢复对照_{tag}',reg,PIDS,[row(r) for r in pamfRows],'本组PAMF接入匹配恢复器前后的变化','PAMF和DURA+PAMF共享同一识别器；本页不把监督与半监督混画，恢复器数据权限与本组一致。',f'{reg}_pamf_restoration',f'{tag}组PAMF：恢复前后',['测试人群','未接入DURA','接入DURA'],[[TEST[t],mainref(reg,'F0','PAMF',t=t),mainref(reg,'F0','DURA+PAMF',t=t)] for t in TASKS]);ps['chart']['delta']=True
        ps['chart']['title']=f'{tag}组PAMF：标准划分三人群六景准确率'
        ps['chart']['note']='标准划分的三类互斥测试人群；图示六种practical准确率均值，不含CLEAN。同一识别器恢复前后配对，逐景结果见下方。'
        timing=[]
        for r in rows(OUT/'tables/PAMF_current_timing.csv'):
            if r['case'].startswith(own+'_'):timing.append(['PAMF',RN[reg],'识别器常规执行' if r['case'].endswith('_F') else 'DURA+PAMF优化图',int(r['batch']),float(r['wall_median_ms']),float(r['wall_P95_ms']),float(r['throughput_IQ_s'])])
        def timeref(b,route):return f'@num:时延效率_{tag}:{5+next(i for i,r in enumerate(timing) if r[2]==route and r[3]==b)}:E'
        add(f'时延效率_{tag}',reg,['F0'],timing,'本组PAMF的实际批量延迟与吞吐','独占RTX3090、FP32、5轮计时；常规识别与完整链优化图分列，执行路径不同，不归因成增加恢复器带来加速。',f'{reg}_timing',f'{tag}组：批量与中位延迟',['批量','PAMF常规执行','DURA+PAMF优化图'],[[str(b),timeref(b,'识别器常规执行'),timeref(b,'DURA+PAMF优化图')] for b in (1,8,32,64,128)],unit='Median latency (ms)',headers=['识别器','学习方式','执行路径','批量','中位延迟（ms）','95%分位延迟（ms）','吞吐（IQ/秒）'],widths=[24,23,32,18,24,27,25],copy=False)
        by[sourceNames[reg]]['chart']['data']=[[TEST[t]]+[mainref(reg,'F0',l,t=t) for l in LABELS] for t in TASKS]
    overview=[['监督',132,132,33,297,165,198],['普通PL半监督',132,132,33,297,165,198]]
    add('实验总览',None,PIDS,overview,'先分学习方式，再分增强开关与推理链','两套平行主比较：监督对监督、普通PL半监督对普通PL半监督。每套均列RIEI、DRIFT、IB-RFF、SDG、PAMF、DURA+PAMF。','coverage','每种学习条件下的逻辑模型覆盖',['类别','每组独立模型数'],[['外部无增强',132],['外部有增强',132],['PAMF',33]],unit='Classifier models',headers=['学习方式','外部无增强模型','外部有增强模型','PAMF模型','全部独立模型','主比较独立识别器','主比较推理链展示项'],widths=[26,24,24,23,23,28,28],copy=False)
    grouped=[]
    for r in rows(OUT/'tables/experiment_groups.csv'):grouped.append([SHORT[r['protocol']],NAMES.get(r['method'],'PAMF'),RN[r['learning_regime']],'开启' if r['channel_augmentation']=='on' else '关闭',r['seed'],'仅目标CLEAN' if r['channel_augmentation']=='off' else 'CLEAN＋全部六种practical','恢复器仅用有标签源IQ' if r['learning_regime']=='supervised' else '恢复器可用全部允许源训练IQ'])
    add('实验登记',None,PIDS,grouped,'逐模型核对实验归组与允许报告范围','登记是已完成模型的目录，不是混合性能表；同一识别器的多个读出不能计为多次训练。','registry','两种学习条件合计的逻辑模型数',['方法','独立模型数'],[[NAMES[m],132] for m in METHODS]+[['PAMF',66]],unit='Classifier models',headers=['数据设置','方法','学习方式','训练信道增强','模型种子','报告场景','DURA源IQ权限'],widths=[26,24,24,24,24,36,46],copy=False)
    for s in spec:decorate(s,contracts)
    spec=sorted(spec,key=lambda s:(0 if s['name']=='实验总览' else 3 if s['name']=='实验登记' else 1 if s['regime']=='supervised' else 2,0 if '标准对比' in s['name'] else 99 if '完整数值' in s['name'] else next((i for i,k in enumerate(('接收机划分','标注比例','源接收机数量','十类规模','CLEAN','全景','DURA泛用性','PAMF恢复','时延')) if k in s['name']),80)))
    for s in spec:s.pop('source_index',None)
    P.mkdir(parents=True,exist_ok=True);payload=dict(sheets=spec,output=str(OUT/'PAMF实验周报数据-监督与半监督分组.xlsx'),source='已冻结独立评分结果；未重新读取IQ、checkpoint或目标真值。')
    (P/'weekly_workbook_data.json').write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8');print(json.dumps([(s['name'],len(s['data'])) for s in spec],ensure_ascii=False),flush=True)
def decorate(s,contracts):
    reg=s['regime'];off='无增强' in s['name'];tag=RN.get(reg,'实验登记与覆盖')
    s['explain']=[['学习条件',tag+'；本页性能图不混入另一学习条件。' if reg else '监督与半监督分别建页；本页只展示模型覆盖，不比较混合性能。'],['实验目的',s['question']],['固定与改变',s['design']],['PAMF训练','监督识别器50轮全部只用有标签源训练，11100更新，不建立无标签训练加载器。' if reg=='supervised' else '普通PL半监督识别器50轮、11100更新；前13轮只用标签，后37轮联合使用有标签与无标签源训练。' if reg else '监督：50轮有标签；半监督：13轮有标签＋37轮联合训练。两者不混称同一学习方式。'],['外部方法','保留原生Full目标、预算和默认头；本组关闭普通PL，不读取无标签训练IQ。' if reg=='supervised' else '保留原生Full目标、预算和默认头；本组附加普通PL，阈值0.95、权重1、整批平均，无教师和复杂门控。' if reg else 'RIEI、DRIFT、IB-RFF、SDG各自明确监督/普通PL半监督归属。'],['增强与测试','本页关闭附加信道增强，仅目标接收机CLEAN；不报告practical，也不接DURA。' if off else '本页有增强结果报告CLEAN及全部六种practical；无增强控制在单独CLEAN页。'],['恢复器权限','本组DURA只用有标签源集合的IQ，不读取发射机标签。完整主系统也不使用无标签训练池。' if reg=='supervised' else '本组DURA可用全部允许源训练IQ，仍不读取发射机标签、识别器输出或目标域数据。' if reg else '监督主系统配仅有标签源IQ恢复器；半监督主系统配全部允许源训练IQ恢复器。'],['数据与日期','源/目标接收机与物理记录按原契约隔离；训练与验证日期为2021年3月8/15/23日，3月1日是更早未见日。'],['统计口径','三个模型种子2026092701/02/03；先种子内平均六景或计算配对差，再报均值±样本标准差。'],['解释边界','监督与半监督之间不直接把完整系统差全部归因于伪标签；保留组内负值、零接受和非单调。'],['冻结证据','本次仅组织现有冻结、独立评分结果；新增训练、预测、目标真值访问和DURA训练均为0。']]
    if '时延' in s['name']:
        s['explain'][3]=['测量条件','冻结PAMF与DURA；独占RTX3090、FP32；批量1/8/32/64/128，每项5轮计时。']
        s['explain'][8]=['统计口径','中位延迟、95%分位与吞吐；5轮计时不等于5个训练种子。']
    s['partitionHeaders']=['数据设置','类别数','源接收机索引','目标接收机索引','有标签训练包','无标签训练包','源验证包','源日期','跨接收机测试包/景','双跨测试包/景','跨日诊断包/景']
    s['partitionData']=[[r['display'],int(r['classes']),r['source_Rx'].replace(';',', '),r['target_Rx'].replace(';',', '),int(r['L_s']),int(r['U_s']),int(r['V']),'3月8/15/23日',int(r['cross_rx']),int(r['cross_both']),'本页不报告' if off else int(r['cross_date'])] for r in (contracts[p] for p in s['protocols'])]
    s['partitionNote']='监督组的数据划分仍保留无标签池计数用于契约说明，但监督训练与监督主系统不读取该池。'
    if reg=='supervised':s['partitionHeaders'][5]='无标签池（本组不读取）'
if __name__=='__main__':main()
