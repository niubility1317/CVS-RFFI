"""Add separate unseen-date/unseen-receiver evidence to each applicable experiment.

Consumes retained independent score summaries and receiver-level counts only.
The existing receiver-only panels and all original workbook data remain intact.
"""
import copy,json,statistics,sys
from pathlib import Path
ROOT=Path('E:/type10-7');sys.path.insert(0,str(ROOT))
from tools.reports.pamf_priority_data import OUT,P,LABELS,METHODS,SCENES,SEEDS,Data,rows,write,dump

JOINT_LABEL='目标跨日且跨接收机'
LEGACY_JOINT='目标日期与接收机双跨'

def main():
    payload=json.loads((P/'base_workbook_data.json').read_text(encoding='utf-8'))
    d=Data();records={r['id']:r for r in payload['records']};original=list(records.values())
    contracts={r['protocol']:r for r in rows(OUT/'frozen_inputs/data_contracts.csv')}
    joint_records={}
    def convert_record(id):
        if id is None:return None
        key='joint|'+id
        if key in joint_records:return key
        r=records[id];n=copy.deepcopy(r);n['id']=key;n['test']='cross_both'
        sup=r['learning_regime']=='supervised';m=METHODS[LABELS.index(r['method'])] if r['method'] in LABELS[:4] else 'supervised' if sup else 'ordinary_PL'
        aug=r['channel_augmentation'];cell=('C' if sup else 'D') if aug else ('A' if sup else 'B')
        if m in ('supervised','ordinary_PL'):cell='native'
        a=[]
        for s in SEEDS:
            if r['scope']=='existing_matched_supervised_no_channel_recipe':v=d.histor['cross_both','accuracy',s]
            elif r['scope']=='six_common_target_receivers' and r['protocol']=='F0':
                v=statistics.mean(d.m5_joint[m,r['readout'],view,s] for view in SCENES[1:]) if r['view']=='six_practical_mean' else d.m5_joint[m,r['readout'],r['view'],s]
            else:v=d.source[r['protocol'],m,cell,'cross_both',r['readout'],r['view'],'accuracy',str(s)]
            a.append(v)
        n.update(mean=statistics.mean(a),sd=statistics.stdev(a),seed_values=';'.join(map(str,a)))
        joint_records[key]=n;return key
    def joint_charts(sp):
        cs=copy.deepcopy(sp['charts'])
        for ch in cs:
            ch['test']='cross_both';ch['title']='双跨 · '+ch['title'].replace('目标跨接收机','目标跨日且跨接收机')
            for s in ch['series']:s['ids']=[convert_record(k) for k in s['ids']]
        return cs
    byname={s['name']:s for s in payload['sheets']}
    def metadata_n(rr):return next(i for i,x in enumerate(rr) if isinstance(x,(float,int)))
    indexes={}
    for s in payload['sheets']:
        if 'data' not in s or '测试人群' not in s.get('headers',[]):continue
        n=metadata_n(s['data'][0]);indexes[s['name']]={tuple(r[:n]):i for i,r in enumerate(s['data'])}
    def joint_token(v):
        if not isinstance(v,str) or not v.startswith('@'):return v
        a=v.split(':');sp=byname[a[1]];r=copy.deepcopy(sp['data'][int(a[2])-5]);j=sp['headers'].index('测试人群');r[j]=LEGACY_JOINT;n=metadata_n(r)
        i=indexes[sp['name']][tuple(r[:n])];a[2]=str(i+5);return ':'.join(a)
    coverage=[]
    for sp in payload['sheets']:
        if sp.get('charts'):
            for ch in sp['charts']:ch['test']='cross_rx'
            sp['jointCharts']=joint_charts(sp)
            sp['jointSlug']=sp['slug']+'_cross_both'
            sp['partitionHeaders'][-1]='跨接收机测试包/景'
            sp['partitionHeaders'].append('跨日且跨接收机测试包/景')
            for pp,pid in zip(sp['partitionData'],sp['protocols']):
                pp.append(36000 if sp['slug'].startswith('source_count') and pid=='F0' else int(contracts[pid]['cross_both']))
            sp['explain'][2][1]='原图为目标接收机D1–D3跨接收机；新增独立图/表为同一目标接收机D0跨日且跨接收机。两组不合并；六景先种子内平均，误差为三模型种子样本标准差。'
            sp['explain'][4][1]='训练/源验证：2021年3月8、15、23日；双跨测试：未参与训练或验证的目标接收机、2021年3月1日（更早未见日）。'
            sp['notes']=(sp.get('notes','')+' 新增双跨图与原跨接收机图单独成块，日期和样本数各自列明。').strip()
            coverage.append(dict(sheet=sp['name'],applicability='ADDED',joint_panels=len(sp['jointCharts']),table='Separate chart source matrix plus full seed statistics',test='cross_both'))
        elif '测试人群' in sp.get('headers',[]) and sp['chart']['unit'] in ('Accuracy (%)','Macro-F1 (%)'):
            ch=copy.deepcopy(sp['chart']);ch['title']='跨日且跨接收机 · '+ch['title'].replace('跨接收机','').replace('三类人群','目标D0').replace('三人群','目标D0')
            if ch['headers'][0]=='测试人群':ch['data']=[r for r in ch['data'] if r[0]==LEGACY_JOINT]
            if '跨接收机CLEAN' in ch['headers']:
                ch['headers']=[ch['headers'][0],'双跨CLEAN'];ch['data']=[[r[0],r[2]] for r in ch['data']]
            ch['data']=[[joint_token(v) for v in r] for r in ch['data']]
            ch['note']='仅目标接收机D0：2021年3月1日，源训练/验证日期为3月8、15、23日。无增强仅CLEAN；有增强保留CLEAN及六景，六景均值不含CLEAN。'
            sp['jointChart']=ch
            j=sp['headers'].index('测试人群');n=metadata_n(sp['data'][0]);has_joint=any(r[j]==LEGACY_JOINT for r in sp['data'])
            if has_joint:jd=[copy.deepcopy(r) for r in sp['data'] if r[j]==LEGACY_JOINT]
            else:
                canonical=byname['补_监督_完整数值' if '普通PL半监督' not in sp['data'][0] else '补_半监督_完整数值'];pool={tuple(r[:n]):r for r in canonical['data']};jd=[]
                for r in sp['data']:
                    rr=copy.deepcopy(r);rr[j]=LEGACY_JOINT;jd.append(copy.deepcopy(pool[tuple(rr[:n])]))
            assert jd and all(r[j]==LEGACY_JOINT for r in jd)
            sp['jointHeaders']=copy.deepcopy(sp['headers']);sp['jointData']=jd
            sp['jointExplanation']='新增独立双跨图与筛选表；原图、原完整数据及来源公式均保留。'+ch['note']
            coverage.append(dict(sheet=sp['name'],applicability='ADDED',joint_panels=1,table_rows=len(jd),test='cross_both'))
        else:
            sp['jointNotApplicable']='本页为模型覆盖、实验登记或独占时延，不属于按目标日期/接收机划分的识别性能实验。'
            coverage.append(dict(sheet=sp['name'],applicability='NOT_APPLICABLE',joint_panels=0,table='No fabricated population-specific latency or model count',test=''))
    records.update(joint_records);payload['records']=list(records.values());payload['output']=str(OUT/'PAMF重点实验与补充数据.xlsx')
    payload['joint_definition']=dict(test='cross_both',display=JOINT_LABEL,date='2021-03-01',train_dates=['2021-03-08','2021-03-15','2021-03-23'],future_extrapolation_claim=False)
    assert len(coverage)==35 and sum(c['applicability']=='ADDED' for c in coverage)==31
    # Receiver-only results stay exactly as delivered in r22.
    assert list(records.values())[:len(original)]==original
    write(OUT/'tables/priority_summary.csv',records.values())
    write(OUT/'tables/joint_summary.csv',joint_records.values())
    write(OUT/'tables/joint_worksheet_coverage.csv',[dict(sheet=c['sheet'],applicability=c['applicability'],joint_panels=c['joint_panels'],test=c['test']) for c in coverage])
    gains=[]
    for l in LABELS[:5]:
        m=METHODS[LABELS.index(l)] if l in LABELS[:4] else 'ordinary_PL';cell='D' if l in LABELS[:4] else 'native'
        a=[d.source['F0',m,cell,'cross_both','raw','six_practical_mean','accuracy',str(s)] for s in SEEDS];b=[d.source['F0',m,cell,'cross_both','DURA-LU','six_practical_mean','accuracy',str(s)] for s in SEEDS];dd=[y-x for x,y in zip(a,b)]
        gains.append(dict(method=l,before_mean=statistics.mean(a),after_mean=statistics.mean(b),delta_pp=statistics.mean(dd),delta_sd=statistics.stdev(dd),positive_seeds=sum(x>0 for x in dd),seed_values=';'.join(map(str,dd))))
    write(OUT/'tables/DURA_generality_cross_both.csv',gains)
    plan=dict(priority=[dict(name=s['name'],slug=s['slug'],panels=len(s['charts']),joint_slug=s['jointSlug'],joint_panels=len(s['jointCharts'])) for s in payload['sheets'] if s.get('priority')],supplement_sheets=26,joint_covered_sheets=31,joint_not_applicable_sheets=4,joint_native_panels=sum(c['joint_panels'] for c in coverage),missing_current_PAMF_PL_no_channel=True,m5_same_main_models_common_six_targets=True)
    dump(OUT/'chart_plan.json',plan);dump(P/'weekly_workbook_data.json',payload)
    dump(P/'joint_coverage.json',coverage)
    print(json.dumps(dict(status='PREPARED',records=len(records),new_joint_records=len(joint_records),joint_experiment_sheets=31,new_native_charts=plan['joint_native_panels'],missing_PL_no_augmentation='Preserved in each applicable population',source_date='2026-10-26'),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
