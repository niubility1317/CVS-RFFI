"""Independent pure-stdlib reproduction from frozen counts and model-seed statistics."""
import argparse,collections,csv,gzip,json,math,re,statistics,zipfile,xml.etree.ElementTree as E
from pathlib import Path
DEFAULT=Path('E:/type10-7/exports/reports/20261026-PAMF-priority-comparisons-r23')
SEEDS=(2026092701,2026092702,2026092703);METHODS={'RIEI':'riei','DRIFT':'drift','IB-RFF':'ibrff','SDG':'sdg'}
def rows(p):
    with (gzip.open if p.suffix=='.gz' else open)(p,'rt',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def readjson(p):
    with (gzip.open if p.suffix=='.gz' else open)(p,'rt',encoding='utf-8-sig') as f:return json.load(f)
def near(a,b):assert isinstance(a,(float,int)) and abs(a-float(b))<1e-9,(a,b)
def col(n):
    s=''
    while n:n,k=divmod(n-1,26);s=chr(65+k)+s
    return s
def verify(root,workbook=None,payload_path=None,layout_path=None):
    if workbook is None and (root/'reproduce/workbook_model.json.gz').exists():
        workbook=root/'PAMF重点实验与补充数据.xlsx';payload_path=root/'reproduce/workbook_model.json.gz';layout_path=root/'reproduce/workbook_layouts.json'
    fields=('protocol','method','cell','test','readout','view','metric','seed');source={tuple(r[k] for k in fields):float(r['percent']) for r in rows(root/'frozen_inputs/per_seed_overall.csv.gz')}
    old={(r['test'],int(r['seed'])):float(r['percent']) for r in rows(root/'frozen_inputs/PAMF_existing_no_channel_supervised.csv') if r['metric']=='accuracy'}
    rec={r['id']:r for r in rows(root/'tables/priority_summary.csv')};cc=collections.defaultdict(list);maximum=0
    for r in rows(root/'frozen_inputs/receiver_only_reference.csv'):
        assert rec[r['id']]==r,'Previously delivered receiver-only statistics changed'
    receivers=rows(root/'frozen_inputs/m5_common_receiver_counts.csv');assert len(receivers)==756
    joint_receivers=rows(root/'frozen_inputs/m5_common_receiver_counts_joint.csv');assert len(joint_receivers)==756
    for test,rs in [('cross_rx',receivers),('cross_both',joint_receivers)]:
        for r in rs:
            assert int(r['count'])==(18000 if test=='cross_rx' else 6000) and 0<=int(r['correct'])<=int(r['count']);cc[test,r['method'],r['readout'],r['view'],int(r['seed'])].append(r)
    m5={}
    for k,rr in cc.items():
        assert len(rr)==6 and len({r['receiver'] for r in rr})==6
        m5[k]=100*sum(int(r['correct']) for r in rr)/sum(int(r['count']) for r in rr)
    scenes=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
    for r in rec.values():
        sup=r['learning_regime']=='supervised';aug=r['channel_augmentation']=='True';own='supervised' if sup else 'ordinary_PL';m=METHODS.get(r['method'],own);cell=('C' if sup else 'D') if aug else ('A' if sup else 'B');cell='native' if m==own else cell
        a=[]
        for s in SEEDS:
            if r['scope']=='existing_matched_supervised_no_channel_recipe':v=old[r['test'],s];assert sup and not aug and r['method']=='PAMF'
            elif r['scope']=='six_common_target_receivers' and r['protocol']=='F0':
                assert not sup and aug;v=statistics.mean(m5[r['test'],m,r['readout'],view,s] for view in scenes) if r['view']=='six_practical_mean' else m5[r['test'],m,r['readout'],r['view'],s]
            else:v=source[r['protocol'],m,cell,r['test'],r['readout'],r['view'],'accuracy',str(s)]
            a.append(v)
        assert a==[float(v) for v in r['seed_values'].split(';')]
        for k,v in [('mean',statistics.mean(a)),('sd',statistics.stdev(a))]:maximum=max(maximum,abs(v-float(r[k])));near(v,r[k])
        assert int(r['seed_n'])==3
    for test,file in [('cross_rx','DURA_generality_priority.csv'),('cross_both','DURA_generality_cross_both.csv')]:
        gains=rows(root/'tables'/file);assert {r['method'] for r in gains}==set(METHODS)|{'PAMF'}
        for r in gains:
            m=METHODS.get(r['method'],'ordinary_PL');cell='native' if m=='ordinary_PL' else 'D';a=[source['F0',m,cell,test,'DURA-LU','six_practical_mean','accuracy',str(s)]-source['F0',m,cell,test,'raw','six_practical_mean','accuracy',str(s)] for s in SEEDS]
            near(statistics.mean(a),r['delta_pp']);near(statistics.stdev(a),r['delta_sd']);assert int(r['positive_seeds'])==sum(v>0 for v in a)
    plan=json.loads((root/'chart_plan.json').read_text(encoding='utf-8'));assert len(plan['priority'])==9 and sum(c['panels'] for c in plan['priority'])==19 and sum(c['joint_panels'] for c in plan['priority'])==19 and plan['supplement_sheets']==26 and plan['joint_covered_sheets']==31 and plan['joint_native_panels']==47
    manifest=json.loads((root/'figures/manifest.json').read_text(encoding='utf-8'));assert len(manifest)==18 and sum(r['chart_type']=='line' for r in manifest)==4
    for r in manifest:
        assert r['PAMF_included'];s=(root/'figures'/(r['name']+'.svg')).read_text(encoding='utf-8');assert 'PAMF' in s and '#c4364e' in s.lower() and 'v2' not in s.lower()
        assert not re.search('[\u4e00-\u9fff]',s),r['name']
        if not r['name'].startswith('main_no_channel'):assert 'PAMF+DURA' in s or r['name'].startswith('DURA_generality')
        if r['name'].endswith('_cross_both'):assert 'Unseen date + RX' in s
    result=dict(status='VERIFIED',source_summary_records=len(rec),reproduced_seed_values=len(rec)*3,receiver_count_rows=len(receivers),joint_receiver_count_rows=len(joint_receivers),common_target_receivers=6,m5_models='Same-seed main models; frozen six-target aggregate',source_counts_all_1_through_6=True,priority_figures=18,priority_panels=38,joint_priority_figures=9,joint_priority_panels=19,joint_experiment_worksheets=31,not_applicable_count_or_timing_worksheets=4,original_receiver_only_statistics_unchanged=True,retained_supplement_figures=26,current_PL_PAMF_no_augmentation='MISSING; explicitly shown in both test populations; user chose no new training',DURA_generality_includes_PAMF=True,maximum_absolute_difference=maximum,new_training=0,new_prediction=0,new_truth_scoring=0)
    if workbook:
        ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main','c':'http://schemas.openxmlformats.org/drawingml/2006/chart','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
        payload=readjson(payload_path);layouts=readjson(layout_path);cells={};errs=[];nums=0;charts=0;linecharts=0;missing=0
        contracts=rows(root/'frozen_inputs/data_contracts.csv');protocols={r['display']:r['protocol'] for r in contracts}
        views=dict(zip(('CLEAN','高郊区','中郊区','低郊区','高城市','中城市','低城市','六种practical平均'),('clean',*scenes,'six_practical_mean')))
        joint_rows_checked=0
        for sp in payload['sheets']:
            for rr in sp.get('jointData',[]):
                r=dict(zip(sp['jointHeaders'],rr));sup=r['学习方式']=='监督';label=r.get('方法',r.get('识别方法'));m=METHODS.get(label,'supervised' if sup else 'ordinary_PL');cell=('C' if sup else 'D') if r.get('训练信道增强','开启')=='开启' else ('A' if sup else 'B');cell='native' if m not in METHODS.values() else cell
                p=protocols[r['数据设置']];v=views[r['场景']];metric='accuracy' if r['指标']=='准确率' else 'macro_f1';rd='DURA-L' if sup else 'DURA-LU'
                def values(readout):return [source[p,m,cell,'cross_both',readout,v,metric,str(s)] for s in SEEDS]
                if '识别方法' in r:
                    a=values('raw');b=values(rd);dif=[y-x for x,y in zip(a,b)]
                    for actual,expect in [(r['直接识别（%）'],statistics.mean(a)),(r['接DURA后（%）'],statistics.mean(b)),(r['配对提升（百分点）'],statistics.mean(dif)),(r['配对差标准差'],statistics.stdev(dif))]:near(expect,actual)
                    assert int(r['提升为正种子数'])==sum(x>0 for x in dif)
                else:
                    a=values('raw' if r['推理链']=='直接识别' else rd);near(statistics.mean(a),r['均值（%）']);near(statistics.stdev(a),r['标准差（%）'])
                joint_rows_checked+=1
        assert sum(bool(s.get('jointCharts') or s.get('jointChart')) for s in payload['sheets'])==31
        with zipfile.ZipFile(workbook) as z:
            assert z.testzip() is None;shared=[]
            if 'xl/sharedStrings.xml' in z.namelist():shared=[''.join(t.itertext()) for t in E.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)]
            names=[s.attrib['name'] for s in E.fromstring(z.read('xl/workbook.xml')).findall('s:sheets/s:sheet',ns)]
            assert names==[s['name'] for s in payload['sheets']] and len(names)==35
            for i,n in enumerate(names,1):
                vals={}
                for c in E.fromstring(z.read(f'xl/worksheets/sheet{i}.xml')).findall('.//s:row/s:c',ns):
                    t=c.attrib.get('t');v=c.find('s:v',ns);v=v.text if v is not None else None
                    if t=='s' and v is not None:v=shared[int(v)]
                    elif t=='inlineStr':v=''.join(c.find('s:is',ns).itertext())
                    elif v is not None and t not in ('str','e'):v=float(v)
                    if t=='e':errs.append((n,c.attrib['r'],v))
                    vals[c.attrib['r']]=v
                cells[n]=vals
            assert not errs,errs
            by={s['name']:s for s in payload['sheets']}
            def resolve(v):
                if not isinstance(v,str) or not v.startswith('@'):return v
                a=v.split(':');rr=by[a[1]]['data'][int(a[2])-5]
                if a[0]=='@num':return resolve(rr[ord(a[3])-65])
                return str(v)
            for sp in payload['sheets']:
                l=layouts[sp['name']];vals=cells[sp['name']]
                for j,rr in enumerate(sp['partitionData'],l['partitionHead']+1):
                    for k,v in enumerate(rr,1):
                        if isinstance(v,(int,float)):near(vals.get(col(k)+str(j)),v);nums+=1
                        else:assert vals.get(col(k)+str(j))==v
                if sp.get('charts'):
                    for id,j in l['index'].items():
                        r=rec[id]
                        for c,k in [('I','mean'),('J','sd')]:near(vals[c+str(j)],r[k]);nums+=1
                        for c,v in zip(('M','N','O'),r['seed_values'].split(';')):near(vals[c+str(j)],v);nums+=1
                        assert ('跨日且跨接收机' in vals['F'+str(j)])==(r['test']=='cross_both')
                    for ch,b in zip(sp['charts']+sp.get('jointCharts',[]),l['blocks']):
                        for si,s in enumerate(ch['series']):
                            for j,id in enumerate(s['ids'],b['head']+1):
                                v=vals.get(col(si+2)+str(j))
                                if id:near(v,rec[id]['mean']);nums+=1
                                else:assert v is None;missing+=1
                else:
                    for j,rr in enumerate(sp['data'],l['dataStart']):
                        for k,v in enumerate(rr,1):
                            if sp.get('copy') and k==sp['copy']['target']:continue
                            if sp.get('gain_formula') and k==sp['gain_formula']['target']:v=float(rr[sp['gain_formula']['after']-1])-float(rr[sp['gain_formula']['before']-1])
                            v=resolve(v)
                            if isinstance(v,(int,float)):near(vals.get(col(k)+str(j)),v);nums+=1
                            elif v is not None and not str(v).startswith('@'):assert vals.get(col(k)+str(j))==v
                    if sp.get('jointChart'):
                        for j,rr in enumerate(sp['jointData'],l['jointTableHead']+1):
                            assert rr[sp['jointHeaders'].index('测试人群')]=='目标日期与接收机双跨'
                            for k,v in enumerate(rr,1):
                                if sp.get('copy') and k==sp['copy']['target']:continue
                                if sp.get('gain_formula') and k==sp['gain_formula']['target']:v=float(rr[sp['gain_formula']['after']-1])-float(rr[sp['gain_formula']['before']-1])
                                if isinstance(v,(int,float)):near(vals.get(col(k)+str(j)),v);nums+=1
                                elif v is not None:assert vals.get(col(k)+str(j))==v
                        for j,rr in enumerate(sp['jointChart']['data'],l['jointChartHead']+1):
                            for k,v in enumerate(rr,1):
                                v=resolve(v)
                                if isinstance(v,(int,float)):near(vals.get(col(k)+str(j)),v);nums+=1
                                elif v is not None:assert vals.get(col(k)+str(j))==v
            paths=[n for n in z.namelist() if re.fullmatch(r'xl/(?:drawings/)?charts/chart\d+\.xml',n)];assert len(paths)==98
            palette=set();expected_units=[]
            for sp in payload['sheets']:expected_units.extend([c['unit'] for c in sp['charts']+sp.get('jointCharts',[])] if sp.get('charts') else [sp['chart']['unit']]+([sp['jointChart']['unit']] if sp.get('jointChart') else []))
            for n in paths:
                c=E.fromstring(z.read(n));bars=c.findall('.//c:barChart',ns);line=c.findall('.//c:lineChart',ns);assert bars or line
                ci=int(re.search(r'chart(\d+)\.xml',n).group(1))-1
                unit=''.join(t.text or '' for t in c.findall('.//c:valAx/c:title//a:t',ns));assert unit==expected_units[ci],(n,unit,expected_units[ci])
                minimum=c.find('.//c:valAx/c:scaling/c:min',ns);assert minimum is not None and float(minimum.attrib['val'])==0,(n,'Axis must start at zero')
                if '(%)' in unit:
                    maximum_node=c.find('.//c:valAx/c:scaling/c:max',ns);assert maximum_node is not None and float(maximum_node.attrib['val'])==100,(n,'Percent axis must end at 100')
                if bars:assert bars[0].find('c:barDir',ns).attrib['val']=='col'
                if line:
                    linecharts+=1
                    dash=line[0].findall('c:ser',ns)[-1].find('c:spPr/a:ln/a:prstDash',ns)
                    assert dash is not None and dash.attrib['val']!='solid','Restored PAMF line must remain distinguishable'
                palette.update(t.attrib['val'] for t in c.findall('.//a:srgbClr',ns))
                for ref in c.findall('.//c:val/c:numRef/c:f',ns):
                    m=re.fullmatch(r"'([^']+)'!\$([A-Z]+)\$(\d+)(?::\$([A-Z]+)\$(\d+))?",ref.text);assert m,ref.text
                    name,c1,r1,c2,r2=m.groups();c2=c2 or c1;r2=r2 or r1;assert c1==c2
                    assert all(cells[name].get(c1+str(j)) is None or isinstance(cells[name].get(c1+str(j)),(int,float)) for j in range(int(r1),int(r2)+1))
                charts+=1
            assert linecharts==4 and 'C4364E' in {p.upper() for p in palette} and 'E29525' in {p.upper() for p in palette} and missing==2
        result.update(workbook=dict(sheets=35,native_charts=charts,line_charts=linecharts,retained_original_worksheets=24,additional_legacy_full_panels=2,priority_worksheets=9,joint_experiment_worksheets=31,new_joint_charts=47,joint_supplement_rows_independently_recomputed=joint_rows_checked,source_numeric_cells_checked=nums,missing_points_preserved=missing,formula_errors=errs,colors_exported=True,axis_units_verified=charts,all_axes_start_at_zero=True,percentage_axes_zero_to_100=True))
    (root/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False),flush=True);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--workbook',type=Path);p.add_argument('--payload',type=Path);p.add_argument('--layouts',type=Path);a=p.parse_args();verify(a.root,a.workbook,a.payload,a.layouts)
