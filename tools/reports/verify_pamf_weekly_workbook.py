"""Read saved XLSX OOXML and compare experimental values to frozen CSV evidence."""
import csv,gzip,json,math,re,statistics,zipfile,xml.etree.ElementTree as E
from pathlib import Path
ROOT=Path('E:/type10-7');OUT=ROOT/'exports/reports/20261025-PAMF-PL-comparison-r20';P=ROOT/'local_artifacts/pamf_pl_comparison_report_20261025_r20'
NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main','c':'http://schemas.openxmlformats.org/drawingml/2006/chart','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
def csvrows(p):
    with (gzip.open if p.suffix=='.gz' else open)(p,'rt',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def col(n):
    s=''
    while n:n,k=divmod(n-1,26);s=chr(65+k)+s
    return s
def main():
    payload=json.loads((P/'weekly_workbook_data.json').read_text(encoding='utf-8'));layouts=json.loads((P/'workbook_layouts.json').read_text(encoding='utf-8'));book=Path(payload['output'])
    contracts={r['display']:r for r in csvrows(OUT/'tables/data_contracts.csv')};ps={k:r['protocol'] for k,r in contracts.items()}
    fields=('protocol','method','cell','test','readout','view','metric');source={tuple(r[k] for k in fields):r for r in csvrows(OUT/'tables/complete_summary.csv')}
    seeds={tuple(r[k] for k in fields)+(int(r['seed']),):float(r['percent']) for r in csvrows(OUT/'tables/per_seed_overall.csv.gz')}
    tests={'源接收机跨日诊断':'cross_date','目标跨接收机':'cross_rx','目标日期与接收机双跨':'cross_both'}
    views={'CLEAN':'clean','高郊区':'practical_high','中郊区':'practical_mid','低郊区':'practical_low_suburban','高城市':'practical_high_urban','中城市':'practical_mid_urban','低城市':'practical_low_urban','六种practical平均':'six_practical_mean'}
    metrics={'准确率':'accuracy','Macro-F1':'macro_f1'};methods={'RIEI':'riei','DRIFT':'drift','IB-RFF':'ibrff','SDG':'sdg'}
    checked=0;parts=0;formula_count=0;double_charts=0;cache_points=0;range_points=0;errors=[];sheets={}
    def near(a,b):
        assert isinstance(a,(int,float)) and abs(a-float(b))<1e-9,(a,b)
    with zipfile.ZipFile(book) as z:
        assert z.testzip() is None
        shared=[]
        if 'xl/sharedStrings.xml' in z.namelist():shared=[''.join(t.itertext()) for t in E.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',NS)]
        names=[s.attrib['name'] for s in E.fromstring(z.read('xl/workbook.xml')).findall('s:sheets/s:sheet',NS)];assert len(names)==24
        for i,name in enumerate(names,1):
            root=E.fromstring(z.read(f'xl/worksheets/sheet{i}.xml'));values={}
            for c in root.findall('.//s:sheetData/s:row/s:c',NS):
                t=c.attrib.get('t');v=c.find('s:v',NS);value=v.text if v is not None else None
                if t=='s' and value is not None:value=shared[int(value)]
                elif t=='inlineStr':value=''.join(c.find('s:is',NS).itertext())
                elif value is not None and t not in ('str','e'):value=float(value)
                if t=='e':errors.append((name,c.attrib['r'],value))
                if c.find('s:f',NS) is not None:formula_count+=1
                values[c.attrib['r']]=value
            sheets[name]=values
        assert not errors,errors
        for sp in payload['sheets']:
            name=sp['name'];v=sheets[name];l=layouts[name];reg=sp['regime'];supervised=reg=='supervised';own='supervised' if supervised else 'ordinary_PL';cell='C' if supervised else 'D';rest='DURA-L' if supervised else 'DURA-LU'
            assert v['A4']=='实验说明' and v['A20'].startswith('数据划分')
            for i,r in enumerate(sp['partitionData'],l['partitionHead']+1):
                c=contracts[v['A'+str(i)]]
                for j,key in [(2,'classes'),(5,'L_s'),(6,'U_s'),(7,'V'),(9,'cross_rx'),(10,'cross_both')]:near(v[col(j)+str(i)],c[key]);parts+=1
                assert v['C'+str(i)]==c['source_Rx'].replace(';',', ') and v['D'+str(i)]==c['target_Rx'].replace(';',', ')
            head=sp['headers'];n=l['dataStart']
            for i in range(len(sp['data'])):
                rownum=n+i;rr=[v.get(col(j+1)+str(rownum)) for j in range(len(head))]
                if reg and '学习方式' in head:assert rr[head.index('学习方式')]==('监督' if supervised else '普通PL半监督')
                if '均值（%）' in head:
                    p=ps[rr[0]];m=methods.get(rr[1],own);cc='native' if m==own else ('A' if supervised else 'B') if rr[3]=='关闭' else cell;rd='raw' if rr[4]=='直接识别' else rest
                    if rr[1]=='DURA+PAMF':assert rd==rest
                    src=source[p,m,cc,tests[rr[5]],rd,views[rr[6]],metrics[rr[7]]]
                    near(rr[8],src['mean']);near(rr[9],src['sd']);checked+=2
                    display=rr[10].split('±');assert len(display)==2
                    assert abs(float(display[0])-rr[8])<=.00500000001 and abs(float(display[1])-rr[9])<=.00500000001
                    if '无增强' in name:assert rr[3]=='关闭' and rr[6]=='CLEAN' and rr[5] in ('目标跨接收机','目标日期与接收机双跨') and rd=='raw'
                elif name.startswith('DURA泛用性'):
                    p=ps[rr[0]];m=methods[rr[1]];t=tests[rr[3]];view=views[rr[4]];metric=metrics[rr[5]]
                    a=source[p,m,cell,t,'raw',view,metric];b=source[p,m,cell,t,rest,view,metric]
                    near(rr[6],a['mean']);near(rr[7],b['mean']);d=[seeds[p,m,cell,t,rest,view,metric,s]-seeds[p,m,cell,t,'raw',view,metric,s] for s in (2026092701,2026092702,2026092703)]
                    near(rr[8],statistics.mean(d));near(rr[9],statistics.stdev(d));assert rr[10]==sum(x>0 for x in d);checked+=4
            for i,r in enumerate(sp['chart']['data'],l['chartHead']+1):
                if sp['chart'].get('delta'):near(v['D'+str(i)],v['C'+str(i)]-v['B'+str(i)])
        charts=[p for p in z.namelist() if re.fullmatch(r'xl/(?:drawings/)?charts/chart\d+\.xml',p)];assert len(charts)==24
        for filename in charts:
            chart=E.fromstring(z.read(filename));bar=chart.find('.//c:barChart',NS);assert bar is not None and bar.find('c:barDir',NS).attrib['val']=='col'
            ci=int(re.search(r'chart(\d+)\.xml',filename).group(1))-1
            title=''.join(t.text or '' for t in chart.findall('.//c:chart/c:title//a:t',NS))
            assert title==payload['sheets'][ci]['chart']['title'],(filename,title)
            series=bar.findall('c:ser',NS);refs=[]
            for s in series:
                nr=s.find('c:val/c:numRef',NS);assert nr is not None
                f=nr.find('c:f',NS).text;refs.append(f);m=re.fullmatch(r"'([^']+)'!\$([A-Z]+)\$(\d+):\$([A-Z]+)\$(\d+)",f);assert m,f
                name,c1,r1,c2,r2=m.groups();assert c1==c2
                expected=[sheets[name][c1+str(i)] for i in range(int(r1),int(r2)+1)]
                assert expected and all(isinstance(x,(int,float)) and math.isfinite(x) for x in expected)
                range_points+=len(expected)
                pts=nr.findall('c:numCache/c:pt',NS)
                if pts:
                    assert len(pts)==len(expected)
                    for pt,x in zip(pts,expected):near(float(pt.find('c:v',NS).text),x);cache_points+=1
            if any('DURA泛用性_' in f or 'PAMF恢复对照_' in f for f in refs):
                assert len(series)==2;double_charts+=1
                labels=[''.join(s.find('c:tx',NS).itertext()) for s in series]
                assert '未接入DURA' in labels[0] and '接入DURA' in labels[1]
        assert double_charts==4
    result=dict(status='VERIFIED',sheets=24,native_editable_column_charts=24,paired_DURA_accuracy_charts=4,source_metric_values_checked=checked,partition_numeric_cells_checked=parts,formulas_with_saved_caches=formula_count,chart_source_range_points_checked=range_points,chart_embedded_cache_points_checked=cache_points,formula_errors=errors,single_learning_regime_per_performance_sheet=True,supervised_and_semi_supervised_separate=True,no_channel_target_CLEAN_only=True,new_training=0,new_prediction=0,new_truth_scoring=0)
    (P/'workbook_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
