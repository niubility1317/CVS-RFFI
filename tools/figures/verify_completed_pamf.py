"""Validate delivered PAMF figures, source aggregates and report sign rendering."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
import math
import re
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
from PIL import Image
from pypdf import PdfReader


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def verify(root,archives=False):
    data=read(root/'source_data/snapshot.json');tables=data['tables'];meta=data['metadata']
    facts=read(root/'analysis/conclusions_data.json');manifest=read(root/'figure_manifest.json')
    completion=meta['frozen_completion']
    assert completion['effective_complete']==2078 and completion['running']==completion['queued']==0
    assert completion['excluded_historical_jobs']==217
    assert meta['final_total_parameters']==meta['final_F_parameters']+meta['final_G_parameters']==499944
    assert meta['no_new_training_inference_or_truth_access']
    assert len(meta['protocols'])==17
    for name,count in {'main':2016,'core':384,'fine':704,'physics_seed':864,'G_seed':189,
                       'gfactor':1152,'external_full_seed':12960}.items():
        assert len(tables[name])==count,(name,len(tables[name]))
    # Independently recompute means and sample SD from the frozen per-seed percentages.
    for seed_table,summary_table,fields in [
        ('external_full_seed','external_full',('protocol','family','regime','variant','readout','test','metric','view')),
        ('physics_seed','physics',('architecture','readout','test','metric','view'))]:
        grouped=defaultdict(list)
        for row in tables[seed_table]:
            assert 0<=row['value']<=100
            grouped[tuple(row[k] for k in fields)].append(row)
        for row in tables[summary_table]:
            group=grouped[tuple(row[k] for k in fields)]
            assert sorted(int(r['seed']) for r in group)==[2026092701,2026092702,2026092703]
            vals=[r['value'] for r in group];mean=math.fsum(vals)/3
            sd=math.sqrt(math.fsum((v-mean)**2 for v in vals)/2)
            assert abs(row['mean']-mean)<1e-10 and abs(row['sd']-sd)<1e-10
        six_groups=defaultdict(list)
        for row in tables[seed_table]:
            key=tuple(row[k] for k in fields if k!='view')+(row['seed'],)
            six_groups[key].append(row)
        for group in six_groups.values():
            six=next(r['value'] for r in group if r['view']=='six_practical_mean')
            scenes=[r['value'] for r in group if r['view'] not in ('clean','six_practical_mean')]
            assert len(scenes)==6 and abs(six-math.fsum(scenes)/6)<1e-10
    assert facts['DURA_protocol_gain']['positive_mean']==17
    assert facts['DURA_clean_change']['nonpositive_mean']==5
    assert facts['native_comparisons']['native_minus_supervised']['nonpositive_mean']==3
    assert facts['native_comparisons']['native_minus_ordinaryPL']['nonpositive_mean']==5
    assert facts['external_Full_DURA']['G-LU_minus_raw']['positive_mean']==86
    assert facts['external_Full_DURA']['G-L_minus_raw']['positive_mean']==84
    assert abs(facts['native_vs_best_external_F0_GLU_pp']-11.466402116402122)<1e-10
    assert facts['fairness']['F_PL_x_G_U_interaction']['positive_mean']==5
    assert facts['fairness']['PL_minus_Sup_at_G_LU']['F0']['mean']<0
    assert sum(r['within_five_pp'] for r in facts['five_other_channel_gaps'])==4
    assert facts['low_urban_goal']['actual_pct']<70
    profiles=meta['timing_profiles']
    assert profiles['fine_timing']['case_rounds']==900 and profiles['structure_timing']['case_rounds']==450
    assert len(tables['fine_timing'])==180 and len(tables['structure_timing'])==90
    for name in ('fine_timing','structure_timing'):
        for r in tables[name]:
            assert r['rounds']==5 and r['samples']==(5000 if r['batch']==1 else 500)
            assert 0<r['median_ms']<=r['P95_ms']
            assert abs(r['throughput_IQ_s']-1000*r['batch']/r['median_ms'])<1e-8
    figures=manifest['figures'];assert manifest['status']=='VERIFIED' and len(figures)==40
    assert manifest['font']=='Times New Roman' and manifest['dpi']==4800
    Image.MAX_IMAGE_PIXELS=1_500_000_000
    fonts=set()
    figure_text={}
    for r in figures:
        stem=r['stem']
        assert r['data_charts']==1 and r['no_subtitle'] and r['no_footer'] and r['canvas_clipping_checked']
        for table in r['source_tables']:
            assert (root/'source_data'/table).is_file() or (root/'analysis'/table).is_file(),table
        for folder,suffix in [('png','.png'),('tiff','.tiff')]:
            with Image.open(root/folder/(stem+suffix)) as im:
                assert im.size==tuple(r['png_pixels']) and im.width==int(183/25.4*4800)
                assert all(abs(d-4800)<.03 for d in im.info['dpi'])
                if folder=='tiff':assert im.info['compression']=='tiff_lzw'
        svg=ET.parse(root/'svg'/(stem+'.svg')).getroot();ns='{http://www.w3.org/2000/svg}'
        assert svg.findall('.//'+ns+'text') and not svg.findall('.//'+ns+'image')
        pdf=PdfReader(root/'pdf'/(stem+'.pdf'));page=pdf.pages[0];text=page.extract_text()
        figure_text[r['number']]=text
        assert len(pdf.pages)==1 and f'Fig. {r["number"]}.' in text
        assert not any(token in text for token in ('Historical','historical','PAMF residual','PAMF SIG_TIME','Whiskers:'))
        assert not re.search(r'\bF(?:0[1-9]|1[0-9]|2[01])\b|T1 \+ G-L|\bG-LU?\b',text),stem
        for item in page['/Resources'].get_object()['/Font'].get_object().values():
            font=str(item.get_object()['/BaseFont']);assert 'TimesNewRoman' in font;fonts.add(font)
    assert len(PdfReader(root/'all_figures.pdf').pages)==40
    main=[r for r in figures if r['role']=='main'];supp=[r for r in figures if r['role']=='supplementary']
    assert len(main)==18 and len(supp)==22
    assert len(PdfReader(root/'main_figures.pdf').pages)==len(main)
    assert len(PdfReader(root/'supplementary_figures.pdf').pages)==len(supp)
    assert next(r for r in figures if r['number']==13)['role']=='main'
    assert all(next(r for r in figures if r['number']==i)['role']=='supplementary' for i in (12,14,15,16,40))
    for i in (2,3,5,6,17,24):
        assert all(k in figure_text[i] for k in ('Before (%)','After (%)','Change (pp)'))
    assert '70.60' in figure_text[2] and '77.65' in figure_text[2]
    for i in (7,8,9,10):assert 'not tested' in figure_text[i]
    for name in ('Complete recognizer','Remove cross-receiver','Remove paired consistency','Remove selective self-'):
        assert name in figure_text[13],name
    assert all(s in figure_text[13] for s in ('77.65','78.06','76.63','77.85'))
    assert '84.39' in figure_text[14] and 'Clean accuracy (%)' in figure_text[14]
    assert 'Accuracy difference' not in figure_text[14] and 'T1 minus variant' not in figure_text[14]
    assert 'no restoration' in figure_text[28] and '→' in figure_text[28]
    with (root/'experiment_names.csv').open(encoding='utf-8-sig') as stream:names=list(csv.DictReader(stream))
    assert len(names)==71 and len({(r['group'],r['original_id']) for r in names})==71
    assert all(r['figure_name'] and r['chinese_name'] and r['comparison_reference'] for r in names)
    report=PdfReader(root/'PAMF_结果总结_20261023.pdf');assert len(report.pages)==12
    text='\n'.join(p.extract_text() for p in report.pages)
    for token in ('84.48','77.65','11.47','-0.1336','-0.1646','2.34','217','1,850'):
        assert token in text,token
    assert all(len(p.extract_text())>300 for p in report.pages)
    assert all(len(p.images)==0 for p in report.pages),'Scientific report figures must remain vectors'
    assert '补充：二十一项识别器细诊断' in report.pages[-1].extract_text()
    guide=PdfReader(root/'PAMF_逐图说明_20261023.pdf');assert len(guide.pages)==41,len(guide.pages)
    assert all(len(p.images)==0 for p in guide.pages)
    explanations=read(root/'analysis/figure_explanations.json');assert len(explanations)==40
    for i,p in enumerate(guide.pages[1:],1):
        text=p.extract_text();assert f'Fig. {i}' in text and all(s in text for s in ('问题：','读法：','结果：','边界：')),(i,text[:80])
        assert all(explanations[i-1][k] for k in ('question','encoding','result','boundary'))
    with (root/'source_data/resource_rankings.csv').open(encoding='utf-8-sig') as stream:
        resources=list(csv.DictReader(stream))
    groups=defaultdict(list)
    for r in resources:groups[r['panel']].append(r)
    assert len(groups)==7 and len(resources)==47
    for panel,rs in groups.items():
        lower=rs[0]['lower_is_better']=='True';values=[float(r['value']) for r in rs]
        assert values==sorted(values,reverse=not lower)
        assert [int(r['rank']) for r in rs]==list(range(1,len(rs)+1))
        for r,v in zip(rs,values):assert abs(float(r['relative_to_best_pct'])-100*v/values[0])<1e-8
    result=dict(status='VERIFIED',as_of_hkt='2026-10-23',figures=40,report_pages=12,dpi=4800,
        raster_pixel_width=int(183/25.4*4800),fonts=sorted(fonts),embedded_report_figures='vector',
        no_subtitles_or_figure_footnotes=True,one_data_chart_per_image=True,
        actual_three_F_seeds=True,six_scene_SD_reconciled=True,negative_effects_retained=True,
        resource_panels=7,resource_rows=47,best_first_true_proportions=True,
        timing_cohorts_separate=True,no_new_models_inference_or_truth_access=True)
    result.update(main_figures=len(main),supplementary_figures=len(supp),guide_pages=len(guide.pages),
        semantic_experiment_names=71,primary_three_module_ablation=True,
        absolute_endpoints_with_paired_differences=True,five_source_condition_explicitly_unmeasured=True,
        all_40_figures_explained=True)
    if archives:
        stems={r['stem'] for r in figures}
        for kind in ('PNG','TIFF'):
            with zipfile.ZipFile(root.with_name(root.name+f'-{kind}-4800dpi.zip')) as z:
                assert len(z.namelist())==40 and {Path(n).stem for n in z.namelist()}==stems
        with zipfile.ZipFile(root.with_suffix('.zip')) as z:
            assert 'PAMF_结果总结_20261023.pdf' in z.namelist()
            assert len([n for n in z.namelist() if n.startswith('svg/')])==40
            assert len([n for n in z.namelist() if n.startswith('pdf/')])==40
        result['archives_verified']=True
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--archives',action='store_true');a=p.parse_args()
    result=verify(a.root,a.archives)
    (a.root/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
