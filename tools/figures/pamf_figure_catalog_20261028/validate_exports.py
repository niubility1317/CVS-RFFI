"""Check exported figure files and numeric correspondence, without accessing truth."""
import argparse,csv,json,math,statistics,xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image,ImageStat
from pypdf import PdfReader

def validate(root):
 c=json.loads((root/'catalog.json').read_text(encoding='utf-8'));issues=[];cells=0;points=0
 for f in c['figures']:
  name=f['id'];actual=list(csv.DictReader((root/'source_data'/(name+'_plot.csv')).open(encoding='utf-8-sig')))
  expected=[dict(panel=chr(97+i),**r) for i,p in enumerate(f['panels']) for r in p['rows']]
  if len(actual)!=len(expected):issues.append([name,'row count mismatch'])
  for a,b in zip(actual,expected):
   for key in ['panel','category','series']:
    if a.get(key,'')!=str(b.get(key,'')):issues.append([name,'key mismatch',key])
   for key in ['estimate','sd','seed_n']:
    if b.get(key) is not None and (not a.get(key) or not math.isclose(float(a[key]),float(b[key]),rel_tol=1e-12,abs_tol=1e-10)):issues.append([name,'value mismatch',key])
   cells+=1
  seed_expected=sum(len(r.get('seed_values',[])) for p in f['panels'] for r in p['rows'])+sum(len(p.get('seeds',[])) for p in f['panels'])
  sf=root/'source_data'/(name+'_seeds.csv')
  seeds=list(csv.DictReader(sf.open(encoding='utf-8-sig'))) if sf.exists() else []
  if len(seeds)!=seed_expected:issues.append([name,'seed export mismatch',len(seeds),seed_expected])
  # Check every seed value provided by either catalog path.
  ep=[]
  for i,p in enumerate(f['panels']):
   for r in p['rows']:ep.extend((chr(97+i),str(r['category']),str(r.get('series','')),float(v)) for v in r.get('seed_values',[]))
   ep.extend((chr(97+i),str(r['category']),str(r.get('series','')),float(r['value'])) for r in p.get('seeds',[]))
  ap=[(r['panel'],r['category'],r.get('series',''),float(r['value'])) for r in seeds]
  if sorted(ep)!=sorted(ap):issues.append([name,'seed values mismatch'])
  points+=len(seeds)
  if len(PdfReader(root/'pdf'/(name+'.pdf')).pages)!=1:issues.append([name,'PDF page count'])
  tree=ET.parse(root/'svg'/(name+'.svg'))
  if not tree.findall('.//{http://www.w3.org/2000/svg}text'):issues.append([name,'SVG text not editable'])
  with Image.open(root/'png'/(name+'.png')) as im:
   if min(im.size)<500 or max(ImageStat.Stat(im.convert('RGB')).stddev)<1:issues.append([name,'blank/small PNG'])
 out=dict(status='VERIFIED' if not issues else 'FAILED',figures=len(c['figures']),plot_data_rows=cells,exported_point_values=points,pdf_svg_png_each=len(c['figures']),issues=issues)
 (root/'export_validation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(out,ensure_ascii=False));assert not issues
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);validate(p.parse_args().root)
