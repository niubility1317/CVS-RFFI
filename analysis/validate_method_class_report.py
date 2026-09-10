"""Check report tables, encodings, local links, and mirrored delivery bytes."""
import csv,gzip,json,math,re,shutil
from pathlib import Path
BASE=Path(__file__).parent
OUT=BASE/'method_class_analysis_20260910'
text=(OUT/'report.md').read_text(encoding='utf-8')
assert '\ufffd' not in text and not text.startswith('\ufeff')
d=json.loads((OUT/'statistics.json').read_text(encoding='utf-8'))
main={r['method']:r for r in d['methods']}
x=[r['source_leo'] for r in d['methods']];y=[r['target_leo'] for r in d['methods']]
mx=sum(x)/len(x);my=sum(y)/len(y)
manual=sum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
assert abs(manual-d['correlations'][0]['pearson'])<1e-12
for line in text.splitlines():
    if not line.startswith('|'):continue
    cells=line.strip('|').split('|')
    if cells[0] in main and len(cells)==7:
        r=main[cells[0]]
        for cell,k in zip(cells[1:],['source_clean','source_leo','target_clean','target_leo','source_leo_delta_vs_B0','target_leo_delta_vs_B0']):
            assert abs(float(cell)-r[k])<=.00500001,(cells[0],k,cell,r[k])
for name,count in [('methods',8),('classes',48),('scene_classes',192),('fixed_epoch_curves',104)]:
    with (OUT/(name+'.csv')).open(encoding='utf-8-sig',newline='') as f:assert len(list(csv.DictReader(f)))==count
with gzip.open(OUT/'full_log_audit.json.gz','rt',encoding='utf-8') as f:audit=json.load(f)
assert sum(r['jsonl_records'] for r in audit['rows'].values())==1600
assert sum(r['stdout_lines'] for r in audit['rows'].values())==72264
for r in audit['rows'].values():assert len(r['records'])==200 and not r['errors']
for link in re.findall(r'\]\(([^)]+)\)',text):
    if link.startswith('http'):continue
    path=re.sub(r':\d+$','',link)
    p=Path(path) if re.match(r'^[A-Z]:',path) else OUT/path
    assert p.exists(),p
dest=Path('E:/type10-7/automation_reports/CV-SincNet/method_class_analysis_20260910')
dest.mkdir(exist_ok=True)
files=[p for p in OUT.iterdir() if p.is_file() and p.name!='full_log_audit.json']
for p in files:
    shutil.copyfile(p,dest/p.name)
    assert (dest/p.name).read_bytes()==p.read_bytes()
print(json.dumps({'status':'VERIFIED','method_rows':8,'class_rows':48,'scene_class_rows':192,'fixed_scores':104,
    'epochs_parsed':1600,'stdout_lines_scanned':72264,'mirrored_files':len(files),'report_characters':len(text)},ensure_ascii=False))
