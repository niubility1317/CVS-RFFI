"""Preserve the complete text evidence and mirror the completed report."""
import json,shutil,zipfile
from pathlib import Path
b=Path('analysis/core90_anchored_results_20260912')
run='core90_anchored_s392005_20260911_r1'
items=[x for x in json.loads((b/'inventory.json').read_text()) if Path(x['path']).suffix in {'.json','.jsonl','.csv','.log','.md'}]
z=b/'source_text_evidence.zip'
with zipfile.ZipFile(z,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as f:
    for x in items:f.write(b/x['path'],x['path'])
with zipfile.ZipFile(z) as f:
    assert f.testzip() is None
    assert len(f.infolist())==len(items)
    assert all(f.getinfo(x['path']).file_size==x['size'] for x in items)
report=Path('automation_reports/CV-SincNet')/run
shutil.copyfile(b/'detailed_results.md',report/'detailed_results_20260912.md')
with (report/'report.md').open('a',encoding='utf-8',newline='\n') as f:
    f.write('\n## 2026-09-12完成结果读回：VERIFIED\n\n2026-09-11 18:56:08已完成14候选source冻结；本轮40次拟合全部E80、3200个epoch。全量日志无运行错误，V重算与已存policy一致，嵌套OOF重算与已存计数一致。A3/A4头部OOF均92.5159%（H0为92.2302%）；A6嵌套OOF92.1190%，救回8、损伤22，净−14。本seed收益与分组保底未通过；多seed仍PENDING_ALL_HEAD_SEEDS。未运行target或新增seed。\n\n详见[detailed_results_20260912.md](detailed_results_20260912.md)；完整统计和文本证据位于analysis/core90_anchored_results_20260912，source_text_evidence.zip含397份原始文本。\n')
root=Path('E:/type10-7')
dest=root/'analysis/core90_anchored_results_20260912';dest.mkdir(parents=True,exist_ok=True)
for p in b.iterdir():
    if p.is_file():shutil.copyfile(p,dest/p.name)
for name in ['report.md','detailed_results_20260912.md']:
    target=root/report/name;shutil.copyfile(report/name,target)
    assert target.read_bytes()==(report/name).read_bytes()
print(json.dumps(dict(evidence_members=len(items),zip_bytes=z.stat().st_size,mirrored=True)))
