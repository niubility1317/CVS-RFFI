"""Validate and package the completed evaluation metadata and result tables."""
import csv,json,shutil,zipfile
from pathlib import Path
from datetime import datetime,timezone
BASE=Path('E:/type10-7');ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
ART=BASE/'local_artifacts/response_matrix_eval_20260917_r1'
WT=BASE/'code/snapshots/native_dr_eg_prepare_20260914_wt'
state=json.loads((ART/'state.json').read_text(encoding='utf-8'))
assert state['status']=='SCORED_COMPLETE' and len(state['rows'])==33
counts={}
for p in OUT.glob('*.csv'):
    with p.open(encoding='utf-8-sig',newline='') as f:counts[p.stem]=len(list(csv.DictReader(f)))
assert counts['summary']==33 and counts['source_curves']==6600 and counts['independent_recount']==132 and counts['coverage36']==36
assert counts['day0_summary']==33 and counts['all_groups']==33*1137
report=(OUT/'report.md').read_text(encoding='utf-8');assert '\ufffd' not in report
assert (OUT/'complete_test_data.xlsx').stat().st_size>10000
check=dict(status='VERIFIED',completed_rows=33,new_rows=17,reused_rows=16,predictions=33*672000,scene_recounts=132,csv_counts=counts,evaluation_seconds=state['finished']-state['created'],source_contract=state['checkpoint_provenance'],scope='fixed inventory; historical benchmark descriptive evaluation')
(ROOT/'verification.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
spec=json.loads((ROOT/'experiment.json').read_text(encoding='utf-8'));spec['checkpoint']['provenance_verdict']=state['checkpoint_provenance']
for row in spec['rows']:row['status']='ANALYZED'
(ROOT/'experiment.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (ROOT/'events.jsonl').open('a',encoding='utf-8') as f:
    for rid in state['rows']:f.write(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),status='ANALYZED',row_id=rid,note='fixed prediction, independent score and scene confusion recount PASS',evidence=[str(ART/rid/'score.json'),'results/independent_recount.csv']))+'\n')
(ROOT/'report.md').write_text('# 响应博弈矩阵测试评估：VERIFIED\n\n33行已完成测试评估，新增17行、复用16行；共22176000条预测，132个场景重计PASS。评估范围冻结时另3行仍在训练，不以中途checkpoint补齐。\n\n[完整报告](results/report.md) · [Excel](results/complete_test_data.xlsx) · [CSV与报告数据包](complete_test_data.zip) · [验证清单](verification.json) · [逐行配置](experiment.json)\n\n本次为既有测试集描述性评估，结果不反馈训练、选模或重跑。最终报告区分全目标集与day0跨天＋跨接收机子集。\n',encoding='utf-8')
archive=ROOT/'complete_test_data.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()):
        if p.is_file():z.write(p,'results/'+p.name)
    for name in ['experiment.json','report.md','verification.json','events.jsonl']:z.write(ROOT/name,name)
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
dst=WT/ROOT.relative_to(BASE);dst.mkdir(parents=True,exist_ok=True)
for p in ROOT.iterdir():
    if p.is_file() and p.suffix!='.zip':shutil.copy2(p,dst/p.name)
shutil.copytree(OUT,dst/'results',dirs_exist_ok=False)
print(json.dumps(check,ensure_ascii=False))
