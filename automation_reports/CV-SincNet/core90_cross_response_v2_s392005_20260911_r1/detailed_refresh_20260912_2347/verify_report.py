from pathlib import Path
import csv
import gzip
import json
import re
P=Path(__file__).resolve().parent
D=json.loads(gzip.decompress((P/'complete_evidence.json.gz').read_bytes()))
rows=list(csv.DictReader((P/'scores_long.csv').open(encoding='utf-8-sig')))
assert len(rows)==1184
for v in D['rows']:
    for scene in ['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak']:
        part=[r for r in rows if r['variant']==v and r['scene']==scene]
        total=next(r for r in part if r['category']=='aggregate')
        for subset in [[r for r in part if r['category']==c] for c in ['class','receiver_all_days']]+[
            [r for r in part if r['category']=='named' and r['group'] in ['test_unseen_day_seen_rx','test_seen_day_unseen_rx','test_unseen_day_unseen_rx']]]:
            assert sum(int(r['correct']) for r in subset)==int(total['correct'])
            assert sum(int(r['total']) for r in subset)==int(total['total'])
    assert D['rows'][v]['audit']['status']=='VERIFIED'
text=(P/'detailed_results.md').read_text(encoding='utf-8')
assert '\ufffd' not in text
for link in re.findall(r'\]\(([^)]+)\)',text): assert (P/link).exists(),link
result=dict(status='VERIFIED',score_rows=len(rows),complete_epochs=sum(len(r['metrics']) for r in D['rows'].values()),
    prediction_rows=sum(r['audit']['prediction_rows'] for r in D['rows'].values()),disjoint_group_sum_checks=96,
    gzip_readback=True,report_links=True)
(P/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(result)
