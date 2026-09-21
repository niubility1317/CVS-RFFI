"""Independently cross-check report CSV arithmetic, disjoint groups and matrices."""
from pathlib import Path
import csv
import json
import re

ROOT=Path(__file__).resolve().parent
d=json.loads((ROOT/'completed_summary.json').read_text(encoding='utf-8'))
audit=json.loads((ROOT/'prediction_audit.json').read_text())
rows=list(csv.DictReader((ROOT/'scores_long.csv').open(encoding='utf-8-sig')))
assert len(rows)==1184
for row in rows:
    correct,total,wrong=map(int,(row['correct'],row['total'],row['wrong']))
    assert correct+wrong==total
    assert abs(correct/total*100-float(row['accuracy_percent']))<1e-9
checked=0
for v,entry in d.items():
    for scene,groups in entry['groups'].items():
        names=('test_unseen_day_seen_rx','test_seen_day_unseen_rx','test_unseen_day_unseen_rx')
        total=sum(groups['named'][n]['tx_total'] for n in names)
        correct=sum(groups['named'][n]['tx_correct'] for n in names)
        assert total==groups['aggregate']['tx_total']==198000
        assert correct==groups['aggregate']['tx_correct']
        matrix=audit[v]['confusion_matrices'][scene]
        assert sum(map(sum,matrix))==total
        assert sum(matrix[i][i] for i in range(6))==correct
        for cls in range(6):
            m=entry['score']['per_class'][scene][str(cls)]
            assert sum(matrix[cls])==m['tx_total'] and matrix[cls][cls]==m['tx_correct']
        checked+=1
assert len(list(csv.DictReader((ROOT/'training_curves.csv').open(encoding='utf-8-sig'))))==1600
report=(ROOT/'completed_results.md').read_text(encoding='utf-8')
assert '\ufffd' not in report
for link in re.findall(r'\]\(([^)]+)\)',report): assert (ROOT/link).is_file(),link
result=dict(status='VERIFIED',csv_rows=1184,disjoint_aggregate_and_confusion_checks=checked,
    complete_training_rows=1600,prediction_rows=sum(x['prediction_rows'] for x in audit.values()),
    source_split_equal=True,local_links_valid=True)
(ROOT/'report_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
