"""Read-only full prediction/log audit; never mutate training or scoring output."""
import json
import math
import sys
from collections import Counter
from pathlib import Path

PROJECT=Path('/home/szu2070436088/2510044040/CV-SincNet')
sys.path.insert(0,str(PROJECT/'releases/core90_cross_response_v2_392005_e5bffaa3/code'))
from cvsrffi.cross_response.final_scoring import _load_complete_predictions, _accuracy

VARIANTS=('U0','U1','U1_mask_off','U3','Ux','Ux_normalized','head_only','permanent_detach')
baseline_manifest=None
baseline_truth=None
for variant in VARIANTS:
    run='core90_cross_response_v2_s392005_20260911_'+('r2' if variant=='head_only' else 'r1')
    directory=PROJECT/'runs'/run/variant
    pred_dir=directory/'final_predictions'
    manifest,predictions=_load_complete_predictions(pred_dir)
    if baseline_manifest is None: baseline_manifest=manifest
    assert all(manifest[k]==baseline_manifest[k] for k in
        ('record_ids','named_ids','record_metadata','scene_seeds','registered_class_count'))
    # All four prediction files have now passed complete-ID/finite-logit checks.
    truth_rows=[json.loads(line) for line in (pred_dir/'scoring_truth.jsonl').read_text().splitlines()]
    truth={row['record_id']:row['label'] for row in truth_rows}
    assert len(truth_rows)==len(truth)==manifest['record_count'] and set(truth)==set(manifest['record_ids'])
    if baseline_truth is None: baseline_truth=truth
    assert truth==baseline_truth
    score=json.loads((pred_dir/'independent_scores.json').read_text())
    groups_checked=0
    confusion={}
    by_day={}
    for scene,pred in predictions.items():
        aggregate=score['test'] if scene=='clean' else score['sat_test_named'][scene]['aggregate']
        named=score['named_test'] if scene=='clean' else score['sat_test_named'][scene]['named']
        groups=[(manifest['record_ids'],aggregate)]
        groups += [(ids,named[name]) for name,ids in manifest['named_ids'].items()]
        groups += [([rid for rid,meta in manifest['record_metadata'].items() if str(meta['receiver'])==rx],s)
            for rx,s in score['per_receiver'][scene].items()]
        groups += [([rid for rid,y in truth.items() if str(y)==cls],s) for cls,s in score['per_class'][scene].items()]
        for ids,stored in groups:
            checked=_accuracy(pred,truth,ids)
            assert checked['tx_correct']==stored['tx_correct'] and checked['tx_total']==stored['tx_total']
            assert abs(checked['tx_acc']-stored['tx_acc'])<1e-10
            groups_checked+=1
        matrix=[[0]*manifest['registered_class_count'] for _ in range(manifest['registered_class_count'])]
        for rid,y in truth.items(): matrix[y][pred[rid]]+=1
        confusion[scene]=matrix
        by_day[scene]={str(day):_accuracy(pred,truth,[rid for rid,m in manifest['record_metadata'].items() if m['day']==day])
            for day in sorted({m['day'] for m in manifest['record_metadata'].values()})}
    files={'independent_scores.json':json.dumps(score),variant+'.log':(PROJECT/'logs'/run/(variant+'.log')).read_text()}
    for name in ('cross_response_activation.json','phase1_resource_summary.json','phase1_terminal_status.json',
                 'phase1_training_completion_receipt.json','metrics_epoch.jsonl','resolved_config.json'):
        path=directory/name
        if path.exists(): files[name]=path.read_text()
    # Scan every diagnostic event while keeping large per-step payloads remote.
    diag=directory/'cross_response_diagnostics.jsonl'
    counts=Counter()
    event_steps=Counter()
    nonfinite=Counter()
    lines=0
    success_lines=0
    if diag.exists():
        with diag.open() as handle:
            for line in handle:
                if not line.strip(): continue
                d=json.loads(line); lines+=1; success_lines+=int(bool(d.get('success')))
                seen=set()
                for event in d.get('records',[]):
                    kind=event.get('kind','UNKNOWN'); counts[kind]+=1; seen.add(kind)
                    if kind=='nonfinite_diagnostic': nonfinite[event.get('reason','UNKNOWN')]+=1
                event_steps.update(seen)
    audit=dict(status='VERIFIED',records=manifest['record_count'],prediction_rows=manifest['rows_written'],
        registered_classes=manifest['registered_class_count'],compared_score_groups=groups_checked,
        same_physical_records_roles_and_scene_seeds=True,same_truth_mapping=True,
        all_predictions_validated_before_truth_read=True,confusion_matrices=confusion,
        confusion_orientation='rows=true, columns=predicted',per_day=by_day,
        diagnostics=dict(all_lines_parsed=lines,successful_lines=success_lines,events=counts,
            steps_per_event=event_steps,nonfinite_reasons=nonfinite,bytes=diag.stat().st_size if diag.exists() else 0),
        checkpoint_exists=(directory/'final_ssdg.pth').is_file(),
        checkpoint_bytes=(directory/'final_ssdg.pth').stat().st_size,
        no_remote_writes=True,run=run)
    print(json.dumps(dict(variant=variant,files=files,audit=audit)),flush=True)
