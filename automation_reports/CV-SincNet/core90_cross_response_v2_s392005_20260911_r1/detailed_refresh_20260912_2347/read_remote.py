"""Read complete logs and validate frozen predictions before reading scoring truth."""
from pathlib import Path
import collections
import json
import sys
import time
ROOT=Path('/home/szu2070436088/2510044040/CV-SincNet')
sys.path.insert(0,str(ROOT/'releases/core90_cross_response_v2_392005_40c1c205/code'))
from cvsrffi.cross_response.final_scoring import _load_complete_predictions, _accuracy
R1='core90_cross_response_v2_s392005_20260911_r1'
R2='core90_cross_response_v2_s392005_20260911_r2'
VARIANTS=['U0','U1','U1_mask_off','U3','Ux','Ux_normalized','head_only','permanent_detach']
result=dict(time=time.time(),rows={},pipelines={},read_only=True)
for run in (R1,R2):
    result['pipelines'][run]=json.loads((ROOT/'runs'/run/'pipeline.json').read_text())
reference=None
for variant in VARIANTS:
    run=R2 if variant=='head_only' else R1
    folder=ROOT/'runs'/run/variant
    row=dict(run=run,root=str(folder),files={},json={},metrics=[],diagnostics={})
    result['rows'][variant]=row
    for path in folder.rglob('*'):
        if path.is_file(): row['files'][str(path.relative_to(folder))]=path.stat().st_size
    for name in ['cross_response_activation.json','phase1_terminal_status.json','phase1_resource_summary.json',
        'phase1_training_completion_receipt.json','final_predictions/independent_scores.json',
        'resolved_config.json','config.json','frozen_phase1_heldout_eval.json']:
        path=folder/name
        if path.exists(): row['json'][name]=json.loads(path.read_text())
    path=folder/'metrics_epoch.jsonl'
    if path.exists(): row['metrics']=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    path=ROOT/'logs'/run/(variant+'.log')
    row['stdout']=path.read_text(errors='replace') if path.exists() else None
    path=folder/'cross_response_diagnostics.jsonl'
    kinds=collections.Counter(); records=0; failed=0; nonfinite=collections.Counter()
    if path.exists():
        with path.open() as handle:
            for line in handle:
                d=json.loads(line); records+=1; failed+=not d.get('success',True)
                for item in d.get('records',[]):
                    kinds[item.get('kind','unknown')]+=1
                    if item.get('kind')=='nonfinite_diagnostic': nonfinite[item.get('reason','unknown')]+=1
    row['diagnostics']=dict(lines=records,failed_steps=failed,kinds=dict(kinds),nonfinite_reasons=dict(nonfinite))
    pred_dir=folder/'final_predictions'
    if not (pred_dir/'prediction_manifest.json').exists():
        row['audit']=dict(status='INCOMPLETE',reason='NO_PREDICTION_MANIFEST'); continue
    try:
        manifest,predictions=_load_complete_predictions(pred_dir)
        if reference is None: reference=manifest
        same=all(manifest[k]==reference[k] for k in ('record_ids','named_ids','record_metadata','scene_seeds','registered_class_count'))
        assert same,'row data/scene mismatch'
        truth_rows=[json.loads(l) for l in (pred_dir/'scoring_truth.jsonl').read_text().splitlines()]
        truth={r['record_id']:r['label'] for r in truth_rows}
        assert len(truth_rows)==len(truth)==manifest['record_count']
        assert set(truth)==set(manifest['record_ids'])
        scores=row['json']['final_predictions/independent_scores.json']
        checked=0; confusion={}
        for scene,pred in predictions.items():
            agg=scores['test'] if scene=='clean' else scores['sat_test_named'][scene]['aggregate']
            named=scores['named_test'] if scene=='clean' else scores['sat_test_named'][scene]['named']
            groups=[(manifest['record_ids'],agg)]+[(ids,named[n]) for n,ids in manifest['named_ids'].items()]
            groups += [([rid for rid,m in manifest['record_metadata'].items() if str(m['receiver'])==rx],score) for rx,score in scores['per_receiver'][scene].items()]
            groups += [([rid for rid,y in truth.items() if str(y)==cls],score) for cls,score in scores['per_class'][scene].items()]
            for ids,stored in groups:
                calc=_accuracy(pred,truth,ids)
                assert calc['tx_correct']==stored['tx_correct'] and calc['tx_total']==stored['tx_total']
                assert abs(calc['tx_acc']-stored['tx_acc'])<1e-10
                checked+=1
            mat=[[0]*manifest['registered_class_count'] for _ in range(manifest['registered_class_count'])]
            for rid,y in truth.items(): mat[y][pred[rid]]+=1
            confusion[scene]=mat
        row['audit']=dict(status='VERIFIED',record_count=manifest['record_count'],prediction_rows=manifest['rows_written'],
            groups=checked,registered_classes=manifest['registered_class_count'],same_physical_ids_and_scenes=same,
            predictions_validated_before_truth=True,confusion=confusion,
            manifest_summary={k:v for k,v in manifest.items() if k not in ('record_ids','record_metadata','named_ids')})
    except Exception as error:
        row['audit']=dict(status='FAILED',error=repr(error))
result['original_head_only_failure']=(ROOT/'logs'/R1/'head_only.log').read_text()
print(json.dumps(result,ensure_ascii=False))
