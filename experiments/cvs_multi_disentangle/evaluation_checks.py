"""Synthetic all-row scoring closure; no dataset or actual target is opened."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.cvs_multi_disentangle import design as d,evaluate as e


def check(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    ids=[];truth={};labels=[]
    for y in range(6):
        for rx in e.HELDOUT_RX.values():
            for day in e.TARGET_DAYS:
                for i in range(1000):
                    key=f'{y}-{rx}-{day}-{i}';ids.append(key);labels.append(y)
                    truth[key]=dict(label=y,receiver=rx,day=day)
    ids=np.asarray(ids);labels=np.asarray(labels)
    configs=[d.config(r) for r in d.rows()]
    e.BASE=output;e.manifest=lambda d:{};e.physical_ids=lambda d:ids
    class FakeDesign:
        TRUTH=str(output/'truth.json');CLASSES=d.CLASSES
        rows=staticmethod(d.rows);config=staticmethod(d.config)
    e.design=lambda:FakeDesign
    e.write(output/'truth.json',truth)
    e.write(output/'source_matrix_frozen.json',dict(status='ALL_SOURCE_FROZEN',run_id=d.RUN,rows=configs,
        target_access=False,selection='ALL_PREREGISTERED_FIXED_CONTROLS'))
    e.source_resources=lambda c:dict(synthetic=True)
    for c in configs:
        out=output/c['row_id']/'prediction'
        e.write(out/'provenance.json',dict(config=c))
        e.write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=e.COUNT,views=list(e.VIEWS),truth_read=False,query_fit=False))
        values=labels.copy();n=(d.ARMS.index(c['arm'])+1)*1000;values[:n]=(values[:n]+1)%6
        np.savez(out/'predictions.npz',ids=ids,**{v:values for v in e.VIEWS})
    e.score();done=e.read(output/'scoring_complete.json')
    assert done['metric_records']==3136 and done['day_metric_records']==896
    pairs=e.read(output/'paired_results.json')['comparisons']
    assert len(pairs)==208
    a=next(r for r in pairs if r['treatment']=='multi_interaction' and r['control']=='legacy_cosine' and r['view']=='six_view_mean' and r['metric']=='accuracy')
    assert abs(a['mean_difference']+7000/e.COUNT)<1e-12 and a['sd_difference']==0
    # Verify the independent local delivery path against synthetic complete logs.
    for c in configs:
        source=output/c['row_id']/'source';source.mkdir(parents=True,exist_ok=True)
        (source/'epoch_metrics.jsonl').write_text(''.join(json.dumps(dict(epoch=i))+'\n' for i in range(1,201)),encoding='utf-8')
        e.write(source/'completion.json',dict(logged_steps=44400,optimizer_steps=44400))
    from experiments.cvs_multi_disentangle.monitor import verify
    delivered=verify(output)
    assert delivered['models']==32 and delivered['full_training_epochs']==6400
    # Truth must stay closed when any registered view/row is absent or invalid.
    bad=output/configs[-1]['row_id']/'prediction/complete.json'
    e.write(bad,dict(status='INCOMPLETE'))
    try:e.prediction_preflight(FakeDesign,configs)
    except (KeyError,ValueError):pass
    else:raise AssertionError('Incomplete prediction accepted')
    result=dict(status='PASS',synthetic_only=True,real_target_read=False,rows=32,views=7,
        metric_records=3136,day_metric_records=896,paired_records=208,truth_last_incomplete_rejected=True,
        local_delivery_independent_recount=delivered['status'])
    e.write(output/'check.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);print(json.dumps(check(p.parse_args().output)))
