"""Truth-side only: recover authorized U labels after a frozen trajectory."""
import gzip
import json
from pathlib import Path
import numpy as np
from experiments.cric_dura_comparison.data import sample_id
from experiments.cric_dura_comparison.io import read,write
from experiments.pamf_repl_weight.pseudo_score import score_observations


def score(c):
    if any(c.get(k) for k in ('target_inputs','target_truth','views_root','tests','model','optimizer')):
        raise ValueError('U truth-side scorer cannot access model, optimizer or target')
    root=Path(c['source_output']);done=read(root/'complete.json')
    if (done['status']!='SOURCE_FROZEN' or done['epoch']!=50 or done['updates']!=11100
            or done['target_access'] or done['U_truth_read']):
        raise ValueError('Complete independently frozen source trajectory required')
    cfg=done['config'];contract=read(cfg['source_contract'])
    if cfg['protocol_id']!=c['protocol_id'] or cfg['row_id']!=c['row_id']:
        raise ValueError('Exact frozen source row required')
    n=cfg['num_classes'];dataset='ManyRx' if c['protocol_id']=='ManyRx' else 'ManySig'
    paths=[root/f'segment{epoch}'/'U_pseudo_observations.jsonl.gz' for epoch in (30,50)]
    if not all(p.is_file() for p in paths):raise ValueError('All online U decisions must be fixed first')
    # This enumerates the possible source TX indices only in the post-freeze
    # scoring process; training receives neither this mapping nor scores.
    with np.load(Path(cfg['source_cache'])/'U_s_metadata.npz',allow_pickle=False) as meta:
        ids=[str(x) for x in meta['ids'].tolist()]
        if ids!=contract['role_ids']['U_s'] or 'y' in meta.files:
            raise ValueError('Exact original blind U source role required')
        truth={}
        for sid,rx,day,eq,packet in zip(ids,meta['rx_i'],meta['day_i'],meta['eq_i'],meta['sig_i']):
            matches=[tx for tx in range(n) if sample_id(dataset,tx,int(rx),int(day),int(eq),int(packet))==sid]
            if len(matches)!=1:raise ValueError('U physical ID did not resolve to exactly one registered class')
            truth[sid]=matches[0]
    results=[score_observations(p,truth=truth,allowed_ids=ids,num_classes=n) for p in paths]
    unique_seen=set();unique_accepted=set()
    for path in paths:
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            for line in stream:
                r=json.loads(line)
                unique_seen.update(r['physical_ids'])
                unique_accepted.update(sid for sid,accepted in zip(r['physical_ids'],r['accepted']) if accepted)
    overall={k:sum(r['overall'][k] for r in results) for k in ('seen','accepted','accepted_correct')}
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False)
    write(out/'scores.json',dict(status='SCORED_U_DIAGNOSTIC',overall=overall,segments=results,
        unique_U_total=len(ids),unique_seen=len(unique_seen),unique_accepted=len(unique_accepted),
        truth_source='physical ID builder inversion, only post-freeze',training_or_selection_use=False))
    write(out/'complete.json',dict(status='SCORED_U_DIAGNOSTIC',truth_last=True,
                                   training_or_selection_use=False,source_epoch=50))
