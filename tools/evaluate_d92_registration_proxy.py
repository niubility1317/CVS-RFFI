"""Source-only registration calibration: hide classifier entries, never claim unseen TX.

All six TX were seen by Phase1. These episodes test old/new competition and
K1 feature weighting, not novel-transmitter generalization. L fits; one V scores.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import numpy as np
from cvsrffi.stage2_d92_support_cv import fit_support_cv


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True,type=Path);p.add_argument('--commit',required=True)
    a=p.parse_args();spec=json.loads(a.spec.read_text(encoding='utf-8'));cfg=spec['development']
    out=Path(spec['execution']['remote_run_root']);out.mkdir(parents=True,exist_ok=False)
    (out/'startup.json').write_text(json.dumps(dict(pid=os.getpid(),argv=sys.argv,cwd=os.getcwd(),python=sys.executable,
        commit=a.commit,spec=spec,time=time.time()),indent=2)+'\n',encoding='utf-8')
    marker=json.loads((Path(cfg['features'])/'complete.json').read_text())
    if marker['target_access'] is not False or marker['status']!='SOURCE_FEATURES_COMPLETE':raise ValueError('Source-only features required')
    data=np.load(Path(cfg['features'])/'features.npz',allow_pickle=False)
    x,y,logits=data['features'],data['labels'],data['logits']
    rows=[]
    with (out/'metrics.jsonl').open('x',encoding='utf-8') as stream:
        for rx in sorted(set(data['receivers'].tolist())):
            for scene in range(3):
                cell=(data['receivers']==rx)&(data['scenes']==scene)
                rng=np.random.default_rng(np.random.SeedSequence([cfg['support_seed'],int(rx),scene]))
                supports=[rng.permutation(np.flatnonzero(cell&(data['roles']=='L_s')&(y==c))) for c in range(6)]
                queries=[np.flatnonzero(cell&(data['roles']=='V')&(y==c))[:cfg['validation_per_class']] for c in range(6)]
                if min(map(len,supports))<20 or min(map(len,queries))<cfg['validation_per_class']:raise ValueError('Missing source coverage')
                q=np.concatenate(queries)
                for old_tuple in itertools.combinations(range(6),3):
                    order=list(old_tuple)+[c for c in range(6) if c not in old_tuple]
                    inverse=np.argsort(order);mapped=inverse[y];old=mapped[q]<3
                    for k in cfg['k']:
                        s=np.concatenate([v[:k] for v in supports])
                        for fw in (cfg['k1_fft_grid'] if k==1 else [None]):
                            state=fit_support_cv(support_features=x[s],support_labels=mapped[s],support_logits=logits[s][:,old_tuple],
                                classes=[marker['classes'][c] for c in order],old_count=3,k1_fft_weight=fw if fw is not None else 0.0)
                            scores=state.score(x[q],logits[q][:,old_tuple]);correct=scores.argmax(1)==mapped[q]
                            oa=float(correct[old].mean());na=float(correct[~old].mean())
                            row=dict(receiver=int(rx),scene=scene,old_role_classes=list(old_tuple),k=k,k1_fft_weight=fw,
                                old_accuracy=oa,new_proxy_accuracy=na,harmonic_mean=2*oa*na/(oa+na) if oa+na else 0.0,
                                selected=state.audit['selected'],fit_seconds=state.audit['fit_seconds'],
                                target_access=False,scope='source_seen_TX_classifier_entry_holdout_NOT_novel_TX')
                            rows.append(row);stream.write(json.dumps(row)+'\n')
                    stream.flush()
                print(json.dumps(dict(receiver=int(rx),scene=scene,completed=len(rows),total=cfg['expected_rows'])),flush=True)
    if len(rows)!=cfg['expected_rows']:raise ValueError('Incomplete fixed proxy matrix')
    summary=[]
    for k in cfg['k']:
        for fw in (cfg['k1_fft_grid'] if k==1 else [None]):
            group=[r for r in rows if r['k']==k and r['k1_fft_weight']==fw]
            summary.append(dict(k=k,k1_fft_weight=fw,count=len(group),**{key:float(np.mean([r[key] for r in group])) for key in ['old_accuracy','new_proxy_accuracy','harmonic_mean']}))
    baseline=next(r for r in summary if r['k']==1 and r['k1_fft_weight']==0.0)
    eligible=[r for r in summary if r['k']==1 and r['old_accuracy']>=baseline['old_accuracy']-0.01 and r['new_proxy_accuracy']>=baseline['new_proxy_accuracy']-0.01]
    selected=max(eligible,key=lambda r:(r['harmonic_mean'],-r['k1_fft_weight']))
    result=dict(status='SOURCE_PROXY_COMPLETE',rows=len(rows),summary=summary,selected_k1_fft_weight=selected['k1_fft_weight'],
        selection_rule='max source V proxy H subject to old and proxy-new accuracy no worse than identity-only by1pp; ties prefer smaller FFT',
        target_access=False,all_tx_previously_seen_by_phase1=True,novel_tx_performance_verified=False)
    (out/'complete.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
