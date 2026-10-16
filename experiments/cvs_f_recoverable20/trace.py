"""Independent saved-metadata replay; no IQ, checkpoint or model imports."""
import gzip,hashlib,json,math,csv
from collections import Counter
from itertools import zip_longest
from pathlib import Path
import numpy as np
FIT=(1,3,4,8)
SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def seed(base,*parts):
    raw=json.dumps([int(base),*parts],ensure_ascii=False,separators=(',',':'))
    return int.from_bytes(hashlib.sha256(raw.encode('utf-8')).digest()[:8],'little')
def eligible(p):
    n=p['added_noise_db']
    return bool(p['scene'] in SCENES[1:] and p['lock'] in ('locked','degraded') and n is not None and math.isfinite(n) and 3<=n<=15)

def history(source):
    rows=read(source/'curriculum_history.json')['records'];mechanism=read(source/'mechanism_progress.json')['records']
    with (source/'metrics_epoch.csv').open(encoding='utf-8-sig',newline='') as f:native=list(csv.DictReader(f))
    if len(rows)!=20 or len(mechanism)!=20 or len(native)!=20:raise ValueError('Complete curriculum/native history required')
    eta=0.;past=[]
    for e,(r,m,n) in enumerate(zip(rows,mechanism,native),1):
        expected={(t,rx,d) for t in range(6) for rx in FIT for d in (1,2,3)};cells=r['cells'];keys=[(c['TX'],c['RX'],c['day']) for c in cells]
        if set(keys)!=expected or len(keys)!=72 or any(c['count']<=0 or not 0<=c['correct']<=c['count'] or not math.isfinite(c['margin_sum']+c['CE_sum']) for c in cells):raise ValueError('All legal fitL cells required')
        count=sum(c['count'] for c in cells)
        if count!=28416:raise ValueError('One primary L128 read per step')
        values=dict(L_packets=count,source_cells=72,all_72_fit_cells=True,accuracy=sum(c['correct'] for c in cells)/count,
            mean_margin=sum(c['margin_sum'] for c in cells)/count,mean_CE=sum(c['CE_sum'] for c in cells)/count,
            p10_cell_accuracy=float(np.quantile([c['correct']/c['count'] for c in cells],.1)),p10_cell_margin=float(np.quantile([c['margin_sum']/c['count'] for c in cells],.1)),gradient=float(n['train_grad_before_clip']))
        if r['metrics']!=values or r['epoch']!=e or r['eta_used']!=eta or any(m['curriculum'][k]!=r[k] for k in r):raise ValueError('Saved primary logits/native gradient aggregate differs')
        if m['curriculum']['held_V_used'] or m['curriculum']['U_truth_read'] or m['curriculum']['target_access'] or m['curriculum']['extra_model_forwards']:raise ValueError('FitL progress only')
        absolute=values['accuracy']>=.7 and values['p10_cell_accuracy']>=.6 and values['p10_cell_margin']>=.02 and values['mean_CE']<=1 and 0<=values['gradient']<=5
        previous=past[-3:]
        stable=not previous or (values['mean_CE']<=1.05*float(np.median([v['mean_CE'] for v in previous])) and values['p10_cell_margin']>=float(np.median([v['p10_cell_margin'] for v in previous]))-.01 and values['gradient']<=2*max(1e-12,float(np.median([v['gradient'] for v in previous]))))
        if e>=5 and absolute and stable:eta=min(.5,round(eta+.1,10))
        if (r['eta_next'],r['absolute_ready'],r['stable'])!=(eta,bool(absolute),bool(stable)):raise ValueError('Fixed source progress rule differs')
        past.append(values)
    return rows

def batch(r,primary,step,etas,counts,visits,coverage):
    e=(step-1)//222+1;eta=etas[e-1];namespace=f'FcrossRX/train/e{e}/step{step}/L_S'
    if (r['step'],r['epoch'],r['eta'])!=(step,e,eta) or len(r['records'])!=128 or len(primary)!=128:raise ValueError('One course record per complete actual L batch')
    for v,p in zip(r['records'],primary):
        if len(p)!=5 or p[0]!=1 or p[1] not in FIT or p[2] not in (1,2,3):raise ValueError('Only legal fitL metadata')
        pid='source:'+str(p[4]);session=f'rx{p[1]}:day{p[2]}:eq1'
        if v['sample_id']!=pid or v['session_id']!=session or v['eta']!=eta:raise ValueError('Physical ID/session differs')
        nominal=str(np.random.default_rng(seed(20261016421,namespace,'scene',pid)).choice(SCENES))
        u=float(np.random.default_rng(seed(20261016651,namespace,'mixture',pid)).random());hard=u<eta
        if v['nominal_scene']!=nominal or v['u']!=u or v['hard_requested']!=hard:raise ValueError('Private TX-blind nominal/mixture stream differs')
        proposals=v['proposals']
        if not 1<=len(proposals)<=5 or (not hard and len(proposals)!=1):raise ValueError('Bounded proposals required')
        for i,prop in enumerate(proposals):
            scene=nominal if i==0 else str(np.random.default_rng(seed(20261016651,namespace,'urban_scene',i,pid)).choice(SCENES[1:]))
            expected=seed(20261016421,namespace,scene,pid) if i==0 else seed(20261016651,namespace,'proposal',i,scene,pid)
            if prop['index']!=i or prop['scene']!=scene or prop['seed']!=expected or prop['eligible']!=eligible(prop) or prop['state'] not in ('LOS','PARTIAL','BLOCKED') or prop['lock'] not in ('locked','degraded','unlocked'):raise ValueError('Full-draw seed/eligibility metadata differs')
        hits=[i for i,x in enumerate(proposals) if eligible(x)] if hard else []
        chosen=hits[0] if hits else 0;accepted=bool(hits)
        expected_length=chosen+1 if accepted else (5 if hard else 1)
        if len(proposals)!=expected_length or v['selected_index']!=chosen or v['hard_accepted']!=accepted:raise ValueError('First eligible or original nominal fallback required')
        counts['packets']+=1;counts['proposal_packets']+=len(proposals);counts['hard_requested']+=int(hard);counts['hard_accepted']+=int(accepted);counts['hard_fallback']+=int(hard and not accepted);counts['nominal_selected']+=int(chosen==0)
        selected=proposals[chosen];coverage[(p[1],p[2],selected['scene'],selected['state'],selected['lock'])]+=1;visits[tuple(p)]+=1

def check_trace(source,result):
    source=Path(source);rows=history(source);etas=[r['eta_used'] for r in rows]
    with gzip.open(source/'training_sequence.jsonl.gz','rt',encoding='utf-8') as f:primary=[json.loads(x)['ids'] for x in f if '"L_s"' in x]
    if len(primary)!=4440:raise ValueError('Complete 4440 L records required')
    counts=Counter();visits=Counter();coverage=Counter();steps=0
    with gzip.open(source/'curriculum_draws.jsonl.gz','rt',encoding='utf-8') as f:
        for ids,line in zip_longest(primary,f):
            if ids is None or line is None:raise ValueError('Whole source course trace required')
            steps+=1;batch(json.loads(line),ids,steps,etas,counts,visits,coverage)
    s=result['training_practical'];recorded=Counter({(v['RX'],v['day'],v['scene'],v['state'],v['lock']):v['count'] for v in s['strata']})
    if any(s[k]!=v for k,v in counts.items()) or s['calls']!=4440 or len(visits)!=5040 or sum(visits.values())!=568320 or coverage!=recorded or s['rejected_channel_draws']!=counts['proposal_packets']-568320 or s['physical_samples_dropped'] or s['TX_used_for_selection'] or s['student_confidence_gate'] or s['nominal_support_floor']!=.5 or s['extra_proposals_max']!=4:raise ValueError('Actual course execution/coverage differs')
    if result['curriculum']['max_eta']!=max(etas) or result['curriculum']['eta_final']!=rows[-1]['eta_next']:raise ValueError('Actual course activation differs')
    return dict(status='VERIFIED',steps=steps,physical_IDs=5040,**counts,eta_used=etas,eta_final=rows[-1]['eta_next'],source_cells_per_epoch=72,all_labeled_physical_samples_retained=True,analysis_IQ_checkpoint_network_access=0)
