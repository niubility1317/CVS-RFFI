"""Independent metadata-only replay. No IQ, checkpoint or model imports."""
from collections import Counter
import gzip,json,math
from itertools import zip_longest
from pathlib import Path
import numpy as np
FIT=(3,4,6,8)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def bank_from_profile(profile):
    p=np.asarray(profile,dtype=np.float64)
    if p.shape!=(4,) or not np.isfinite(p).all():raise ValueError('Four finite source profile values')
    return np.asarray([min(.02,abs(float(a-b))/2) for i,a in enumerate(p) for b in p[i+1:]])
def parameter(seed,epoch,rx,day,bank):
    if epoch not in range(1,21) or rx not in FIT or day not in (1,2,3):raise ValueError('Registered source sessions only')
    offset=int(np.random.default_rng(np.random.SeedSequence([seed,0,rx,day])).integers(6));index=(offset+epoch-1)%6
    angle=float(np.random.default_rng(np.random.SeedSequence([seed,epoch,rx,day,2])).uniform(-np.pi,np.pi))
    return float(bank[index]*np.cos(angle)),float(bank[index]*np.sin(angle)),index
def record(r,primary,step,seed,bank,visits,counts):
    epoch=(step-1)//222+1
    if r['step']!=step or r['epoch']!=epoch or r['ids']!=primary or len(primary)!=128 or len(r['bank_indices'])!=128:raise ValueError('RXMIX must follow complete actual L128 sequence')
    for p,k in zip(primary,r['bank_indices']):
        if len(p)!=5 or p[0]!=1 or p[1] not in FIT or p[2] not in (1,2,3) or any(int(v)!=v or v<0 for v in p):raise ValueError('Only legal fitL packet metadata')
        if k!=parameter(seed,epoch,p[1],p[2],bank)[2]:raise ValueError('TX-blind RX/day/epoch amplitude cycle differs')
        visits[tuple(p)]+=1;counts[(p[1],p[2],k)]+=1
def check_trace(source,result,seed):
    source=Path(source);saved=read(source/'rxmix_sessions.json');metadata=read(source.parent.parent/'private_source_ellipticity_profile.json')
    if metadata['statistics_root']!=saved['source_statistic'] or metadata['source_summary_status']!='PASS' or metadata['fit_RX']!=list(FIT) or metadata['L_used']!=5040 or metadata['target_access'] or metadata['held_RX1_used']:raise ValueError('Matching source-only statistic required')
    bank=bank_from_profile(metadata['profile'])
    if not np.array_equal(np.asarray(saved['amplitude_bank']),bank):raise ValueError('Actual bank must derive from legal fitRX profile and fixed cap')
    expected={(e,r,d) for e in range(1,21) for r in FIT for d in (1,2,3)};seen=set()
    for p in saved['sessions']:
        key=(p['epoch'],p['RX'],p['day'])
        if key in seen or key not in expected:raise ValueError('Exactly240 source-only sessions required')
        seen.add(key);br,bi,k=parameter(seed,*key,bank)
        if p['bank_index']!=k or not math.isclose(p['b_real'],br,abs_tol=1e-14) or not math.isclose(p['b_imag'],bi,abs_tol=1e-14) or math.hypot(p['b_real'],p['b_imag'])>.02+1e-14:raise ValueError('Actual private TX-blind session parameter differs')
    if seen!=expected:raise ValueError('All20epochs x4RX x3day sessions')
    with gzip.open(source/'training_sequence.jsonl.gz','rt',encoding='utf-8') as f:primary=[json.loads(line)['ids'] for line in f if '"L_s"' in line]
    if len(primary)!=4440:raise ValueError('Whole actual primary L sequence required')
    visits=Counter();counts=Counter();steps=0
    with gzip.open(source/'rxmix_sequence.jsonl.gz','rt',encoding='utf-8') as f:
        for ids,line in zip_longest(primary,f):
            if ids is None or line is None:raise ValueError('One RXMIX record per actual source L batch')
            steps+=1;record(json.loads(line),ids,steps,seed,bank,visits,counts)
    if len(visits)!=5040 or sum(counts.values())!=568320:raise ValueError('Complete original physical L coverage')
    coverage=result['rxmix_TX_RX_day_coverage'];by_rx=Counter()
    if {(v['TX'],v['RX'],v['day'],v['bank_index']) for v in coverage}!={(t,r,d,k) for t in range(6) for r in FIT for d in (1,2,3) for k in range(6)} or sum(v['count'] for v in coverage)!=568320:raise ValueError('Complete TX/RX/day/amplitude coverage')
    for v in coverage:by_rx[(v['RX'],v['day'],v['bank_index'])]+=v['count']
    s=result['rxmix'];checks=dict(calls=s['calls']==4440,packets=s['packets']==568320,physical=s['physical_IDs']==5040,sessions=s['sessions']==240,
        bank=s['bank_entries']==6 and s['max_b_abs']==float(bank.max()),coverage=by_rx==counts,
        inverse=math.isfinite(s['max_inverse_error']) and s['max_inverse_error']>=0,power=math.isfinite(s['max_pointwise_power_bound_error']) and s['max_pointwise_power_bound_error']>=0,
        no_label=not s['TX_used_for_selection'],RNG=not s['global_RNG_advance'],no_packet_RMS=not s['per_packet_normalization'],boundary=not s['boundary_change'],noise=not s['added_noise'],no_TX_inverse=not s['unknown_TX_inversion'])
    if not all(checks.values()):raise ValueError(('Runtime RXMIX proof differs',checks))
    return dict(status='VERIFIED',steps=steps,physical_IDs=5040,session_count=240,IQ_exposures=568320,max_b_abs=float(bank.max()),
        max_inverse_error=s['max_inverse_error'],max_power_bound_error=s['max_pointwise_power_bound_error'],checks=checks,analysis_IQ_checkpoint_network_access=0)
