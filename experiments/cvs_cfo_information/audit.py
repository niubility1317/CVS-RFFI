"""Source-only information probe for the coordinate removed by synchronization.

Fixed ridge probes are diagnostic, not new CVS candidates. Only L_s fits any
state; all V is read-only. No checkpoint, query or target score is consumed.
"""
import argparse,json,math,os,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
RUN='20261002-diagnostic-cvs-cfo-information-source-manysig-m1-r01'
RELEASE='cvs_cfo_information_source_20261002_r01'
WINDOWS=((0,80),(80,160),(160,240))
FEATURES=('cfo80','coherence80','cfo_coherence80','three_window')
RX=(1,3,4,6,8)
RIDGE=1e-3


def write(p,value):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2)


def packet_features(iq):
    x=np.asarray(iq,dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(2,256) or not np.isfinite(x).all():raise ValueError('Finite IQ[B,2,256] required')
    z=x[:,0]+1j*x[:,1];angles=[];coherences=[];validities=[]
    for start,end in WINDOWS:
        block=z[:,start:end].reshape(-1,4,20)
        C=(block[:,1:]*block[:,:-1].conj()).sum((1,2))
        e=np.sqrt(np.maximum((np.abs(block[:,1:])**2).sum((1,2))*(np.abs(block[:,:-1])**2).sum((1,2)),1e-24))
        valid=(e>1e-12)&(np.abs(C)>1e-6*e)
        angles.append(np.angle(np.where(valid,C,1+0j)))
        coherences.append(np.abs(C)/e);validities.append(valid)
    a=np.stack(angles,1);c=np.stack(coherences,1);v=np.stack(validities,1)
    circle=np.stack((np.cos(a),np.sin(a)),2)
    sets=dict(cfo80=circle[:,1],coherence80=c[:,1:2],cfo_coherence80=np.column_stack((circle[:,1],c[:,1])),
              three_window=np.column_stack((circle.reshape(-1,6),c)))
    telemetry=dict(angle=a,coherence=c,valid=v,hz=a*25000000/(2*math.pi*20))
    return sets,telemetry


def fit_probe(x,y):
    """Train standardization and all six coefficient columns using L_s only."""
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.int64)
    if x.ndim!=2 or y.shape!=(len(x),) or not np.isfinite(x).all() or set(y.tolist())!=set(range(6)):raise ValueError('Incomplete/invalid L_s probe fit')
    mean=x.mean(0);sd=np.maximum(x.std(0),1e-8);a=(x-mean)/sd
    onehot=np.eye(6)[y];prior=onehot.mean(0)
    w=np.linalg.solve(a.T@a/len(a)+RIDGE*np.eye(a.shape[1]),a.T@(onehot-prior)/len(a))
    return dict(mean=mean.tolist(),scale=sd.tolist(),weights=w.tolist(),prior=prior.tolist(),training_count=len(y),ridge=RIDGE,
                fit_role='L_s',V_fit=False,class_count=6)


def predict_probe(state,x):
    a=(np.asarray(x,dtype=np.float64)-np.asarray(state['mean']))/np.asarray(state['scale'])
    logits=a@np.asarray(state['weights'])+np.asarray(state['prior'])
    if not np.isfinite(logits).all():raise ValueError('Nonfinite probe')
    return logits.argmax(1)


def metrics(y,pred):
    cm=np.zeros((6,6),dtype=np.int64);np.add.at(cm,(y,pred),1)
    den=cm.sum(0)+cm.sum(1);tp=np.diag(cm)
    return dict(count=len(y),accuracy=float(tp.sum()/len(y)),macro_f1=float(np.divide(2*tp,den,out=np.zeros(6),where=den>0).mean()),confusion=cm.tolist())


def factor_association(x,y,rx,day):
    """Balanced source crossed-factor variance on circular coordinates, not Hz.

    Describes association. Does not claim isolated physical TX/RX causality.
    """
    x=np.asarray(x,dtype=np.float64);mu=x.mean(0);total=float(((x-mu)**2).sum())
    def centers(labels):return {k:x[labels==k].mean(0) for k in np.unique(labels)}
    ty,rr,dd=centers(y),centers(rx),centers(day)
    terms={name:sum(int((labels==k).sum())*float(((means[k]-mu)**2).sum()) for k in means)
           for name,labels,means in [('TX',y,ty),('RX',rx,rr),('day',day,dd)]}
    interaction=0.
    for t in ty:
        for r in rr:
            mask=(y==t)&(rx==r)
            interaction+=int(mask.sum())*float(((x[mask].mean(0)-ty[t]-rr[r]+mu)**2).sum())
    terms['TX_RX_interaction']=interaction;terms['remaining']=max(0.,total-sum(terms.values()))
    return dict(total_squared_variation=total,association_fraction={k:v/total if total>1e-20 else None for k,v in terms.items()},
                claim='Balanced-source circular-coordinate association;not causal hardware decomposition')


def validate_config(c):
    if any(c.get(k) for k in ('target','target_truth','p1_capsule','checkpoint','teacher','resume','initial_checkpoint','augmentation')):raise ValueError('Source-only diagnostic refuses target/weights/augmentation')
    expected=dict(method='cvs_cfo_information_source',run_id=RUN,source_receivers=list(RX),source_days=[1,2,3],windows=[list(w) for w in WINDOWS],
                  split_seed=392005,feature_sets=list(FEATURES),ridge=RIDGE,source_contract=PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json',
                  dataset=PROJECT+'/Dataset_WigSig/ManySig.pkl',output_root=PROJECT+'/runs/'+RUN+'/source')
    if any(c.get(k)!=v for k,v in expected.items()):raise ValueError('Fixed source probe configuration differs')
    return c


def run(c):
    validate_config(c)
    import torch
    from baselines.common.practical_source import build_contract_split
    from baselines.common.cvs_data import make_cvs_loader
    from experiments.cvs_identity_ce.source import source_args
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    split=build_contract_split(source_args(c))
    if split.test or split.named_tests or split.split_info['counts']!={'L_s':6300,'U_s':56700,'V':27000}:raise ValueError('Unexpected source/target roles')
    actual=json.loads((out/'source_contract.json').read_text());expected=json.loads(Path(c['source_contract']).read_text())
    if any(actual.get(k)!=v for k,v in expected.items()):raise ValueError('Actual physical role mismatch')
    resolved=dict(c,pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,python=sys.executable,commit=(ROOT/'release_commit.txt').read_text().strip(),
                  source_counts=split.split_info['counts'],target_access=False,target_constructed=False,checkpoint_access=False,
                  formal_CVS_training=False,probe_fit_role='L_s',probe_V_fit=False,source_roles_exact_match=True,U_s_use='unused',device='cpu',
                  optimizer_steps=0,scope='Diagnostic closed-form identity probes;not formal CE candidates or TX parameter estimators')
    write(out/'resolved_config.json',resolved);print('RESOLVED_CONFIG '+json.dumps(resolved),flush=True)
    start=time.perf_counter();roles={};groups=[];associations={}
    for role,dataset in [('L_s',split.train),('V',split.val)]:
        values={k:[] for k in FEATURES};ys=[];rs=[];ds=[];tele={k:[] for k in ('angle','coherence','valid','hz')}
        for batch in make_cvs_loader(dataset,batch_size=128,shuffle=False,num_workers=0,device=torch.device('cpu'),drop_last=False):
            f,t=packet_features(batch['iq'].tolist())
            for k in FEATURES:values[k].append(f[k])
            for k in tele:tele[k].append(t[k])
            ys+=batch['label'].tolist();rs+=batch['receiver'].tolist();ds+=batch['day'].tolist()
        y,rx,day=map(np.asarray,(ys,rs,ds));values={k:np.concatenate(v) for k,v in values.items()};tele={k:np.concatenate(v) for k,v in tele.items()}
        if len(y)!=(6300 if role=='L_s' else 27000) or set(rx.tolist())!=set(RX) or set(day.tolist())!={1,2,3}:raise ValueError('Source coverage differs')
        roles[role]=dict(features=values,y=y,rx=rx,day=day)
        for ty in range(6):
            for r in RX:
                for d in [1,2,3]:
                    m=(y==ty)&(rx==r)&(day==d);n=int(m.sum())
                    if n!=(70 if role=='L_s' else 300):raise ValueError('Source crossed cell incomplete')
                    groups.append(dict(role=role,tx=ty,receiver=r,day=d,count=n,
                        cfo_hz_mean=tele['hz'][m].mean(0).tolist(),cfo_hz_sd=tele['hz'][m].std(0).tolist(),
                        circular_mean=np.column_stack((np.cos(tele['angle'][m]),np.sin(tele['angle'][m]))).mean(0).tolist(),
                        coherence_mean=tele['coherence'][m].mean(0).tolist(),fallback_fraction=(~tele['valid'][m]).mean(0).tolist()))
        associations[role]=factor_association(values['cfo80'],y,rx,day)
        print('ROLE '+json.dumps(dict(role=role,count=len(y),groups=90,seconds=time.perf_counter()-start)),flush=True)
    fitted=[];records=[]
    L,V=roles['L_s'],roles['V']
    for name in FEATURES:
        for held in [None,*RX]:
            train=np.ones(len(L['y']),dtype=bool) if held is None else L['rx']!=held
            evaluate=np.ones(len(V['y']),dtype=bool) if held is None else V['rx']==held
            state=fit_probe(L['features'][name][train],L['y'][train]);state.update(feature=name,held_source_rx=held)
            fitted.append(state);pred=predict_probe(state,V['features'][name][evaluate]);y=V['y'][evaluate]
            row=dict(feature=name,held_source_rx=held,fit_receivers=list(RX) if held is None else [r for r in RX if r!=held],
                     fit_count=int(train.sum()),fit_role='L_s',evaluation_role='V',diagnostic_only=True,**metrics(y,pred))
            row['per_source_rx']={str(r):metrics(y[(V['rx'][evaluate]==r)],pred[(V['rx'][evaluate]==r)]) for r in np.unique(V['rx'][evaluate])}
            records.append(row);print('PROBE '+json.dumps({k:v for k,v in row.items() if k not in ('confusion','per_source_rx')}),flush=True)
    result=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',run_id=RUN,counts={'L_s':6300,'V':27000},groups=groups,probes=records,
                fitted_probes=fitted,associations=associations,target_access=False,checkpoint_access=False,V_fit=False,
                source_only=True,formal_CVS_training=False,unseen_source_rx_scope='Leave-one-source-RX-out L_s fit;existing V masked only for scoring;not target test',
                claim='Frequency/coherence source association diagnostic;not recovered TX oscillator, causal decomposition or final clean performance',elapsed_seconds=time.perf_counter()-start)
    write(out/'information_statistics.json',result)
    write(out/'completion.json',dict(status=result['status'],counts=result['counts'],groups=len(groups),probes=len(records),target_access=False,V_fit=False,elapsed_seconds=result['elapsed_seconds']))
    print('COMPLETE '+json.dumps(dict(counts=result['counts'],groups=len(groups),probes=len(records),elapsed_seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(json.loads(Path(p.parse_args().config).read_text(encoding='utf-8')))
