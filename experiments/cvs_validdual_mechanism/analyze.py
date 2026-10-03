"""Balanced source-factor decomposition and public continuous-history probes."""
import argparse,csv,itertools,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from experiments.cvs_validdual_clean.contracts import expected_rows,prediction_config,SOURCE_COMMIT,SOURCE_RUN
from experiments.cvs_validdual_clean.provenance import validate_payload

RUN='20261003-diagnostic-cvs-validdual-mechanism-source-public-m8-r01'
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'local_artifacts/cvs_validdual_clean_20261003_r01/completed/source'
OUTPUT=ROOT/'local_artifacts'/RUN
FACTORS=('TX','RX','day')
AXES=(tuple(range(6)),(1,3,4,6,8),(1,2,3))

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,d):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)

def decompose(cells):
    """Exact descriptive balanced ANOVA, population moments, no p-values."""
    if len(cells)!=90:raise ValueError('Expected all90 balanced source cells')
    x=np.empty((6,5,3,66),dtype=np.float64);within=[];seen=set()
    for c in cells:
        key=(c['tx'],c['receiver'],c['day'])
        if key in seen or any(v not in axis for v,axis in zip(key,AXES)) or c['count']!=300:raise ValueError('Duplicate, missing, non-source or unbalanced cell')
        seen.add(key);a=np.asarray(c['coefficient_mean'],dtype=np.float64);v=c['coefficient_trace_variance']
        if a.shape!=(66,) or not np.isfinite(a).all() or not np.isfinite(v) or v< -1e-12:raise ValueError('Invalid coefficient moments')
        x[tuple(axis.index(k) for axis,k in zip(AXES,key))]=a;within.append(max(0.,v))
    mean=x.mean((0,1,2),keepdims=True);effects={():np.broadcast_to(mean,x.shape)};energy={}
    for size in range(1,4):
        for subset in itertools.combinations(range(3),size):
            reduced=tuple(k for k in range(3) if k not in subset)
            m=x.mean(reduced,keepdims=True) if reduced else x
            effect=np.broadcast_to(m,x.shape).copy()
            for other,value in effects.items():
                if set(other)<set(subset):effect-=value
            effects[subset]=effect;energy[':'.join(FACTORS[k] for k in subset)]=float(np.mean(np.sum(effect*effect,axis=-1)))
    between=float(np.mean(np.sum((x-mean)**2,axis=-1)));noise=float(np.mean(within));total=between+noise
    if not np.isclose(sum(energy.values()),between,rtol=1e-10,atol=1e-14):raise ValueError('ANOVA conservation failed')
    return dict(between_cell_variance=between,within_cell_variance=noise,total_variance=total,
        numerical_floor=1e-12,degenerate=total<=1e-12,components=energy,
        fractions_of_total={k:(v/total if total>1e-12 else None) for k,v in energy.items()},
        within_fraction=noise/total if total>1e-12 else None,cell_means=x)

def public_signals():
    records=[];waves=[]
    for family in ('noise','qpsk','tone','periodic16'):
        for s in range(4):
            rng=np.random.default_rng(91001+s);n=np.arange(320)
            if family=='noise':z=rng.normal(size=320)+1j*rng.normal(size=320)
            elif family=='qpsk':z=np.exp(1j*(np.pi/4+np.pi/2*rng.integers(0,4,320)))
            elif family=='tone':z=np.exp(1j*(.04+.013*s)*n+1j*.17*s)
            else:z=np.tile(rng.normal(size=16)+1j*rng.normal(size=16),20)
            waves.append(z);records.append(dict(family=family,public_seed=91001+s))
    return records,np.stack(waves)

def received_views(waves):
    out={}
    for name in ('identity','phase','delay1','delay4','delay16'):
        if name=='identity':z=waves[:,64:].copy()
        elif name=='phase':z=waves[:,64:]*np.exp(.4j)
        else:
            lag=int(name[5:]);z=waves[:,64:]+(.23+.09j)*waves[:,64-lag:320-lag]
        z=z/np.sqrt(np.mean(abs(z)**2,axis=-1,keepdims=True))
        out[name]=np.stack((z.real,z.imag),axis=1).astype(np.float32)
    return out

def drift(a,b):
    aa=np.asarray(a,dtype=np.float64).reshape(len(a),-1);bb=np.asarray(b,dtype=np.float64).reshape(len(b),-1)
    return np.linalg.norm(aa-bb,axis=1)/np.maximum(np.linalg.norm(aa,axis=1),1e-12)

def probe(model,views):
    arrays={};state={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.eval().requires_grad_(False)
    with torch.no_grad():
        for name,x in views.items():
            z,raw,changed,g,c=model.parts(torch.from_numpy(x))
            for key,value in dict(z=z,raw=raw,corrected=g,coefficients=c).items():arrays[name+'_'+key]=value.numpy().copy()
    if any(not torch.equal(v,state[k]) for k,v in model.state_dict().items()):raise ValueError('Frozen model state changed')
    metrics=[]
    for name in views:
        if name=='identity':continue
        di=drift(views['identity'],views[name]);dg=drift(arrays['identity_corrected'],arrays[name+'_corrected'])
        dz=drift(arrays['identity_z'],arrays[name+'_z']);dr=drift(arrays['identity_raw'],arrays[name+'_raw'])
        dc=np.linalg.norm((arrays['identity_coefficients']-arrays[name+'_coefficients']).reshape(16,-1).astype(np.float64),axis=1)
        for i in range(16):metrics.append(dict(view=name,public_index=i,input_drift=float(di[i]),corrected_drift=float(dg[i]),raw_feature_drift=float(dr[i]),fused_feature_drift=float(dz[i]),coefficient_change=float(dc[i]),
            correction_ratio=float(dg[i]/di[i]) if di[i]>1e-6 else None,feature_ratio=float(dz[i]/dr[i]) if dr[i]>1e-6 else None))
    return arrays,metrics

def run():
    output=OUTPUT/'analysis';output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();torch.set_num_threads(2)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    write(output/'launch.json',dict(pid=os.getpid(),argv=sys.argv,cwd=str(Path.cwd()),commit=commit,source_commit=SOURCE_COMMIT,python=sys.executable,torch=torch.__version__,numpy=np.__version__,device='cpu',target_access=False))
    public,waves=public_signals();views=received_views(waves)
    np.savez_compressed(output/'public_inputs.npz',continuous=waves,**views)
    write(output/'public_records.json',public)
    expected=read(OUTPUT/'weights/source_contract.json');all_factors=[];all_metrics=[]
    frozen=read(SOURCE/'frozen_source_matrix.json')
    if frozen['status']!='ALL_E200_FROZEN' or frozen['commit']!=SOURCE_COMMIT or frozen['target_access'] is not False:raise ValueError('Source freeze differs')
    if {(r['variant'],r['seed']) for r in frozen['records']}!=set(expected_rows().values()) or len(frozen['records'])!=8:raise ValueError('Frozen matrix differs')
    for rid,(variant,seed) in sorted(expected_rows().items()):
        folder=SOURCE/rid;resolved=read(folder/'resolved_config.json')
        payload=torch.load(OUTPUT/'weights'/rid/'last.pt',map_location='cpu',weights_only=False)
        model=validate_payload(prediction_config(rid),read(folder/'completion.json'),read(folder/'initialization.json'),read(folder/'source_contract.json'),expected,resolved,payload)
        model.eval().requires_grad_(False)
        with torch.no_grad():
            if not torch.isfinite(model(torch.zeros(2,2,256))).all():raise ValueError('Actual checkpoint no-query smoke failed')
        diag=read(folder/'source_final_diagnostics.json')
        if diag['role']!='V' or diag['count']!=27000 or any(diag.get(k) is not False for k in ('target_access','used_for_training','used_for_selection')):raise ValueError('Not frozen source-only diagnostics')
        result=decompose(diag['validdual_groups']);cells=result.pop('cell_means')
        result.update(row_id=rid,variant=variant,model_seed=seed);all_factors.append(result)
        arrays,metrics=probe(model,views);np.savez_compressed(output/(rid+'.npz'),cell_means=cells,**arrays)
        for m in metrics:m.update(row_id=rid,variant=variant,model_seed=seed,**public[m['public_index']]);all_metrics.append(m)
        print(json.dumps(dict(row_id=rid,status='FROZEN_DIAGNOSTIC_COMPLETE',public_pairs=len(metrics))),flush=True)
    write(output/'source_factors.json',dict(rows=all_factors,scope='720 source cells,8 reused frozen networks,27000 distinct V IDs per network; no new fit or target access'))
    write(output/'public_metrics.json',dict(rows=all_metrics,claim='Conditional within-network sensitivity, not independently retrained ablation or transmitter accuracy'))
    with (output/'public_metrics.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_metrics[0]));w.writeheader();w.writerows(all_metrics)
    write(output/'complete.json',dict(status='ARTIFACTS_COMPLETE',rows=8,source_cells=720,public_pairs=512,elapsed_seconds=time.perf_counter()-started,target_access=False,new_checkpoints=False,independent_recount_pending=True))

if __name__=='__main__':run()
