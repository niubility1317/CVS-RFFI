"""Fixed public TX/channel/RX cascade; not augmentation or RF classification."""
import itertools,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from experiments.cvs_rff_physics.known_excitation import template
from .model import ReferenceResidual,VARIANTS

ROOT=Path(__file__).resolve().parents[2]
RUN='20261003-diagnostic-cvs-reference-residual-public-m2-r01'
OUT=ROOT/'local_artifacts'/RUN
TX=('ideal','linear_d1','iq_mirror','pa3','pa5','memory3')
CHANNELS=('identity','delay1','delay4','delay16')
RX=('identity','iq_mirror','pa3')

def write(p,d):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def transmit(n,kind):
    s=template(n)
    if kind=='ideal':return s
    if kind=='linear_d1':return s+(.08+.03j)*template(n-1)
    if kind=='iq_mirror':return s+(.06+.02j)*s.conj()
    if kind=='pa3':return s+(.04+.01j)*s*abs(s)**2
    if kind=='pa5':return s+(.007+.003j)*s*abs(s)**4
    if kind=='memory3':
        previous=template(n-1);return s+(.04+.01j)*previous*abs(previous)**2
    raise ValueError('Unknown fixed TX')

def signals():
    records=[];iq=[];n=np.arange(256)
    for tx,channel,rx in itertools.product(TX,CHANNELS,RX):
        x=transmit(n,tx)
        if channel!='identity':x=x+(.23+.09j)*transmit(n-int(channel[5:]),tx)
        if rx=='iq_mirror':x=x+(.04-.01j)*x.conj()
        if rx=='pa3':x=x+(.015+.01j)*x*abs(x)**2
        x=x/np.sqrt(np.mean(abs(x)**2));iq.append(np.stack((x.real,x.imag)))
        records.append(dict(tx=tx,channel=channel,rx=rx))
    return records,np.asarray(iq,dtype=np.float32)

def paired_metrics(records,iq,features):
    lookup={(r['tx'],r['channel'],r['rx']):i for i,r in enumerate(records)};out=[]
    for tx,channel,rx in itertools.product(TX[1:],CHANNELS,RX):
        i=lookup[(tx,channel,rx)];base=lookup[('ideal',channel,rx)]
        anchor=lookup[(tx,'identity','identity')];anchorbase=lookup[('ideal','identity','identity')]
        d=features[i].astype(np.float64)-features[base].astype(np.float64)
        ref=features[anchor].astype(np.float64)-features[anchorbase].astype(np.float64)
        raw=iq[i].astype(np.float64)-iq[base].astype(np.float64)
        # RMS changes have unlike lengths (80 vs256), so normalize each by its number of real entries.
        size=np.sqrt(np.mean(d*d));rawsize=np.sqrt(np.mean(raw*raw));refsize=np.sqrt(np.mean(ref*ref))
        out.append(dict(tx=tx,channel=channel,rx=rx,TX_signature_rms=float(size),raw_TX_difference_rms=float(rawsize),
            TX_retention_ratio=float(size/rawsize) if rawsize>1e-8 else None,
            relative_signature_drift=float(np.sqrt(np.mean((d-ref)**2))/refsize) if refsize>1e-8 else None))
    return out

def run():
    OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);records,iq=signals()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();started=time.perf_counter()
    write(OUT/'launch.json',dict(pid=os.getpid(),argv=sys.argv,cwd=str(Path.cwd()),commit=commit,python=sys.executable,torch=torch.__version__,numpy=np.__version__,device='cpu',target_access=False,training=False))
    write(OUT/'records.json',records);np.savez_compressed(OUT/'inputs.npz',iq=iq)
    metrics=[]
    for variant in VARIANTS:
        model=ReferenceResidual(variant).eval().requires_grad_(False)
        with torch.no_grad():values=model.components(torch.from_numpy(iq))
        arrays={k:v.numpy() for k,v in values.items()};arrays['ridge']=model.ridge().detach().numpy()
        np.savez_compressed(OUT/(variant+'.npz'),**arrays)
        rows=paired_metrics(records,iq,arrays['features'])
        for row in rows:row['variant']=variant;metrics.append(row)
        print(json.dumps(dict(variant=variant,public_inputs=len(iq),TX_signature_pairs=len(rows))),flush=True)
    write(OUT/'metrics.json',dict(rows=metrics,claim='Untrained analytic branch initialization only; neither trained network identity accuracy nor TX/RX disentanglement'))
    write(OUT/'complete.json',dict(status='ARTIFACTS_COMPLETE',variants=2,public_inputs_per_variant=72,TX_signature_pairs=120,elapsed_seconds=time.perf_counter()-started,target_access=False,training=False))

if __name__=='__main__':run()
