"""Frozen operator diagnostic; no source/target IQ, optimization or model selection."""
import argparse,copy,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'code')]
from experiments.cvs_mirror_mechanism.diagnose import validate_payload,validate_source_contract,load_verified_state
from experiments.cvs_mirror_subspace_identity.model import build,VARIANTS
from experiments.cvs_mirror_subspace_identity.collect import validate_completed,SOURCE_COMMIT
from experiments.cvs_spectral_relation_identity.physics import unit_rms,multiply_pair

RUN='20261003-diagnostic-cvs-mirror-history-public-m8-r01'
PARENT='20261003-diagnostic-cvs-mirror-mechanism-public-m8-r02'
SEEDS=(2026092701,2026092702,2026092703,2026092704)
DELAYS=(2,8,16)
GROUPS=('noise','tone','periodic','dc','zero')
PUBLIC_SEED=2026100321
HISTORY=16

def write(p,d):
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)

def public_history():
    """Generate -16..255 jointly, normalize the entire record by observed-crop RMS."""
    g=torch.Generator().manual_seed(PUBLIC_SEED)
    n=torch.arange(-HISTORY,256,dtype=torch.float64)[None];a=torch.arange(1,7,dtype=torch.float64)[:,None]
    phase=(.071+.021*a)*n+.13*a
    noise=torch.randn(6,2,256+HISTORY,generator=g,dtype=torch.float64)
    tone=torch.stack((phase.cos(),phase.sin()),1)
    env=1+.22*torch.cos(2*torch.pi*n/20)+.11*torch.sin(4*torch.pi*n/20+.1*a)
    periodic=torch.stack((env*(phase+.1*env).cos(),env*(phase+.1*env).sin()),1)
    dc=torch.stack((torch.ones_like(phase)*(.1*a).cos(),torch.ones_like(phase)*(.1*a).sin()),1)
    z=torch.cat((noise,tone,periodic,dc,torch.zeros_like(noise)))
    scale=torch.rsqrt(z[...,HISTORY:].square().sum(1).mean(-1,keepdim=True).clamp_min(1e-12))[:,None]
    return (z*scale).float()

def paired_fir(history,delay,continuous):
    if history.shape!=(30,2,272) or delay not in DELAYS or type(continuous) is not bool:raise ValueError('Fixed public FIR domain differs')
    past=history if continuous else torch.cat((torch.zeros_like(history[...,:HISTORY]),history[...,HISTORY:]),-1)
    return past[...,HISTORY:]+multiply_pair(past[...,HISTORY-delay:HISTORY+256-delay],.23,.09)

@torch.no_grad()
def diagnose(model):
    model.eval();model.requires_grad_(False)
    before={k:v.clone() for k,v in model.state_dict().items()}
    h=public_history();x=h[...,HISTORY:];arrays={'history_iq':h.numpy(),
        'window':model.core.mirror_relation.window.numpy(),
        'mix_real':model.core.mirror_relation.mix_real.numpy(),'mix_imag':model.core.mirror_relation.mix_imag.numpy()}
    for precision,block,history in [('fp32',model.core.mirror_relation,h),('fp64',copy.deepcopy(model.core.mirror_relation).double(),h.double())]:
        base=history[...,HISTORY:];arrays[precision+'_q_base']=block.statistics(base).numpy()
        arrays[precision+'_spectra_base']=block.spectral(base).numpy()
        for delay in DELAYS:
            for mode in ('zero','continuous'):
                raw=paired_fir(history,delay,mode=='continuous');received=unit_rms(raw)
                prefix=f'{precision}_d{delay}_{mode}_'
                arrays[prefix+'raw_iq']=raw.numpy();arrays[prefix+'iq']=received.numpy()
                arrays[prefix+'raw_spectra']=block.spectral(raw).numpy()
                spectra=block.spectral(received);arrays[prefix+'spectra']=spectra.numpy()
                v=block.mix_spectra(spectra);components=block.components(v)
                arrays[prefix+'q']=block.relations(v).numpy()
                arrays[prefix+'energy_floor']=components['energy_floor_active'].numpy()
                arrays[prefix+'determinant_floor']=components['determinant_floor_active'].numpy()
    arrays['embedding_base']=F.normalize(model.features(x),dim=1,eps=1e-4).numpy()
    for delay in DELAYS:
        for mode in ('zero','continuous'):
            received=torch.from_numpy(arrays[f'fp32_d{delay}_{mode}_iq'])
            arrays[f'embedding_d{delay}_{mode}']=F.normalize(model.features(received),dim=1,eps=1e-4).numpy()
    if any(not torch.equal(before[k],v) for k,v in model.state_dict().items()):raise ValueError('Frozen model state changed')
    if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite public diagnostic')
    return arrays

def run(output):
    spec=json.loads((ROOT/'automation_reports/CV-SincNet'/RUN/'experiment.json').read_text(encoding='utf-8'))
    if spec['status']!='LOCAL_VERIFIED':raise ValueError('Preregistered local validation required')
    source=json.loads((ROOT/'local_artifacts/cvs_mirror_subspace_identity_20261003_r01/source_research_complete.json').read_text(encoding='utf-8'))
    validate_completed(source);rows={r['row_id']:r for r in source['rows']}
    expected={v+'-s'+str(s) for v in VARIANTS for s in SEEDS}
    if set(rows)!=expected or {r['row_id'] for r in spec['rows']}!=expected:raise ValueError('All eight frozen models required')
    cache=ROOT/'local_artifacts'/PARENT;reference=json.loads((cache/'original_source_contract.json').read_text(encoding='utf-8'))
    if reference['role_ids']['V']!=source['source_v_ids']:raise ValueError('Original data contract differs')
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    write(output/'launch.json',dict(run_id=RUN,pid=os.getpid(),cwd=str(ROOT),python=sys.executable,torch=torch.__version__,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_commit=SOURCE_COMMIT,
        checkpoint_cache=str(cache),device='cpu',public_seed=PUBLIC_SEED,source_IQ_access=False,target_access=False,optimizer_steps=0))
    results=[]
    for rid,row in sorted(rows.items()):
        folder=cache/rid;payload=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        contract=json.loads((folder/'source_contract.json').read_text(encoding='utf-8'))
        if validate_payload(payload,row)!=contract:raise ValueError('Checkpoint physical contract differs')
        validate_source_contract(contract,reference,row)
        model=build(row['resolved']['variant']);load_verified_state(model,payload['model'],row['resolved']['mirror_relation_actual'])
        out=output/rid;out.mkdir();started=time.time();arrays=diagnose(model)
        np.savez_compressed(out/'paired_arrays.npz',**arrays)
        result=dict(row_id=rid,variant=row['resolved']['variant'],model_seed=row['resolved']['model_seed'],status='VERIFIED',
            frozen_checkpoint=str(folder/'last.pt'),source_commit=SOURCE_COMMIT,model_state_unchanged=True,public_packets=30,
            cases=180,optimizer_steps=0,source_IQ_access=False,target_access=False,seconds=time.time()-started)
        write(out/'completion.json',result);results.append(result);print(json.dumps(result),flush=True)
    write(output/'completion.json',dict(status='VERIFIED',run_id=RUN,rows=results,public_packets_per_model=30,paired_cases=1440,target_access=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
