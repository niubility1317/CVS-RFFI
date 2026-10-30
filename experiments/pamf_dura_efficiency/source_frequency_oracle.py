"""Source-only FFT-bin replacement probes; no deployable estimator or fitting."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from experiments.pamf_dura_direct_train import runner as training,f_guided
from experiments.pamf_dura_diagnostics.runner import source_indices,tensor,numpy_values
from .phase_sensitivity import metrics

RUN='20261031-diagnostic-DURA-source-frequency-oracle-s2026092701-r01'
SOURCE_RUN='20261030-diagnostic-DURA-source-phase-oracle-s2026092701-m2-r01'
CASES=('native','complex_Q1','complex_Q2','complex_Q3','complex_Q4','phase_Q4','magnitude_Q4')


def replace_bins(clean,restored,quartile,component='complex'):
    if clean.shape!=restored.shape or clean.ndim!=3 or clean.shape[1:]!=(2,256):
        raise ValueError('Matching [N,2,256] IQ required')
    if quartile not in range(4) or component not in ('complex','phase','magnitude'):
        raise ValueError('Fixed four quartiles and three components required')
    x=clean[:,0].astype(np.float64)+1j*clean[:,1].astype(np.float64)
    z=restored[:,0].astype(np.float64)+1j*restored[:,1].astype(np.float64)
    xf=np.fft.fft(x,norm='ortho');zf=np.fft.fft(z,norm='ortho')
    power=np.abs(xf)**2;order=np.argsort(-power,axis=1,kind='stable')
    mask=np.zeros_like(power,dtype=bool);np.put_along_axis(mask,order[:,quartile*64:(quartile+1)*64],True,axis=1)
    if component=='complex':replacement=xf
    elif component=='phase':
        # A zero reference coefficient has no phase; leave that coefficient
        # unchanged rather than inventing a reference phase.
        unit=np.divide(xf,np.abs(xf),out=np.zeros_like(xf),where=np.abs(xf)>0)
        replacement=np.where(np.abs(xf)>0,np.abs(zf)*unit,zf)
    else:
        # For zero restored coefficients phase is undefined; retain zero.
        unit=np.divide(zf,np.abs(zf),out=np.zeros_like(zf),where=np.abs(zf)>0)
        replacement=np.abs(xf)*unit
    modified=np.where(mask,replacement,zf)
    value=np.fft.ifft(modified,norm='ortho')
    iq=np.stack((value.real,value.imag),axis=1).astype(np.float32)
    energy=power.sum(1)
    if np.any(energy<=0) or not np.isfinite(iq).all():raise ValueError('Finite nonzero source clean required')
    current=np.fft.fft(iq[:,0].astype(np.float64)+1j*iq[:,1].astype(np.float64),norm='ortho')
    return iq,dict(clean_energy_share=(power*mask).sum(1)/energy,
        native_iq_nmse=(np.abs(zf-xf)**2).sum(1)/energy,
        iq_nmse=(np.abs(current-xf)**2).sum(1)/energy,
        replaced_bin_count=mask.sum(1).astype(np.float64))


def aggregate(values,mask):
    return dict(count=int(mask.sum()),teacher_choices=int(values['teacher_agreement'][mask].sum()),
        metrics={key:dict(mean=float(value[mask].mean(dtype=np.float64)),
                         median=float(np.median(value[mask])),p95=float(np.quantile(value[mask],.95)))
                 for key,value in values.items()})


@torch.no_grad()
def run(args):
    output=args.root/'runs'/RUN
    if output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('New CPU source-only root required')
    p=training.paths(args.root);clean,roles=training.source_training(p)
    ids=f_guided.source_ids(p,roles);positions=source_indices(roles)
    source=args.root/'runs'/SOURCE_RUN;metadata=training.read_json(source/'results.json')
    if (metadata.get('status')!='ANALYZED_SOURCE_ONLY' or metadata.get('F_teacher')!=f_guided.teacher_reference()
        or any(metadata.get(k)!=0 for k in ('optimizer_updates','TX_labels_read','V_reads','target_reads'))):
        raise ValueError('Complete legal original paired source output required')
    arrays={}
    for model in ('noVar','DURA_default'):
        for view in ('1','20'):
            with np.load(source/f'{model}-{view}.npz',allow_pickle=False) as pack:
                a={k:pack[k] for k in ('physical_ids','source_positions','roles','rx','day','scene','state','lock',
                    'clean','native','F_features','F_logits')}
            if (not np.array_equal(a['physical_ids'],ids[positions]) or not np.array_equal(a['source_positions'],positions)
                or not np.array_equal(a['roles'],roles[positions]) or not np.array_equal(a['clean'],clean[positions])
                or a['native'].shape!=(512,2,256) or a['native'].dtype!=np.float32
                or a['F_features'].shape!=(3,512,160) or a['F_logits'].shape!=(3,512,6)
                or set(a['scene'].tolist())!=set(training.VIEWS)):
                raise ValueError('Original matched source512 and seven source scenes required')
            if any(not np.isfinite(a[k]).all() for k in ('native','F_features','F_logits')):raise ValueError('Nonfinite source array')
            arrays[model,view]=a
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    before={k:v.clone() for k,v in reader.state_dict().items()}
    output.mkdir()
    training.write_json(output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],CPU_threads=2,GPU_workers=0,F_teacher=f_guided.teacher_reference()))
    reference=arrays['noVar','1']
    smoke=reader(tensor(reference['clean'][:2],'cpu'))
    torch.testing.assert_close(smoke,tensor(reference['F_logits'][0,:2],'cpu'),atol=2e-5,rtol=2e-5)
    print(json.dumps(dict(event='REAL_SOURCE_F_CHECKPOINT_SMOKE_PASS',Ls=2,G_reads=0,target_reads=0)),flush=True)
    started=time.perf_counter();all_f=[];all_z=[]
    for offset in range(0,512,32):
        f=reader.features(tensor(reference['clean'][offset:offset+32],'cpu'))
        all_f.append(f);all_z.append(reader.classify_features(f))
    rf=torch.cat(all_f);rz=torch.cat(all_z)
    torch.testing.assert_close(rf,tensor(reference['F_features'][0],'cpu'),atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(rz,tensor(reference['F_logits'][0],'cpu'),atol=2e-5,rtol=2e-5)
    summary={};inputs=512
    for (model,view),a in arrays.items():
        torch.testing.assert_close(tensor(a['F_features'][0],'cpu'),rf,atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(tensor(a['F_logits'][0],'cpu'),rz,atol=2e-5,rtol=2e-5)
        vectors={};saved_iq=[];saved_f=[];saved_z=[];cell={}
        for case in CASES:
            if case=='native':
                iq=a['native'];energy=np.square(a['clean'].astype(np.float64)).sum((1,2))
                terms=dict(iq_nmse=np.square(iq.astype(np.float64)-a['clean']).sum((1,2))/energy,
                    native_iq_nmse=np.square(iq.astype(np.float64)-a['clean']).sum((1,2))/energy,
                    clean_energy_share=np.zeros(512),replaced_bin_count=np.zeros(512))
            else:
                component,q=case.split('_');iq,terms=replace_bins(a['clean'],a['native'],int(q[1:])-1,component)
            fs=[];zs=[];local={}
            for offset in range(0,512,32):
                f=reader.features(tensor(iq[offset:offset+32],'cpu'));z=reader.classify_features(f)
                for key,value in metrics(f,z,rf[offset:offset+32],rz[offset:offset+32]).items():
                    local.setdefault(key,[]).append(numpy_values(value))
                fs.append(numpy_values(f));zs.append(numpy_values(z));inputs+=len(f)
            values={k:np.concatenate(v) for k,v in local.items()};values.update(terms)
            if any(v.shape!=(512,) or not np.isfinite(v).all() for v in values.values()):raise FloatingPointError('Nonfinite source metrics')
            if case.startswith('complex_') and np.any(values['iq_nmse']>values['native_iq_nmse']+1e-7):
                raise ValueError('Complex source replacement must not raise IQ MSE')
            if case=='native':
                np.testing.assert_allclose(np.concatenate(zs),a['F_logits'][1],atol=2e-5,rtol=2e-5)
            masks={'all':np.ones(512,dtype=bool)}
            for key in ('roles','rx','day','scene','state','lock'):
                for v in np.unique(a[key]):masks[f'{key}:{v}']=a[key]==v
            cell[case]={key:aggregate(values,mask) for key,mask in masks.items()}
            vectors.update({case+'/'+k:v for k,v in values.items()})
            saved_iq.append(iq);saved_f.append(np.concatenate(fs));saved_z.append(np.concatenate(zs))
            print(json.dumps(dict(event='SOURCE_FREQUENCY_CASE_COMPLETE',model=model,view=view,case=case,F_inputs=inputs)),flush=True)
        np.savez(output/f'{model}-{view}.npz',**{k:a[k] for k in ('physical_ids','source_positions','roles','rx','day','scene','state','lock','clean')},
            case_names=np.asarray(CASES),IQ=np.stack(saved_iq),F_features=np.stack(saved_f),F_logits=np.stack(saved_z),
            reference_F_features=numpy_values(rf),reference_F_logits=numpy_values(rz),**vectors)
        summary[model+'/'+view]=cell
    if inputs!=14848 or any(not torch.equal(v,before[k]) for k,v in reader.state_dict().items()) or any(v.grad is not None for v in reader.parameters()):
        raise ValueError('Fixed F budget/state changed')
    training.write_json(output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',run=RUN,commit=args.commit,
        summary=summary,source_run=SOURCE_RUN,source_records=512,L_s=128,U_s=384,
        source_views=['1','20'],models=['noVar','DURA_default'],cases=list(CASES),F_unchanged=True,
        F_analysis_inputs=inputs,F_smoke_inputs=2,G_inputs=0,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,
        seconds=time.perf_counter()-started,selection='Descriptive source oracle;no automated training or model selection',
        scope='Pairedsourceclean FFT oracle,not deployable/not actualTXaccuracy/not a certified recoverability bound'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
