"""Source-only known-action Gaussian reference, never a recovery upper bound."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from scipy.linalg import cho_factor,cho_solve

RUN='20261031-diagnostic-DURA-known-action-Gaussian-s2026092701-r01'
PREVIOUS='20261030-diagnostic-DURA-source-phase-oracle-s2026092701-m2-r01'


def real_operator(kernel,coefficients):
    h=kernel[0].astype(np.float64)+1j*kernel[1].astype(np.float64)
    a=coefficients[0].astype(np.float64)+1j*coefficients[1].astype(np.float64)
    b=coefficients[2].astype(np.float64)+1j*coefficients[3].astype(np.float64)
    if h.shape!=(64,) or a.shape!=(256,) or b.shape!=(256,):raise ValueError('Original source action support required')
    t=np.arange(256)[:,None];k=np.arange(64)[None,:]
    H=np.zeros((256,256),dtype=np.complex128)
    np.add.at(H,(np.broadcast_to(t,(256,64)),abs(t-k)),np.broadcast_to(h,(256,64)))
    p=a[:,None]*H;q=b[:,None]*H.conj()
    return np.block([[(p+q).real,-(p-q).imag],[(p+q).imag,(p-q).real]])


def noise_covariance(coefficients):
    # Same noEQ source simulator: w~CN(0,10^-3), n=a*w+b*conj(w).
    # Actual AGC is noise-dependent; fixed realized coefficients are a plug-in
    # reference, NOT the exact conditional law given those coefficients.
    ar,ai,br,bi=np.asarray(coefficients,dtype=np.float64)
    u,v,w,z=ar+br,-ai+bi,ai+bi,ar-br
    N=np.zeros((512,512));i=np.arange(256)
    N[i,i]=.0005*(u*u+v*v);N[i+256,i+256]=.0005*(w*w+z*z)
    N[i,i+256]=N[i+256,i]=.0005*(u*w+v*z)
    return N


def gaussian_reference(received,A,N,mean,covariance):
    cross=covariance@A.T
    S=A@cross+N
    factor=cho_factor((S+S.T)*.5,lower=True,check_finite=True)
    return mean+cross@cho_solve(factor,received.reshape(512)-A@mean,check_finite=True)


def fit_prior(clean,indices):
    count=len(indices);total=np.zeros(512);second=np.zeros((512,512))
    for offset in range(0,count,1024):
        x=np.asarray(clean[indices[offset:offset+1024]],dtype=np.float64).reshape(-1,512)
        total+=x.sum(0);second+=x.T@x
    mean=total/count;covariance=second/count-np.outer(mean,mean)
    covariance=(covariance+covariance.T)*.5
    ridge=1e-6*np.trace(covariance)/512
    if not np.isfinite(ridge) or ridge<=0:raise ValueError('Nondegenerate source covariance required')
    covariance+=ridge*np.eye(512)
    cho_factor(covariance,lower=True,check_finite=True)
    return mean,covariance,float(ridge)


def summary(iq,features,logits,clean,teacher_features,teacher_logits,mask):
    error=np.sum((iq-clean)**2,axis=(1,2),dtype=np.float64)/np.sum(clean**2,axis=(1,2),dtype=np.float64)
    unit=lambda v:v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-8)
    cosine=1-np.sum(unit(features)*unit(teacher_features),axis=1)
    choice=logits.argmax(1)==teacher_logits.argmax(1)
    return dict(count=int(mask.sum()),mean_packet_NMSE=float(error[mask].mean()),
        median_packet_NMSE=float(np.median(error[mask])),p90_packet_NMSE=float(np.quantile(error[mask],.9)),
        F_clean_choice_agreement=int(choice[mask].sum()),F_cosine_error=float(cosine[mask].mean()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only source diagnostic required')
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    import torch
    from experiments.pamf_dura_direct_train import runner as base,f_guided
    from experiments.pamf_dura_diagnostics.runner import source_indices,tensor,numpy_values
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    torch.set_num_threads(2);p=base.paths(args.root);clean,roles=base.source_training(p);ids=f_guided.source_ids(p,roles)
    probe=source_indices(roles);fit=np.setdiff1d(np.arange(63000),probe)
    if len(fit)!=62488 or len(probe)!=512:raise ValueError('Original disjoint fit/probe source cohorts required')
    output=args.root/'runs'/RUN;output.mkdir()
    base.write_json(output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,
        CUDA_VISIBLE_DEVICES='',CPU_threads=2,target_reads=0,TX_labels_read=0))
    previous=args.root/'runs'/PREVIOUS
    previous_result=base.read_json(previous/'results.json')
    if previous_result['F_teacher']!=f_guided.teacher_reference() or not previous_result['F_unchanged']:
        raise ValueError('Matching original frozenF baseline required')
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    before={k:v.detach().clone() for k,v in reader.state_dict().items()}
    with torch.no_grad():
        smoke=reader(tensor(clean[np.flatnonzero(roles=='L_s')[:2]],'cpu'))
        if smoke.shape!=(2,6) or not torch.isfinite(smoke).all():raise ValueError('Actual originalF source2Ls smoke failed')
    print(json.dumps(dict(event='REAL_SOURCE_F_NOQUERY_SMOKE_PASS',Ls=2,G_inputs=0)),flush=True)
    started=time.perf_counter();mean,covariance,ridge=fit_prior(clean,fit)
    np.savez(output/'source_prior.npz',mean=mean,covariance=covariance,fit_positions=fit,probe_positions=probe,ridge=ridge)
    all_results={};maximum_replay=0.
    for epoch in (1,20):
        folder=p['cache']/'epochs'/f'{epoch:02d}'
        values={k:np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False)[probe] for k in ('received','kernel','coefficients')}
        with np.load(folder/'groups.npz',allow_pickle=False) as g:scene=g['scene'][probe]
        x=np.asarray(clean[probe],dtype=np.float64);restored=[]
        for index in range(512):
            A=real_operator(values['kernel'][index],values['coefficients'][index])
            # Independent torch native action versus the dense operator.
            if index<2:
                from experiments.pamf_dura_efficiency.model import supervised_action
                action=supervised_action(tensor(values['kernel'][index:index+1],'cpu'),tensor(values['coefficients'][index:index+1],'cpu'))
                tx=torch.complex(tensor(x[index:index+1,0],'cpu'),tensor(x[index:index+1,1],'cpu'))
                with torch.no_grad():ty=action(tx)
                dense=(A@x[index].reshape(512)).reshape(2,256)
                actual=numpy_values(torch.stack((ty.real,ty.imag),1))[0]
                delta=float(np.max(abs(dense-actual)));maximum_replay=max(maximum_replay,delta)
                if delta>2e-5:raise ValueError('Dense known action differs from source native action')
            if scene[index]=='clean':result=values['received'][index].copy()
            else:result=gaussian_reference(values['received'][index],A,noise_covariance(values['coefficients'][index]),mean,covariance).reshape(2,256)
            if not np.isfinite(result).all():raise FloatingPointError('Nonfinite known-action reference')
            restored.append(result.astype(np.float32))
            if (index+1)%64==0:print(json.dumps(dict(event='KNOWN_ACTION_REFERENCE_PROGRESS',epoch=epoch,count=index+1,seconds=time.perf_counter()-started)),flush=True)
        restored=np.stack(restored);ff=[];zz=[]
        with torch.no_grad():
            for offset in range(0,512,32):
                f=reader.features(tensor(restored[offset:offset+32],'cpu'));ff.append(numpy_values(f));zz.append(numpy_values(reader.classify_features(f)))
        features=np.concatenate(ff);logits=np.concatenate(zz);baselines={}
        for name in ('noVar','DURA_default'):
            with np.load(previous/f'{name}-{epoch}.npz',allow_pickle=False) as a:
                if not np.array_equal(a['source_positions'],probe) or not np.array_equal(a['physical_ids'],ids[probe]) or not np.array_equal(a['scene'],scene):
                    raise ValueError('Source baseline physical pairing differs')
                np.testing.assert_allclose(a['clean'],x,atol=0,rtol=0)
                baselines[name]=(a['native'],a['F_features'][1],a['F_logits'][1])
                teacher_features=a['F_features'][0];teacher_logits=a['F_logits'][0]
        arrays=dict(known_action_Gaussian=(restored,features,logits),**baselines)
        masks={'all':np.ones(512,dtype=bool)}
        for label in np.unique(scene):masks['scene:'+label]=scene==label
        result={name:{group:summary(iq,f,z,x,teacher_features,teacher_logits,mask) for group,mask in masks.items()} for name,(iq,f,z) in arrays.items()}
        all_results[str(epoch)]=result
        np.savez(output/f'view{epoch}.npz',source_positions=probe,physical_ids=ids[probe],scene=scene,
            received=values['received'],clean=x,oracle=restored,oracle_features=features,oracle_logits=logits,
            reference_features=teacher_features,reference_logits=teacher_logits,**{name:iq for name,(iq,_,_) in baselines.items()})
    if any(not torch.equal(v,before[k]) for k,v in reader.state_dict().items()) or any(v.grad is not None for v in reader.parameters()):
        raise ValueError('Frozen F changed')
    base.write_json(output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',summary=all_results,seconds=time.perf_counter()-started,
        source_fit=62488,source_probe=512,source_probe_L=128,source_probe_U=384,source_views=[1,20],prior_ridge=ridge,
        dense_native_replay_max_abs=maximum_replay,F_teacher=f_guided.teacher_reference(),F_unchanged=True,F_new_inputs=1026,
        G_new_inputs=0,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,
        scope='Known realized action plus source marginalGaussian prior and plug-in noise covariance;nonGaussian IQ and noise-dependent AGC violate exactconditionalGaussian assumptions;not deployable/not certified recovery or accuracyupperbound;teacherchoices not TXtruth'))
    print(json.dumps(dict(event='KNOWN_ACTION_REFERENCE_COMPLETE',seconds=time.perf_counter()-started)),flush=True)


if __name__=='__main__':main()
