"""Exact current PAMF branches; default joint feature includes all current paths."""
from pathlib import Path
import time
import numpy as np
from .io import activate,read,write,require

NAMES=('z_id','t_emb','f_emb','pa_local','fft')

def tensor(array,device):
    import torch
    return torch.tensor(np.asarray(array,dtype=np.float32).tolist(),device=device,dtype=torch.float32)

def fft96(iq):
    """Unchanged D92 historical spectral_logmag_sketch arithmetic."""
    raw=np.asarray(iq,dtype=np.float32)
    require(raw.ndim==3 and raw.shape[1:]==(2,256) and np.isfinite(raw).all(),'Invalid FFT IQ')
    result=np.empty((len(raw),96),dtype=np.float32)
    source=np.linspace(0.,1.,256,dtype=np.float64);target=np.linspace(0.,1.,96,dtype=np.float64)
    for index,row in enumerate(raw):
        z=row[0].astype(np.float64)+1j*row[1].astype(np.float64);z-=np.mean(z)
        rms=float(np.sqrt(np.mean(np.abs(z)**2)))
        if rms>1e-8:z/=rms
        spectrum=np.log1p(np.abs(np.fft.fftshift(np.fft.fft(z*np.hanning(256)))))
        sketch=np.interp(target,source,spectrum).astype(np.float32)
        sketch-=np.mean(sketch,dtype=np.float64).astype(np.float32)
        sketch/=max(float(np.linalg.norm(sketch)),1e-8)
        result[index]=sketch
    return result

def branches(reader,x):
    import torch
    capture=[]
    h=reader.core.id_backbone.register_forward_hook(lambda module,args,out:capture.append(out))
    try:z=reader.features(x)
    finally:h.remove()
    require(len(capture)==1 and isinstance(capture[0],dict),'Expected one actual current PAMF auxiliary forward')
    values=dict(z_id=z,**{k:capture[0][k] for k in ('t_emb','f_emb','pa_local')})
    require(all(v.shape==(len(x),160) and bool(torch.isfinite(v).all()) for v in values.values()),'Wrong branch dimensions')
    logits=reader.classify_features(z)
    return values,logits

def load_reader(rt,seed,device):
    from experiments.pamf_repl_view_shuffle.source import load_reader as load
    folder=rt['PAMF_sources'][str(seed)]
    contract=read(rt['source_contract'])
    reader,c=load(folder,contract,device)
    require(c['model_seed']==seed and c['view_comparison']['view_count']==3,'Wrong current PAMF source seed/view')
    require(sum(v.numel() for v in reader.parameters())==232993,'Current PAMF parameters differ')
    return reader

def load_G(rt,device):
    from experiments.cric_dura_comparison.frozen_inference import load_G as load
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    model=load(rt['DURA'],read(rt['source_contract']),device)
    return InferenceEngine(model,reuse=True,graph=True,native=False)

def source_smoke(rt,out,device='cuda:0'):
    import torch
    activate(rt)
    from experiments.cric_dura_comparison.f_data import SourceDataset
    c=read(rt['source_contract'])
    # Two legal source L samples; no target or query input handle is consumed.
    x=SourceDataset(rt['source_cache'],'L_s',c).x[:2].to(device)
    reader=load_reader(rt,rt['model_seeds'][0],device)
    before={k:v.detach().clone() for k,v in reader.state_dict().items()}
    with torch.inference_mode():
        v,logits=branches(reader,x)
        reference=reader.features(x)
        require(torch.equal(v['z_id'],reference),'Composite identity feature mismatch')
        require(logits.shape==(2,6),'Wrong default head')
        engine=load_G(rt,device);restored=engine(x)[0]
        a=engine(x[:1])[0];b=engine(x[1:])[0]
        require(torch.allclose(restored,torch.cat((a,b)),rtol=2e-4,atol=2e-5),'DURA depends on query batch')
        vr,_=branches(reader,restored)
        if rt.get('kernel_method') in ('signature_kernel','task_signature','fisher_signature','nested_fisher','source_bridge','paired_cca','balanced_h','kernel_margin','class_whiten','identity_holdout'):
            from .signature_features import capture
            sv,sl=capture(reader,x)
            require(torch.equal(sv['z_id'],v['z_id']) and torch.equal(sl,logits),'Descriptor hook changed frozen F')
            v=sv
        if rt.get('kernel_method') in ('phase_orbit','phase_kernel','nonlinear_pair','registered_scale','local_anchor','teacher_kernel','prior_fusion','confidence_prior','native_skip','time_orbit','episodic_metric','ridge_prior','residual_skip','center_prior','phase8_prior','orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble','rf_class_ensemble'):
            from .phase_orbit import extract,rotate,moments,stack_views
            options=dict(aggregate=stack_views if rt['kernel_method']=='phase_kernel' else moments)
            if rt['kernel_method'] in ('teacher_kernel','prior_fusion','confidence_prior','native_skip','time_orbit','episodic_metric','ridge_prior','residual_skip','center_prior','phase8_prior','orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble','rf_class_ensemble'):
                from .teacher_features import extract,probabilities
                if rt['kernel_method']=='time_orbit':
                    from .teacher_features import extract_time as extract
                elif rt['kernel_method']=='phase8_prior':
                    from .teacher_features import extract_phase8 as extract
                elif rt['kernel_method'] in ('orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble','rf_class_ensemble'):
                    from .teacher_features import extract_orbit_ensemble as extract
                options={}
            iq=np.asarray(x.cpu().tolist(),np.float32)
            g_before={k:v.detach().clone() for k,v in engine.pipeline.model.state_dict().items()}
            base,_=extract(reader,iq,device,**options);shift,_=extract(reader,rotate(iq,1),device,**options)
            single=[extract(reader,iq[i:i+1],device,**options)[0] for i in range(2)]
            require(all(np.allclose(np.roll(value,-1,axis=1) if rt['kernel_method']=='phase_kernel' or key=='teacher_phase_logits' else value,shift[key],rtol=2e-4,atol=2e-5) for key,value in base.items()),'C4 orbit invariance failed')
            if rt['kernel_method'] in ('teacher_kernel','prior_fusion','confidence_prior','native_skip','time_orbit','episodic_metric','ridge_prior','residual_skip','center_prior','phase8_prior','orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble','rf_class_ensemble'):
                probability=probabilities(base['teacher_phase_logits'])
                require(probability.shape==(2,6) and np.allclose(probability.sum(1),1),'Frozen default-head probabilities differ')
            require(all(np.allclose(value,np.concatenate([s[key] for s in single]),rtol=2e-4,atol=2e-5) for key,value in base.items()),'Orbit feature depends on other packets')
            if rt['kernel_method'] in ('phase8_prior','orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble','rf_class_ensemble'):
                from .phase_orbit import rotate_eighth
                from .teacher_features import extract as extract_C4
                turned,_=extract(reader,rotate_eighth(iq,1),device)
                require(all(np.allclose(value,turned[key],rtol=2e-4,atol=2e-5) for key,value in base.items() if key!='teacher_phase_logits' and not key.startswith('C4_')),'C8 RF invariance failed')
                old_C4,_=extract_C4(reader,iq,device)
                require(np.allclose(base['teacher_phase_logits'],old_C4['teacher_phase_logits'],rtol=2e-4,atol=2e-5),'C8 changed original C4 head prior')
                if rt['kernel_method'] in ('orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble','rf_class_ensemble'):
                    require(all(np.allclose(base['C4_'+key],value,rtol=2e-4,atol=2e-5) for key,value in old_C4.items() if key!='teacher_phase_logits'),'Ensemble changed original C4 RF component')
            if rt['kernel_method']=='time_orbit':
                shifted,_=extract(reader,np.roll(iq,64,axis=2),device)
                require(all(np.allclose(value,shifted[key],rtol=2e-4,atol=2e-5) for key,value in base.items() if key!='teacher_phase_logits'),'Temporal orbit RF invariance failed')
            via_G,_=extract(reader,iq,device,engine,**options)
            direct,_=extract(reader,np.asarray(restored.cpu().tolist(),np.float32),device,**options)
            require(all(np.allclose(value,direct[key],rtol=2e-4,atol=2e-5) for key,value in via_G.items()),'G must precede the orbit')
            require(all(torch.equal(g_before[k],value) for k,value in engine.pipeline.model.state_dict().items()),'Frozen G changed')
            v=base
        require(all(torch.equal(before[k],value) for k,value in reader.state_dict().items()),'Frozen F state changed')
    write(out,dict(status='PASS',real_checkpoint=True,source_rows=2,target_rows=0,query_rows=0,
        optimizer_updates=0,PAMF_parameters=232993,DURA_parameters=266951,branch_dimensions={k:list(v.shape) for k,v in v.items()},
        actual_composite_joint=True,DURA_individual_packet_check=True))

def export(rt,seed,input_path,output,device='cuda:0'):
    import torch
    activate(rt);root=Path(output);root.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    reader=load_reader(rt,seed,device)
    capsule=Path(rt['data_root'])/'capsule';m=read(capsule/'manifest.json')
    require(m['protocol_schema']=='p2_min_v1' and m['phase2_data_status']=='VALIDATED_ONCE','Unvalidated received input')
    require(input_path in ('P0','P1'),'Unknown input path')
    iq=np.load(capsule/'received.npy' if input_path=='P0' else Path(rt['data_root'])/'DURA/received.npy',mmap_mode='r')
    ids=np.load(capsule/'ids.npy',allow_pickle=False)
    require(iq.shape==(len(ids),2,256),'Wrong received shape')
    before={k:v.detach().clone() for k,v in reader.state_dict().items()}
    arrays={k:np.lib.format.open_memmap(root/(k+'.npy'),mode='w+',dtype=np.float32,shape=(len(ids),96 if k=='fft' else 160)) for k in NAMES}
    pred=np.lib.format.open_memmap(root/'default_prediction.npy',mode='w+',dtype=np.int64,shape=(len(ids),))
    with torch.inference_mode():
        for start in range(0,len(ids),128):
            raw=np.array(iq[start:start+128],copy=True);values,logits=branches(reader,tensor(raw,device))
            for k,v in values.items():arrays[k][start:start+len(raw)]=np.asarray(v.cpu().tolist(),dtype=np.float32)
            arrays['fft'][start:start+len(raw)]=fft96(raw);pred[start:start+len(raw)]=logits.argmax(1).cpu().tolist()
            if start%8192==0:print('FEATURES',seed,input_path,start,len(ids),flush=True)
    for v in arrays.values():v.flush()
    pred.flush()
    require(all(torch.equal(before[k],v) for k,v in reader.state_dict().items()),'Frozen F state changed')
    np.save(root/'ids.npy',ids,allow_pickle=False)
    write(root/'complete.json',dict(status='FEATURES_COMPLETE',capsule_id=m['capsule_id'],model_seed=seed,input_path=input_path,
          source=rt['PAMF_sources'][str(seed)],count=len(ids),truth_read=False,query_fit=False,source_runtime_fit=False,
          feature_dims={k:v.shape[1] for k,v in arrays.items()},seconds=time.monotonic()-started))

def restore(rt,output,device='cuda:0'):
    import torch
    activate(rt);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    capsule=Path(rt['data_root'])/'capsule';m=read(capsule/'manifest.json')
    require(m['phase2_data_status']=='VALIDATED_ONCE','Unvalidated received input')
    iq=np.load(capsule/'received.npy',mmap_mode='r')
    result=np.lib.format.open_memmap(root/'received.npy',mode='w+',dtype=np.float32,shape=iq.shape)
    engine=load_G(rt,device);started=time.monotonic()
    for start in range(0,len(iq),128):
        with torch.inference_mode():y=engine(tensor(iq[start:start+128],device))[0]
        result[start:start+len(y)]=np.asarray(y.cpu().tolist(),dtype=np.float32)
        if start%8192==0:print('DURA',start,len(iq),flush=True)
    result.flush()
    write(root/'complete.json',dict(status='RESTORED_COMPLETE',capsule_id=m['capsule_id'],count=len(iq),new_physical_observations=0,
         weight_updates=0,query_statistics_updates=0,truth_read=False,reader_used=False,seconds=time.monotonic()-started))
