"""Frozen source-train/source-V SIG diagnostics; no TX labels or target."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from experiments.pamf_dura_direct_train import runner as base,f_guided
from experiments.pamf_dura_direct_train.antithetic_noise import reuse_targets
from experiments.pamf_dura_diagnostics.runner import source_indices,tensor,numpy_values
from .signature_iq import signature_features
from .branch_features import components
from .head_features import head_feature_basis
from .signature_generalization import errors,summarize,decision,SCENES

RUN='20261031-diagnostic-DURA-source-SIG-generalization-s2026092701-m3-r01'
MODULE='experiments.pamf_dura_efficiency.signature_generalization_probe'
MODELS={
 'context':('20261031-phase1-DURA-SIGcontextPrior-E30-s2026092701-r01','4452e09285b650536ab5e9dcc2acbfd9268c00e8',344295),
 'linear':('20261031-phase1-DURA-SIGdeepLinear-E30-s2026092701-r01','df7fe90d16910f7373d79ee93291f265506a8434',406055),
 'mlp':('20261031-phase1-DURA-SIGdeepMLP-E30-s2026092701-r01','df7fe90d16910f7373d79ee93291f265506a8434',457575)}
RULE=('Fixed3Gs-final30/7410/context-linear-mlp/CPU2/native. Source-train512rolewise128L384U/cache01and20;'
 'fixedsourceV1800/sevenreceivedviews/0TXlabels. H1BOTHnewmodels:>=10percentpredictorMSEgain in>=4of6train scenes BOTHcachedviews,'
 'andnoVgain in>=4of6Vscenes. H2separate:BOTHnewmodels >=10percentVtotalMSEgain in>=4of6scenes butnoVhead-visiblegain in>=4of6.'
 'Bothhypothesesreported/noalternativeprimary/noautomatictraining;Vdiagnosisonly/notfitting/stateupdate/targetorTXaccuracy/recoverybound.')


def capture(model,x):
    records=[];handles=[]
    try:
        for prior in model.prior:
            handles.append(prior.signature_conditioner.register_forward_pre_hook(lambda module,args:records.append(args[0].detach().clone())))
        iq=model(x)
    finally:
        for handle in handles:handle.remove()
    if len(records)!=2 or any(v.shape!=(len(x),160) for v in records) or not torch.isfinite(iq).all():raise ValueError('Two finite actual conditioner predictions required')
    return iq,torch.stack(records)


def configure(key):
    if key=='context':
        from experiments.pamf_dura_direct_train import signature_context_prior as method
        method.configure()
    elif key in ('linear','mlp'):
        from experiments.pamf_dura_direct_train import signature_deep_prior as method
        method.configure(mode=key)
    else:raise ValueError('Fixed context,linear,mlp only')


def validation_targets(args,p,reader,ids,clean):
    folder=args.output/'source_V_SIG';new_inputs=0
    if args.key=='context':
        folder.mkdir();values=[];started=time.perf_counter()
        for start in range(0,1800,64):values.append(numpy_values(signature_features(reader,tensor(clean[start:start+64],'cpu'))))
        values=np.concatenate(values)
        if values.shape!=(1800,160) or values.dtype!=np.float32 or not np.isfinite(values).all():raise ValueError('Finite frozen source-V SIG required')
        np.save(folder/'targets.npy',values,allow_pickle=False);np.save(folder/'physical_ids.npy',ids,allow_pickle=False)
        base.write_json(folder/'complete.json',dict(status='SOURCE_V_SIG_DIAGNOSTIC_COMPLETE',count=1800,shape=[1800,160],numeric_bytes=1152000,
            F_teacher=f_guided.teacher_reference(),source_contract='original7/63/30',role='V_diagnostic_only',F_prefix_inputs=1800,F_updates=0,
            TX_label_reads=0,V_label_reads=0,target_reads=0,V_training=False,seconds=time.perf_counter()-started))
        new_inputs=1800
    meta=base.read_json(folder/'complete.json')
    expected=dict(status='SOURCE_V_SIG_DIAGNOSTIC_COMPLETE',count=1800,shape=[1800,160],numeric_bytes=1152000,F_teacher=f_guided.teacher_reference(),
        source_contract='original7/63/30',role='V_diagnostic_only',F_prefix_inputs=1800,F_updates=0,TX_label_reads=0,V_label_reads=0,target_reads=0,V_training=False)
    if any(meta.get(k)!=v for k,v in expected.items()) or not np.array_equal(np.load(folder/'physical_ids.npy',allow_pickle=False),ids):raise ValueError('Matching fixed source-V SIG reference required')
    return np.load(folder/'targets.npy',mmap_mode='r',allow_pickle=False),meta,new_inputs


@torch.no_grad()
def worker(args):
    if args.output!=args.root/'runs'/RUN or not args.output.is_dir() or (args.output/args.key).exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Independent immutable CPU-only source diagnostic row required')
    configure(args.key);p=base.paths(args.root);run,commit,parameters=MODELS[args.key]
    model=base.load_model(args.root/'runs'/run/'scratch_joint/final.pt','cpu',arm='scratch_joint',commit=commit,epoch=30)
    if sum(v.numel() for v in model.parameters())!=parameters:raise ValueError('Complete registered frozen G required')
    clean,roles=base.source_training(p);ids=f_guided.source_ids(p,roles);take=source_indices(roles);targets,reuse=reuse_targets(p,None,clean,roles)
    from experiments.pamf_fcrossrx_readout.source_diagnostic import source_reference
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    validation=p['cache']/'validation';vids,source,index=source_reference(validation,p['source_cache'])
    if set(vids.tolist())&set(ids.tolist()):raise ValueError('Source V cannot overlap L/U records')
    vclean=np.load(validation/'clean.npy',mmap_mode='r',allow_pickle=False)
    if vclean.shape!=(1800,2,256) or vclean.dtype!=np.float32:raise ValueError('Existing source-V clean reference required')
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    beforeG={k:v.clone() for k,v in model.state_dict().items()};beforeF={k:v.clone() for k,v in reader.state_dict().items()}
    basis=numpy_values(head_feature_basis(reader));weight=numpy_values(reader.core.id_backbone.cls_head.weight)
    output=args.output/args.key;output.mkdir();base.write_json(output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,G_run=run,G_commit=commit,CPU_threads=2,GPU_workers=0))
    variance=np.load(p['cache']/'epochs/01/variance.npy',mmap_mode='r',allow_pickle=False);smoke=np.flatnonzero((roles=='L_s')&(variance>np.float32(1e-8)))[:2]
    if len(smoke)!=2:raise ValueError('Two actual nonclean source L_s required')
    received=np.load(p['cache']/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    restored,pred=capture(model,tensor(received[smoke],'cpu'));torch.testing.assert_close(model(tensor(received[smoke],'cpu')),restored,rtol=0,atol=0)
    f,parts=components(reader,tensor(clean[smoke],'cpu'));torch.testing.assert_close(parts['signature'],tensor(targets[smoke],'cpu'),rtol=2e-5,atol=3e-5)
    scores=reader(restored)
    if reader.classify_features(f).shape!=(2,6) or scores.shape!=(2,6) or not torch.isfinite(scores).all():raise ValueError('Original F/defaulthead smoke failed')
    if any(q.signature_conditioner._forward_pre_hooks for q in model.prior):raise ValueError('Observer hooks remain after smoke')
    base.write_json(output/'smoke.json',dict(status='PASS',real_nonclean_Ls=2,G_inputs=4,F_full_inputs=4,strict_final30=True,
        conditioner_observer_preserves_IQ=True,source_clean_SIG_cache_match=True,TX_label_reads=0,V_label_reads=0,target_reads=0,optimizer_updates=0))
    print(json.dumps(dict(event='REAL_SOURCE_SIG_GENERALIZATION_SMOKE_PASS',model=args.key)),flush=True)
    vt,vmeta,new_prefix=validation_targets(args,p,reader,vids,vclean);started=time.perf_counter();summary={'train':{},'V':{}};records=[];observations=0
    trx=np.load(p['cache']/'train/rx.npy',allow_pickle=False);tday=np.load(p['cache']/'train/day.npy',allow_pickle=False)
    for pool,views in [('train',['1','20']),('V',['clean',*SCENES])]:
        for view in views:
            if pool=='train':
                folder=p['cache']/'epochs'/f'{int(view):02d}';values=np.asarray(np.load(folder/'received.npy',mmap_mode='r',allow_pickle=False)[take]);reference=np.asarray(targets[take]);paired=np.asarray(clean[take])
                with np.load(folder/'groups.npz',allow_pickle=False) as group:scene=group['scene'][take]
                metadata=dict(physical_ids=ids[take],source_positions=take,rx=trx[take],day=tday[take],roles=roles[take],scene=scene)
                if set(scene)!=set(['clean',*SCENES]):raise ValueError('All fixed train scenes required')
            else:
                values=np.load(validation/(view+'.npy'),mmap_mode='r',allow_pickle=False);reference=np.asarray(vt);paired=vclean
                metadata=dict(physical_ids=vids,source_positions=index,rx=source.rx[index],day=source.day[index],roles=np.asarray(['V']*1800))
            n=len(reference)
            if values.shape!=(n,2,256) or values.dtype!=np.float32:raise ValueError('Existing finite FP32 received source handle required')
            outputs=[];predictions=[];features=[]
            for start in range(0,n,32):
                iq,sig=capture(model,tensor(values[start:start+32],'cpu'));features.append(numpy_values(signature_features(reader,iq)))
                outputs.append(numpy_values(iq));predictions.append(numpy_values(sig))
            iq=np.concatenate(outputs);pred=np.concatenate(predictions,axis=1);features=np.concatenate(features)
            metrics=errors(pred,features,reference,iq,np.asarray(paired),basis);masks={'all':np.ones(n,bool)}
            for field in ['rx','day','roles']+(['scene'] if pool=='train' else []):
                for value in np.unique(metadata[field]):masks[field+':'+str(value)]=metadata[field]==value
            groups={name:dict(count=int(mask.sum()),**summarize(metrics,mask)) for name,mask in masks.items()};summary[pool][view]=groups
            for name,row in groups.items():records.append(dict(pool=pool,view=view,group=name,**row))
            np.savez(output/(pool+'-'+view+'.npz'),**metadata,IQ=iq,SIG_predictions=pred,restored_SIG=features,SIG_reference=reference,head_basis=basis,head_weight=weight)
            observations+=n
            print(json.dumps(dict(event='SOURCE_SIG_GENERALIZATION_VIEW_COMPLETE',model=args.key,pool=pool,view=view,count=n,overall=groups['all'],seconds=time.perf_counter()-started)),flush=True)
    if observations!=13624 or len(records)!=106:raise ValueError('Complete fixed train/V diagnostic matrix required')
    if any(not torch.equal(v,beforeG[k]) for k,v in model.state_dict().items()) or any(not torch.equal(v,beforeF[k]) for k,v in reader.state_dict().items()):raise ValueError('Frozen G/F state changed')
    if any(p.grad is not None for module in [model,reader] for p in module.parameters()) or any(q.signature_conditioner._forward_pre_hooks for q in model.prior):raise ValueError('Frozen gradients or observer hooks remain')
    base.write_json(output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',key=args.key,G_run=run,G_commit=commit,G_parameters=parameters,G_unchanged=True,F_unchanged=True,
        F_teacher=f_guided.teacher_reference(),source_contract='original7/63/30',source_training_unique=512,source_V_unique=1800,source_observations=13624,
        source_training_views=[1,20],source_V_views=['clean',*SCENES],summary=summary,records=records,head_contrast_rank=int(basis.shape[1]),
        G_inputs=13624,F_prefix_restored_inputs=13624,new_F_clean_V_prefix_inputs=new_prefix,G_smoke_inputs=4,F_smoke_full_inputs=4,
        source_training_SIG_cache_reuse=reuse,V_SIG_cache=vmeta,V_diagnostic_IQ_inputs=12600,V_labels_read=0,V_training=False,TX_labels_read=0,target_reads=0,optimizer_updates=0,head_fit_updates=0,
        seconds=time.perf_counter()-started,scope='SameknownsourceRX/heldphysicalV/seenGtrainingpool/differentfixedcorruptiondraws/notnewRXtest/notTXaccuracy/targetbenefit/modelselection'))
    base.write_json(output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',observations=13624,group_cells=106))


def run(args):
    if args.output!=args.root/'runs'/RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('New immutable CPU-only source diagnostic required')
    args.output.mkdir();base.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,rule=RULE,CPU_threads=2,GPU_workers=0))
    rows={};started=time.perf_counter()
    try:
        for key in MODELS:
            command=[sys.executable,'-u','-m',MODULE,'worker','--root',str(args.root),'--output',str(args.output),'--commit',args.commit,'--key',key]
            process=subprocess.Popen(command,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,start_new_session=True)
            base.write_json(args.output/('submitted-'+key+'.json'),dict(pid=process.pid,command=command))
            code=process.wait()
            if code:raise RuntimeError('Source diagnostic worker failed: '+key+' exit='+str(code))
            row=base.read_json(args.output/key/'results.json');complete=base.read_json(args.output/key/'complete.json')
            if row['status']!=complete['status'] or complete['group_cells']!=106:raise ValueError('Complete source diagnostic row required')
            rows[key]=row
        models={key:row['summary'] for key,row in rows.items()};verdict=decision(models)
        base.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',run=RUN,rule=RULE,models=rows,hypotheses=verdict,
            source_unique_physical_records=2312,source_observations_per_model=13624,G_inputs=40872,F_prefix_restored_inputs=40872,new_F_clean_V_prefix_inputs=1800,
            G_smoke_inputs=12,F_smoke_full_inputs=12,group_cells=318,TX_labels_read=0,V_labels_read=0,target_reads=0,V_training=False,optimizer_updates=0,seconds=time.perf_counter()-started,
            scope='Frozenmechanismdiagnosis;VrecordgeneralizationknownsourceRX/notnewRX/notTXtruth/targetbenefit/recoverybound/automatictraining'))
        base.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',models=list(rows),hypotheses=verdict,formal_predictions=0,goal_complete=False))
    except BaseException as error:
        base.write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,completed_models=list(rows)))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['run','worker'])
    for key in ['root','output']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);parser.add_argument('--key',choices=list(MODELS));args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned diagnostic release required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='run':run(args)
    else:
        if args.key is None:parser.error('Fixed model key required')
        worker(args)


if __name__=='__main__':main()
