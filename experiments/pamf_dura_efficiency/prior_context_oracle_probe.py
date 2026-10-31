"""SOURCE-only fixed-weight prior-context intervention; no TX/V/target."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from experiments.pamf_dura_direct_train import signature_context_prior as method
from experiments.pamf_dura_direct_train.antithetic_noise import reuse_targets
from experiments.pamf_dura_diagnostics.runner import source_indices,tensor,numpy_values
from .prior_context_oracle import decoder_context,decision,CASES
from .periodic_utility_probe import stats

RUN='20261031-diagnostic-DURA-source-priorContext-oracle-s2026092701-r01'
G_RUN=method.RUN
G_COMMIT='4452e09285b650536ab5e9dcc2acbfd9268c00e8'
REFERENCE='20261030-diagnostic-DURA-source-phase-oracle-s2026092701-m2-r01'
RULE=('FixedG-SIGcontextPrior-final30/source512128L384U/cache01and20/CPU2. '
 'Caseslearned/disabled/clean_SIG/shifted_SIG;allweightsfixed. Primaryclean_SIG BOTHviews: '
 'cleanFchoicegain>=4vslearnedand>=2vsglobal512-cyclicshift1cleanSIG;5RXnet>=0/clean-scenenet>=0/eachnoncleanscenenet>=-1/fullFcosnotworse. '
 'Disabled descriptiveonly/noalternateprimary. SourcecleanSIGatconditioneronly isprivileged/notdeployable/notTXtruth/targetbenefit/recoverybound;noautomaticGtraining.')


def evaluate(model,received,target,shifted):
    iq=[];predictions=[];contexts=[];modulations=[]
    for mode in CASES:
        projected=[];used=[];raw=[];handles=[]
        reference=shifted if mode=='shifted_SIG' else target
        with decoder_context(model,mode,reference):
            try:
                for prior in model.prior:
                    handles.append(prior.signature_projection.register_forward_hook(lambda module,args,value:projected.append(value.detach().clone())))
                    handles.append(prior.signature_conditioner.register_forward_pre_hook(lambda module,args:used.append(args[0].detach().clone())))
                    handles.append(prior.signature_conditioner.register_forward_hook(lambda module,args,value:raw.append(value.detach().clone())))
                output=model(received)
            finally:
                for handle in handles:handle.remove()
        if len(projected)!=2 or len(used)!=2 or len(raw)!=2 or not torch.isfinite(output).all():raise ValueError('Two finite real prior paths required')
        iq.append(output);predictions.append(torch.stack(projected));contexts.append(torch.stack(used));modulations.append(torch.stack(raw))
    return torch.stack(iq),torch.stack(predictions),torch.stack(contexts),torch.stack(modulations)


@torch.no_grad()
def run(args):
    if args.output!=args.root/'runs'/RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable CPU-only source diagnostic required')
    method.configure();base=method.base;p=base.paths(args.root)
    clean,roles=base.source_training(p);ids=method.f_guided.source_ids(p,roles);take=source_indices(roles)
    targets,reuse=reuse_targets(p,None,clean,roles);target=np.asarray(targets[take]);shifted=np.roll(target,1,axis=0).copy()
    previous=args.root/'runs'/REFERENCE;meta=base.read_json(previous/'results.json')
    if meta['F_teacher']!=method.f_guided.teacher_reference() or not meta['F_unchanged']:raise ValueError('Matching source frozen F clean cache required')
    rx=np.load(p['cache']/'train/rx.npy',allow_pickle=False);day=np.load(p['cache']/'train/day.npy',allow_pickle=False)
    archives={}
    for view in ['1','20']:
        folder=p['cache']/'epochs'/f'{int(view):02d}'
        with np.load(previous/f'DURA_default-{view}.npz',allow_pickle=False) as f:
            a={k:f[k] for k in ['source_positions','physical_ids','roles','clean','rx','day','scene']}
            a['reference_features']=f['F_features'][0];a['reference_logits']=f['F_logits'][0]
        for key,value in dict(source_positions=take,physical_ids=ids[take],roles=roles[take],clean=clean[take],rx=rx[take],day=day[take]).items():
            if not np.array_equal(a[key],value):raise ValueError('Original source order/roles required: '+key)
        with np.load(folder/'groups.npz',allow_pickle=False) as groups:
            if not np.array_equal(a['scene'],groups['scene'][take]):raise ValueError('Original received scene order required')
        a['received']=np.asarray(np.load(folder/'received.npy',mmap_mode='r',allow_pickle=False)[take])
        if a['received'].shape!=(512,2,256) or a['reference_features'].shape!=(512,160) or a['reference_logits'].shape!=(512,6):raise ValueError('Complete original source512 shapes required')
        if not all(a[k].dtype==np.float32 and np.isfinite(a[k]).all() for k in ['clean','received','reference_features','reference_logits']):raise ValueError('Finite source FP32 required')
        if (int((a['roles']=='L_s').sum()),int((a['roles']=='U_s').sum()))!=(128,384) or set(a['rx'])!={1,3,4,6,8} or len(set(a['scene']))!=7:raise ValueError('Fixed source roles/RX/scenes required')
        archives[view]=a
    if target.shape!=(512,160) or not np.isfinite(target).all():raise ValueError('Complete paired source SIG required')
    # The released G has matching source-only F-guided provenance; it is not
    # an initializer for training. Use its exact saved metadata and configuration.
    model=method.parent.load_model(args.root/'runs'/G_RUN/'scratch_joint/final.pt','cpu',arm='scratch_joint',commit=G_COMMIT,epoch=30)
    if sum(v.numel() for v in model.parameters())!=344295:raise ValueError('Complete fixed context G required')
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    beforeG={k:v.clone() for k,v in model.state_dict().items()};beforeF={k:v.clone() for k,v in reader.state_dict().items()}
    weights=np.stack([numpy_values(q.signature_conditioner.weight) for q in model.prior]);bias=np.stack([numpy_values(q.signature_conditioner.bias) for q in model.prior])
    args.output.mkdir();base.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,G_run=G_RUN,G_commit=G_COMMIT,rule=RULE,CPU_threads=2,GPU_workers=0))
    a=archives['1'];smoke=np.flatnonzero((a['roles']=='L_s')&(a['scene']!='clean'))[:2]
    if len(smoke)!=2:raise ValueError('Two actual nonclean L_s required')
    sx=tensor(a['received'][smoke],'cpu');st=tensor(target[smoke],'cpu');sd=tensor(shifted[smoke],'cpu');original=model(sx)
    values=evaluate(model,sx,st,sd)
    torch.testing.assert_close(values[0][0],original,rtol=0,atol=0);torch.testing.assert_close(model(sx),original,rtol=0,atol=0)
    if not torch.equal(values[3][1],torch.zeros_like(values[3][1])):raise ValueError('Disabled context not zero')
    for ci,reference in [(2,st),(3,sd)]:torch.testing.assert_close(values[2][ci],reference[None].expand(2,-1,-1),rtol=0,atol=0)
    features,parts=method.parent.components(reader,tensor(a['clean'][smoke],'cpu'))
    torch.testing.assert_close(parts['signature'],st,rtol=2e-5,atol=3e-5)
    torch.testing.assert_close(reader.classify_features(features),tensor(a['reference_logits'][smoke],'cpu'),rtol=3e-5,atol=3e-5)
    scores=reader(original)
    if scores.shape!=(2,6) or not torch.isfinite(scores).all():raise ValueError('Original F/defaulthead smoke failed')
    base.write_json(args.output/'smoke.json',dict(status='PASS',real_nonclean_Ls=2,G_inputs=12,F_full_inputs=4,source_SIG_and_F_clean_reference_match=True,
        regular_IQ_restored_after_interventions=True,actual_conditioner_inputs_and_disabled_outputs_checked=True,strict_final30_load=True,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0))
    print(json.dumps(dict(event='REAL_SOURCE_PRIOR_CONTEXT_ORACLE_SMOKE_PASS')),flush=True)
    started=time.perf_counter();summary={};records=[];descriptor={}
    for view,a in archives.items():
        outputs=[];preds=[];contexts=[];raws=[];features=[];logits=[]
        for start in range(0,512,32):
            stop=start+32;values=evaluate(model,tensor(a['received'][start:stop],'cpu'),tensor(target[start:stop],'cpu'),tensor(shifted[start:stop],'cpu'))
            outputs.append(numpy_values(values[0]));preds.append(numpy_values(values[1]));contexts.append(numpy_values(values[2]));raws.append(numpy_values(values[3]))
            fs=[];zs=[]
            for iq in values[0]:
                f=reader.features(iq);z=reader.classify_features(f)
                if not torch.isfinite(f).all() or not torch.isfinite(z).all():raise FloatingPointError('Finite frozen F outputs required')
                fs.append(numpy_values(f));zs.append(numpy_values(z))
            features.append(np.stack(fs));logits.append(np.stack(zs))
        iq=np.concatenate(outputs,axis=1);pred=np.concatenate(preds,axis=2);context=np.concatenate(contexts,axis=2);raw=np.concatenate(raws,axis=2)
        fs=np.concatenate(features,axis=1);zs=np.concatenate(logits,axis=1)
        if not np.array_equal(context[2],np.broadcast_to(target,(2,512,160))) or not np.array_equal(context[3],np.broadcast_to(shifted,(2,512,160))) or np.any(raw[1]!=0):raise ValueError('Actual fixed intervention input/output differs')
        masks={'all':np.ones(512,bool)}
        for key in ['rx','day','roles','scene']:
            for value in np.unique(a[key]):masks[key+':'+str(value)]=a[key]==value
        groups={}
        for name,mask in masks.items():
            groups[name]=dict(count=int(mask.sum()),cases={case:stats(iq[i],fs[i],zs[i],a['clean'],a['reference_features'],a['reference_logits'],zs[0],mask) for i,case in enumerate(CASES)})
            records.append(dict(view=view,group=name,**groups[name]))
        summary[view]=groups
        descriptor[view]={case:dict(prior_relative_SIG_MSE=np.mean(np.sum((pred[i].astype(np.float64)-target[None])**2,axis=2)/np.maximum(np.sum(target.astype(np.float64)**2,axis=1),1e-4),axis=1).tolist(),
            mean_absolute_bounded_modulation=float(np.mean(.1*np.abs(np.tanh(raw[i].astype(np.float64))))),maximum_absolute_bounded_modulation=float(np.max(.1*np.abs(np.tanh(raw[i].astype(np.float64)))))) for i,case in enumerate(CASES)}
        np.savez(args.output/f'view{view}.npz',**a,SIG_targets=target,shifted_SIG_targets=shifted,IQ=iq,F_features=fs,F_logits=zs,
            SIG_predictions=pred,conditioner_SIG=context,modulation_raw=raw,conditioner_weights=weights,conditioner_bias=bias)
        print(json.dumps(dict(event='SOURCE_PRIOR_CONTEXT_ORACLE_VIEW_COMPLETE',view=view,overall=groups['all'],seconds=time.perf_counter()-started)),flush=True)
    if any(not torch.equal(v,beforeG[k]) for k,v in model.state_dict().items()) or any(not torch.equal(v,beforeF[k]) for k,v in reader.state_dict().items()):raise ValueError('Frozen G/F state changed')
    if any(v.grad is not None for module in [model,reader] for v in module.parameters()):raise ValueError('Frozen model gradients populated')
    if any(q.signature_conditioner._forward_hooks or q.signature_conditioner._forward_pre_hooks or q.signature_projection._forward_hooks for q in model.prior):raise ValueError('Temporary hooks remain')
    verdict=decision(summary)
    base.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',run=RUN,rule=RULE,G_run=G_RUN,G_commit=G_COMMIT,G_epochs=30,G_updates=7410,G_parameters=344295,
        G_unchanged=True,F_unchanged=True,F_teacher=method.f_guided.teacher_reference(),source_cache_run=REFERENCE,source_SIG_cache_reuse=reuse,
        cases=list(CASES),source_records=512,source_views=[1,20],source_observations=1024,source_unique_physical_records=512,source_L_unique=128,source_U_unique=384,
        summary=summary,records=records,descriptor=descriptor,decision=verdict,G_inputs=4096,G_smoke_inputs=12,F_full_inputs=4096,F_smoke_full_inputs=4,
        G_checkpoint_reads=1,F_checkpoint_reads=1,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,seconds=time.perf_counter()-started,
        scope='Same-trained-weights interventions may shift the learned context distribution; sourcecleanFchoice notTXtruth; pairedcleanoracle notdeployable/recoverybound/automatictraining. Oldnegatives/defaultDURA unchanged.'))
    base.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',decision=verdict,formal_predictions=0,goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['root','output']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned diagnostic release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
