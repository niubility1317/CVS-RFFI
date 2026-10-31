"""SOURCE-only receiver-mean signature directions; no model/identity fitting."""
import numpy as np

RUN='20261031-diagnostic-DURA-source-SIG-RX-subspace-s2026092701-r01'
SOURCE='20261031-diagnostic-DURA-source-branchFeatures-s2026092701-r01'
CACHE='20261031-phase1-DURA-signatureGuide-E30-s2026092701-r01'
VIEWS=('1','20')
CASES=('native','full_SIG','RX_orthogonal_SIG','RX_mean_SIG')
RULE=('Source-only new receiver-mean hypothesis, not a change to prior FAIL. '
 'Fit four-dimensional centered five-RX clean-SIG mean span on 62488 source L/U records; '
 'exclude all fixed512 diagnostic records before statistics. Equal-RX mean center, FP64 SVD, fixed rank4/no rank search. '
 'DefaultDURA cached source512128L384U/caches01and20. Compare full cleanSIG replacement with '
 'clean-minus-native SIG correction projected onto the orthogonal complement; RX-mean-only descriptive. '
 'BOTH views orthogonal cleanF agreement at least2 more than fullSIG, every5RX and cleanscene no loss versus fullSIG, '
 'each nonclean scene at most1 loss versus fullSIG; overall full-feature relativeMSE at most1.01 times native. '
 'Only PASS supports separately registered source IQ feasibility, NEVER automatic G training or target inference. '
 'RX means may contain class-composition/device differences; their span is not proven receiver nuisance. '
 'No TX labels/V/target/G checkpoint/model optimization; original F/defaulthead frozen. '
 'Latent clean-reference intervention is not realizable IQ/trueTX accuracy/recovery ceiling.')


def receiver_basis(features,receivers,excluded):
    """Exclude evaluation records, then use the fixed centered five-RX mean span."""
    features=np.asarray(features);receivers=np.asarray(receivers);excluded=np.asarray(excluded)
    if (features.ndim!=2 or features.shape[1]<4 or receivers.shape!=(len(features),)
            or excluded.ndim!=1 or not np.issubdtype(excluded.dtype,np.integer)
            or len(np.unique(excluded))!=len(excluded) or np.any(excluded<0) or np.any(excluded>=len(features))):
        raise ValueError('Source features, public RX and unique held-out positions required')
    keep=np.ones(len(features),bool);keep[excluded]=False;groups=np.unique(receivers[keep])
    if len(groups)!=5:raise ValueError('Exactly the five source RX required')
    means=np.stack([np.mean(features[keep&(receivers==rx)],axis=0,dtype=np.float64) for rx in groups])
    center=means.mean(0);centered=means-center
    if not np.isfinite(centered).all():raise FloatingPointError('Nonfinite source RX statistics')
    _,singular,right=np.linalg.svd(centered,full_matrices=False)
    if singular[3]<=1e-10*singular[0]:raise ValueError('Four independent source RX mean directions required')
    basis=right[:4].T
    total=0.;projected=0.
    for start in range(0,len(features),4096):
        use=keep[start:start+4096];x=np.asarray(features[start:start+4096][use],dtype=np.float64)-center
        total+=float(np.square(x).sum());projected+=float(np.square(x@basis).sum())
    return dict(basis=basis,center=center,means=means,receivers=groups,singular_values=singular,
        counts=np.asarray([np.sum(keep&(receivers==rx)) for rx in groups]),fit_records=int(keep.sum()),
        held_out_records=len(excluded),fit_centered_energy_fraction=projected/max(total,1e-20))


def feature_cases(actual,native_signature,clean_signature,basis):
    actual=np.asarray(actual);native_signature=np.asarray(native_signature);clean_signature=np.asarray(clean_signature);basis=np.asarray(basis)
    if (actual.ndim!=2 or native_signature.shape!=actual.shape or clean_signature.shape!=actual.shape
            or basis.shape!=(actual.shape[1],4)):
        raise ValueError('Matched source signatures and fixed rank4 basis required')
    np.testing.assert_allclose(basis.T@basis,np.eye(4),atol=1e-10,rtol=1e-10)
    change=np.asarray(clean_signature,dtype=np.float64)-native_signature;rx_change=(change@basis)@basis.T
    return np.stack((actual,actual+change,actual+change-rx_change,actual+rx_change)).astype(np.float32)


def summaries(values,mask):
    result={}
    for i,name in enumerate(CASES):
        agreement=int(values['agreement'][i,mask].sum());full=int(values['agreement'][1,mask].sum())
        result[name]=dict(count=int(mask.sum()),cleanF_agreement=agreement,gain_vs_full_SIG=agreement-full,
            gain_vs_native=agreement-int(values['agreement'][0,mask].sum()),
            reference_margin_mean=float(values['reference_margin'][i,mask].mean(dtype=np.float64)),
            feature_relative_MSE_mean=float(values['feature_relative_MSE'][i,mask].mean(dtype=np.float64)))
    return result


def decision(groups):
    o=groups[0]['cases'];q=o['RX_orthogonal_SIG']
    checks=dict(extra_agreement=q['gain_vs_full_SIG']>=2,
        RX_protection=all(g['cases']['RX_orthogonal_SIG']['gain_vs_full_SIG']>=0 for g in groups if g['dimension']=='RX'),
        clean_protection=all(g['cases']['RX_orthogonal_SIG']['gain_vs_full_SIG']>=0 for g in groups if g['dimension']=='scene' and g['value']=='clean'),
        scene_protection=all(g['cases']['RX_orthogonal_SIG']['gain_vs_full_SIG']>=-1 for g in groups if g['dimension']=='scene' and g['value']!='clean'),
        feature_error=q['feature_relative_MSE_mean']<=1.01*o['native']['feature_relative_MSE_mean'])
    return dict(eligible=all(checks.values()),checks=checks)


def run(args):
    import json,os,sys,time
    import torch
    from experiments.pamf_dura_direct_train import runner as base,f_guided
    from experiments.pamf_dura_diagnostics.runner import tensor,numpy_values,source_indices
    from .branch_features import components,BRANCHES
    if args.output!=args.root/'runs'/RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable CPU-only source output required')
    p=base.paths(args.root);clean,roles=base.source_training(p);ids=f_guided.source_ids(p,roles);indices=source_indices(roles)
    rx=np.load(p['cache']/'train/rx.npy',allow_pickle=False);day=np.load(p['cache']/'train/day.npy',allow_pickle=False)
    folder=args.root/'runs'/SOURCE;meta=base.read_json(folder/'results.json')
    if (meta['F_teacher']!=f_guided.teacher_reference() or not meta['F_unchanged'] or meta['branches']!=list(BRANCHES)
            or meta['source_records']!=512 or meta['decision']!=dict(eligible=False,automatic_training=False,primary='DURA_default',candidate='PA')
            or any(meta[k]!=0 for k in ['new_G_inputs','G_checkpoint_reads','optimizer_updates','TX_labels_read','V_reads','target_reads'])):
        raise ValueError('Original source-only branch observations and unchanged PA FAIL required')
    identity=dict(source_positions=indices,physical_ids=ids[indices],roles=roles[indices],rx=rx[indices],day=day[indices])
    with np.load(folder/'clean_components.npz',allow_pickle=False) as f:
        for key,value in dict(identity,clean=clean[indices]).items():
            if not np.array_equal(f[key],value):raise ValueError('Matching source diagnostic records required')
        reference={key:f[key] for key in ['features','parts','logits','normalized_head_weight','head_scale']}
    cache=args.root/'runs'/CACHE/'source_signature_cache';cache_meta=base.read_json(cache/'complete.json')
    if cache_meta['F_teacher']!=f_guided.teacher_reference() or cache_meta['status']!='SOURCE_ACTUAL_SIG_COMPLETE':
        raise ValueError('Same previously verified original source SIG cache required')
    signatures=np.load(cache/'targets.npy',mmap_mode='r',allow_pickle=False)
    if signatures.shape!=(63000,160) or not np.array_equal(np.load(cache/'physical_ids.npy',allow_pickle=False),ids):
        raise ValueError('Matching source-only SIG cache order required')
    archives={}
    for view in VIEWS:
        with np.load(folder/f'DURA_default-{view}.npz',allow_pickle=False) as f:
            for key,value in identity.items():
                if not np.array_equal(f[key],value):raise ValueError('Matching cached defaultDURA source observations required')
            archives[view]={key:f[key] for key in ['scene','state','lock','features','parts','case_logits']}
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    before={key:value.clone() for key,value in reader.state_dict().items()}
    args.output.mkdir();base.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,rule=RULE,CPU_threads=2,GPU_workers=0))
    if not np.all(roles[indices[:2]]=='L_s'):raise ValueError('Two real source L_s smoke records required')
    with torch.no_grad():
        actual,parts=components(reader,tensor(clean[indices[:2]],'cpu'))
        torch.testing.assert_close(actual,tensor(reference['features'][:2],'cpu'),atol=3e-5,rtol=2e-5)
        torch.testing.assert_close(parts['signature'],tensor(signatures[indices[:2]],'cpu'),atol=3e-5,rtol=2e-5)
        torch.testing.assert_close(reader.classify_features(actual),tensor(reference['logits'][:2],'cpu'),atol=3e-5,rtol=2e-5)
    base.write_json(args.output/'smoke.json',dict(status='PASS',real_Ls=2,F_inputs=2,head_inputs=2,new_G_inputs=0,TX_labels_read=0,V_reads=0,target_reads=0))
    print(json.dumps(dict(event='REAL_SOURCE_RX_SIGNATURE_SMOKE_PASS')),flush=True)
    started=time.perf_counter();fit=receiver_basis(signatures,rx,indices)
    if fit['fit_records']!=62488 or fit['held_out_records']!=512:raise ValueError('Fixed source leave-512-out fit required')
    np.savez(args.output/'source_basis.npz',**fit,excluded=indices)
    basis=fit['basis'];np.savez(args.output/'reference.npz',**identity,clean_features=reference['features'],
        clean_signature=reference['parts'][3],clean_logits=reference['logits'],normalized_head_weight=reference['normalized_head_weight'],head_scale=reference['head_scale'])
    results={};records=[]
    with torch.no_grad():
        for view,a in archives.items():
            features=feature_cases(a['features'],a['parts'][3],reference['parts'][3],basis)
            logits=numpy_values(reader.classify_features(tensor(features.reshape(-1,160),'cpu'))).reshape(4,512,6)
            np.testing.assert_allclose(logits[0],a['case_logits'][0],atol=3e-5,rtol=2e-5)
            np.testing.assert_allclose(logits[1],a['case_logits'][4],atol=3e-5,rtol=2e-5)
            choice=reference['logits'].argmax(1);selected=np.take_along_axis(logits,choice[None,:,None],axis=2).squeeze(2)
            other=logits.copy();other[:,np.arange(512),choice]=-np.inf
            values=dict(case_features=features,case_logits=logits,agreement=(logits.argmax(2)==choice[None]).astype(np.float32),
                reference_margin=selected-other.max(2),feature_relative_MSE=np.square(features-reference['features'][None]).sum(2)/np.maximum(np.square(reference['features']).sum(1),1e-4)[None])
            if any(not np.isfinite(value).all() for value in values.values()):raise FloatingPointError('Nonfinite source RX intervention')
            delta=reference['parts'][3].astype(np.float64)-a['parts'][3]
            fraction=float(np.square(delta@basis).sum()/max(np.square(delta).sum(),1e-20))
            np.savez(args.output/f'DURA_default-{view}.npz',**identity,**{key:a[key] for key in ['scene','state','lock']},**values)
            masks=[('overall','all',np.ones(512,bool))]
            for dim,group in dict(RX=rx[indices],day=day[indices],role=roles[indices],scene=a['scene'],state=a['state'],lock=a['lock']).items():
                masks.extend((dim,str(value),group==value) for value in np.unique(group))
            groups=[dict(view=view,dimension=dim,value=value,cases=summaries(values,mask)) for dim,value,mask in masks]
            verdict=decision(groups);results[view]=dict(overall=groups[0]['cases'],decision=verdict,SIG_error_energy_in_RX_span=fraction);records.extend(groups)
            print(json.dumps(dict(event='SOURCE_RX_SIGNATURE_CELL_COMPLETE',view=view,overall=groups[0]['cases'],decision=verdict)),flush=True)
    if any(not torch.equal(value,before[key]) for key,value in reader.state_dict().items()) or any(p.grad is not None for p in reader.parameters()):
        raise ValueError('Frozen original F changed')
    verdict=dict(eligible=all(value['decision']['eligible'] for value in results.values()),automatic_training=False)
    base.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',run=RUN,rule=RULE,source_run=SOURCE,source_signature_cache=CACHE,
        source_records=512,source_L=128,source_U=384,fit_records=62488,held_out_records=512,basis_rank=4,
        source_RX=fit['receivers'].tolist(),source_RX_fit_counts=fit['counts'].tolist(),fit_singular_values=fit['singular_values'].tolist(),
        fit_centered_energy_fraction=fit['fit_centered_energy_fraction'],views=list(VIEWS),cases=list(CASES),results=results,records=records,decision=verdict,
        F_teacher=f_guided.teacher_reference(),F_unchanged=True,main_F_inputs=0,head_only_inputs=4096,smoke_F_inputs=2,smoke_head_inputs=2,
        new_G_inputs=0,G_checkpoint_reads=0,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,seconds=time.perf_counter()-started,
        scope='SourceRXmeanassociation/notcausalRXnuisance;latentpairedcleanfeatureoracle/notIQrealizability/trueTX/physicalceiling;originalcacheandF50costinherited'))
    base.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',decision=verdict,formal_predictions=0,goal_complete=False))


def main():
    import argparse
    from pathlib import Path
    import torch
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['root','output']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
