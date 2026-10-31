"""SOURCE-only tail24 identity utility versus equal-energy global IQ repair."""
import numpy as np

RUN='20261031-diagnostic-DURA-source-tail24-utility-s2026092701-r01'
SOURCE='20261031-diagnostic-DURA-source-branchFeatures-s2026092701-r01'
VIEWS=('1','20');CASES=('native','tail24_clean','equal_energy_global','all_clean')
RULE=('New source boundary hypothesis from earlier source tail24/prefix232 per-sample NMSE3.20x. '
 'FixeddefaultDURA/source512128L384U/caches01and20;no source/model/segment/length search. '
 'Tail24pairedcleanreplacement versus closestglobalpairedclean IQ at EXACTSAME correctionL2energy. '
 'Allclean cached positive control tautological/notaccuracyceiling. '
 'BOTHviews tail vsnative cleanF agreement>=4, versus equalenergyglobal>=2;each5RX/cleanscene no loss versusglobal; '
 'eachnoncleansceneloss<=1versusglobal;meanfullFrelativeMSE<=native. '
 'PASS onlysupports separately registered boundary-aware G/sourcefeasibility hypothesis;noautomatictraining. '
 'PairedcleanIQoracles notdeployableinputs/trueTXaccuracy/physicalceiling. No G/TX/V/target/modelupdates.')


def tail_cases(restored,clean):
    restored=np.asarray(restored);clean=np.asarray(clean)
    if restored.ndim!=3 or restored.shape[1:]!=(2,256) or clean.shape!=restored.shape:
        raise ValueError('Matched source IQ B2x256 required')
    error=clean.astype(np.float64)-restored;tail=error.copy();tail[:,:,:232]=0
    total=np.sum(error**2,(1,2));amount=np.sum(tail**2,(1,2))
    fraction=np.divide(amount,total,out=np.zeros_like(amount),where=total>0)
    scale=np.sqrt(fraction)
    return np.stack((restored,restored+tail,restored+scale[:,None,None]*error,clean)).astype(np.float32)


def summarize(values,mask):
    result={}
    for i,name in enumerate(CASES):
        result[name]=dict(count=int(mask.sum()),cleanF_agreement=int(values['agreement'][i,mask].sum()),
            gain_vs_native=int(values['agreement'][i,mask].sum()-values['agreement'][0,mask].sum()),
            gain_vs_equal_energy_global=int(values['agreement'][i,mask].sum()-values['agreement'][2,mask].sum()),
            reference_margin_mean=float(values['reference_margin'][i,mask].mean(dtype=np.float64)),
            feature_relative_MSE_mean=float(values['feature_relative_MSE'][i,mask].mean(dtype=np.float64)),
            IQ_NMSE_mean=float(values['IQ_NMSE'][i,mask].mean(dtype=np.float64)))
    return result


def decision(groups):
    o=groups[0]['cases'];t=o['tail24_clean']
    checks=dict(native_gain=t['gain_vs_native']>=4,equal_energy_gain=t['gain_vs_equal_energy_global']>=2,
        RX_protection=all(r['cases']['tail24_clean']['gain_vs_equal_energy_global']>=0 for r in groups if r['dimension']=='RX'),
        clean_protection=all(r['cases']['tail24_clean']['gain_vs_equal_energy_global']>=0 for r in groups if r['dimension']=='scene' and r['value']=='clean'),
        scene_protection=all(r['cases']['tail24_clean']['gain_vs_equal_energy_global']>=-1 for r in groups if r['dimension']=='scene' and r['value']!='clean'),
        feature_error=t['feature_relative_MSE_mean']<=o['native']['feature_relative_MSE_mean'])
    return dict(eligible=all(checks.values()),checks=checks)


def run(args):
    import json,os,sys,time,torch
    from experiments.pamf_dura_direct_train import runner as base,f_guided
    from experiments.pamf_dura_diagnostics.runner import tensor,numpy_values,source_indices
    if args.output!=args.root/'runs'/RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable CPU-only source output required')
    p=base.paths(args.root);clean,roles=base.source_training(p);indices=source_indices(roles);ids=f_guided.source_ids(p,roles)
    rx=np.load(p['cache']/'train/rx.npy',allow_pickle=False)[indices];day=np.load(p['cache']/'train/day.npy',allow_pickle=False)[indices]
    identity=dict(source_positions=indices,physical_ids=ids[indices],roles=roles[indices],rx=rx,day=day)
    folder=args.root/'runs'/SOURCE;meta=base.read_json(folder/'results.json')
    if (meta['F_teacher']!=f_guided.teacher_reference() or not meta['F_unchanged'] or meta['source_records']!=512
            or meta['decision']!=dict(eligible=False,automatic_training=False,primary='DURA_default',candidate='PA')
            or any(meta[k]!=0 for k in ['new_G_inputs','G_checkpoint_reads','optimizer_updates','TX_labels_read','V_reads','target_reads'])):
        raise ValueError('Original source branch observations and preserved PA FAIL required')
    with np.load(folder/'clean_components.npz',allow_pickle=False) as f:
        for key,value in dict(identity,clean=clean[indices]).items():
            if not np.array_equal(f[key],value):raise ValueError('Matching original source reference required')
        reference={key:f[key] for key in ['features','logits','normalized_head_weight','head_scale']}
    archives={}
    for view in VIEWS:
        with np.load(folder/f'DURA_default-{view}.npz',allow_pickle=False) as f:
            for key,value in identity.items():
                if not np.array_equal(f[key],value):raise ValueError('Matching defaultDURA source observations required')
            archives[view]={key:f[key] for key in ['scene','state','lock','restored','features','case_logits']}
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False);before={k:v.clone() for k,v in reader.state_dict().items()}
    args.output.mkdir();base.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,rule=RULE,CPU_threads=2,GPU_workers=0))
    if not np.all(roles[indices[:2]]=='L_s'):raise ValueError('Two source L_s smoke records required')
    with torch.no_grad():
        actual=reader.features(tensor(clean[indices[:2]],'cpu'))
        torch.testing.assert_close(actual,tensor(reference['features'][:2],'cpu'),atol=3e-5,rtol=2e-5)
        torch.testing.assert_close(reader.classify_features(actual),tensor(reference['logits'][:2],'cpu'),atol=3e-5,rtol=2e-5)
    base.write_json(args.output/'smoke.json',dict(status='PASS',real_Ls=2,F_inputs=2,head_inputs=2,new_G_inputs=0,TX_labels_read=0,V_reads=0,target_reads=0))
    print(json.dumps(dict(event='REAL_SOURCE_TAIL_UTILITY_SMOKE_PASS')),flush=True)
    started=time.perf_counter();np.savez(args.output/'reference.npz',**identity,clean=clean[indices],**reference)
    results={};records=[]
    with torch.no_grad():
        for view,a in archives.items():
            IQ=tail_cases(a['restored'],clean[indices]);features=[]
            for start in range(0,512,32):
                sl=slice(start,start+32);new=reader.features(tensor(IQ[1:3,sl].reshape(-1,2,256),'cpu')).reshape(2,32,160)
                features.append(numpy_values(new))
            features=np.concatenate(features,axis=1)
            features=np.concatenate((a['features'][None],features,reference['features'][None]),axis=0)
            logits=numpy_values(reader.classify_features(tensor(features.reshape(-1,160),'cpu'))).reshape(4,512,6)
            np.testing.assert_allclose(logits[0],a['case_logits'][0],atol=3e-5,rtol=2e-5)
            np.testing.assert_allclose(logits[3],reference['logits'],atol=3e-5,rtol=2e-5)
            choice=reference['logits'].argmax(1);selected=np.take_along_axis(logits,choice[None,:,None],axis=2).squeeze(2)
            other=logits.copy();other[:,np.arange(512),choice]=-np.inf
            values=dict(IQ=IQ,features=features,logits=logits,agreement=(logits.argmax(2)==choice[None]).astype(np.float32),reference_margin=selected-other.max(2),
                feature_relative_MSE=np.square(features-reference['features'][None]).sum(2)/np.maximum(np.square(reference['features']).sum(1),1e-4)[None],
                IQ_NMSE=np.mean(np.square(IQ-clean[indices][None]),(2,3))/np.maximum(np.mean(clean[indices]**2,(1,2)),1e-4)[None],
                correction_energy=np.sum(np.square(IQ.astype(np.float64)-a['restored'][None]),(2,3)))
            if any(not np.isfinite(v).all() for v in values.values()) or not np.all(values['agreement'][3]==1):raise ValueError('Finite complete source tail outputs and all-clean control required')
            np.testing.assert_allclose(values['correction_energy'][1],values['correction_energy'][2],atol=1e-8,rtol=3e-5)
            np.savez(args.output/f'DURA_default-{view}.npz',**identity,**{key:a[key] for key in ['scene','state','lock']},**values)
            masks=[('overall','all',np.ones(512,bool))]
            for dim,group in dict(RX=rx,day=day,role=roles[indices],scene=a['scene'],state=a['state'],lock=a['lock']).items():
                masks.extend((dim,str(v),group==v) for v in np.unique(group))
            groups=[dict(view=view,dimension=dim,value=value,cases=summarize(values,mask)) for dim,value,mask in masks]
            verdict=decision(groups);error=(a['restored'].astype(np.float64)-clean[indices])**2
            geometry=dict(tail24_aggregate_error_energy_fraction=float(error[:,:,232:].sum()/max(error.sum(),1e-20)),
                tail24_prefix232_per_sample_error_ratio=float(error[:,:,232:].mean()/max(error[:,:,:232].mean(),1e-20)))
            results[view]=dict(overall=groups[0]['cases'],geometry=geometry,decision=verdict);records.extend(groups)
            print(json.dumps(dict(event='SOURCE_TAIL_UTILITY_CELL_COMPLETE',view=view,overall=groups[0]['cases'],geometry=geometry,decision=verdict)),flush=True)
    if any(not torch.equal(value,before[key]) for key,value in reader.state_dict().items()) or any(p.grad is not None for p in reader.parameters()):raise ValueError('Frozen original F changed')
    verdict=dict(eligible=all(r['decision']['eligible'] for r in results.values()),automatic_training=False)
    base.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',run=RUN,rule=RULE,source_run=SOURCE,source_records=512,source_L=128,source_U=384,
        views=list(VIEWS),cases=list(CASES),results=results,records=records,decision=verdict,F_teacher=f_guided.teacher_reference(),F_unchanged=True,
        main_F_inputs=2048,head_only_inputs=4096,smoke_F_inputs=2,smoke_head_inputs=2,new_G_inputs=0,G_checkpoint_reads=0,optimizer_updates=0,
        TX_labels_read=0,V_reads=0,target_reads=0,seconds=time.perf_counter()-started,
        scope='Source tailclean IQ oracle and equal-energy closestglobalclean control;notdeployableIQ/trueTXaccuracy/causalboundaryproof/physicalceiling'))
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
