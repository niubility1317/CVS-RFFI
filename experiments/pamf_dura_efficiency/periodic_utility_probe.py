"""SOURCE-only periodic/residual error utility with an equal-energy control."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from experiments.pamf_dura_direct_train import runner as base,f_guided
from experiments.pamf_dura_diagnostics.runner import source_indices,tensor,numpy_values
from .periodic_utility import periodic_projection,counterfactuals,decision,CASES,START,STOP,PERIOD

RUN='20261031-diagnostic-DURA-source-period20-utility-s2026092701-r01'
PREVIOUS='20261030-diagnostic-DURA-source-phase-oracle-s2026092701-m2-r01'
RULE=('FixeddefaultDURA/source512L128U384/cache01and20/nativeIQ-and-Fcache;window80:160/four20pointcycles. '
 'PrimaryresidualoracleBOTHviews:cleanFchoicegain>=4vsnativeand>=2vsequalcorrectionenergyglobalnearestclean;'
 'each5RXnet>=0,clean-scenenet>=0,eachnoncleanscenenet>=-1,overallFcosineerrornotabove native. '
 'Periodicmeanoracle descriptiveonly/noalternateprimary-window-period-threshold. NoG/TX-V-target/updates;'
 'sourceclean counterfactualnotdeployable/notTXtruth/hardware/noiseattribution/recoverybound;PASSonlyseparatelearninghypothesis/notautomaticG.')


def stats(iq,features,logits,clean,reference_features,reference_logits,native_logits,mask):
    a=iq.astype(np.float64);x=clean.astype(np.float64);f=features.astype(np.float64);t=reference_features.astype(np.float64)
    nmse=np.mean((a-x)**2,(1,2))/np.maximum(np.mean(x*x,(1,2)),1e-4)
    unit=lambda v:v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-8)
    error=1-(unit(f)*unit(t)).sum(1)
    correct=logits.argmax(1)==reference_logits.argmax(1);old=native_logits.argmax(1)==reference_logits.argmax(1)
    return dict(teacher_correct=int(correct[mask].sum()),iq_nmse=float(nmse[mask].mean()),feature_cosine_error=float(error[mask].mean()),
        repaired=int((correct&~old&mask).sum()),damaged=int((~correct&old&mask).sum()),net_repaired=int(correct[mask].sum()-old[mask].sum()))


@torch.no_grad()
def run(args):
    if args.output!=args.root/'runs'/RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable CPU-only source diagnostic required')
    p=base.paths(args.root);clean,roles=base.source_training(p);ids=f_guided.source_ids(p,roles);take=source_indices(roles)
    previous=args.root/'runs'/PREVIOUS;meta=base.read_json(previous/'results.json')
    if meta['F_teacher']!=f_guided.teacher_reference() or not meta['G_unchanged'] or not meta['F_unchanged']:
        raise ValueError('Original legal frozen DURA/source-F cache required')
    rx=np.load(p['cache']/'train/rx.npy',allow_pickle=False);day=np.load(p['cache']/'train/day.npy',allow_pickle=False)
    archives={}
    for view in ['1','20']:
        with np.load(previous/f'DURA_default-{view}.npz',allow_pickle=False) as f:
            a={k:f[k] for k in ['source_positions','physical_ids','roles','clean','rx','day','scene','native','F_features','F_logits']}
        # Only the established clean/native rows are consumed from the old
        # counterfactual archive; no other old oracle is a candidate here.
        a['F_features']=a['F_features'][:2];a['F_logits']=a['F_logits'][:2]
        for key,value in dict(source_positions=take,physical_ids=ids[take],roles=roles[take],clean=clean[take],rx=rx[take],day=day[take]).items():
            if not np.array_equal(a[key],value):raise ValueError('Original paired source order and roles required: '+key)
        if a['native'].shape!=(512,2,256) or a['F_features'].shape!=(2,512,160) or a['F_logits'].shape!=(2,512,6):
            raise ValueError('Complete original source native/clean F cache required')
        if not all(a[k].dtype==np.float32 and np.isfinite(a[k]).all() for k in ['native','clean','F_features','F_logits']):raise ValueError('Finite original FP32 source data required')
        if int((a['roles']=='L_s').sum())!=128 or int((a['roles']=='U_s').sum())!=384:raise ValueError('Original source512 L/U roles required')
        if set(a['rx'])!={1,3,4,6,8} or len(set(a['scene']))!=7:raise ValueError('All source RX/scenes required')
        archives[view]=a
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False);before={k:v.clone() for k,v in reader.state_dict().items()}
    args.output.mkdir();base.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,rule=RULE,CPU_threads=2,GPU_workers=0))
    a=archives['1'];smoke=np.flatnonzero((a['roles']=='L_s')&(a['scene']!='clean'))[:2]
    if len(smoke)!=2:raise ValueError('Two actual nonclean source L_s required')
    for key,cache_index in [('clean',0),('native',1)]:
        scores=reader(tensor(a[key][smoke],'cpu'))
        torch.testing.assert_close(scores,tensor(a['F_logits'][cache_index,smoke],'cpu'),atol=3e-5,rtol=3e-5)
    base.write_json(args.output/'smoke.json',dict(status='PASS',real_nonclean_Ls=2,F_full_inputs=4,G_inputs=0,original_F_clean_and_native_logits_replayed=True,
        optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0))
    print(json.dumps(dict(event='REAL_SOURCE_PERIOD20_UTILITY_SMOKE_PASS')),flush=True)
    started=time.perf_counter();summaries={};records=[];energy_summary={}
    for view,a in archives.items():
        cases,details=counterfactuals(tensor(a['native'],'cpu'),tensor(a['clean'],'cpu'))
        iq=np.stack([numpy_values(cases[name]) for name in CASES]);features=[a['F_features'][0],a['F_features'][1]];logits=[a['F_logits'][0],a['F_logits'][1]]
        for name in CASES[1:]:
            fs=[];zs=[]
            for offset in range(0,512,32):
                value=reader.features(cases[name][offset:offset+32]);score=reader.classify_features(value)
                if not torch.isfinite(value).all() or not torch.isfinite(score).all():raise FloatingPointError('Finite frozen F outputs required')
                fs.append(numpy_values(value));zs.append(numpy_values(score))
            features.append(np.concatenate(fs));logits.append(np.concatenate(zs))
        features=np.stack(features);logits=np.stack(logits);d={k:numpy_values(v) for k,v in details.items()}
        for name in ['periodic','residual']:
            if not np.array_equal(iq[CASES.index(name),:,:,:START],a['native'][:,:,:START]) or not np.array_equal(iq[CASES.index(name),:,:,STOP:],a['native'][:,:,STOP:]):
                raise ValueError('Oracle changed outside the fixed source window')
        norm=lambda value:np.linalg.norm(value.astype(np.float64).reshape(512,-1),axis=1)
        clean_norm=norm(a['clean']);res_norm=norm(iq[2]-a['native']);matched_norm=norm(iq[3]-a['native'])
        mismatch=np.abs(res_norm-matched_norm)/np.maximum(clean_norm,1.)
        if mismatch.max()>2e-6:raise ValueError('Actual equal-energy control differs beyond FP32/input-norm tolerance')
        window=d['window_error'].astype(np.float64);per=d['periodic_delta'].astype(np.float64);res=d['residual_delta'].astype(np.float64)
        window_energy=float(np.sum(window*window));periodic_energy=float(np.sum(per*per));residual_energy=float(np.sum(res*res))
        clean_periodic=numpy_values(periodic_projection(tensor(a['clean'],'cpu')))
        clean_center=a['clean'][:,:,START:STOP].astype(np.float64)
        clean_residual=clean_center-clean_periodic[:,:,START:STOP].astype(np.float64)
        energy_summary[view]=dict(window_error_energy=window_energy,periodic_error_energy=periodic_energy,residual_error_energy=residual_energy,
            residual_fraction_of_window_error=residual_energy/max(window_energy,1e-20),orthogonality_dot=float(np.sum(per*res)),
            maximum_equal_energy_norm_difference_relative_input=float(mismatch.max()),
            source_clean_residual_fraction=float(np.sum(clean_residual**2)/max(np.sum(clean_center**2),1e-20)))
        masks={'all':np.ones(512,bool)}
        for key in ['rx','day','roles','scene']:
            for value in np.unique(a[key]):masks[key+':'+str(value)]=a[key]==value
        group={}
        for name,mask in masks.items():
            group[name]=dict(count=int(mask.sum()),cases={case:stats(iq[i],features[i+1],logits[i+1],a['clean'],features[0],logits[0],logits[1],mask) for i,case in enumerate(CASES)})
            records.append(dict(view=view,group=name,**group[name]))
        summaries[view]=group
        np.savez(args.output/f'view{view}.npz',**{k:a[k] for k in ['source_positions','physical_ids','roles','clean','rx','day','scene']},
            IQ=iq,F_features=features,F_logits=logits,**d)
        print(json.dumps(dict(event='SOURCE_PERIOD20_UTILITY_VIEW_COMPLETE',view=view,overall=group['all'],seconds=time.perf_counter()-started)),flush=True)
    if any(not torch.equal(v,before[k]) for k,v in reader.state_dict().items()) or any(v.grad is not None for v in reader.parameters()):raise ValueError('Frozen F changed')
    verdict=decision(summaries)
    base.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',run=RUN,rule=RULE,source_cache_run=PREVIOUS,source_G_identity='DefaultDURA_R69/nativeIQcacheonly/noGload',
        F_teacher=f_guided.teacher_reference(),F_unchanged=True,source_records=512,source_views=[1,20],source_observations=1024,source_unique_physical_records=512,
        source_L_unique=128,source_U_unique=384,window=[START,STOP],period=PERIOD,cases=list(CASES),summary=summaries,records=records,energy_summary=energy_summary,decision=verdict,
        F_full_inputs=3072,F_smoke_full_inputs=4,G_inputs=0,G_checkpoint_reads=0,F_checkpoint_reads=1,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,
        seconds=time.perf_counter()-started,scope='Source-training-pool pairedclean oracle/notdeployable/notTXtruth/RXnoise-hardwareattribution orrecoverybound;equal-energyutilitynotobservationrecoverability;noautomatictraining'))
    base.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',decision=verdict,formal_predictions=0,goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['root','output']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
