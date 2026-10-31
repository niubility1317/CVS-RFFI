"""Fixed SOURCE comparison of common IQ/SIG steps and prior SIG-only steps."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from experiments.pamf_dura_direct_train import runner as base,f_guided
from experiments.pamf_dura_diagnostics.runner import tensor,numpy_values,source_indices
from .branch_features import components
from .signature_iq import signature_features
from .signature_cone import common_signature_steps,common_step

RUN='20261031-diagnostic-DURA-source-commonSIG-step-s2026092701-r01'
CONTROL='20261031-diagnostic-DURA-source-signatureLossStep-s2026092701-r01'
VIEWS=('1','20');CASES=('native','IQ_oracle','SIG_control','common_SIG')
RULE=('New source-only common IQ/signature update, no change to prior source FAIL. '
 'FixeddefaultDURA/source512128L384U/caches01and20;reusepreviousSIG-onlyloss-scaled4stepcontrol. '
 'At each of4steps sum unit IQ and SIG derivatives;zero/opposed/undefined direction gives no step. '
 'Scale by min(exactnonnegativeIQline-minimizer, SIGloss/dotSIGdirection), perstep cap.25percentnativeRMS/totalproject1percent. '
 'No backtracking/rank/alpha/step-count/search. Finite IQ no worse than anchor, only first-order SIG descent. '
 'BOTHviews common cleanF agreement>=SIGcontrol+2, every5RX/cleanscene no loss versusSIGcontrol, '
 'eachnonclean scene loss<=1 versusSIGcontrol, meanSIGrelativeMSE<=.9native, eachrecordIQNMSE<=native+tolerance. '
 'PASS onlysupports separately registered scratchjointE30 hypothesis;never automatictraining/deployment. '
 'Sourcepairedclean/Freference inaccessibleatdeployment;proxy nottrueTX/hardware/physicalceiling.')


def summaries(values,mask):
    out={}
    for i,name in enumerate(CASES):
        out[name]=dict(count=int(mask.sum()),cleanF_agreement=int(values['agreement'][i,mask].sum()),
            gain_vs_native=int(values['agreement'][i,mask].sum()-values['agreement'][0,mask].sum()),
            gain_vs_SIG_control=int(values['agreement'][i,mask].sum()-values['agreement'][2,mask].sum()),
            reference_margin_mean=float(values['reference_margin'][i,mask].mean(dtype=np.float64)),
            signature_relative_MSE_mean=float(values['signature_relative_MSE'][i,mask].mean(dtype=np.float64)),
            IQ_NMSE_mean=float(values['IQ_NMSE'][i,mask].mean(dtype=np.float64)))
    return out


def decision(groups,values):
    o=groups[0]['cases'];c=o['common_SIG']
    checks=dict(extra_agreement=c['gain_vs_SIG_control']>=2,
        RX_protection=all(r['cases']['common_SIG']['gain_vs_SIG_control']>=0 for r in groups if r['dimension']=='RX'),
        clean_protection=all(r['cases']['common_SIG']['gain_vs_SIG_control']>=0 for r in groups if r['dimension']=='scene' and r['value']=='clean'),
        scene_protection=all(r['cases']['common_SIG']['gain_vs_SIG_control']>=-1 for r in groups if r['dimension']=='scene' and r['value']!='clean'),
        signature_error=c['signature_relative_MSE_mean']<=.9*o['native']['signature_relative_MSE_mean'],
        IQ_protection=bool(np.all(values['IQ_NMSE'][3]<=values['IQ_NMSE'][0]*(1+3e-5)+1e-10)))
    return dict(eligible=all(checks.values()),checks=checks)


def run(args):
    if args.output!=args.root/'runs'/RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable CPU-only source output required')
    p=base.paths(args.root);clean,roles=base.source_training(p);indices=source_indices(roles);ids=f_guided.source_ids(p,roles)
    rx=np.load(p['cache']/'train/rx.npy',allow_pickle=False)[indices];day=np.load(p['cache']/'train/day.npy',allow_pickle=False)[indices]
    identity=dict(source_positions=indices,physical_ids=ids[indices],roles=roles[indices],rx=rx,day=day)
    folder=args.root/'runs'/CONTROL;meta=base.read_json(folder/'results.json')
    if (meta['F_teacher']!=f_guided.teacher_reference() or not meta['F_unchanged'] or meta['source_records']!=512
            or meta['cases']!=['native','IQ_oracle','signature4'] or meta['decision']['eligible']
            or any(meta[k]!=0 for k in ['new_G_inputs','G_checkpoint_reads','model_optimizer_updates','TX_labels_read','V_reads','target_reads'])):
        raise ValueError('Original source-only SIG control and preserved FAIL required')
    with np.load(folder/'reference.npz',allow_pickle=False) as f:
        for key,value in dict(identity,clean=clean[indices]).items():
            if not np.array_equal(f[key],value):raise ValueError('Matching original source reference required')
        reference={key:f[key] for key in ['clean_features','clean_signature','clean_logits','normalized_head_weight']}
    archives={}
    for view in VIEWS:
        with np.load(folder/f'DURA_default-{view}.npz',allow_pickle=False) as f:
            for key,value in identity.items():
                if not np.array_equal(f[key],value):raise ValueError('Matching original source control records required')
            archives[view]={key:f[key] for key in ['scene','state','lock','IQ','features','signature','logits']}
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    before={k:v.clone() for k,v in reader.state_dict().items()}
    args.output.mkdir();base.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,rule=RULE,CPU_threads=2,GPU_workers=0))
    if not np.all(roles[indices[:2]]=='L_s'):raise ValueError('Two source L_s smoke records required')
    with torch.no_grad():
        feature,parts=components(reader,tensor(clean[indices[:2]],'cpu'))
        torch.testing.assert_close(parts['signature'],tensor(reference['clean_signature'][:2],'cpu'),atol=3e-5,rtol=2e-5)
        torch.testing.assert_close(reader.classify_features(feature),tensor(reference['clean_logits'][:2],'cpu'),atol=3e-5,rtol=2e-5)
    r=tensor(archives['1']['IQ'][0,:2],'cpu').requires_grad_();x=tensor(clean[indices[:2]],'cpu');target=tensor(reference['clean_signature'][:2],'cpu')
    error=(signature_features(reader,r)-target).square().sum(1)/target.square().sum(1).clamp_min(1e-4)
    gradient=torch.autograd.grad(error.sum(),r)[0];step,info=common_step(r.detach(),x,error.detach(),gradient.detach())
    if (not step.abs().sum()>0 or not torch.isfinite(step).all() or (gradient*step).sum((1,2)).max()>1e-6
            or ((r.detach()+step-x).square().sum((1,2))>(r.detach()-x).square().sum((1,2))*(1+3e-5)+1e-10).any()):
        raise ValueError('Actual source common-step derivative/IQ smoke failed')
    base.write_json(args.output/'smoke.json',dict(status='PASS',real_Ls=2,F_inputs=2,F_prefix_inputs=2,F_input_VJP_inputs=2,head_inputs=2,
        actual_common_derivative_and_finite_IQ=True,new_G_inputs=0,TX_labels_read=0,V_reads=0,target_reads=0))
    print(json.dumps(dict(event='REAL_SOURCE_COMMON_SIG_SMOKE_PASS')),flush=True)
    started=time.perf_counter();results={};records=[]
    np.savez(args.output/'reference.npz',**identity,clean=clean[indices],**reference)
    for view,a in archives.items():
        batches=[]
        for start in range(0,512,32):
            sl=slice(start,start+32);anchor=tensor(a['IQ'][0,sl],'cpu');truth=tensor(clean[indices[sl]],'cpu')
            target=tensor(reference['clean_signature'][sl],'cpu');value,history,geometry=common_signature_steps(reader,anchor,target,truth)
            with torch.no_grad():
                new,parts=components(reader,value)
                IQ=torch.cat((tensor(a['IQ'][:,sl],'cpu'),value[None]))
                features=torch.cat((tensor(a['features'][:,sl],'cpu'),new[None]))
                signatures=torch.cat((tensor(a['signature'][:,sl],'cpu'),parts['signature'][None]))
                logits=reader.classify_features(features.flatten(0,1)).reshape(4,32,6)
                torch.testing.assert_close(logits[:3],tensor(a['logits'][:,sl],'cpu'),atol=3e-5,rtol=2e-5)
                choice=tensor(reference['clean_logits'][sl],'cpu').argmax(1)
                selected=logits.gather(2,choice[None,:,None].expand(4,-1,1)).squeeze(2);other=logits.clone();other.scatter_(2,choice[None,:,None].expand(4,-1,1),-torch.inf)
                values=dict(IQ=IQ,features=features,signature=signatures,logits=logits,step_losses=history,
                    agreement=(logits.argmax(2)==choice[None]).float(),reference_margin=selected-other.max(2).values,
                    signature_relative_MSE=(signatures-target[None]).square().sum(2)/target.square().sum(1).clamp_min(1e-4)[None],
                    IQ_NMSE=(IQ-truth[None]).square().mean((2,3))/truth.square().mean((1,2)).clamp_min(1e-4)[None],
                    shift_relative_MSE=(IQ-anchor[None]).square().mean((2,3))/anchor.square().mean((1,2)).clamp_min(1e-16)[None],
                    **{'step_'+key:value for key,value in geometry.items()})
                if values['shift_relative_MSE'].max()>.00010001:raise ValueError('Source one-percent RMS ball exceeded')
                batches.append({key:numpy_values(value) for key,value in values.items()})
        values={key:np.concatenate([b[key] for b in batches],axis=1) for key in batches[0]}
        if any(not np.isfinite(v).all() for v in values.values()):raise FloatingPointError('Nonfinite source common-step output')
        np.savez(args.output/f'DURA_default-{view}.npz',**identity,**{key:a[key] for key in ['scene','state','lock']},**values)
        masks=[('overall','all',np.ones(512,bool))]
        for dim,group in dict(RX=rx,day=day,role=roles[indices],scene=a['scene'],state=a['state'],lock=a['lock']).items():
            masks.extend((dim,str(v),group==v) for v in np.unique(group))
        groups=[dict(view=view,dimension=dim,value=value,cases=summaries(values,mask)) for dim,value,mask in masks]
        verdict=decision(groups,values);geometry=dict(negative_IQ_SIG_cosine_fraction=float((values['step_cosine']<0).mean()),
            cap_half_baseline_SIG_increase_first_order_fraction=float((values['step_baseline_feature_first_order']<0).mean()),
            common_active_fraction=float(values['step_active'].mean()),IQ_NMSE_violations=int((values['IQ_NMSE'][3]>values['IQ_NMSE'][0]*(1+3e-5)+1e-10).sum()))
        results[view]=dict(overall=groups[0]['cases'],geometry=geometry,decision=verdict);records.extend(groups)
        print(json.dumps(dict(event='SOURCE_COMMON_SIG_CELL_COMPLETE',view=view,overall=groups[0]['cases'],geometry=geometry,decision=verdict)),flush=True)
    if any(not torch.equal(value,before[key]) for key,value in reader.state_dict().items()) or any(p.grad is not None for p in reader.parameters()):raise ValueError('Original frozen F changed')
    verdict=dict(eligible=all(r['decision']['eligible'] for r in results.values()),automatic_training=False)
    base.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',run=RUN,rule=RULE,source_control=CONTROL,source_records=512,source_L=128,source_U=384,
        views=list(VIEWS),cases=list(CASES),results=results,records=records,decision=verdict,F_teacher=f_guided.teacher_reference(),F_unchanged=True,
        main_F_inputs=1024,F_prefix_inputs=4096,F_input_VJP_inputs=4096,head_only_inputs=4096,smoke_F_inputs=2,smoke_prefix_inputs=2,smoke_VJP_inputs=2,smoke_head_inputs=2,
        new_G_inputs=0,G_checkpoint_reads=0,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,seconds=time.perf_counter()-started,
        scope='Source IQ oracle and first-order geometry only/notGparametergradient-guarantee/trueTX/hardware/physicalceiling;priorFAILunchanged'))
    base.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',decision=verdict,formal_predictions=0,goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['root','output']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
