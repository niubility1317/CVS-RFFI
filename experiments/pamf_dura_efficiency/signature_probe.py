"""SOURCE-only bounded-IQ feasibility of the original frozen time signature."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from experiments.pamf_dura_direct_train import runner as base,f_guided
from experiments.pamf_dura_diagnostics.runner import tensor,numpy_values,source_indices
from .branch_features import components,BRANCHES
from .signature_iq import signature_features,signature_steps,project

RUN='20261031-diagnostic-DURA-source-signatureIQ-s2026092701-r01'
SOURCE='20261031-diagnostic-DURA-source-branchFeatures-s2026092701-r01'
VIEWS=('1','20')
CASES=('native','IQ_oracle','signature4')
RULE=('New independent source IQ feasibility hypothesis motivated by descriptive signature damage; prior PA FAIL unchanged. '
 'DefaultDURA only/source512128L384U/caches01and20/fixed final4steps of0.25percent nativeIQ RMS/project1percent ball. '
 'Loss only pairedsourceclean actual weightedTripleSIG relativeMSE/no classloss. '
 'IQ_oracle=closest pairedclean waveform in SAME 1percent ball;descriptive comparator. '
 'BOTHviews signature4 cleanFchoicegain>=4vsnative and>=2vsIQoracle;all5sourceRXgain>=0;eachnonclean scene gain>=-1;clean gain>=0; '
 'mean signature relativeMSE <=0.9native;meanIQNMSE<=1.01native. No stepsize/iterations/reference/parent/threshold search. '
 'PASS supports separate scratchE30 candidate registration,not automatictraining or targetclaim. Sourceclean reference unavailableatdeployment;notphysicalrecoveryceiling/trueTXaccuracy.')


def summaries(values,mask):
    result={}
    for i,name in enumerate(CASES):
        result[name]=dict(count=int(mask.sum()),cleanF_agreement=int(values['agreement'][i,mask].sum()),
            gain_vs_native=int(values['agreement'][i,mask].sum()-values['agreement'][0,mask].sum()),
            reference_margin_mean=float(values['reference_margin'][i,mask].mean(dtype=np.float64)),
            signature_relative_MSE_mean=float(values['signature_relative_MSE'][i,mask].mean(dtype=np.float64)),
            IQ_NMSE_mean=float(values['IQ_NMSE'][i,mask].mean(dtype=np.float64)),
            shift_relative_MSE_max=float(values['shift_relative_MSE'][i,mask].max()))
    return result


def decision(groups):
    o=groups[0]['cases'];s=o['signature4'];n=o['native'];q=o['IQ_oracle']
    checks=dict(gain=s['gain_vs_native']>=4,vs_waveform_oracle=s['cleanF_agreement']-q['cleanF_agreement']>=2,
        RX_protection=all(g['cases']['signature4']['gain_vs_native']>=0 for g in groups if g['dimension']=='RX'),
        scene_protection=all(g['cases']['signature4']['gain_vs_native']>=-1 for g in groups if g['dimension']=='scene' and g['value']!='clean'),
        clean_protection=all(g['cases']['signature4']['gain_vs_native']>=0 for g in groups if g['dimension']=='scene' and g['value']=='clean'),
        signature_error=s['signature_relative_MSE_mean']<=.9*n['signature_relative_MSE_mean'],
        IQ_error=s['IQ_NMSE_mean']<=1.01*n['IQ_NMSE_mean'])
    return dict(eligible=all(checks.values()),checks=checks)


def run(args, *, run_id=RUN, rule=RULE, iq_steps=signature_steps):
    if args.output!=args.root/'runs'/run_id or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable CPU source diagnostic required')
    p=base.paths(args.root);clean,roles=base.source_training(p);indices=source_indices(roles);ids=f_guided.source_ids(p,roles)
    folder=args.root/'runs'/SOURCE;meta=base.read_json(folder/'results.json')
    if (meta['F_teacher']!=f_guided.teacher_reference() or meta['source_records']!=512 or not meta['F_unchanged']
        or meta['models']!=['DURA_default','pureG_E30'] or meta['views']!=list(VIEWS) or meta['branches']!=list(BRANCHES)
        or meta['decision']!=dict(eligible=False,automatic_training=False,primary='DURA_default',candidate='PA')
        or any(meta[k]!=0 for k in ['new_G_inputs','G_checkpoint_reads','optimizer_updates','TX_labels_read','V_reads','target_reads'])):
        raise ValueError('Matching original source branch observations and unchanged PA FAIL required')
    rx=np.load(p['cache']/'train/rx.npy',allow_pickle=False)[indices];day=np.load(p['cache']/'train/day.npy',allow_pickle=False)[indices]
    identity=dict(source_positions=indices,physical_ids=ids[indices],roles=roles[indices],rx=rx,day=day)
    with np.load(folder/'clean_components.npz',allow_pickle=False) as f:
        for k,v in dict(identity,clean=clean[indices]).items():
            if not np.array_equal(f[k],v):raise ValueError('Exact original clean source roles/order required')
        reference={k:f[k] for k in ['features','parts','logits']}
    archives={}
    for view in VIEWS:
        with np.load(folder/f'DURA_default-{view}.npz',allow_pickle=False) as f:
            for k,v in identity.items():
                if not np.array_equal(f[k],v):raise ValueError('Exact original native DURA source roles/order required')
            archives[view]={k:f[k] for k in ['scene','state','lock','restored','features','parts','case_logits']}
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    before={k:v.clone() for k,v in reader.state_dict().items()}
    args.output.mkdir();base.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,rule=rule,CPU_threads=2,GPU_workers=0))
    if not np.all(roles[indices[:2]]=='L_s'):raise ValueError('Real smoke must use two L_s')
    x=tensor(clean[indices[:2]],'cpu').requires_grad_(True)
    actual,parts=components(reader,x);expected=parts['signature'];direction=torch.linspace(-1,1,160)[None]
    original_vjp=torch.autograd.grad((expected*direction).sum(),x)[0]
    prefix=signature_features(reader,x);prefix_vjp=torch.autograd.grad((prefix*direction).sum(),x)[0]
    torch.testing.assert_close(prefix,expected,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(prefix_vjp,original_vjp,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(prefix,tensor(reference['parts'][3,:2],'cpu'),atol=3e-5,rtol=2e-5)
    torch.testing.assert_close(actual,tensor(reference['features'][:2],'cpu'),atol=3e-5,rtol=2e-5)
    torch.testing.assert_close(reader.classify_features(actual),tensor(reference['logits'][:2],'cpu'),atol=3e-5,rtol=2e-5)
    if any(m._forward_hooks for m in reader.modules()):raise ValueError('Prefix hooks leaked')
    base.write_json(args.output/'smoke.json',dict(status='PASS',real_Ls=2,full_F_inputs=2,prefix_inputs=2,input_VJP_records=4,
        head_only_inputs=2,exact_signature_output_and_input_VJP=True,TX_labels_read=0,V_reads=0,target_reads=0))
    print(json.dumps(dict(event='REAL_SOURCE_SIGNATURE_PREFIX_SMOKE_PASS')),flush=True)
    started=time.perf_counter();results={};records=[]
    normalized_weight=numpy_values(torch.nn.functional.normalize(reader.core.id_backbone.cls_head.weight,dim=1,eps=1e-4))
    np.savez(args.output/'reference.npz',**identity,clean=clean[indices],clean_features=reference['features'],
        clean_signature=reference['parts'][3],clean_logits=reference['logits'],normalized_head_weight=normalized_weight)
    for view in VIEWS:
        a=archives[view];batches=[]
        for start in range(0,512,32):
            sl=slice(start,start+32);anchor=tensor(a['restored'][sl],'cpu');truth=tensor(clean[indices[sl]],'cpu')
            target=tensor(reference['parts'][3,sl],'cpu')
            signature_iq,history=iq_steps(reader,anchor,target)
            with torch.no_grad():
                iq_oracle=project(anchor,truth-anchor)
                iq=torch.stack((anchor,iq_oracle,signature_iq));new_feature,new_parts=components(reader,iq[1:].flatten(0,1))
                feature=torch.cat((tensor(a['features'][sl],'cpu')[None],new_feature.reshape(2,32,160)))
                signatures=torch.cat((tensor(a['parts'][3,sl],'cpu')[None],new_parts['signature'].reshape(2,32,160)))
                logits=reader.classify_features(feature.flatten(0,1)).reshape(3,32,6)
                torch.testing.assert_close(logits[0],tensor(a['case_logits'][0,sl],'cpu'),atol=3e-5,rtol=2e-5)
                reference_choice=tensor(reference['logits'][sl],'cpu').argmax(1)
                chosen=logits.gather(2,reference_choice[None,:,None].expand(3,-1,1)).squeeze(-1)
                other=logits.clone();other.scatter_(2,reference_choice[None,:,None].expand(3,-1,1),-torch.inf)
                values=dict(IQ=iq,features=feature,signature=signatures,logits=logits,step_losses=history,
                    agreement=(logits.argmax(2)==reference_choice[None]).float(),reference_margin=chosen-other.max(2).values,
                    signature_relative_MSE=(signatures-target[None]).square().sum(2)/target.square().sum(1).clamp_min(1e-4)[None],
                    IQ_NMSE=(iq-truth[None]).square().mean((2,3))/truth.square().mean((1,2)).clamp_min(1e-4)[None],
                    shift_relative_MSE=(iq-anchor[None]).square().mean((2,3))/anchor.square().mean((1,2)).clamp_min(1e-16)[None])
                if values['shift_relative_MSE'].max()>.00010001:raise ValueError('One percent RMS IQ bound exceeded')
                batches.append({k:numpy_values(v) for k,v in values.items()})
        values={k:np.concatenate([v[k] for v in batches],axis=1) for k in batches[0]}
        if any(not np.isfinite(v).all() for v in values.values()):raise FloatingPointError('Nonfinite source IQ feasibility observation')
        np.savez(args.output/f'DURA_default-{view}.npz',**identity,**{k:a[k] for k in ['scene','state','lock']},**values)
        masks=[('overall','all',np.ones(512,bool))]
        for dim,group in dict(RX=rx,day=day,role=roles[indices],scene=a['scene'],state=a['state'],lock=a['lock']).items():
            masks.extend((dim,str(v),group==v) for v in np.unique(group))
        groups=[dict(view=view,dimension=dim,value=value,cases=summaries(values,mask)) for dim,value,mask in masks]
        verdict=decision(groups);results[view]=dict(overall=groups[0]['cases'],decision=verdict);records.extend(groups)
        print(json.dumps(dict(event='SOURCE_SIGNATURE_IQ_CELL_COMPLETE',view=view,overall=groups[0]['cases'],decision=verdict)),flush=True)
    if any(not torch.equal(v,before[k]) for k,v in reader.state_dict().items()) or any(p.grad is not None for p in reader.parameters()):
        raise ValueError('Frozen original F changed')
    verdict=dict(eligible=all(r['decision']['eligible'] for r in results.values()),automatic_training=False)
    base.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',run=run_id,source_run=SOURCE,rule=rule,
        source_records=512,source_L=128,source_U=384,views=list(VIEWS),cases=list(CASES),results=results,records=records,decision=verdict,
        F_teacher=f_guided.teacher_reference(),F_unchanged=True,full_F_inputs=2048,prefix_inputs=4096,input_VJP_records=4096,
        head_only_inputs=3072,smoke_full_F_inputs=2,smoke_prefix_inputs=2,smoke_input_VJP_records=4,smoke_head_only_inputs=2,
        new_G_inputs=0,G_checkpoint_reads=0,model_optimizer_updates=0,IQ_steps_per_record=4,
        TX_labels_read=0,V_reads=0,target_reads=0,seconds=time.perf_counter()-started,
        scope='SourcepairedcleanIQoraclefeasibility/notdeployableinference/trueTXaccuracy/physicalrecoveryceiling;priorPAFAILunchanged'))
    base.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',decision=verdict,formal_predictions=0,goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['root','output']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
