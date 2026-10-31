"""Fixed SOURCE PA+SIG counterfactual through the unchanged original head."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from experiments.pamf_dura_direct_train import runner as base,f_guided
from experiments.pamf_dura_diagnostics.runner import tensor,numpy_values,source_indices
from .branch_features import components,BRANCHES
from .joint_branches import paired_cases,CASES,cases_for

RUN='20261031-diagnostic-DURA-source-PA-SIG-s2026092701-r01'
SOURCE='20261031-diagnostic-DURA-source-branchFeatures-s2026092701-r01'
VIEWS=('1','20')
RULE=('Fixed defaultDURA/source512128L384U/caches01and20;cachedactualweightedfourFbranches. '
 'Cases native/PAclean/SIGclean/PAplusSIGclean/allclean,otheridentity-responseunchanged,originalfrozenF50/defaulthead. '
 'PrimaryPAplusSIGmustBOTHviews:gainvsnative>=6/512,gainvsSIG>=2/512,all5RXgainvsnative>=0,eachRXgainvsSIG>=-1,atleast4of5RXgainvsSIG>=0; '
 'eachnonclean scene gainvsnative>=-1,clean gainvsnative>=0. No branch/parent/threshold search. '
 'Passsupportsseparatelearninghypothesisonly/notautomaticGtraining. OldPA/sourceIQFAILspreserved. '
 'Thislatentpairedsourcecleanoracle isnotrealizableIQ/deployablemethod/trueTXaccuracy/physicalrecoveryceiling.')


def summary(values,mask,cases=CASES):
    return {name:dict(count=int(mask.sum()),cleanF_agreement=int(values['agreement'][i,mask].sum()),
        gain_vs_native=int(values['agreement'][i,mask].sum()-values['agreement'][0,mask].sum()),
        gain_vs_signature=int(values['agreement'][i,mask].sum()-values['agreement'][2,mask].sum()),
        reference_margin_mean=float(values['reference_margin'][i,mask].mean(dtype=np.float64)),
        feature_relative_MSE_mean=float(values['feature_relative_MSE'][i,mask].mean(dtype=np.float64))) for i,name in enumerate(cases)}


def decision(groups,joint='PA_signature'):
    overall=groups[0]['cases'][joint];rx=[r['cases'][joint] for r in groups if r['dimension']=='RX']
    checks=dict(gain=overall['gain_vs_native']>=6,extra_vs_signature=overall['gain_vs_signature']>=2,
        RX_native=all(r['gain_vs_native']>=0 for r in rx),RX_signature_loss=all(r['gain_vs_signature']>=-1 for r in rx),
        RX_signature_majority=sum(r['gain_vs_signature']>=0 for r in rx)>=4,
        scene_protection=all(r['cases'][joint]['gain_vs_native']>=-1 for r in groups if r['dimension']=='scene' and r['value']!='clean'),
        clean_protection=all(r['cases'][joint]['gain_vs_native']>=0 for r in groups if r['dimension']=='scene' and r['value']=='clean'))
    return dict(supports_joint_hypothesis=all(checks.values()),checks=checks)


def run(args,*,run_id=RUN,auxiliary='PA',rule=RULE):
    cases=cases_for(auxiliary);joint=auxiliary+'_signature'
    if args.output!=args.root/'runs'/run_id or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable CPU-only source diagnostic required')
    p=base.paths(args.root);clean,roles=base.source_training(p);indices=source_indices(roles);ids=f_guided.source_ids(p,roles)
    rx=np.load(p['cache']/'train/rx.npy',allow_pickle=False)[indices];day=np.load(p['cache']/'train/day.npy',allow_pickle=False)[indices]
    identity=dict(source_positions=indices,physical_ids=ids[indices],roles=roles[indices],rx=rx,day=day)
    folder=args.root/'runs'/SOURCE;meta=base.read_json(folder/'results.json')
    if (meta['F_teacher']!=f_guided.teacher_reference() or not meta['F_unchanged'] or meta['branches']!=list(BRANCHES)
        or meta['decision']!=dict(eligible=False,automatic_training=False,primary='DURA_default',candidate='PA')
        or any(meta[k]!=0 for k in ['new_G_inputs','G_checkpoint_reads','optimizer_updates','TX_labels_read','V_reads','target_reads'])):
        raise ValueError('Original actual source branches and unchanged PA FAIL required')
    with np.load(folder/'clean_components.npz',allow_pickle=False) as f:
        for k,v in dict(identity,clean=clean[indices]).items():
            if not np.array_equal(f[k],v):raise ValueError('Exact original clean source roles/order required')
        ref={k:f[k] for k in ['features','parts','logits','normalized_head_weight']}
    archives={}
    for view in VIEWS:
        with np.load(folder/f'DURA_default-{view}.npz',allow_pickle=False) as f:
            for k,v in identity.items():
                if not np.array_equal(f[k],v):raise ValueError('Matched default DURA source branches required')
            archives[view]={k:f[k] for k in ['scene','state','lock','features','parts','case_logits']}
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False);before={k:v.clone() for k,v in reader.state_dict().items()}
    args.output.mkdir();base.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,rule=rule,CPU_threads=2,GPU_workers=0))
    if not np.all(roles[indices[:2]]=='L_s'):raise ValueError('Two real L_s smoke records required')
    with torch.no_grad():
        actual,parts=components(reader,tensor(clean[indices[:2]],'cpu'))
        torch.testing.assert_close(actual,tensor(ref['features'][:2],'cpu'),atol=3e-5,rtol=2e-5)
        for i,k in enumerate(BRANCHES):torch.testing.assert_close(parts[k],tensor(ref['parts'][i,:2],'cpu'),atol=3e-5,rtol=2e-5)
        torch.testing.assert_close(reader.classify_features(actual),tensor(ref['logits'][:2],'cpu'),atol=3e-5,rtol=2e-5)
    base.write_json(args.output/'smoke.json',dict(status='PASS',real_Ls=2,F_inputs=2,head_only_inputs=2,actual_weighted_components_and_original_head_replayed=True,
        new_G_inputs=0,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0))
    print(json.dumps(dict(event='REAL_SOURCE_JOINT_SIG_SMOKE_PASS',auxiliary=auxiliary)),flush=True)
    started=time.perf_counter();results={};records=[]
    with torch.no_grad():
        weight=numpy_values(torch.nn.functional.normalize(reader.core.id_backbone.cls_head.weight,dim=1,eps=1e-4))
        np.testing.assert_allclose(weight,ref['normalized_head_weight'],atol=1e-7,rtol=1e-7)
        np.savez(args.output/'clean_reference.npz',**identity,features=ref['features'],parts=ref['parts'],logits=ref['logits'],normalized_head_weight=weight)
        for view in VIEWS:
            a=archives[view];actual=tensor(a['features'],'cpu');parts={k:tensor(a['parts'][i],'cpu') for i,k in enumerate(BRANCHES)}
            cleanF=tensor(ref['features'],'cpu');cleanparts={k:tensor(ref['parts'][i],'cpu') for i,k in enumerate(BRANCHES)}
            interventions=paired_cases(actual,parts,cleanF,cleanparts,auxiliary);features=torch.stack([interventions[k] for k in cases])
            logits=reader.classify_features(features.flatten(0,1)).reshape(5,512,6)
            auxiliary_index=2 if auxiliary=='PA' else 3
            for new,old in [(0,0),(1,auxiliary_index),(2,4),(4,5)]:torch.testing.assert_close(logits[new],tensor(a['case_logits'][old],'cpu'),atol=3e-5,rtol=2e-5)
            target=tensor(ref['logits'],'cpu').argmax(1);chosen=logits.gather(2,target[None,:,None].expand(5,-1,1)).squeeze(-1)
            other=logits.clone();other.scatter_(2,target[None,:,None].expand(5,-1,1),-torch.inf)
            values=dict(native_features=actual,native_parts=torch.stack([parts[k] for k in BRANCHES]),case_features=features,case_logits=logits,
                agreement=(logits.argmax(2)==target[None]).float(),reference_margin=chosen-other.max(2).values,
                feature_relative_MSE=(features-cleanF[None]).square().sum(2)/cleanF.square().sum(1).clamp_min(1e-4)[None])
            values={k:numpy_values(v) for k,v in values.items()}
            if any(not np.isfinite(v).all() for v in values.values()) or not np.all(values['agreement'][-1]==1):raise ValueError('Finite source values/allclean control required')
            np.savez(args.output/f'DURA_default-{view}.npz',**identity,**{k:a[k] for k in ['scene','state','lock']},**values)
            masks=[('overall','all',np.ones(512,bool))]
            for dim,v in dict(RX=rx,day=day,role=roles[indices],scene=a['scene'],state=a['state'],lock=a['lock']).items():masks.extend((dim,str(x),v==x) for x in np.unique(v))
            groups=[dict(view=view,dimension=dim,value=value,cases=summary(values,mask,cases)) for dim,value,mask in masks]
            verdict=decision(groups,joint);results[view]=dict(overall=groups[0]['cases'],decision=verdict);records.extend(groups)
            print(json.dumps(dict(event='SOURCE_JOINT_SIG_CELL_COMPLETE',auxiliary=auxiliary,view=view,overall=groups[0]['cases'],decision=verdict)),flush=True)
    if any(not torch.equal(v,before[k]) for k,v in reader.state_dict().items()) or any(p.grad is not None for p in reader.parameters()):raise ValueError('Frozen original F changed')
    verdict=dict(supports_joint_hypothesis=all(r['decision']['supports_joint_hypothesis'] for r in results.values()),automatic_training=False)
    base.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',run=run_id,source_run=SOURCE,rule=rule,source_records=512,source_L=128,source_U=384,
        primary='defaultDURA_'+auxiliary+'_plus_SIG',views=list(VIEWS),cases=list(cases),results=results,records=records,decision=verdict,
        F_teacher=f_guided.teacher_reference(),F_unchanged=True,F_inputs=0,head_only_inputs=5120,F_smoke_inputs=2,head_smoke_inputs=2,
        new_G_inputs=0,G_checkpoint_reads=0,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,seconds=time.perf_counter()-started,
        scope='Latentpairedsourcecleanoracle/notrealizableIQ/deployment/trueTXaccuracy/physicalrecoveryceiling;oldFAILsunchanged'))
    base.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',decision=verdict,formal_predictions=0,goal_complete=False))


def main(*,run_id=RUN,auxiliary='PA',rule=RULE):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['root','output']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args,run_id=run_id,auxiliary=auxiliary,rule=rule)


if __name__=='__main__':main()
