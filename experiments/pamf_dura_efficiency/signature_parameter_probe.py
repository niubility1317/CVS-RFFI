"""SOURCE physical versus capped SIG derivatives in the unchanged G parameters."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from experiments.pamf_dura_direct_train import signature_guided as guide,f_guided
from experiments.pamf_dura_direct_train.antithetic_noise import reuse_targets
from .receiver_gradient_probe import parameter_layout,SOURCE,SOURCE_COMMIT,VIEWS
from .signature_parameters import split_parameters,component_summary,decision

base=guide.base
RUN='20261031-diagnostic-DURA-source-SIG-parameter-s2026092701-r01'
COHORT='20261031-diagnostic-DURA-source-RX-gradients-s2026092701-r01'
RULE=('Fixed original SIG-onlyG/source47591/E30/7410 andoriginalF50/defaulthead;reuseRX-gradientcohort840visits/477records/cache01and20. '
 'SplitactualfourphysicaltermsandcappedSIGsurrogateparameterderivativesfromsameGforward;sumreplayoldfullgradientrelL2<=2e-5. '
 'PrimarysmallSIGcontributiononlyifBOTHviews:fullGphysicalnormdefinedforall5RXandraw equalRXpool;atleast4of5RX SIG/physicalnormratio<.1andpooledratio<.1. '
 'Modulesdescriptiveonly/noalternateprimary/threshold/parentsearch. Nooptimizer/TX-V-target/G-Fstateupdate. '
 'Passsupportsseparateparameterbalancinghypothesisonly/notautomaticGtraining. '
 'Localfixedcaps/rawgradient-notAdamWstep/trajectory/TXimportance/hardware/physicalceiling;smallratioisnotproofboostinghelps.')


def collect(model,reader,parameters,received,clean,action,variance,target):
    generator=torch.Generator().manual_seed(2026103112)
    probes=[torch.complex(torch.randn(len(clean),256,generator=generator),torch.randn(len(clean),256,generator=generator)) for _ in range(2)]
    loss,terms,restored=guide.objective(model,reader,received,clean,action,variance,probes,target)
    physical=terms['iq_nmse']+.1*terms['prior_nmse']+.05*terms['spectral_error']+.05*terms['action_probe_nmse']
    torch.testing.assert_close(loss,physical+terms[guide.GUIDE],atol=2e-7,rtol=2e-6)
    p,s,pu,su=split_parameters(physical,terms[guide.GUIDE],parameters)
    values={k:float(v.detach().mean()) for k,v in dict(loss=loss,physical_loss=physical,**terms).items()}
    return base.numpy_values(p),base.numpy_values(s),values,pu,su,restored.detach()


def run(args):
    if args.output!=args.root/'runs'/RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable CPU-only source decomposition required')
    guide.configure();p=base.paths(args.root);clean,roles=base.source_training(p);ids=f_guided.source_ids(p,roles)
    cohort=args.root/'runs'/COHORT;meta=base.read_json(cohort/'results.json')
    if (meta['source_G_run']!=SOURCE or meta['source_G_commit']!=SOURCE_COMMIT or meta['G_parameters']!=266951
        or meta['F_teacher']!=f_guided.teacher_reference() or not meta['G_unchanged'] or not meta['F_unchanged']
        or meta['source_observations']!=840 or meta['source_unique_physical_records']!=477
        or meta['decision']['supports_RX_conflict_hypothesis']
        or any(meta[k]!=0 for k in ['optimizer_updates','TX_labels_read','V_reads','target_reads'])):
        raise ValueError('Fixed legal SIG-G/source-RX cohort and old FAIL required')
    model=guide.load_model(p['output']/'scratch_joint/final.pt','cpu',arm='scratch_joint',commit=SOURCE_COMMIT).eval().requires_grad_(True)
    reader,rng_info=guide.load_teacher_preserving_rng(p,'cpu');targets,reuse=reuse_targets(p,reader,clean,roles)
    parameters,layout,tags=parameter_layout(model);beforeG={k:v.clone() for k,v in model.state_dict().items()};beforeF={k:v.clone() for k,v in reader.state_dict().items()}
    with np.load(cohort/'source_plan.npz',allow_pickle=False) as f:plan={k:f[k] for k in f.files}
    for k,v in layout.items():
        if not np.array_equal(v,plan[k]):raise ValueError('Matched original unique G parameter layout required')
    archives={}
    for view in VIEWS:
        with np.load(cohort/f'gradients-{view}.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
        indices=plan['indices_'+view]
        if indices.shape!=(5,84) or not np.array_equal(indices,a['source_positions']):raise ValueError('Exact fixed source cohort required')
        for k,v in [('physical_ids',ids),('roles',roles)]:
            if not np.array_equal(a[k],v[indices]):raise ValueError('Original source roles/IDs changed')
        if a['gradients'].shape!=(5,266951) or not np.isfinite(a['gradients']).all():raise ValueError('Finite original total G gradients required')
        archives[view]=a
    args.output.mkdir();base.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,rule=RULE,CPU_threads=2,GPU_workers=0))
    np.savez(args.output/'source_plan.npz',**plan)
    def batch(view,indices):
        folder=p['cache']/'epochs'/f'{int(view):02d}'
        values={k:base.tensor(np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False)[indices],'cpu') for k in ['received','kernel','coefficients','variance']}
        return values,base.tensor(clean[indices],'cpu'),base.tensor(targets[indices],'cpu')
    a=archives['1'];flat=a['source_positions'].flatten();take=flat[(a['roles'].flatten()=='L_s')&(a['scene'].flatten()!='clean')][:2]
    if len(take)!=2:raise ValueError('Two real nonclean L_s in fixed source cohort required')
    values,x,target=batch('1',take)
    with torch.no_grad():
        features,parts=guide.components(reader,x);torch.testing.assert_close(parts['signature'],target,atol=3e-5,rtol=2e-5)
        scores_clean=reader.classify_features(features)
    pg,sg,_,_,_,restored=collect(model,reader,parameters,values['received'],x,base.supervised_action(values['kernel'],values['coefficients']),values['variance'],target)
    with torch.no_grad():scores=reader(restored)
    if (scores.shape!=(2,6) or scores_clean.shape!=scores.shape or not torch.isfinite(scores).all()
        or np.linalg.norm(pg)<=1e-12 or np.linalg.norm(sg)<=1e-12):raise ValueError('Actual physical/SIG derivative and original head smoke failed')
    base.write_json(args.output/'smoke.json',dict(status='PASS',real_nonclean_Ls=2,strict_G_E30=True,actual_original_SIG_and_head=True,
        physical_and_SIG_nonzero_finite=True,G_inputs=2,F_full_inputs=4,F_prefix_inputs=2,F_input_VJP_inputs=2,G_parameter_VJP_inputs=4,
        optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0))
    print(json.dumps(dict(event='REAL_SOURCE_SIG_PARAMETER_SMOKE_PASS')),flush=True)
    started=time.perf_counter();results={};records=[]
    for view,a in archives.items():
        physical=[];signature=[];losses=[];unused=[]
        for indices in a['source_positions']:
            values,x,target=batch(view,indices)
            pg,sg,terms,pu,su,_=collect(model,reader,parameters,values['received'],x,base.supervised_action(values['kernel'],values['coefficients']),values['variance'],target)
            physical.append(pg);signature.append(sg);losses.append(terms);unused.append(dict(physical=pu,SIG=su))
        physical=np.stack(physical);signature=np.stack(signature);total=physical+signature;old=a['gradients']
        relative=float(np.linalg.norm((total-old).astype(np.float64))/max(np.linalg.norm(old.astype(np.float64)),1e-12))
        if relative>2e-5:raise ValueError('Physical plus SIG no longer reproduces fixed source total gradient')
        summaries={'full':component_summary(physical,signature)}
        for name in ['coarse','action','prior']:summaries[name]=component_summary(physical[:,tags==name],signature[:,tags==name])
        np.savez(args.output/f'decomposition-{view}.npz',physical_gradients=physical,SIG_gradients=signature,
            **{k:a[k] for k in ['source_positions','physical_ids','roles','rx','day','scene']})
        results[view]=dict(components=summaries,losses={str(r):v for r,v in zip(meta['source_RX'],losses)},unused_parameter_tensors=unused,
            total_gradient_replay_relative_L2_error=relative,total_gradient_replay_maximum_abs_error=float(np.abs(total-old).max()))
        records.extend(dict(view=view,RX=r,source_records=84,losses=loss) for r,loss in zip(meta['source_RX'],losses))
        print(json.dumps(dict(event='SOURCE_SIG_PARAMETER_VIEW_COMPLETE',view=view,full=summaries['full'],total_replay_relative_error=relative)),flush=True)
    if (any(not torch.equal(v,beforeG[k]) for k,v in model.state_dict().items()) or any(not torch.equal(v,beforeF[k]) for k,v in reader.state_dict().items())
        or any(p.grad is not None for p in [*model.parameters(),*reader.parameters()])):raise ValueError('Frozen G/F changed or accumulated parameter gradients')
    verdict=decision({v:r['components']['full'] for v,r in results.items()})
    base.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',run=RUN,rule=RULE,source_cohort_run=COHORT,source_G_run=SOURCE,source_G_commit=SOURCE_COMMIT,
        G_parameters=266951,parameter_group_counts=meta['parameter_group_counts'],source_views=list(VIEWS),source_RX=meta['source_RX'],source_scenes=meta['source_scenes'],
        source_observations=840,source_unique_physical_records=477,source_L_visits=86,source_U_visits=754,results=results,records=records,decision=verdict,
        G_inputs=840,F_full_inputs=0,F_prefix_inputs=840,F_input_VJP_inputs=840,G_parameter_VJP_inputs=1680,G_smoke_inputs=2,F_smoke_full_inputs=4,F_smoke_prefix_inputs=2,F_smoke_VJP_inputs=2,G_smoke_parameter_VJP_inputs=4,
        G_checkpoint_reads=1,F_checkpoint_reads=1,F_teacher=f_guided.teacher_reference(),G_unchanged=True,F_unchanged=True,source_SIG_cache=reuse,teacher_load=rng_info,
        optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,seconds=time.perf_counter()-started,
        scope='Fixedcaprawlocalbatchmeanparameterderivatives/mayincludeJacobianandwithinbatchcancellation/notAdamW/TXimportance/trajectory/hardware/recoverylimit;oldFAILspreserved'))
    base.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',decision=verdict,formal_predictions=0,goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['root','output']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
