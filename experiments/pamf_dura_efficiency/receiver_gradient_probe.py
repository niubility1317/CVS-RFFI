"""Frozen SIG-G SOURCE RX gradients, balanced over seven simulated scenes."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from experiments.pamf_dura_direct_train import signature_guided as guide,f_guided
from experiments.pamf_dura_direct_train.antithetic_noise import reuse_targets
from .receiver_gradients import balanced_indices,geometry,decision,RECEIVERS,SCENES,PER_CELL

base=guide.base
RUN='20261031-diagnostic-DURA-source-RX-gradients-s2026092701-r01'
SOURCE=guide.RUN
SOURCE_COMMIT='47591cdfe2e6e6dd7534cdea846badd942321b8f'
VIEWS=('1','20')
RULE=('Fixed original SIG-only G E30/7410/source47591, original frozenF50/defaulthead. '
 'For eachcache01/20 select12records per5sourceRXx7scene fromonefixed sourcepermutation2026103111;noTXselection. '
 'EachRX84records,420/view,840visits;actualfourphysical+SIGalpha1/cap.5training-surrogate rawparametergradients. '
 'Bothviews require>=3of10RXpairs cosine<-.05 and>=1of5RX cosinewithraw equalRXmeangradient<-.05,alldirectionsdefined. '
 'Nooptimizer/G-Fstateupdate/TX-V-target. Coarse-action-priormodules descriptiveonly;noalternateprimary. '
 'PasssupportsseparateRX-balancedlearninghypothesis,notautomaticGtraining. '
 'Scene-balancednotTX-balanced;localfixedcap derivativesnotAdamWstep/trajectory/trueTXgain/hardware/physicalrecoverylimit.')


def parameter_layout(model):
    names=[];sizes=[];groups=[];parameters=[]
    for name,p in model.named_parameters():
        group=('coarse' if name.startswith('frozen.coarse.') else
               'action' if name.startswith('frozen.estimator.') else
               'prior' if name.startswith('prior.') else None)
        if group is None:raise ValueError('Original three-module G only')
        names.append(name);sizes.append(p.numel());groups.append(group);parameters.append(p)
    if sum(sizes)!=266951:raise ValueError('Exact original SIG G required')
    tags=np.repeat(np.asarray(groups),sizes)
    return parameters,dict(parameter_names=np.asarray(names),parameter_sizes=np.asarray(sizes,np.int64),parameter_groups=np.asarray(groups)),tags


def gradients(model,reader,parameters,received,clean,action,variance,target):
    generator=torch.Generator().manual_seed(2026103112)
    probes=[torch.complex(torch.randn(len(clean),256,generator=generator),torch.randn(len(clean),256,generator=generator)) for _ in range(2)]
    loss,terms,restored=guide.objective(model,reader,received,clean,action,variance,probes,target)
    derivative=torch.autograd.grad(loss.mean(),parameters,allow_unused=True)
    flat=torch.cat([(torch.zeros_like(p) if g is None else g.detach()).flatten() for p,g in zip(parameters,derivative)])
    if not torch.isfinite(flat).all():raise ValueError('Finite raw G gradient required')
    return base.numpy_values(flat),{k:float(v.detach().mean()) for k,v in dict(loss=loss,**terms).items()},sum(g is None for g in derivative),restored.detach()


def run(args):
    if args.output!=args.root/'runs'/RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('New immutable CPU-only source diagnostic required')
    guide.configure();p=base.paths(args.root);clean,roles=base.source_training(p);ids=f_guided.source_ids(p,roles)
    if p['output'].name!=SOURCE:raise ValueError('Fixed original SIG source G only')
    done=base.read_json(p['output']/'scratch_joint/training_complete.json')
    expected=dict(status='SOURCE_TRAINED',epoch=30,updates=7410,parameters=266951,G_ancestor_epochs=0,F_unchanged=True,training_TX_labels_read=0,V_reads=0,target_reads=0)
    if any(done.get(k)!=v for k,v in expected.items()) or done['F_teacher']!=f_guided.teacher_reference():raise ValueError('Legal fixed source SIG checkpoint required')
    model=guide.load_model(p['output']/'scratch_joint/final.pt','cpu',arm='scratch_joint',commit=SOURCE_COMMIT).eval().requires_grad_(True)
    reader,rng_info=guide.load_teacher_preserving_rng(p,'cpu');targets,reuse=reuse_targets(p,reader,clean,roles)
    parameters,layout,tags=parameter_layout(model);beforeG={k:v.clone() for k,v in model.state_dict().items()};beforeF={k:v.clone() for k,v in reader.state_dict().items()}
    rx=np.asarray(np.load(p['cache']/'train/rx.npy',allow_pickle=False),np.int64);day=np.load(p['cache']/'train/day.npy',allow_pickle=False)
    order=np.random.default_rng(2026103111).permutation(63000);matrices={};scenes={}
    for view in VIEWS:
        with np.load(p['cache']/'epochs'/f'{int(view):02d}'/'groups.npz',allow_pickle=False) as f:scenes[view]=f['scene']
        matrices[view]=balanced_indices(rx,scenes[view],order)
    args.output.mkdir();base.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit,rule=RULE,CPU_threads=2,GPU_workers=0))
    # Matrices are fixed from source metadata before any new model output.
    np.savez(args.output/'source_plan.npz',**layout,**{f'indices_{v}':m for v,m in matrices.items()})
    smoke_indices=np.flatnonzero((roles=='L_s')&(scenes['1']!='clean'))[:2]
    if len(smoke_indices)!=2:raise ValueError('Two real nonclean source L_s required')
    def batch(view,indices):
        folder=p['cache']/'epochs'/f'{int(view):02d}'
        values={k:base.tensor(np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False)[indices],'cpu') for k in ['received','kernel','coefficients','variance']}
        return values,base.tensor(clean[indices],'cpu'),base.tensor(targets[indices],'cpu')
    values,x,target=batch('1',smoke_indices)
    with torch.no_grad():
        features,parts=guide.components(reader,x);torch.testing.assert_close(parts['signature'],target,atol=3e-5,rtol=2e-5)
        clean_logits=reader.classify_features(features)
    g,_,_,restored=gradients(model,reader,parameters,values['received'],x,base.supervised_action(values['kernel'],values['coefficients']),values['variance'],target)
    if np.linalg.norm(g)<=1e-12:raise ValueError('Real source smoke must exercise a nonzero G gradient')
    with torch.no_grad():scores=reader(restored)
    if clean_logits.shape!=scores.shape or scores.shape!=(2,6) or not torch.isfinite(scores).all():raise ValueError('Original F/defaulthead smoke failed')
    base.write_json(args.output/'smoke.json',dict(status='PASS',real_nonclean_Ls=2,strict_G_E30=True,actual_original_SIG_and_head=True,nonzero_finite_G_gradient=True,
        G_inputs=2,F_full_inputs=4,F_prefix_inputs=2,F_input_VJP_inputs=2,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0))
    print(json.dumps(dict(event='REAL_SOURCE_RX_GRADIENT_SMOKE_PASS')),flush=True)
    started=time.perf_counter();results={};records=[]
    for view in VIEWS:
        matrix=matrices[view];all_gradients=[];losses=[];unused=[]
        for receiver,indices in zip(RECEIVERS,matrix):
            values,x,target=batch(view,indices)
            g,terms,n,_=gradients(model,reader,parameters,values['received'],x,base.supervised_action(values['kernel'],values['coefficients']),values['variance'],target)
            all_gradients.append(g);losses.append(terms);unused.append(n)
        all_gradients=np.stack(all_gradients);summaries={'full':geometry(all_gradients)}
        for name in ('coarse','action','prior'):summaries[name]=geometry(all_gradients[:,tags==name])
        np.savez(args.output/f'gradients-{view}.npz',gradients=all_gradients,source_positions=matrix,physical_ids=ids[matrix],roles=roles[matrix],rx=rx[matrix],day=day[matrix],scene=scenes[view][matrix])
        results[view]=dict(geometry=summaries,losses={str(r):v for r,v in zip(RECEIVERS,losses)},unused_parameter_tensors=unused,
            L_visits=int((roles[matrix]=='L_s').sum()),U_visits=int((roles[matrix]=='U_s').sum()))
        records.extend(dict(view=view,RX=receiver,source_records=84,losses=loss,unused_parameter_tensors=u) for receiver,loss,u in zip(RECEIVERS,losses,unused))
        print(json.dumps(dict(event='SOURCE_RX_GRADIENT_VIEW_COMPLETE',view=view,full=summaries['full'])),flush=True)
    if (any(not torch.equal(v,beforeG[k]) for k,v in model.state_dict().items()) or any(not torch.equal(v,beforeF[k]) for k,v in reader.state_dict().items())
        or any(p.grad is not None for p in [*model.parameters(),*reader.parameters()])):raise ValueError('Frozen G/F changed or accumulated parameter gradients')
    verdict=decision({v:r['geometry']['full'] for v,r in results.items()})
    unique=np.unique(np.concatenate([m.flatten() for m in matrices.values()])).size
    base.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',run=RUN,rule=RULE,source_G_run=SOURCE,source_G_commit=SOURCE_COMMIT,
        G_parameters=266951,parameter_group_counts={name:int((tags==name).sum()) for name in ('coarse','action','prior')},source_views=list(VIEWS),source_RX=list(RECEIVERS),source_scenes=list(SCENES),
        per_RX_scene=PER_CELL,source_observations=840,source_unique_physical_records=int(unique),source_L_visits=sum(r['L_visits'] for r in results.values()),source_U_visits=sum(r['U_visits'] for r in results.values()),
        selection_seed=2026103111,probe_seed=2026103112,probe_reset_per_RX_view=True,results=results,records=records,decision=verdict,
        G_inputs=840,F_full_inputs=0,F_prefix_inputs=840,F_input_VJP_inputs=840,G_smoke_inputs=2,F_smoke_full_inputs=4,F_smoke_prefix_inputs=2,F_smoke_VJP_inputs=2,
        G_checkpoint_reads=1,F_checkpoint_reads=1,F_teacher=f_guided.teacher_reference(),G_unchanged=True,F_unchanged=True,source_SIG_cache=reuse,teacher_load=rng_info,
        optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,seconds=time.perf_counter()-started,
        scope='ScenebalancednotTXbalanced;rawlocalfixedcaptraining-surrogateparametergradients/notAdamWstep/trajectory/TXaccuracy/hardware/recoverylimit;oldFAILspreserved'))
    base.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',decision=verdict,formal_predictions=0,goal_complete=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['root','output']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);run(args)


if __name__=='__main__':main()
