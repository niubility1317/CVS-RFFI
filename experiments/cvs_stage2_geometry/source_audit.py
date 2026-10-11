"""Complete fixed-parent source geometry before any B0--B3 identity update."""
import argparse
import json
from pathlib import Path
import time
import torch
from . import design as d
from .features import identity
from .geometry_audit import extract_responses,analyze,summarize
from experiments.cvs_stage2_decouple.common import verify_parent,build_model,IdentityView,clean
from experiments.cvs_stage2_decouple.source_counterfactual_audit import ReadOnlyState
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags


def run(seed,output=None,device='cuda:0'):
    from .actions import fit_actions,load_actions
    from .receiver import fit_receiver,trusted_covariance
    from .physics_audit import PhysicalAudit,algebraic_audit
    from experiments.cvs_phase1_stack.recover import verify_contract
    if seed not in d.SEEDS:raise ValueError('Unknown geometry seed')
    out=Path(output).parent if output else d.AUDIT_BASE/f's{seed}'
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);started=time.perf_counter();c=d.parent_config(seed)
    provenance=verify_parent(c)
    old=Path(c['preparation']);done=d.read(old/'preparation_complete.json')
    if done['config']!=c or done['target_access'] or done['status']!='PREPARATION_COMPLETE':
        raise ValueError('Original source cache provenance differs')
    packed=torch.load(old/'source_cache.pt',map_location='cpu',weights_only=False)
    verify_contract(packed['contract'],d.read(d.SOURCE));cache=packed['labeled'];vcache=packed['validation']
    if len(cache['ids'])!=6300 or sorted(cache['ids'])!=sorted(packed['contract']['role_ids']['L_s']):
        raise ValueError('Source L physical IDs differ')
    resolved=dict(seed=seed,parent_contract=c,parent_provenance=provenance,source_count=6300,
        validation_count=len(vcache['ids']),target_access=False,identity_updates=0,
        audit_run=d.AUDIT_RUN,code_commit=(d.ROOT/'release_commit.txt').read_text().strip(),
        action_fit_passes=2,receiver_rank=2,receiver_ridge=.1,
        action_risk='parent margin search heuristic, separately audited actual task risk',
        validation='forward and fixed L-fitted readout only; no V fit or parameter selection',
        device=device,hardware=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else 'CPU')
    d.write(out/'resolved_config.json',resolved);print('RESOLVED '+json.dumps(resolved),flush=True)
    with numerical_context(d.FULL_FP32_POLICY):
        model=build_model(c,device)
        with ReadOnlyState(model) as state:
            model.eval()
            response=extract_responses(model,cache,seed,device)
            cache['parent_feature']=response['parent_feature']
            actual_margin=response['parent_logits'].gather(1,cache['y'][:,None]).squeeze(1)-response['parent_logits'].masked_fill(
                torch.nn.functional.one_hot(cache['y'],6).bool(),-torch.inf).max(1).values
            if not torch.allclose(actual_margin,cache['parent_margin'],atol=2e-4,rtol=2e-5):
                raise ValueError('New joint160 forward differs from frozen original parent margin')
            action_started=time.perf_counter()
            action=fit_actions(IdentityView(model),cache,seed,device,passes=2,batch_size=128)
            torch.save(action,out/'actions.pt')
            action_seconds=time.perf_counter()-action_started
            receiver=fit_receiver(cache,seed);torch.save(receiver,out/'receiver.pt')
            physics=PhysicalAudit(device=device,dtype=torch.float64)
            pf=[];ps={};vp=[];vz=[];vl=[]
            model.eval()
            with torch.no_grad():
                for start in range(0,6300,128):
                    value=physics.features(cache['x'][start:start+128].to(device))
                    pf.append(value['features'].float().cpu())
                    for name,values in value['stats'].items():ps.setdefault(name,[]).append(values.cpu())
                    stability=physics.linear_stability(cache['x'][start:start+128].to(device),clean=value)
                    for name,values in stability['stats'].items():ps.setdefault(name,[]).append(values.cpu())
                physical=torch.cat(pf)
                for start in range(0,len(vcache['x']),256):
                    x=vcache['x'][start:start+256].to(device)
                    logits,feature=identity(model,x,verify=True);vz.append(feature.cpu());vl.append(logits.cpu())
                    vp.append(physics.features(x)['features'].float().cpu())
            validation=dict(feature=torch.cat(vz),physical=torch.cat(vp),y=vcache['y'],rx=vcache['rx'],day=vcache['day'])
            geometry=analyze(cache,response,trusted_covariance(receiver),physical,
                model.id_backbone.encoder.core.id_backbone.cls_head.weight.detach().cpu(),validation,value['feature_groups'])
            action_bank=load_actions(action,device);action_audit=[]
            with torch.no_grad():
                for role,observations,parent_logits in [('L_s',cache,response['parent_logits']),('V',vcache,torch.cat(vl))]:
                    py=observations['y'];pm=parent_logits.gather(1,py[:,None]).squeeze(1)-parent_logits.masked_fill(torch.nn.functional.one_hot(py,6).bool(),-torch.inf).max(1).values
                    for policy in ('coverage','conditional'):
                        for kind in ('linear','temporal'):
                            risk=[];marg=[];parameters=[];iq_change=[]
                            for start in range(0,len(py),128):
                                x=observations['x'][start:start+128].to(device);y=py[start:start+128].to(device)
                                ax,stats=action_bank.transform(x,kind,seed+811000+start,policy=policy)
                                logit,_=identity(model,ax)
                                margin=logit.gather(1,y[:,None]).squeeze(1)-logit.masked_fill(torch.nn.functional.one_hot(y,6).bool(),-torch.inf).max(1).values
                                risk.append((logit.argmax(1)!=y).cpu());marg.append(margin.cpu());parameters.append(stats)
                                iq_change.append(((ax-x).flatten(1).norm(dim=1)/(x.flatten(1).norm(dim=1)+1e-8)).cpu())
                            margins=torch.cat(marg);deficit=(pm.clamp_max(d.MARGIN_REQUIRED)-margins).clamp_min(0)/d.MARGIN_REQUIRED
                            action_audit.append(dict(role=role,policy=policy,kind=kind,count=len(py),errors=int(torch.cat(risk).sum()),
                                margin=summarize(margins),finite_task_violations=int((deficit>0).sum()),finite_loss=summarize(.5*deficit.square()),
                                iq_relative_change=summarize(torch.cat(iq_change)),batch_diagnostics=parameters))
                            print('ACTION_SOURCE_AUDIT',seed,role,policy,kind,len(py),flush=True)
        if not all(state.report.values()):raise ValueError('Parent modified by source geometry preparation')
        resolved['backend_flags']=actual_flags()
    # Cache only source roles, retaining full parent vectors for all160 R loss.
    torch.save(packed,out/'source_cache.pt')
    torch.save(dict(jacobian=response['jacobian'],physical_features=physical,
                    validation=validation,source_ids=cache['ids']),out/'source_responses.pt')
    result=dict(status='SOURCE_GEOMETRY_AUDIT_COMPLETE',seed=seed,count=6300,
        validation_count=len(vcache['ids']),target_access=False,model_state_restored=True,
        identity_updates=0,auxiliary_fit=True,action_fit_seconds=action_seconds,
        parent_state_invariance=state.report,parent_provenance=provenance,
        feature_contract='normalized actual final joint160; unchanged cosine logits exactly reproduced within FP32 tolerance',
        physical_algebra=algebraic_audit(),physical_contract=value['contract'],
        physical_feature_groups=value['feature_groups'],physical_stability_contract=stability['contract'],
        receiver=dict(folds=[{k:v for k,v in fold.items() if k not in ('standard_mean','standard_scale','standard_fit_ids','basis','regression','matrix')}
                             for fold in receiver['folds']],
            blocks=receiver['blocks'],diagnostics={k:v for k,v in receiver['diagnostics'].items() if k!='trusted_covariance'},
            fit_seconds=receiver['fit_seconds'],parent_feature_bytes=receiver['parent_feature_bytes']),
        physical_statistics={name:summarize(torch.cat(values)) for name,values in ps.items()},
        geometry=geometry,response_contract=response['contract'],finite_difference_convergence=response['convergence'],
        linearization=response['linearization'],interaction=response['interactions'],actions=action_audit,
        elapsed_seconds=time.perf_counter()-started,resolved_config=resolved)
    d.write(out/'audit.json',clean(result));d.write(out/'preparation_complete.json',dict(
        status='SOURCE_GEOMETRY_AUDIT_COMPLETE',seed=seed,count=6300,target_access=False,
        model_state_restored=True,source_contract=packed['contract'],parent_contract=c,
        audit=str(out/'audit.json'),identity_updates=0))
    print(json.dumps(dict(status=result['status'],seed=seed,count=6300,elapsed_seconds=result['elapsed_seconds'],target_access=False)),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',required=True,type=int)
    parser.add_argument('--output');parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();run(args.seed,args.output,args.device)
