"""Frozen original E20 default head on full legal V; never query/target input."""
import gzip
import json
import os
from pathlib import Path
import sys
import time
from collections import Counter
import torch
from torch.nn import functional as F
from . import probe_design as d
from . import response as r
from experiments.cvs_mixed_iq20.probe import validate_source,summary


@torch.no_grad()
def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Registered CPU-only source probe')
    if d.BASE.exists():raise FileExistsError('Preserve existing finite-IQ source output')
    d.validate(d.read(d.ROOT/'experiments/cvs_source_response20/probe_registration.json')['rows'][0]['config'])
    torch.set_num_threads(2)
    from experiments.cvs_sigtime_rdrop_legacy import design as legacy
    from experiments.cvs_sigtime_rdrop_legacy.provenance import own_model,source_proof
    from experiments.cvs_phase1_stack.recover import verify_contract
    from experiments.cvs_equivariant_identity.precision import numerical_context
    c=legacy.config(next(v for v in legacy.rows() if v['row_id']=='E20_RDROP-s2026092701'))
    if c['output_root']!=d.PARENT or c['epochs']!=20:raise ValueError('Wrong fixed E20 source parent')
    source_proof(c)
    d.BASE.mkdir();started=time.perf_counter()
    d.write(d.BASE/'dispatcher.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
    try:
        with numerical_context(legacy.FULL_FP32_POLICY):
            model=own_model(c,'cpu')
            smoke=model.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
            if smoke.shape!=(2,6) or not bool(torch.isfinite(smoke).all()):raise ValueError('Real default-head smoke failed')
            d.write(d.BASE/'source_smoke.json',dict(status='PASS',query_read=False,forward_packets=2,checkpoint=d.PARENT+'/final_ssdg.pth'))
            print('SMOKE PASS real E20 RDrop default head; source probe continues',flush=True)
            prepared=d.read(Path(d.PREPARATION)/'preparation_complete.json')
            if prepared.get('target_access') is not False or prepared.get('identity_updates')!=0 or prepared.get('status')!='SOURCE_GEOMETRY_AUDIT_COMPLETE':
                raise ValueError('Source preparation differs')
            packed=torch.load(Path(d.PREPARATION)/'source_cache.pt',map_location='cpu',weights_only=False)
            contract=d.read(d.SOURCE);verify_contract(packed['contract'],contract);verify_contract(prepared['source_contract'],contract)
            V=validate_source({k:packed['validation'][k] for k in ('x','y','rx','day','ids')},contract);del packed
            args=d.read(Path(d.PARENT)/'actual_native_args.json');args=args.get('args',args)
            if str(args.get('wisig_equalized'))!='1' or args.get('wisig_out_len')!=256:raise ValueError('Source IQ contract differs')
            source=d.read(d.ROOT/'source_bank_input.json')
            if source['source_run']!=d.config()['source_bank_run'] or source['primary']!='rectangular/DCT1to8' or source['L_count']!=6300 or source['U_used'] or source['V_used'] or source['target_access']:
                raise ValueError('Only predeclared L-derived prediction array permitted')
            bank=r.bank(source['predicted_contrasts'])
            d.write(d.BASE/'finite_bank.json',{k:dict(real=v.real.tolist(),imag=v.imag.tolist()) for k,v in bank.items() if k in ('empirical','rotated')})
            model.eval().requires_grad_(False);state={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
            d.write(d.BASE/'resolved_config.json',dict(config=d.config(),commit=(d.ROOT/'release_commit.txt').read_text().strip(),
                checkpoint_provenance='VERIFIED fixed scratch E20/fullL6300/U56700/V27000/final-only; frozen diagnosis, no inheritance',
                parent_training_code=d.read(Path(d.PARENT)/'resolved_config.json')['commit'],parent_config=c,
                module_files={k:str(sys.modules[k].__file__) for k in ('model','leo_practical.channel','leo_practical.residual') if k in sys.modules},
                identity_input_policy=model._identity_input_policy,python=sys.executable,torch_version=torch.__version__,source_role='V',
                U_labels_used=False,target_access=False,source_bank=source['source_run'],CUDA_VISIBLE_DEVICES=''))
            outputs={};waves={};sessions={};coverage=Counter();forwards=2
            for condition in d.CONDITIONS:
                predictions=[];margins=[];power_error=mse=interior_error=0.
                kind='empirical' if condition.startswith('EMP') else 'rotated';epoch=1 if '_E1_' in condition else 20
                boundary='edge' if condition.endswith('_EDGE') else 'reflect'
                for start in range(0,27000,256):
                    end=min(start+256,27000);x=V['x'][start:end];y=V['y'][start:end]
                    if condition=='RAW':value=x
                    else:
                        keys=[(epoch,int(rx),int(day),1,0) for rx,day in zip(V['rx'][start:end].tolist(),V['day'][start:end].tolist())]
                        h=r.kernels(bank,kind,keys);value=r.apply_torch(x,h,boundary)
                        for key,tx in zip(keys,y.tolist()):
                            name=f'E{epoch}/RX{key[1]}/D{key[2]}/eq1/view0';parameters=list(r.session(*key))
                            if name in sessions and sessions[name]!=parameters:raise ValueError('Session changed across TX/condition')
                            sessions[name]=parameters;coverage[(condition,key[1],key[2],int(tx))]+=1
                        if start==0 or start%2048==0:
                            first=r.apply_torch(x,h,'reflect',False);other=r.apply_torch(x,h,'edge',False)
                            interior_error=max(interior_error,float((first[:,:,16:240]-other[:,:,16:240]).abs().max()))
                            if interior_error>1e-6:raise ValueError('Unscaled boundary leaked into valid region')
                    p=x.square().sum(1).mean(1);q=value.square().sum(1).mean(1)
                    power_error=max(power_error,float(((p-q).abs()/p.clamp_min(1e-12)).max()));mse+=float((value-x).square().sum())
                    logits=model.forward_identity_only(value)['tx_logits']
                    if logits.shape!=(len(x),6) or not bool(torch.isfinite(logits).all()):raise ValueError('Default head nonfinite')
                    wrong=logits.masked_fill(F.one_hot(y,6).bool(),-torch.inf).max(1).values
                    predictions.append(logits.argmax(1));margins.append((logits.gather(1,y[:,None]).squeeze(1)-wrong)/30);forwards+=len(x)
                    if start%2048==0:d.write(d.BASE/'progress.json',dict(condition=condition,packets=end,forward_packets=forwards,seconds=time.perf_counter()-started))
                outputs[condition]=(torch.cat(predictions),torch.cat(margins));waves[condition]=dict(max_relative_power_error=power_error,
                    waveform_MSE=mse/(27000*2*256),unscaled_boundary_interior_max_abs_error=interior_error)
                print('SOURCE '+condition+' complete forward_packets='+str(forwards),flush=True)
            if forwards!=d.FORWARDS or not torch.equal(rng,torch.get_rng_state()):raise ValueError('Forward budget or RNG changed')
            if any(not torch.equal(v,state[k]) for k,v in model.state_dict().items()) or any(p.grad is not None for p in model.parameters()):raise ValueError('Frozen state changed')
            if len(coverage)!=6*15*6 or set(coverage.values())!={300}:raise ValueError('Exact V session/TX coverage differs')
            reference=outputs['RAW'][0];results={}
            with gzip.open(d.BASE/'sample_decisions.jsonl.gz','xt',encoding='utf-8') as stream:
                for name,(pred,margin) in outputs.items():
                    strata=[]
                    for field,tensor in (('RX',V['rx']),('day',V['day']),('TX',V['y'])):
                        for group in tensor.unique().tolist():
                            mask=tensor==group;strata.append(dict(dimension=field,value=int(group),**summary(pred[mask],V['y'][mask],margin[mask],reference[mask])))
                    results[name]=dict(overall=summary(pred,V['y'],margin,reference),strata=strata,waveform=waves[name])
                    for i,pid in enumerate(V['ids']):stream.write(json.dumps(dict(condition=name,physical_id=pid,RX=int(V['rx'][i]),day=int(V['day'][i]),
                        TX=int(V['y'][i]),prediction=int(pred[i]),correct=bool(pred[i]==V['y'][i]),margin=float(margin[i])),allow_nan=False)+'\n')
            value=dict(status='ANALYZED',source_only=True,target_access=False,conditions=results,source_gate=d.gate(results),virtual_sessions=sessions,
                coverage=[dict(condition=k[0],RX=k[1],day=k[2],TX=k[3],count=v) for k,v in sorted(coverage.items())],
                forward_packets=forwards,seconds=time.perf_counter()-started,optimizer_updates=0,EMA_updates=0,head_fit_updates=0,
                model_state_unchanged=True,random_state_unchanged=True,new_models=0,target_predictions=0,
                commit=(d.ROOT/'release_commit.txt').read_text().strip(),config=d.config(),
                parent_cost=dict(epochs=20,updates=4440,original_training_seconds=d.read(Path(d.PARENT)/'completion.json').get('elapsed_seconds')))
            d.write(d.BASE/'results.json',value);d.write(d.BASE/'complete.json',{k:value[k] for k in ('status','source_only','target_access','source_gate','forward_packets','seconds','optimizer_updates')})
            print('FINITE_IQ_SOURCE_COMPLETE '+json.dumps(value['source_gate']),flush=True)
    except Exception as exc:
        d.write(d.BASE/'failure.json',dict(status='FAILED',error=str(exc),no_automatic_retry=True,target_access=False));raise
