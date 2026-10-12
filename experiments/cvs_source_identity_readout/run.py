import os,sys,time,json
from pathlib import Path
import torch
from . import design as d
from .core import extract,summarize
from .input_policy import apply
from experiments.cvs_source_conditional_geometry.run import load_L
from experiments.cvs_e20_routed_fishr.model import build
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags

def snapshot(model):
    return {kind+':'+name:v.detach().clone() for kind,iterator in (('parameter',model.named_parameters()),('buffer',model.named_buffers())) for name,v in iterator}

def run(row_id):
    c=d.validate(d.config(next(r for r in d.rows() if r['row_id']==row_id)));out=Path(c['output']);out.mkdir(exist_ok=False)
    torch.set_num_threads(2);started=time.perf_counter();L,contract=load_L(c)
    with numerical_context(d.FULL_FP32_POLICY):
        model=build(c['parent_build_config'],'cuda:0').eval().requires_grad_(False);proof=model._parent_provenance
        assert proof['status']=='VERIFIED' and Path(proof['checkpoint']).as_posix()==c['checkpoint'] and not proof['training_target_access']
        resolved=dict(config=c,provenance=proof,source_contract=contract,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),hardware=torch.cuda.get_device_name(0),backend_flags=actual_flags(),CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'])
        d.write(out/'resolved_config.json',resolved);d.write(out/'provenance.json',proof);print('RESOLVED '+json.dumps(resolved),flush=True)
        before=snapshot(model);rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state_all();torch.cuda.reset_peak_memory_stats();results=[];budgets=[]
        for policy in c['input_policies']:
            input_evidence=dict(explicit_CFO_identity_input=True,historical_checkpoint_training_policy='explicit_CFO') if policy=='historical_explicit_CFO' else apply(model)
            def progress(v):
                value=dict(policy=policy,**v);d.write(out/'progress.json',value);print('SOURCE_READOUT '+json.dumps(value),flush=True)
            features,logits,budget=extract(model,L,'cuda:0',c['batch_size'],progress)
            assert budget['model_forwards']==50 and budget['L_occurrences']==6300
            result=summarize(features,logits,L,c);result.update(input_policy=policy,input_evidence=input_evidence);results.append(result);budgets.append(budget)
            d.write(out/f'{policy}_statistics.json',result);del features,logits
            print('READOUT_STATISTICS '+json.dumps(dict(policy=policy,records=len(result['records']),geometry=len(result['geometry']),crossfits=len(result['crossfit']))),flush=True)
        after=snapshot(model);assert before.keys()==after.keys() and all(torch.equal(before[k],after[k]) for k in before)
        assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
        assert not any(m.training for m in model.modules()) and not any(p.grad is not None for p in model.parameters())
        done=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',run_id=d.RUN,row_id=row_id,config=c,identity_updates=0,head_updates=0,target_access=False,U_forward=0,V_forward=0,
            budget=dict(model_forwards=sum(v['model_forwards'] for v in budgets),L_occurrences=sum(v['L_occurrences'] for v in budgets),first_order_derivative_calls=0,parameters_buffers_gradients_RNG_modes_unchanged=True),
            input_policies=c['input_policies'],statistics_files=[p+'_statistics.json' for p in c['input_policies']],seconds=time.perf_counter()-started,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),provenance=proof)
        d.write(out/'completion.json',done);print('SOURCE_DIAGNOSTIC_COMPLETE '+json.dumps(done),flush=True)
    return done
