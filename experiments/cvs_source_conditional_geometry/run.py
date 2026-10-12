from pathlib import Path
import os,sys,time
import torch
from . import design as d
from .core import readout,summarize
from experiments.cvs_e20_routed_fishr.model import build
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags

def load_L(c):
    from experiments.cvs_phase1_stack.recover import verify_contract
    prepared=Path(c['parent_build_config']['preparation']);done=d.read(prepared/'preparation_complete.json')
    if done['target_access'] or done['identity_updates'] or done['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE':raise ValueError('Existing source preparation differs')
    packed=torch.load(prepared/'source_cache.pt',map_location='cpu',weights_only=False)
    verify_contract(packed['contract'],d.read(d.SOURCE));verify_contract(done['source_contract'],packed['contract'])
    L=dict(role='L_s',**{k:packed['labeled'][k] for k in ('x','y','rx','day','ids')});contract=packed['contract']
    if len(L['ids'])!=6300 or len(set(L['ids']))!=6300 or sorted(L['ids'])!=sorted(contract['role_ids']['L_s']):raise ValueError('Wrong source-L physical IDs')
    del packed
    return L,contract

def run(row_id):
    c=d.validate(d.config(next(r for r in d.rows() if r['row_id']==row_id)));out=Path(c['output']);out.mkdir(exist_ok=False)
    torch.set_num_threads(2);started=time.perf_counter();L,contract=load_L(c)
    with numerical_context(d.FULL_FP32_POLICY):
        model=build(c['parent_build_config'],'cuda:0')
        proof=model._parent_provenance
        if proof['status']!='VERIFIED' or Path(proof['checkpoint']).as_posix()!=c['checkpoint'] or proof['training_target_access']:raise ValueError('Wrong original E20 provenance')
        resolved=dict(config=c,provenance=proof,source_contract=contract,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,
            commit=(d.ROOT/'release_commit.txt').read_text().strip(),hardware=torch.cuda.get_device_name(0),backend_flags=actual_flags(),CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'])
        d.write(out/'resolved_config.json',resolved);d.write(out/'provenance.json',proof)
        print('RESOLVED '+__import__('json').dumps(resolved),flush=True);torch.cuda.reset_peak_memory_stats()
        def progress(v):
            d.write(out/'progress.json',v);print('SOURCE_READOUT '+__import__('json').dumps(v),flush=True)
        values,budget=readout(model,L,c['model_seed'],'cuda:0',c['batch_size'],progress);result=summarize(values,L,diagnostic_seed=c['diagnostic_seed'])
        assert budget['model_forwards']==50 and budget['L_occurrences']==6300 and budget['first_order_derivative_calls']==100
        assert result['cell_count']==90 and len(result['records'])==18
        d.write(out/'statistics.json',result)
        done=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',run_id=d.RUN,row_id=row_id,config=c,identity_updates=0,head_updates=0,target_access=False,U_forward=0,V_forward=0,
            records=18,cell_count=90,budget=budget,seconds=time.perf_counter()-started,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),provenance=proof)
        d.write(out/'completion.json',done);print('SOURCE_DIAGNOSTIC_COMPLETE '+__import__('json').dumps(done),flush=True)
    return done
