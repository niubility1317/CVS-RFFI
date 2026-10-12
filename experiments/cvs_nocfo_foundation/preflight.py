import argparse,tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from . import design as d
from .model import build,groups
from .source import expected_counts

def local():
    results=[];torch.set_num_threads(2)
    for row in d.rows():
        c=d.validate(d.config(row));assert not c['unlabeled'] and c['teacher']['source']=='none' and not c['checkpoint_sources'] and not c['multilevel']['enabled'] and c['label_smoothing']==0
        counts=expected_counts(c);assert counts['identity_updates']==4440 and counts['L_packets']==568320 and counts['U_packets']==counts['auxiliary_forwards']==counts['sam_steps']==counts['hidden_mix_steps']==0
        with patch('torch.load',side_effect=AssertionError('Student initial weights loaded')):model=build(c,'cpu')
        pg,_=groups(model,c);model.eval();assert not hasattr(model,'multilevel') and not model._identity_input_policy['explicit_CFO_identity_input']
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'own.pt';payload=dict(model=model.state_dict(),baseline_args=model._stage2_baseline_args,config=c,model_seed=c['model_seed'],epoch=20,total_epochs=20,parent_training_epochs=0,incremental_epochs=20,optimizer_steps=4440,parent_checkpoint=None,checkpoint_selection='final_only',target_access=False,run_id=c['run_id'],candidate_id=c['row_id'],parent_provenance=model._parent_provenance,scratch_only=True,scratch_scope='student_identity_only',reference_E20_loaded=False,teacher_provenance={},checkpoint_sources=[])
            torch.save(payload,p);restored=build(c,'cpu',p).eval()
            with torch.no_grad():assert torch.equal(model(torch.zeros(2,2,256)),restored(torch.zeros(2,2,256)))
            from .evaluate import checkpoint_provenance
            checkpoint_provenance(c,payload)
        results.append(dict(row_id=row['row_id'],counts=counts,no_initial_checkpoint=True,strict_own_checkpoint_roundtrip=True,explicit_CFO=False))
    return dict(status='PASS',rows=results,target_access=False)

def preflight():
    from .source import load_source
    from .objective import objective
    from .update import take_update
    from experiments.cvs_equivariant_identity.precision import numerical_context
    result=local();spec=d.read(d.ROOT/'registration/experiment.json');assert spec['run_id']==d.RUN and [r['config'] for r in spec['rows']]==[d.config(r) for r in d.rows()]
    evidence=[]
    with numerical_context(d.FULL_FP32_POLICY):
        for row in d.rows():
            c=d.config(row);L,U,V,contract=load_source(c);assert U is None
            model=build(c,'cpu');pg,_=groups(model,c);model.train();optimizer=torch.optim.AdamW(pg,weight_decay=c['optimizer']['weight_decay']);idx=torch.arange(32)
            loss,_,_,details,_,norm,update=take_update(c,model,optimizer,lambda:objective(c,model,None,L,idx,None,None,1,'cpu'))
            assert torch.isfinite(loss) and not update['sam_applied'] and details['U_packets']==details['auxiliary_forwards']==0 and all(int(v['step'])==1 for v in optimizer.state.values())
            evidence.append(dict(row_id=row['row_id'],actual_source_L=len(L['ids']),actual_V_count=len(V['ids']),V_backward=False,U_cache_open=False,explicit_CFO=False,real_L_backward=True,updates=1,gradient_norm=norm))
    result.update(actual_source_evidence=evidence);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',required=True);a=p.parse_args();d.write(a.output,preflight() if a.remote_preflight else local())
