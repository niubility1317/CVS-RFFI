import argparse,torch
from . import design as d
from .run import modules,load_model,batch
from experiments.cvs_equivariant_identity.precision import numerical_context


def preflight():
    spec=d.read(d.ROOT/'registration/experiment.json')
    if spec['run_id']!=d.RUN or [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Preregistration differs')
    torch.set_num_threads(2);proofs=[];audit=None
    first=d.config(d.rows()[0])['models'][0]
    L,U,V,contract=modules(first['family'])['source'].load_source(first['config']);del V
    if 'y' in U:raise ValueError('U truth forbidden')
    with numerical_context(d.FULL_FP32_POLICY):
        for row in d.rows():
            for entry in d.config(row)['models']:
                model,saved,m,proof=load_model(entry,'cpu');model.eval()
                with torch.no_grad():logits=model(L['x'][:2])
                if logits.shape!=(2,6) or not bool(torch.isfinite(logits).all()):raise ValueError('Real source forward failed')
                proof.update(model_key=entry['key'],seed=row['model_seed'],source_smoke_shape=list(logits.shape));proofs.append(proof)
                if audit is None:
                    model.train();bank=m['peer_cache'].load(entry['config'],U)
                    audit=batch(model,saved,m,entry['config'],L,U,bank,0,'cpu');del bank
                del model,saved
    return dict(status='PASS',models=proofs,source_contract=contract,real_source_batch=audit,
        model_forwards=18+audit['model_forwards'],L_occurrences=36+audit['L_occurrences'],U_occurrences=audit['U_occurrences'],
        temporary_optimizer_steps=8,persistent_identity_updates=0,target_access=False,U_truth=False,V_forward=False,V_deserialized_then_discarded=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',required=True);a=p.parse_args();d.write(a.output,preflight())
