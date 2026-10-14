import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import torch
from . import design as d
from experiments.cvs_source_self_challenge import audit as core
from experiments.cvs_receiver_dro_seeds import model as parent_model
from experiments.cvs_signature_stats50.predict import source_proof,own_model
from experiments.cvs_source_layer_order.probe import cohorts
from experiments.cvs_backbone_fishr50.objective import validate_layout
from experiments.cvs_equivariant_identity.precision import numerical_context


def model(c):
    d.validate(c)
    return own_model(c['source_config'],'cpu',design=d.parent,build_fn=parent_model.build)


def measure(c,L):
    d.validate(c);p=Path(c['output_root']);p.mkdir();started=time.time()
    old=cohorts(L);indices=[torch.cat((old[2*j],old[2*j+1])) for j in range(5)]
    with numerical_context(d.FULL_FP32_POLICY),torch.random.fork_rng(devices=[]):
        m=model(c);before={k:v.clone() for k,v in m.state_dict().items()};raw=[]
        d.write(p/'resolved_config.json',dict(config=c,execution_commit=(d.ROOT/'release_commit.txt').read_text().strip(),
            checkpoint_parent_run=d.parent.RUN,source_roles='EXACT_MATCH',target_read=False,
            head_fit_updates=0,optimizer_steps=0))
        for j,ix in enumerate(indices):
            y,rx,day=(L[k][ix] for k in ('y','rx','day'));ids=[L['ids'][i] for i in ix.tolist()]
            validate_layout(y,rx,day,ids,'L_s')
            result=core.measure_batch(m,L['x'][ix],y,d.MASK_SEED+j)
            raw.append(dict(cohort=j,y=y.cpu(),rx=rx.cpu(),day=day.cpu(),source_physical_ids=ids,**result))
            print(json.dumps(dict(row=c['row_id'],source_cohorts=j+1,source_packets=(j+1)*180,forward_packets=(j+1)*720)),flush=True)
        if any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()) or any(p.grad is not None for p in m.parameters()):
            raise ValueError('Read-only remaining source diagnostic changed model')
    summary=core.summarize(raw)
    report=dict(status='ANALYZED',config=c,summary=summary,qualification=core.qualification(summary),
        model_parameters_buffers_unchanged=True,source_packets=900,forward_packets=3600,
        head_fit_updates=0,optimizer_steps=0,U_IQ_read=False,U_truth_read=False,V_forward=False,
        V_gradient=False,query_read=False,first_seed_retrained=False,seconds=time.time()-started,
        limitation='Remaining2 predefined frozen source models, same900 physical packets; no training or target result')
    torch.save(raw,p/'actual_source_gradients.pt');d.write(p/'analysis.json',report)
    d.write(p/'completion.json',dict(status='ANALYZED',source_packets=900,source_only=True))
    return report


def recount(root):
    values=[];physical=[]
    for row in d.rows():
        values.append(dict(row_id=row['row_id'],**core.recount(root,design=d,row=row)))
        raw=torch.load(root/row['row_id']/'actual_source_gradients.pt',map_location='cpu',weights_only=False)
        physical.append(set(i for r in raw for i in r['source_physical_ids']))
        if any(r['mask_seed']!=d.MASK_SEED+r['cohort'] for r in raw):
            raise ValueError('Remaining source mask random seed differs')
    if physical[0]!=physical[1] or len(physical[0])!=900:
        raise ValueError('Remaining models used different source physical cohort')
    return dict(status='VERIFIED',models=2,rows=values,source_packet_model_observations=1800,
        physical_unique_packets=900,source_forward_packets=7200,locating_gradient_tensors=10,
        parameter_layer_records=40,saved_parameter_vectors=120,
        model_forward_by_recount=0,head_fit_updates=0,optimizer_steps=0,query_read=False)


def main():
    from experiments.cvs_nocfo_foundation.source import load_source
    from experiments.cvs_nocfo_foundation import design as nd
    torch.set_num_threads(2);first_source=d.require_source()
    spec=d.read(d.ROOT/'registration/experiment.json')
    if spec['run_id']!=d.RUN or [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:
        raise ValueError('Registered remaining source matrix differs')
    if d.BASE.exists():raise FileExistsError('Immutable remaining source output exists; reconcile')
    d.BASE.mkdir(parents=True)
    try:
        L,U,V,contract=load_source(nd.config(nd.rows()[0]))
        if U is not None:raise ValueError('Unlabelled tensor materialized')
        del V
        reports=[measure(d.config(r),L) for r in d.rows()]
        subprocess.run([sys.executable,'-u','-m','experiments.cvs_source_self_challenge_seeds.audit',
            '--recount',str(d.BASE)],check=True)
        decision=dict(status='PASS' if all(r['qualification']['status']=='PASS' for r in reports) else 'FAIL',
            source_only=True,fixed_rules=d.QUALIFICATION,target_metrics_used=False,
            first_source_qualification=first_source,first_seed_retrained=False,
            models=2,passed_models=sum(r['qualification']['status']=='PASS' for r in reports),
            no_automatic_training_release=True)
        d.write(d.BASE/'source_qualification.json',decision)
        d.write(d.BASE/'analysis.json',dict(status='ANALYZED',run_id=d.RUN,models=2,
            reports=reports,qualification=decision,first_seed_retrained=False,
            source_packet_model_observations=1800,physical_unique_packets=900,
            source_forward_packets=7200,head_fit_updates=0,optimizer_steps=0,query_read=False))
        d.write(d.BASE/'completion.json',d.COMPLETION);print(json.dumps(d.COMPLETION),flush=True)
    except Exception as exc:
        d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),
            no_automatic_retry=True,outputs_preserved=True));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recount',type=Path);a=p.parse_args()
    if a.recount:
        torch.set_num_threads(2);value=recount(a.recount);d.write(a.recount/'independent_artifact_recount.json',value);print(json.dumps(value))
    else:main()
