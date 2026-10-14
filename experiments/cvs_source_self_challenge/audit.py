import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import torch
from . import design as d
from .objective import feature_mask,selected_masks,ce,gradient_stats
from experiments.cvs_receiver_dro50 import model as parent_model
from experiments.cvs_signature_stats50.predict import source_proof,own_model
from experiments.cvs_source_layer_order.probe import cohorts
from experiments.cvs_backbone_fishr50.objective import validate_layout
from experiments.cvs_equivariant_identity.precision import numerical_context


def source_model():
    return own_model(d.source_config(),'cpu',design=d.parent,build_fn=parent_model.build)


def prepare(model):
    model.eval().requires_grad_(False)
    named=dict(model.named_parameters());params=[named[n] for n in d.PARAMETERS]
    for p in params:p.requires_grad_(True)
    return params


def measure_batch(model,x,y,seed):
    params=prepare(model);module=dict(model.named_modules())[d.config(d.rows()[0])['layer']]
    state={'mask':None,'activation':None}
    def hook(module,inputs,output):
        if output.shape!=(len(x),96,32) or not torch.isfinite(output).all():
            raise ValueError('Actual deployed t2 shape or values differ')
        state['activation']=output
        return output if state['mask'] is None else output*state['mask'][:,None,:]
    handle=module.register_forward_hook(hook)
    try:
        original=model.forward_identity_only(x)['tx_logits']
        activation=state['activation']
        if (activation<0).any():raise ValueError('Actual post-ReLU t2 is signed; source RSC layer assumption differs')
        locating=torch.autograd.grad(original.gather(1,y[:,None]).sum(),activation,retain_graph=True)[0]
        mask=feature_mask(locating)
        logits={'ORIGINAL':original.detach()};losses={'ORIGINAL':float(ce(original,y).detach())}
        vectors={'ORIGINAL':[v.detach().flatten().cpu() for v in torch.autograd.grad(ce(original,y),params)]}
        state['mask']=mask
        with torch.no_grad():spatial=model.forward_identity_only(x)['tx_logits']
        logits['SPATIAL_ALL']=spatial.detach()
        chosen,random,selected,drop=selected_masks(logits['ORIGINAL'],spatial,y,mask,seed)
        for name,m in (('RSC_SELECTED',chosen),('MATCHED_RANDOM',random)):
            state['mask']=m
            out=model.forward_identity_only(x)['tx_logits'];loss=ce(out,y)
            logits[name]=out.detach();losses[name]=float(loss.detach())
            vectors[name]=[v.detach().flatten().cpu() for v in torch.autograd.grad(loss,params)]
            if not all(torch.isfinite(v).all() for v in vectors[name]):
                raise ValueError('Nonfinite actual source parameter derivative')
        return dict(locating_gradients=locating.detach().cpu(),spatial_mask=mask.cpu(),
            selected_mask=chosen.cpu(),random_mask=random.cpu(),selected_packets=selected.cpu(),
            locating_probability_drop=drop.cpu(),mask_seed=seed,
            logits={k:v.cpu() for k,v in logits.items()},losses=losses,vectors=vectors,
            activation_nonnegative=True,activation_min=float(activation.detach().min()))
    finally:
        handle.remove()


def summarize(raw):
    layers=[]
    for k,name in enumerate(d.PARAMETERS):
        reference=torch.stack([r['vectors']['ORIGINAL'][k].double() for r in raw]).mean(0)
        entry=dict(parameter=name,cohorts=len(raw))
        for key in ('RSC_SELECTED','MATCHED_RANDOM'):
            other=torch.stack([r['vectors'][key][k].double() for r in raw]).mean(0)
            delta=other-reference
            entry[key]=dict(selected=gradient_stats(reference,other),delta=gradient_stats(reference,delta),
                nonzero_delta_cohorts=sum(float((r['vectors'][key][k]-r['vectors']['ORIGINAL'][k]).norm())>1e-12 for r in raw))
        layers.append(entry)
    rows=[]
    for r in raw:
        y=r['y'];true_probability={key:logits.softmax(1).gather(1,y[:,None]).squeeze(1)
            for key,logits in r['logits'].items()}
        rows.append(dict(cohort=r['cohort'],packets=len(y),selected_packets=int(r['selected_packets'].sum()),
            masked_time_positions=int((~r['selected_mask']).sum()),
            random_masked_time_positions=int((~r['random_mask']).sum()),losses=r['losses'],
            RSC_CE_minus_random=r['losses']['RSC_SELECTED']-r['losses']['MATCHED_RANDOM'],
            RSC_true_probability_drop=float((true_probability['ORIGINAL']-true_probability['RSC_SELECTED']).mean()),
            random_true_probability_drop=float((true_probability['ORIGINAL']-true_probability['MATCHED_RANDOM']).mean()),
            activation_nonnegative=r['activation_nonnegative']))
    return dict(gradient_summary=layers,cohort_summary=rows)


def qualification(summary):
    rr=summary['cohort_summary'];gg=summary['gradient_summary']
    ce_gap=sum(r['RSC_CE_minus_random'] for r in rr)/5
    probability_gap=sum(r['RSC_true_probability_drop']-r['random_true_probability_drop'] for r in rr)/5
    passed=(len(rr)==5 and len(gg)==4 and ce_gap>0 and probability_gap>0
        and all(r['RSC_SELECTED']['nonzero_delta_cohorts']==5 for r in gg[:3]))
    return dict(status='PASS' if passed else 'FAIL',fixed_rules=d.QUALIFICATION,
        source_only=True,target_metrics_used=False,mean_RSC_CE_minus_random=ce_gap,
        mean_RSC_probability_drop_minus_random=probability_gap,
        no_automatic_training_release=True)


def recount(root, *, design=d, row=None):
    context=design;row=row or context.rows()[0]
    if row not in context.rows():raise ValueError('Unregistered source recount row')
    p=root/row['row_id'];report=context.read(p/'analysis.json');context.validate(report['config'])
    if report['config']!=context.config(row):raise ValueError('Saved source model differs from requested row')
    raw=torch.load(p/'actual_source_gradients.pt',map_location='cpu',weights_only=False)
    if len(raw)!=5 or [r['cohort'] for r in raw]!=list(range(5)):
        raise ValueError('Fixed source cohorts incomplete')
    ids=[]
    for r in raw:
        validate_layout(r['y'],r['rx'],r['day'],r['source_physical_ids'],'L_s')
        ids.extend(r['source_physical_ids'])
        if r['locating_gradients'].shape!=(180,96,32):raise ValueError('Saved actual locating derivative differs')
        mask=feature_mask(r['locating_gradients'])
        selected,random,packets,drop=selected_masks(r['logits']['ORIGINAL'],r['logits']['SPATIAL_ALL'],r['y'],mask,r['mask_seed'])
        if not all(torch.equal(a,b) for a,b in ((mask,r['spatial_mask']),(selected,r['selected_mask']),
            (random,r['random_mask']),(packets,r['selected_packets']),(drop,r['locating_probability_drop']))):
            raise ValueError('Independent mask and matched exposure recount differs')
        for key in ('ORIGINAL','RSC_SELECTED','MATCHED_RANDOM'):
            if abs(float(ce(r['logits'][key],r['y']))-r['losses'][key])>1e-7:
                raise ValueError('Actual masked default CE recount differs')
            if [v.numel() for v in r['vectors'][key]]!=[3456,6912,576,960] or not all(torch.isfinite(v).all() for v in r['vectors'][key]):
                raise ValueError('Actual source parameter-gradient coordinates differ')
    summary=summarize(raw)
    if len(ids)!=900 or len(set(ids))!=900 or summary!=report['summary'] or qualification(summary)!=report['qualification']:
        raise ValueError('Source coverage, derivatives or decision differs')
    return dict(status='VERIFIED',source_packets=900,forward_packets_recounted=3600,
        cohorts=5,locating_gradient_tensors=5,parameter_layer_records=20,saved_parameter_vectors=60,
        head_fit_updates=0,model_forward_by_recount=0,optimizer_steps=0,query_read=False)


def main():
    from experiments.cvs_nocfo_foundation.source import load_source
    from experiments.cvs_nocfo_foundation import design as nd
    torch.set_num_threads(2);c=d.config(d.rows()[0]);d.validate(c)
    spec=d.read(d.ROOT/'registration/experiment.json')
    if spec['run_id']!=d.RUN or [r['config'] for r in spec['rows']]!=[c]:
        raise ValueError('Registered fixed source diagnostic differs')
    if d.BASE.exists():raise FileExistsError('Immutable output exists; reconcile, never retry')
    d.BASE.mkdir(parents=True);p=Path(c['output_root']);p.mkdir();started=time.time()
    try:
        L,U,V,contract=load_source(nd.config(nd.rows()[0]))
        if U is not None:raise ValueError('U tensor materialized')
        del V
        source_proof(c['source_config'],design=d.parent)
        old=cohorts(L);indices=[torch.cat((old[2*j],old[2*j+1])) for j in range(5)]
        with numerical_context(d.FULL_FP32_POLICY),torch.random.fork_rng(devices=[]):
            model=source_model();before={k:v.clone() for k,v in model.state_dict().items()}
            raw=[]
            d.write(p/'resolved_config.json',dict(config=c,execution_commit=(d.ROOT/'release_commit.txt').read_text().strip(),
                checkpoint_parent_run=d.parent.RUN,source_roles='EXACT_MATCH',target_read=False,
                head_fit_updates=0,optimizer_steps=0))
            for j,ix in enumerate(indices):
                y,rx,day=(L[k][ix] for k in ('y','rx','day'));ids=[L['ids'][i] for i in ix.tolist()]
                validate_layout(y,rx,day,ids,'L_s')
                result=measure_batch(model,L['x'][ix],y,d.MASK_SEED+j)
                raw.append(dict(cohort=j,y=y.cpu(),rx=rx.cpu(),day=day.cpu(),source_physical_ids=ids,**result))
                print(json.dumps(dict(source_cohorts=j+1,source_packets=(j+1)*180,forward_packets=(j+1)*720)),flush=True)
            if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()) or any(p.grad is not None for p in model.parameters()):
                raise ValueError('Source diagnostic changed checkpoint or accumulated parameter gradients')
        summary=summarize(raw);report=dict(status='ANALYZED',config=c,summary=summary,
            qualification=qualification(summary),model_parameters_buffers_unchanged=True,
            source_packets=900,forward_packets=3600,head_fit_updates=0,optimizer_steps=0,
            U_IQ_read=False,U_truth_read=False,V_forward=False,V_gradient=False,query_read=False,
            seconds=time.time()-started,limitation=c['limitation'])
        torch.save(raw,p/'actual_source_gradients.pt');d.write(p/'analysis.json',report)
        subprocess.run([sys.executable,'-u','-m','experiments.cvs_source_self_challenge.audit','--recount',str(d.BASE)],check=True)
        d.write(d.BASE/'source_qualification.json',report['qualification']);d.write(d.BASE/'analysis.json',report)
        d.write(p/'completion.json',dict(status='ANALYZED',source_packets=900,source_only=True))
        done=dict(status='ANALYZED',models=1,source_packets=900,forward_packets=3600,
            gradient_batches=5,layer_records=20,saved_vectors=60,source_only=True,
            query_read=False,U_truth_read=False,head_updates=0,optimizer_steps=0)
        d.write(d.BASE/'completion.json',done);print(json.dumps(done),flush=True)
    except Exception as exc:
        d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),
            no_automatic_retry=True,outputs_preserved=True));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--recount',type=Path);a=parser.parse_args()
    if a.recount:
        torch.set_num_threads(2);value=recount(a.recount);d.write(a.recount/'independent_artifact_recount.json',value);print(json.dumps(value))
    else:main()
