import json,os,subprocess,sys,time,traceback
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from . import design as d
from .objective import symmetric_kl,gradient_stats
from experiments.cvs_evidence_responsibility_train20 import provenance as proof,model as model_api
from experiments.cvs_equivariant_identity.precision import numerical_context

def layout(y,rx,day,ids,per_cell):
    count=90*per_cell
    if y.shape!=rx.shape or y.shape!=day.shape or y.shape!=(count,) or len(ids)!=count or len(set(ids))!=count:
        raise ValueError('Unique balanced source physical packets required')
    for tx in range(6):
        for dd in d.DAYS:
            for receiver in d.RX:
                if int(((y==tx)&(rx==receiver)&(day==dd)).sum())!=per_cell:raise ValueError('Source TX/RX/day layout differs')

def cohorts(L):
    layout(L['y'],L['rx'],L['day'],L['ids'],70)
    order=torch.randperm(6300,generator=torch.Generator().manual_seed(d.COHORT_SEED));cells=[]
    for tx in range(6):
        for dd in d.DAYS:
            for receiver in d.RX:cells.append(order[(L['y'][order]==tx)&(L['day'][order]==dd)&(L['rx'][order]==receiver)])
    ranks=torch.stack(cells).T
    return [torch.cat((ranks[2*j],ranks[2*j+1])) for j in range(5)]

def prepare(model):
    if any(isinstance(module,nn.modules.batchnorm._BatchNorm) for module in model.modules()):raise ValueError('Stateful batch normalization is outside this read-only diagnostic')
    model.train().requires_grad_(False);named=dict(model.named_parameters());params=[named[name] for name in d.PARAMETERS]
    if [p.numel() for p in params]!=list(d.PARAMETER_SIZES) or model_api.classifier(model).weight is not params[-1]:
        raise ValueError('Actual original feature/default-head gradient coordinates differ')
    for p in params:p.requires_grad_(True)
    return params

def derivative(loss,params):
    grads=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=True)
    result=[torch.zeros_like(p).flatten() if value is None else value.detach().flatten() for p,value in zip(params,grads)]
    if any(not torch.isfinite(value).all() for value in result):raise ValueError('Nonfinite source derivative')
    return result

def measure_batch(model,x,y,seed):
    params=prepare(model)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        a=model.forward_identity_only(x.clone());b=model.forward_identity_only(x.clone())
        ce=.5*(F.cross_entropy(a['tx_logits'],y)+F.cross_entropy(b['tx_logits'],y));kl=symmetric_kl(a['tx_logits'],b['tx_logits'])
        vectors={name:derivative(loss,params) for name,loss in (('PAIRED_CE',ce),('SYMMETRIC_KL',kl))}
        model.eval()
        with torch.no_grad():reference=model.forward_identity_only(x.clone())
        result=dict(logits1=a['tx_logits'].detach().cpu(),logits2=b['tx_logits'].detach().cpu(),eval_logits=reference['tx_logits'].cpu(),
            features1=a['id_feat_joint'].detach().cpu(),features2=b['id_feat_joint'].detach().cpu(),eval_features=reference['id_feat_joint'].cpu(),
            losses=dict(PAIRED_CE=float(ce.detach()),SYMMETRIC_KL=float(kl.detach())),vectors=vectors)
    if any(not torch.isfinite(result[key]).all() for key in ('logits1','logits2','eval_logits','features1','features2','eval_features')):
        raise ValueError('Nonfinite source default predictions/features')
    return result

def summarize(raw):
    layers=[];records=[]
    for index,name in enumerate(d.PARAMETERS):
        ce=torch.stack([row['vectors']['PAIRED_CE'][index].double() for row in raw]).mean(0)
        kl=torch.stack([row['vectors']['SYMMETRIC_KL'][index].double() for row in raw]).mean(0)
        layers.append(dict(parameter=name,cohorts=len(raw),**gradient_stats(ce,kl),KL_nonzero_cohorts=sum(float(row['vectors']['SYMMETRIC_KL'][index].norm())>1e-12 for row in raw)))
    for row in raw:
        a,b,e=row['logits1'],row['logits2'],row['eval_logits'];y=row['y'];z1,z2=row['features1'],row['features2']
        records.append(dict(cohort=row['cohort'],packets=len(y),losses=row['losses'],stochastic1_correct=int((a.argmax(1)==y).sum()),
            stochastic2_correct=int((b.argmax(1)==y).sum()),eval_correct=int((e.argmax(1)==y).sum()),stochastic_disagreements=int((a.argmax(1)!=b.argmax(1)).sum()),
            joint160_pair_cosine_mean=float(F.cosine_similarity(z1.double(),z2.double(),dim=1).mean()),joint160_pair_L2_mean=float((z1.double()-z2.double()).norm(dim=1).mean())))
    return dict(layers=layers,cohorts=records)

def qualify(summary):
    passed=(len(summary['cohorts'])==5 and len(summary['layers'])==4 and all(row['losses']['SYMMETRIC_KL']>0 for row in summary['cohorts'])
        and all(row['KL_nonzero_cohorts']==5 and row['cosine'] is not None and row['cosine']>=0 for row in summary['layers'][:3]))
    return dict(status='PASS' if passed else 'FAIL',fixed_rules=d.QUALIFICATION,source_only=True,target_metrics_used=False,training_efficacy_proven=False)

def recount(row_id):
    row=next((row for row in d.rows() if row['row_id']==row_id),None)
    if row is None:raise ValueError('Unregistered recount row')
    root=d.BASE/row_id;report=d.read(root/'analysis.json');d.validate(report['config'])
    if report['config']!=d.config(row):raise ValueError('Wrong saved source parent')
    raw=torch.load(root/'actual_source_gradients.pt',map_location='cpu',weights_only=False)
    if len(raw)!=5 or [item['cohort'] for item in raw]!=list(range(5)):raise ValueError('Five complete source cohorts required')
    ids=[]
    for item in raw:
        layout(item['y'],item['rx'],item['day'],item['source_physical_ids'],2);ids+=item['source_physical_ids']
        for name in ('PAIRED_CE','SYMMETRIC_KL'):
            values=item['vectors'][name]
            if [value.numel() for value in values]!=list(d.PARAMETER_SIZES) or any(not torch.isfinite(value).all() for value in values):raise ValueError('Saved derivative coordinates differ')
        ce=.5*(F.cross_entropy(item['logits1'],item['y'])+F.cross_entropy(item['logits2'],item['y']));kl=symmetric_kl(item['logits1'],item['logits2'])
        if abs(float(ce)-item['losses']['PAIRED_CE'])>1e-7 or abs(float(kl)-item['losses']['SYMMETRIC_KL'])>1e-7:raise ValueError('Saved CE/KL recount differs')
        if any(item[key].shape!=(180,160) or not torch.isfinite(item[key]).all() for key in ('features1','features2','eval_features')):raise ValueError('Saved actual joint160 witness differs')
    summary=summarize(raw)
    if len(ids)!=900 or len(set(ids))!=900 or summary!=report['summary'] or qualify(summary)!=report['qualification']:raise ValueError('Source coverage/gradient summary recount differs')
    result=dict(status='VERIFIED',source_packets=900,model_forward_packets_recounted=2700,gradient_batches=5,parameter_layer_records=20,saved_vectors=40,
        new_forwards=0,new_gradients=0,optimizer_steps=0,head_fit_updates=0,query_read=False)
    d.write(root/'independent_recount.json',result);return result

def smoke(model):
    before={key:value.clone() for key,value in model.state_dict().items()}
    with torch.no_grad():out=model.forward_identity_only(torch.zeros(2,2,256))
    if out['id_feat_joint'].shape!=(2,160) or out['tx_logits'].shape!=(2,6) or not torch.isfinite(out['tx_logits']).all() or any(not torch.equal(value,model.state_dict()[key]) for key,value in before.items()):
        raise ValueError('Real checkpoint no-query smoke failed')
    return dict(status='PASS',real_checkpoint_no_query_smoke=True,synthetic_IQ_packets=2,query_read=False,optimizer_steps=0,head_fit_updates=0)

def run(request_path,entry):
    torch.set_num_threads(2)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_available():raise ValueError('Fixed CPU-only source diagnostic required')
    if d.read(request_path)!=d.request():raise ValueError('Fixed source matrix differs')
    spec=d.expand(d.read(d.ROOT/'registration/experiment.json'))
    if spec['run_id']!=d.RUN or [row['config'] for row in spec['rows']]!=[d.config(row) for row in d.rows()]:raise ValueError('Source preregistration differs')
    if (Path(d.DEPENDENCY_RELEASE)/'release_commit.txt').read_text().strip()!=d.DEPENDENCY_COMMIT:raise ValueError('Original E20 dependency Git identity differs')
    if d.BASE.exists():raise FileExistsError('Existing immutable source output; reconcile original run')
    d.BASE.mkdir(parents=True);results=[];L=None;all_ids=None
    try:
        for row in d.rows():
            c=d.config(row);d.validate(c);root=Path(c['output_root']);root.mkdir();started=time.time()
            with numerical_context(d.FULL_FP32_POLICY),torch.random.fork_rng(devices=[]):
                model=proof.own_model(d.parent_config(c['model_seed']),'cpu')
                preflight=smoke(model);d.write(root/'preflight.json',preflight);print('REAL_CHECKPOINT_SMOKE_PASS '+row['row_id'],flush=True)
                if L is None:
                    from experiments.cvs_nocfo_foundation.source import load_source
                    from experiments.cvs_nocfo_foundation import design as nd
                    loader_config=nd.config(nd.rows()[0])
                    if loader_config['unlabeled'] is not False:raise ValueError('U materialization must be disabled before loading source cache')
                    L,U,V,contract=load_source(loader_config)
                    if U is not None:raise ValueError('Source diagnostic materialized U')
                    del V
                    indices=cohorts(L)
                before={key:value.clone() for key,value in model.state_dict().items()};raw=[]
                d.write(root/'resolved_config.json',dict(config=c,execution_commit=(d.ROOT/'release_commit.txt').read_text().strip(),
                    parent_source_roles='EXACT_MATCH',source_only=True,V_cache_materialized_then_discarded=True,U_tensor_materialized=False,query_read=False))
                for index,ix in enumerate(indices):
                    y,rx,day=(L[key][ix] for key in ('y','rx','day'));ids=[L['ids'][int(i)] for i in ix];layout(y,rx,day,ids,2)
                    raw.append(dict(cohort=index,y=y.cpu(),rx=rx.cpu(),day=day.cpu(),source_physical_ids=ids,
                        **measure_batch(model,L['x'][ix],y,d.DROPOUT_SEED+index)))
                    print(json.dumps(dict(row_id=row['row_id'],source_cohorts=index+1,model_forward_packets=(index+1)*540)),flush=True)
                if any(not torch.equal(value,model.state_dict()[key]) for key,value in before.items()) or any(p.grad is not None for p in model.parameters()):raise ValueError('Read-only source diagnostic changed model state/gradients')
                model.eval().requires_grad_(False)
            ids=[physical for item in raw for physical in item['source_physical_ids']]
            if all_ids is None:all_ids=ids
            elif ids!=all_ids:raise ValueError('Matched source physical cohorts changed between model seeds')
            summary=summarize(raw);qualification=qualify(summary)
            report=dict(status='ANALYZED',config=c,summary=summary,qualification=qualification,model_parameters_buffers_unchanged=True,
                source_packets=900,model_forward_packets=2700,gradient_batches=5,parameter_layer_records=20,saved_vectors=40,head_fit_updates=0,optimizer_steps=0,
                U_tensor_materialized=False,U_truth_read=False,V_forward=False,V_gradient=False,query_read=False,elapsed_seconds=time.time()-started)
            torch.save(raw,root/'actual_source_gradients.pt');d.write(root/'analysis.json',report)
            subprocess.run([sys.executable,'-u',str(entry),str(request_path),'--recount',row['row_id']],check=True)
            d.write(root/'completion.json',dict(status='ANALYZED',config=c,source_only=True,query_read=False,optimizer_steps=0,head_fit_updates=0,independent_recount='VERIFIED'))
            results.append(dict(row_id=row['row_id'],model_seed=row['model_seed'],summary=summary,qualification=qualification))
            if len(results)==1 and qualification['status']=='FAIL':break
        done=dict(status='ANALYZED',run_id=d.RUN,results=results,actual_model_seeds=len(results),planned_model_seeds=3,
            all_three_source_confirmed=len(results)==3 and all(row['qualification']['status']=='PASS' for row in results),
            first_seed_source_falsified=len(results)==1 and results[0]['qualification']['status']=='FAIL',
            unique_source_physical_packets=900,source_packet_observations=900*len(results),model_forward_packets=2700*len(results),
            gradient_batches=5*len(results),parameter_layer_records=20*len(results),saved_vectors=40*len(results),
            synthetic_smoke_packets=2*len(results),training_epochs=0,optimizer_steps=0,head_fit_updates=0,query_read=False,U_truth_read=False,
            default_head_unchanged=True,training_efficacy_proven=False,goal_completed=False)
        d.write(d.BASE/'matrix_result.json',done);d.write(d.BASE/'completion.json',done);print('SOURCE_RDROP_MATRIX_COMPLETE '+json.dumps(done),flush=True)
    except Exception as error:
        d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,no_automatic_retry=True));raise
