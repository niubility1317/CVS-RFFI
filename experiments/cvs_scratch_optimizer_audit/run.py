from copy import deepcopy
from importlib import import_module
from pathlib import Path
from unittest.mock import patch
import csv,json,os,sys,time
import torch
from torch.nn import functional as F
from . import design as d
from .geometry import add,geometry,clone_step
from experiments.cvs_scratch_identity_style.joint_cap import parameter_groups
from experiments.cvs_equivariant_identity.precision import numerical_context

def modules(family):
    prefix='experiments.cvs_scratch_'+('calibrated' if family=='calibrated' else 'identity')+'_style'
    return {name:import_module(prefix+'.'+name) for name in ('model','source','objective','peer_cache','evaluate','calibration')}

def tree_equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return a==b

def load_model(entry,device):
    c=entry['config'];m=modules(entry['family']);source=m['evaluate'].source_provenance(c)
    if source/'final_ssdg.pth'!=Path(entry['checkpoint']):raise ValueError('Wrong fixed final source')
    frozen=d.read(source.parent/'source_frozen.json')
    if frozen['config']!=c or frozen['optimizer_steps']!=4440 or frozen['target_access']:raise ValueError('Source freeze differs')
    payload=torch.load(entry['checkpoint'],map_location='cpu',weights_only=False)
    m['evaluate'].checkpoint_provenance(c,payload)
    if not payload.get('optimizer',{}).get('state'):raise ValueError('Saved AdamW history missing')
    model=m['model'].build(c,device,entry['checkpoint']).train()
    calibration=d.read(source/'calibration.json')
    model.multilevel.fixed_coefficients=dict(calibration['fixed_coefficients'])
    m['calibration'].require_calibration(c,model,calibration,1332)
    return model,payload['optimizer'],m,dict(checkpoint=entry['checkpoint'],run=c['run_id'],row=c['row_id'],
        source_contract='EXACT_MATCH',checkpoint_sources=payload['checkpoint_sources'],teacher_provenance=payload['teacher_provenance'],
        calibration=calibration,optimizer_group_metadata=[{k:v for k,v in g.items() if k!='params'} for g in payload['optimizer']['param_groups']],
        frozen_final20=True,target_training_access=False)

def gradient(loss,parameters):
    return tuple(None if g is None else g.detach().clone() for g in torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True))

def batch(model,saved,m,c,L,U,bank,index,device):
    counts=[];original=model.forward_identity_only
    def counted(x):
        counts.append(len(x))
        return original(x)
    modes=[(v,v.training) for v in model.modules()]
    try:
        with patch.object(model,'forward_identity_only',side_effect=counted):
            result=_batch(model,saved,m,c,L,U,bank,index,device)
        if len(counts)!=result['model_forwards'] or sum(counts)!=result['L_occurrences']+result['U_occurrences']:
            raise ValueError('Measured forward/readout budget differs from objective counters')
        result['measured_forward_batch_sizes']=counts
        return result
    finally:
        for module,mode in modes:module.training=mode


def _batch(model,saved,m,c,L,U,bank,index,device):
    step=1332+4*index;seed=c['model_seed']+880301+index*1597
    gen=torch.Generator().manual_seed(seed)
    order=torch.randperm(len(L['ids']),generator=gen);li,reference=order[:128],order[128:256]
    ui=torch.randperm(len(U['ids']),generator=gen)[:128]
    if set(L['ids'][i] for i in li.tolist())&set(L['ids'][i] for i in reference.tolist()):raise ValueError('Reference/main L overlap')
    # This factory first freezes native nuisance parameters. Match the exact
    # trainable identity set before collecting tensors for autograd or cloning.
    groups,_=m['model'].groups(model,c)
    named,partition=parameter_groups(model);parameters=tuple(p for _,p in named)
    by_id={id(p):i for i,p in enumerate(parameters)}
    indices={key:[by_id[id(p)] for _,p in values] for key,values in partition.items()};indices['all']=list(range(len(named)))
    initial={k:v.detach().clone() for k,v in model.state_dict().items()};modes=[(v,v.training) for v in model.modules()]
    original_grads=[None if p.grad is None else p.grad.detach().clone() for p in parameters]
    captured={};original_cap=m['objective'].apply_joint_cap
    def observe(*args):
        captured['raw']=args[2]
        return original_cap(*args)
    devices=[torch.device(device).index or 0] if str(device).startswith('cuda') else []
    with torch.random.fork_rng(devices=devices):
        torch.default_generator.manual_seed(seed)
        if devices:
            with torch.cuda.device(device):torch.cuda.manual_seed(seed)
        with patch.object(m['objective'],'apply_joint_cap',side_effect=observe):
            total,parts,weighted,detail=m['objective'].objective(c,model,bank,L,li,U,ui,step,device)
        identity=sum(weighted[k] for k in m['objective'].BASE_WEIGHTED_KEYS if k!='fishr')
        raw=captured.get('raw',weighted['fishr']);scalar=detail['joint_gradient_cap']['scalar']
        grads={k:gradient(v,parameters) for k,v in dict(CE=weighted['ce'],KL=weighted['KL'],STYLE=weighted['style_ce'],R=raw,IDENTITY=identity).items()}
        grads['FULL']=add(grads['IDENTITY'],grads['R'],scalar)
        grads['UNCAPPED']=add(grads['IDENTITY'],grads['R'])
        direct=gradient(total,parameters)
        error=geometry(direct,add(direct,grads['FULL'],-1.),indices['all'])
        if error['norm_b']>1e-7+2e-5*error['norm_a']:raise ValueError('Decomposed full gradient differs from actual objective')
        grads['STYLE_NEW']=add(grads['STYLE'],grads['CE'],-c['multilevel']['style_ce_weight'])
        # Original model has no normalization-buffer update in this architecture.
        if not tree_equal(initial,model.state_dict()):raise ValueError('Diagnostic objective mutated model state')
        model.eval();ref_logits=model(L['x'][reference].to(device));ref_ce=F.cross_entropy(ref_logits,L['y'][reference].to(device))
        reference_gradient=gradient(ref_ce,parameters)
        for module,mode in modes:module.training=mode
        records=[]
        for mode in d.MODES:
            deltas={};metadata={}
            for objective in d.OBJECTIVES:
                deltas[objective],metadata[objective]=clone_step(named,groups,saved,grads[objective],mode,c['optimizer']['weight_decay'])
            for objective in d.OBJECTIVES:
                per_group={}
                for group,ix in indices.items():
                    delta=deltas[objective];difference=add(delta,deltas['IDENTITY'],-1.)
                    per_group[group]=dict(gradient_vs_identity=geometry(grads['IDENTITY'],grads[objective],ix),
                        displacement_vs_identity=geometry(deltas['IDENTITY'],delta,ix),
                        displacement_change_vs_identity=geometry(deltas['IDENTITY'],difference,ix),
                        displacement_vs_parameters=geometry(tuple(p.detach() for p in parameters),delta,ix),
                        source_reference_directional_derivative=geometry(reference_gradient,delta,ix)['dot'])
                records.append(dict(mode=mode,objective=objective,groups=per_group,optimizer=metadata[objective]))
        scalar_losses={k:float(v.detach()) for k,v in weighted.items()}
    if not tree_equal(initial,model.state_dict()) or any(not tree_equal(p.grad,g) for p,g in zip(parameters,original_grads)):
        raise ValueError('Checkpoint model changed during diagnostic')
    if not all(module.training==mode for module,mode in modes):raise ValueError('Model modes changed')
    return dict(batch=index,step_position=step,main_L_ids=[L['ids'][i] for i in li.tolist()],U_ids=[U['ids'][i] for i in ui.tolist()],
        reference_L_ids=[L['ids'][i] for i in reference.tolist()],losses=scalar_losses,reference_L_CE=float(ref_ce.detach()),
        coefficient=c['multilevel']['style_ce_weight'],fixed_fishr_coefficients=model.multilevel.fixed_coefficients,
        gradient_decomposition_error=error,gradient_geometry={group:{key:geometry(grads['IDENTITY'],grads[key],ix) for key in ('CE','KL','STYLE','STYLE_NEW','R')} for group,ix in indices.items()},
        cap=detail['joint_gradient_cap'],style=detail['L_style'],cases=records,source_reference_is_first_order_only=True,
        original_parameters_buffers_grads_modes_unchanged=True,private_torch_RNG=True,
        model_forwards=7+detail['style_search_forwards'],L_occurrences=504+detail['style_search_L'],U_occurrences=256,
        temporary_optimizer_steps=8,persistent_identity_updates=0,target_access=False)

def run(row_id,device='cuda:0'):
    c=d.validate(d.config(next(r for r in d.rows() if r['row_id']==row_id)));out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);started=time.perf_counter();first=c['models'][0];m=modules(first['family'])
    L,U,V,contract=m['source'].load_source(first['config']);del V
    if 'y' in U:raise ValueError('U TX truth forbidden')
    resolved=dict(config=c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),source_contract=contract,L=6300,U=56700,V_forward=False,V_deserialized_then_discarded=True,target_loader_constructed=False)
    d.write(out/'resolved_config.json',resolved);print('RESOLVED '+json.dumps(c),flush=True)
    count=0;proofs=[];resources=dict(model_forwards=0,L_occurrences=0,U_occurrences=0)
    fields=['model_key','model_seed','batch','mode','objective','group','cap_scalar','gradient_cosine','gradient_norm_ratio','displacement_cosine','displacement_norm_ratio','displacement_change_ratio','relative_parameter_displacement','source_reference_directional_derivative','seconds']
    with numerical_context(d.FULL_FP32_POLICY), (out/'records.jsonl').open('x',encoding='utf-8') as stream, (out/'records.csv').open('x',encoding='utf-8',newline='') as compact:
        writer=csv.DictWriter(compact,fieldnames=fields);writer.writeheader()
        for entry in c['models']:
            pc=entry['config']
            if any(pc[k]!=first['config'][k] for k in ('preparation','unlabeled_cache')):raise ValueError('Shared source role inputs differ')
            model,saved,m,proof=load_model(entry,device);saved_before=deepcopy(saved);proofs.append(dict(key=entry['key'],**proof))
            bank=m['peer_cache'].load(pc,U)
            for i in range(12):
                tick=time.perf_counter();record=batch(model,saved,m,pc,L,U,bank,i,device)
                if str(device).startswith('cuda'):torch.cuda.synchronize(device)
                record.update(model_key=entry['key'],model_seed=c['model_seed'],seconds=time.perf_counter()-tick)
                stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();count+=1
                for key in resources:resources[key]+=record[key]
                for case in record['cases']:
                    for group,v in case['groups'].items():
                        writer.writerow(dict(model_key=entry['key'],model_seed=c['model_seed'],batch=i,mode=case['mode'],objective=case['objective'],group=group,cap_scalar=record['cap']['scalar'],
                            gradient_cosine=v['gradient_vs_identity']['cosine'],gradient_norm_ratio=v['gradient_vs_identity']['norm_ratio_b_over_a'],
                            displacement_cosine=v['displacement_vs_identity']['cosine'],displacement_norm_ratio=v['displacement_vs_identity']['norm_ratio_b_over_a'],
                            displacement_change_ratio=v['displacement_change_vs_identity']['norm_ratio_b_over_a'],relative_parameter_displacement=v['displacement_vs_parameters']['norm_ratio_b_over_a'],
                            source_reference_directional_derivative=v['source_reference_directional_derivative'],seconds=record['seconds']))
                compact.flush()
                print('BATCH '+json.dumps(dict(model=entry['key'],batch=i,losses=record['losses'],cap=record['cap']['scalar'],
                    cases=[dict(mode=x['mode'],objective=x['objective'],all=x['groups']['all']) for x in record['cases']],seconds=record['seconds'])),flush=True)
            if not tree_equal(saved,saved_before):raise ValueError('Original optimizer history changed')
            del model,saved,saved_before,bank
            if str(device).startswith('cuda'):torch.cuda.empty_cache()
    if count!=72:raise ValueError('Incomplete fixed diagnostic matrix')
    d.write(out/'provenance.json',dict(models=proofs,source_contract=contract,target_access=False))
    d.write(out/'completion.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',config=c,models=6,batches=count,temporary_optimizer_steps=count*8,
        identity_updates=0,persistent_identity_updates=0,target_access=False,U_truth=False,V_forward=False,new_checkpoints=0,
        **resources,source_originals_unchanged=True,
        elapsed_seconds=time.perf_counter()-started,torch_version=torch.__version__,hardware=torch.cuda.get_device_name() if str(device).startswith('cuda') else 'CPU',
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None))
