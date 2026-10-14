"""Fixed source-L AdamW counterfactuals; no training or target access."""
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .coordinate import convert_model, convert_optimizer, PREFIX
from .held_credit import assess, fresh_parameter_caches, physical_ids
from experiments.cvs_crossrx_credit50.capture import capture
from experiments.cvs_crossrx_credit50.objective import objective
from experiments.cvs_u_schedule50.counterfactual import _equal


def norm(grads):
    return math.sqrt(sum(float(v.detach().double().square().sum()) for v in grads if v is not None))


def kernel_changes(model, names, proposal):
    sinc=model.get_submodule(PREFIX)
    initial=copy.deepcopy(sinc.state_dict())
    def materialize(after):
        part=copy.deepcopy(sinc)
        state=dict(initial)
        state.update({name[len(PREFIX)+1:]:p for name,p in zip(names,after)
                      if name.startswith(PREFIX+'.')})
        part.load_state_dict(state,strict=True); part.eval()
        with fresh_parameter_caches(part),torch.no_grad():
            filters=part._filters(torch.device('cpu'),torch.float32).detach().clone()
            low=part.low_hz_.detach().clone();band=part.band_hz_.detach().clone()
        return filters,low,band
    with fresh_parameter_caches(sinc),torch.no_grad():
        before=sinc._filters(torch.device('cpu'),torch.float32).detach().clone()
    base,bl,bb=materialize(proposal['base_after'])
    total,tl,tb=materialize(proposal['total_after'])
    changed=lambda a,b:int((a.flatten(1)!=b.flatten(1)).any(1).sum())
    return dict(base_changed_kernels=changed(base,before),total_changed_kernels=changed(total,before),
        auxiliary_changed_kernels=changed(total,base),kernel_count=len(before),
        base_kernel_change_max=float((base-before).abs().max()),
        auxiliary_kernel_change_max=float((total-base).abs().max()),
        auxiliary_low_hz_change_max=float((tl-bl).abs().max()),
        auxiliary_band_hz_change_max=float((tb-bb).abs().max()))


def run(request):
    d.validate(request)
    out=Path(d.OUTPUT)
    if out.exists():raise FileExistsError('Preserve prior source diagnostic; no relaunch')
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);started=time.perf_counter()
    runtime=dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,device='cpu',GPU_allocations=0)
    d.write(out/'runtime.json',runtime)
    print('RUNTIME '+json.dumps(runtime),flush=True)
    from experiments.cvs_crossrx_credit50 import design as parent,model as api
    from experiments.cvs_crossrx_credit50.provenance import source_proof
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_phase1_stack.fast_execution import configure
    config=parent.config(next(r for r in parent.rows() if r['row_id']==request['source_row']))
    source=source_proof(config)  # Actual roles, ancestry and final-only selection before load.
    raw=api.build(config,torch.device('cpu'))
    original_sinc=raw.get_submodule(PREFIX)
    origins={name:getattr(original_sinc,name).detach().clone() for name in ('low_hz_','band_hz_')}
    ck=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
    if (ck['epoch']!=50 or ck['run_id']!=parent.RUN or ck['candidate_id']!=config['row_id']
            or ck['checkpoint_selection']!='final_only' or ck['a1_ema_successful_updates']!=11100):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: fixed source BASE final50')
    raw.load_state_dict(ck['model'],strict=True);raw.eval()
    raw_named=[(n,p) for n,p in raw.named_parameters() if p.requires_grad]
    if len(raw_named)!=188 or len(ck['optimizer']['param_groups'])!=1:
        raise ValueError('Exact native optimizer parameter traversal required')
    models={mode:convert_model(raw,initial_physical=origins,mode=mode) for mode in d.COORDINATES}
    opts={mode:convert_optimizer(ck['optimizer'],raw_named,models[mode],mode=mode) for mode in d.COORDINATES}
    for mode in d.COORDINATES:
        with torch.no_grad(),fresh_parameter_caches(models[mode]),capture(models[mode]) as observed:
            logits=models[mode](torch.zeros(2,2,256))
        if logits.shape!=(2,6) or observed['z'].shape!=(2,160) or not torch.isfinite(logits).all():
            raise ValueError('Real-checkpoint no-query smoke failed')
    print('SMOKE PASS strict own BASE checkpoint RAW/CENTER; query_read=false',flush=True)
    # The builder reuses already validated source roles; only L dataset packets are read.
    with installed(config) as native:
        context=native._build_ssdg_wisig_data(configure(parent.make_args(config,'cpu')),torch.device('cpu'))
        from cvsrffi.xuc_fusion.native import role_ids_from_native
        if context['named_test_loaders'] or role_ids_from_native(context)!=parent.read(parent.SOURCE)['role_ids']:
            raise ValueError('Original source-only role IDs differ')
    dataset=context['train_loader'].dataset
    if len(dataset)!=6300 or getattr(dataset,'hide',True):raise ValueError('Visible original L6300 required')
    metadata=[];cells={}
    for i,it in enumerate(dataset.index):
        identity=(int(it.eq_i),int(it.rx_i),int(it.day_i),int(it.sig_i),int(dataset.selected[i]))
        metadata.append(dict(y=int(it.tx_i),rx=int(it.rx_i),day=int(it.day_i),id=identity))
        cells.setdefault((int(it.tx_i),int(it.rx_i),int(it.day_i)),[]).append(i)
    physical_ids([v['id'] for v in metadata],6300)
    if len(cells)!=90 or any(len(v)!=70 for v in cells.values()):raise ValueError('Original 90x70 source cells differ')
    keys=sorted(cells)
    cohorts={o:[cells[k][o] for k in keys] for o in d.ORDINALS}
    held=[cells[k][o] for k in keys for o in d.HELD_ORDINALS]
    selected=sorted(set(held).union(*(set(v) for v in cohorts.values())))
    if len(selected)!=540 or len(held)!=180:raise ValueError('Fixed distinct source packet counts differ')
    packets={}
    for index in selected:
        x,y,_,meta=dataset[index]
        actual=tuple(int(meta[k]) for k in ('eq_i','rx_i','day_i','sig_i','base_index'))
        if actual!=metadata[index]['id'] or int(y)!=metadata[index]['y']:
            raise ValueError('Metadata and L packet identity/label differ')
        packets[index]=x
    def batch(indices):
        return (torch.stack([packets[i] for i in indices]),
                torch.tensor([metadata[i]['y'] for i in indices]),
                torch.tensor([metadata[i]['rx'] for i in indices]),
                torch.tensor([metadata[i]['day'] for i in indices]),
                [metadata[i]['id'] for i in indices])
    hx,hy,hr,_,hids=batch(held)
    records=[];forwards=4;temporary=0
    for mode in d.COORDINATES:
        model=models[mode];opt=opts[mode]
        parameters=[p for group in opt.param_groups for p in group['params']]
        by_id={id(p):n for n,p in model.named_parameters()};names=[by_id[id(p)] for p in parameters]
        head=api.classifier(model);wnames=[n for n,p in model.named_parameters() if p is head.weight]
        if len(wnames)!=1 or head.weight.shape!=(6,160):raise ValueError('Default 6x160 head changed')
        default_w=wnames[0];wi=names.index(default_w)
        gain=[i for i,n in enumerate(names) if n.endswith('sequence_pool.gain_raw')]
        if len(gain)!=1:raise ValueError('Original SIG_TIME gain differs')
        before=copy.deepcopy(model.state_dict());optimizer_before=copy.deepcopy(opt.state_dict())
        for ordinal in d.ORDINALS:
            x,y,rx,day,ids=batch(cohorts[ordinal])
            with fresh_parameter_caches(model),capture(model) as observed:
                logits=model(x)
            base_loss=F.cross_entropy(logits,y)
            base=torch.autograd.grad(base_loss,parameters,retain_graph=True,allow_unused=True)
            order=objective(observed['z'],y,rx,day,[json.dumps(v) for v in ids],'ORDER')
            record=dict(coordinate=mode,ordinal=ordinal,base_L_CE=float(base_loss),
                        base_gradient_norm=norm(base),ORDER=order['summary'],groups=[])
            if base[wi] is None or not bool(base[wi].abs().sum()>0):raise ValueError('Native main W must remain jointly trainable')
            for receiver in d.SOURCE_RX:
                auxiliary=torch.autograd.grad(order['loss_by_RX'][str(receiver)],parameters,
                                              retain_graph=True,allow_unused=True)
                if auxiliary[wi] is not None or auxiliary[gain[0]] is not None:
                    raise ValueError('Pre-gain auxiliary must not fit default W or gain')
                for evaluation in d.MODES:
                    ex,ey,er,eids=(x,y,rx,ids) if evaluation=='SAME_L' else (hx,hy,hr,hids)
                    result=assess(model,opt,base,auxiliary,ex,ey,er,receiver,
                                  default_weight_name=default_w,evaluation_ids=eids,current_ids=ids,
                                  mode=evaluation)
                    summary=result['summary'];summary['auxiliary_gradient_norm']=norm(auxiliary)
                    summary['physical_Sinc']=kernel_changes(model,names,result['proposal'])
                    record['groups'].append(summary)
                    forwards+=summary['extra_IQ_forwards'];temporary+=summary['temporary_AdamW_steps']
            records.append(record);forwards+=90
            d.write(out/f'{mode}-ordinal{ordinal}.json',record)
            print('SOURCE_PROBE '+json.dumps(dict(coordinate=mode,ordinal=ordinal,
                  accepted={evaluation:sum(g['accepted'] for g in record['groups']
                    if g['evaluation_mode']==evaluation) for evaluation in d.MODES})),flush=True)
        if (not _equal(before,model.state_dict()) or not _equal(optimizer_before,opt.state_dict())
                or any(p.grad is not None for p in model.parameters())):
            raise ValueError('Diagnostic modified live parameters/buffers/optimizer/gradients')
    active=[];physical=[]
    for record in records:
        if record['coordinate']!='CENTER_25K':continue
        accepted=[g for g in record['groups'] if g['evaluation_mode']=='HELD_L' and g['accepted']
                  and g['auxiliary_gradient_norm']>1e-8]
        if accepted:active.append(record['ordinal'])
        if any(g['physical_Sinc']['auxiliary_changed_kernels']>0 for g in accepted):physical.append(record['ordinal'])
    passed=len(active)>=2 and len(physical)>=2
    if forwards!=18004 or temporary!=160 or sum(len(r['groups']) for r in records)!=80:
        raise ValueError('Fixed source matrix execution counts differ')
    summary=dict(status='SOURCE_SCREEN_PASSED' if passed else 'SOURCE_SCREEN_NOT_PASSED',run_id=d.RUN,
        analysis_code_commit=request['analysis_code_commit'],source_falsification=d.GATE,
        CENTER_held_active_cohorts=active,CENTER_accepted_auxiliary_kernel_change_cohorts=physical,
        checkpoint_source=dict(run_id=parent.RUN,row_id=config['row_id'],epoch=50,seed=2026092701,
          checkpoint=str(source/'final_ssdg.pth'),source_roles='EXACT_MATCH',ancestors=[],
          source_selection='fixed BASE final50 control; not target ranking',target_training_selection=False),
        coordinate_origin='Exact original constructor frequency buffers, not rebased frozen weights',
        optimizer_mapping='Existing same-order AdamW step preserved; CENTER exp_avg x25000 and exp_avg_sq x25000^2. Hypothetical coordinate switch, not old CENTER training trajectory.',
        source_metadata_packets=6300,source_IQ_packet_reads=540,anchor_packets=360,held_L_packets=180,
        previously_trained_source_L=True,current_anchor_held_packets_disjoint=True,
        network_IQ_forwards=forwards,temporary_AdamW_steps=temporary,kernel_syntheses=240,optimizer_updates=0,
        head_fit_updates=0,EMA_updates=0,V_IQ_reads=0,U_IQ_reads=0,U_truth_read=False,target_access=False,
        live_parameters_buffers_optimizer_gradients_unchanged=True,records=records,
        cache_policy='Each temporary state clears parameter-derived Sinc and default W caches, restores originals',
        wall_seconds=time.perf_counter()-started,runtime=runtime,
        limitation='Fixed seen-L final50 L-only hypothetical steps, not contemporaneous U-containing training, unseenRX, never-trained holdout, efficacy, or a three-seed result.')
    d.write(out/'summary.json',summary)
    print('SOURCE_COMPLETE '+json.dumps(dict(status=summary['status'],network_IQ_forwards=forwards,
          temporary_AdamW_steps=temporary,active_cohorts=active,physical_cohorts=physical)),flush=True)
    return summary
