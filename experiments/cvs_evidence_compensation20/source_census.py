"""Full frozen source-V proxy and hidden-U census, without fitting."""
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader,RandomSampler
from . import design as d
from .census import Census
from .objective import readouts
from experiments.cvs_evidence_recomposition20.capture import forward
from .original_gate import gate
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches,physical_ids
from experiments.cvs_u_schedule50.counterfactual import _equal,_rng_snapshot

FORBIDDEN={'tx','tx_i','true_tx_i','tx_label','y_tx'}


def blind_metadata(raw,count):
    """V labels are retained only downstream; no TX fields reach the gate."""
    result={k:v for k,v in raw.items() if k not in FORBIDDEN}
    result['tx_label_visible']=torch.zeros(count,dtype=torch.bool)
    return result


def source_gate(results):
    if set(results)!={'V_proxy','U_s'}:raise ValueError('Both complete source roles required')
    v,u=results['V_proxy'],results['U_s'];vs,us=v['summary'],u['summary']
    criteria=dict(U_EVIDENCE_active=us['EVIDENCE_active_count']>=d.GATE['U_EVIDENCE_active_min'],
        U_new_vs_SELF=us['new_active_vs_SELF_count']>=d.GATE['U_EVIDENCE_new_vs_SELF_min'],
        V_EVIDENCE_active=vs['EVIDENCE_active_count']>=d.GATE['V_EVIDENCE_active_min'],
        V_EVIDENCE_active_precision=vs['EVIDENCE_active_precision'] is not None
            and vs['EVIDENCE_active_precision']>=d.GATE['V_EVIDENCE_active_precision_min'])
    for receiver in d.SOURCE_RX:
        row=v['by_RX'][str(receiver)]
        criteria['V_RX'+str(receiver)+'_active']=row['EVIDENCE_active_count']>=d.GATE['V_each_RX_EVIDENCE_active_min']
        criteria['V_RX'+str(receiver)+'_precision']=row['EVIDENCE_active_precision'] is not None and row['EVIDENCE_active_precision']>=d.GATE['V_each_RX_EVIDENCE_active_precision_min']
    return dict(passed=all(criteria.values()),criteria=criteria,
        interpretation=d.GATE['interpretation'])


def run(request):
    d.validate(request);out=Path(d.OUTPUT)
    if out.exists():raise FileExistsError('Preserve existing census, reconcile before any retry')
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);started=time.perf_counter()
    runtime=dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,device='cpu',GPU_allocations=0)
    d.write(out/'runtime.json',runtime);print('RUNTIME '+json.dumps(runtime),flush=True)
    from experiments.cvs_crossrx_sharpness20 import design as parent,model as api
    from experiments.cvs_crossrx_sharpness20.provenance import source_proof,validate_checkpoint
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_phase1_stack.fast_execution import configure
    c=parent.config(next(r for r in parent.rows() if r['row_id']==request['source_row']))
    gc=parent.config(next(r for r in parent.rows() if r['row_id']==request['gate_recipe_row']))
    source=source_proof(c) # Actual own roles, scratch ancestry and final-only source selection BEFORE load.
    checkpoint=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
    validate_checkpoint(checkpoint,c)
    student=api.build(c,torch.device('cpu')).eval().requires_grad_(False)
    student.load_state_dict(checkpoint['model'],strict=True)
    teacher=copy.deepcopy(student).eval().requires_grad_(False)
    with torch.no_grad():
        for label,model in [('student',student),('own_copy_teacher',teacher)]:
            observed=forward(model,torch.zeros(2,2,256));ro=readouts(model,observed)
            if observed['logits'].shape!=(2,6) or not torch.equal(ro['self']['joint'],observed['joint']):
                raise ValueError('Real checkpoint/query-free smoke failed')
            print('SMOKE PASS '+label+' strict own final20/query_read=false',flush=True)
    before=[copy.deepcopy(model.state_dict()) for model in (student,teacher)]
    with installed(c) as native:
        context=native._build_ssdg_wisig_data(configure(parent.make_args(c,'cpu')),torch.device('cpu'))
        from cvsrffi.xuc_fusion.native import role_ids_from_native
        if context['named_test_loaders'] or role_ids_from_native(context)!=parent.read(parent.SOURCE)['role_ids']:
            raise ValueError('Unchanged legal source role handles differ')
    uloader=context['unlabeled_loader'];vds=context['val_loader'].dataset
    if (len(uloader.dataset)!=56700 or not getattr(uloader.dataset,'hide',False)
        or not isinstance(uloader.sampler,RandomSampler) or uloader.batch_size!=256 or uloader.drop_last
        or len(vds)!=27000 or getattr(vds,'hide',False)):
        raise ValueError('Original hidden U shuffle/full source V loader differs')
    vloader=DataLoader(vds,batch_size=256,shuffle=True,drop_last=False,num_workers=0,
        collate_fn=context['val_loader'].collate_fn,
        generator=torch.Generator().manual_seed(request['sampling_seed']+1))
    args=parent.make_args(gc,'cpu');results={};gate_counts={};all_bases=set();forwards=4
    rng_before=_rng_snapshot()
    with torch.random.fork_rng(devices=[]),torch.no_grad():
        torch.manual_seed(request['sampling_seed'])
        for role,loader in [('V_proxy',vloader),('U_s',uloader)]:
            census=Census(role=role);counts={};batches=0;packets=0
            for batch in loader:
                x,truth,domain,raw=batch
                if role=='U_s' and ((truth!=-1).any() or any(k in raw for k in FORBIDDEN)):
                    raise ValueError('U truth or TX metadata exposed')
                metadata=blind_metadata(raw,len(x))
                x,sentinels,extra=native.move_batch((x,torch.full_like(truth,-1),domain,metadata),torch.device('cpu'))
                if (sentinels!=-1).any() or any(k in native._meta_from_extra(extra) for k in FORBIDDEN):
                    raise ValueError('Truth entered inference/gate')
                ids=physical_ids([tuple(int(metadata[k][i]) for k in ('eq_i','rx_i','day_i','sig_i','base_index')) for i in range(len(x))],len(x))
                bases={pid[-1] for pid in ids}
                if bases&all_bases:raise ValueError('Duplicate physical sample across batches/roles')
                all_bases.update(bases)
                du=native.domain_from_extra(extra,context['domain_label_map'],torch.device('cpu'))
                with fresh_parameter_caches(teacher):weak=teacher.forward_identity_only(x)['tx_logits']
                strong_x=native._strong_augment(x,float(args.strong_noise_std))
                observed=forward(student,strong_x)
                pseudo,masks=gate(native,args,weak,observed['logits'],extra,du)
                ro=readouts(student,observed)
                probabilities={name:ro[key]['logits'].softmax(1).gather(1,pseudo[:,None]).squeeze(1)
                    for name,key in [('SELF','self'),('PA_WEAK','PAweak'),('REF_WEAK','REFweak')]}
                # Only after all predictions, challenge views and q are fixed does V truth enter counts.
                census.add(ids,metadata['rx_i'],metadata['day_i'],pseudo,masks['accepted'],probabilities,
                    source_truth=truth if role=='V_proxy' else None)
                for key,mask in masks.items():counts[key]=counts.get(key,0)+int(mask.sum())
                batches+=1;packets+=len(x);forwards+=2*len(x)
                progress=dict(role=role,packets=packets,batches=batches,mask_counts=counts,
                    network_IQ_forwards=forwards,wall_seconds=time.perf_counter()-started)
                d.write(out/'progress.json',progress)
                print('SOURCE_CENSUS '+json.dumps(progress),flush=True)
            if batches!={'V_proxy':106,'U_s':222}[role]:raise ValueError('Complete batch count differs')
            results[role]=census.finish(request['counts'][role]);gate_counts[role]=counts
            d.write(out/(role+'.json'),results[role])
    if not _equal(rng_before,_rng_snapshot()):raise ValueError('Frozen census altered outside RNG')
    if (any(not _equal(state,model.state_dict()) for state,model in zip(before,(student,teacher)))
        or any(p.grad is not None or p.requires_grad for model in (student,teacher) for p in model.parameters())
        or any(module.training for model in (student,teacher) for module in model.modules())):
        raise ValueError('Frozen census mutated model/buffers/gradients/flags')
    if forwards!=167404 or len(all_bases)!=83700:raise ValueError('Full physical/count budget differs')
    verdict=source_gate(results)
    summary=dict(status='SOURCE_SCREEN_PASSED' if verdict['passed'] else 'SOURCE_SCREEN_NOT_PASSED',
        run_id=d.RUN,analysis_code_commit=request['analysis_code_commit'],gate_result=verdict,
        source_falsification=d.GATE,config=request,roles=results,mask_counts=gate_counts,
        checkpoint_source=dict(run_id=parent.RUN,row_id=c['row_id'],seed=2026092701,epoch=20,
            source_roles='EXACT_MATCH',ancestors=[],selection='fixed BASE_CE firstseed final20; no target ranking',
            teacher_origin='same source student deepcopy; no EMA state or updates'),
        V_IQ_reads=27000,U_IQ_reads=56700,source_unique_physical_packets=83700,
        network_IQ_forwards=forwards,zero_IQ_smoke_forwards=4,classifier_readouts_per_student_forward=3,
        extra_IQ_forward_for_challenge=0,actual_optimizer_updates=0,head_fit_updates=0,EMA_updates=0,
        U_truth_read=False,target_access=False,default_head_structure_unchanged=True,
        parameters_buffers_gradients_flags_unchanged=True,outside_census_RNG_unchanged=True,
        runtime=runtime,wall_seconds=time.perf_counter()-started,
        limitation='Frozen eval student and copied final20 teacher. Original gate predicate/native U shuffle preserved; masks differ from train-dropout/contemporaneous EMA training. V random256 is a source proxy, not hiddenU precision or efficacy. No gradients/optimizer/target decisions here.')
    d.write(out/'summary.json',summary);print('SOURCE_COMPLETE '+json.dumps(verdict),flush=True)
    return summary
