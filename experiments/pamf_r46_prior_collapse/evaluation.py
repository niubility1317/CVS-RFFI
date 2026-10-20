import json
import time
from types import SimpleNamespace
import numpy as np
import torch
from . import design as d
from .runtime import source_root
from experiments.pamf_uncertainty_restore import evaluation as base
from experiments.pamf_restoration.runner import read_json, tensor_array
from experiments.pamf_r46_multipoint_fast.tensor_io import array_tensor
from experiments.pamf_restoration.data import write_json
from experiments.pamf_r46_prior_preserve.evaluation import compact_acceptance
from tools.analyze_pamf_r17_task_results import summary_metrics, compare
from tools.r17_compact_acceptance import target_acceptance


def row_args(args):
    if args.arm not in d.ARMS or args.output.name != d.RUN:
        raise ValueError('Registered collapse row/root required before IO')
    return SimpleNamespace(**dict(vars(args), output=args.output/args.arm, kind=d.KIND))


def reference(args, role):
    if args.output.name not in d.ARMS or args.output.parent.name != d.RUN or args.kind != d.KIND:
        raise ValueError('Registered collapse row required before IO')
    if role not in ('source','target'):
        raise ValueError('Only fixed source/target roles')
    if read_json(args.output.parent/'comparison_plan.json') != d.configuration():
        raise ValueError('Fixed collapse matrix required')
    ready = read_json(args.output/'runtime_ready.json')
    if ready != dict(status='FROZEN_RUNTIME_READY', arm=args.output.name, parameters=d.PARAMETERS[args.output.name],
                     new_training_updates=0, F_parameter_updates=0, head_fit_updates=0):
        raise ValueError('Fixed unchanged source checkpoint/runtime required')
    if role == 'source':
        return base.reference(args, role)
    if read_json(args.output/'source_complete.json').get('status') != 'ANALYZED':
        raise ValueError('Complete source diagnostics required; no accuracy qualification')
    manifest = read_json(args.formal_reference/'manifest.json')
    required = dict(status='FORMAL_REFERENCE_READY', count=168000, views=list(d.VIEWS),
                    original_IQ_bytes_reused=True, TX_truth_reads=0)
    if any(manifest.get(k) != v for k,v in required.items()):
        raise ValueError('Original formal reference handles required')
    ids = np.load(args.formal_reference/'physical_ids.npy', allow_pickle=False)
    if ids.shape != (168000,) or len(np.unique(ids)) != 168000:
        raise ValueError('Fixed original physical population required')
    return args.formal_reference, ids, None, None


@torch.no_grad()
def predict(args, role, pipeline):
    current = row_args(args)
    folder, ids, _, _ = reference(current, role)
    row = base.row_folder(current, role)
    row.mkdir(parents=True, exist_ok=True)
    output = row/'prediction'
    output.mkdir()
    times = {}
    started_all = time.perf_counter()
    for view in d.VIEWS:
        received = np.load(folder/(view+'.npy'), mmap_mode='r', allow_pickle=False)
        if received.shape != (len(ids),2,256):
            raise ValueError('Original received IQ dimensions differ')
        arrays = {key:np.lib.format.open_memmap(output/(view+'.restored.'+key+'.npy'), mode='w+',
                  dtype=np.float32, shape=shape) for key,shape in (
                    ('iq',(len(ids),2,256)), ('features',(len(ids),160)), ('logits',(len(ids),6)))}
        variance = np.lib.format.open_memmap(output/(view+'.variance_mean.npy'), mode='w+', dtype=np.float32, shape=(len(ids),))
        started = time.perf_counter()
        for begin in range(0,len(ids),d.BATCH):
            end = min(begin+d.BATCH,len(ids))
            x = array_tensor(received[begin:end], 'cuda:0')
            iq, var, features, logits = pipeline(x)
            expected = ((iq,(len(x),2,256)), (var,(len(x),)), (features,(len(x),160)), (logits,(len(x),6)))
            if any(tuple(v.shape) != shape or not torch.isfinite(v).all() for v,shape in expected) or (var <= 0).any():
                raise FloatingPointError('Invalid collapse IQ/variance/original F output')
            for key,value in (('iq',iq),('features',features),('logits',logits)):
                arrays[key][begin:end] = tensor_array(value)
            variance[begin:end] = tensor_array(var)
        for value in (*arrays.values(),variance):
            value.flush()
        torch.cuda.synchronize()
        times[view] = time.perf_counter()-started
        print(json.dumps(dict(event='COLLAPSE_VIEW_COMPLETE',arm=args.arm,role=role,view=view,count=len(ids))),flush=True)
    np.save(output/'physical_ids.npy',ids,allow_pickle=False)
    write_json(output/'complete.json',dict(status='PREDICTIONS_COMPLETE',kind=d.KIND,role=role,count=len(ids),
        views=list(d.VIEWS),readouts=['restored'],TX_truth_reads=0,state_updates=0,head_fit_updates=0,
        F_run=d.F_RUN,F_seed=d.F_SEED,original_default_head=True,per_view_seconds=times,
        seconds=time.perf_counter()-started_all,execution='prior_collapsed_native_FP32',batch=d.BATCH,
        setup=[],native_fallback=False))


def native_agreement(args, role):
    current = row_args(args)
    _, ids, _, _ = reference(current, role)
    new_folder = base.row_folder(current,role)/'prediction'
    old_row = source_root(args,args.arm)/d.SOURCE_ROWS[args.arm]
    old_folder = old_row/d.KIND/'prediction' if role == 'source' else old_row/'target_confirmation'/d.KIND/'prediction'
    new = base.prediction_arrays(new_folder,ids,d.KIND,role)
    old = base.prediction_arrays(old_folder,ids,d.KIND,role)
    result = {}
    for view in d.VIEWS:
        values = {}
        for key in ('iq','features','logits'):
            x,y = new[view]['restored'][key],old[view]['restored'][key]
            maximum = 0.
            unequal = 0
            for begin in range(0,len(ids),2048):
                a,b = x[begin:begin+2048],y[begin:begin+2048]
                maximum = max(maximum,float(np.abs(a-b).max()))
                unequal += int(np.count_nonzero(a!=b))
            values[key] = dict(max_abs_difference=maximum,unequal_elements=unequal)
        values['argmax_disagreements'] = int(np.count_nonzero(new[view]['restored']['logits'].argmax(1)!=old[view]['restored']['logits'].argmax(1)))
        result[view] = values
    value = dict(status='COMPARED',role=role,count=len(ids),views=result,TX_truth_reads=0,
                 native_predictions_reused=True,native_target_reruns=0,batch_note='Original128 vs collapsed128;floating operation order differs;all differences reported')
    write_json(base.row_folder(current,role)/'native_agreement.json',value)
    return value


def score(args, role):
    if __import__('os').environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Independent CPU truth-side scorer required')
    # Validate ALL collapse and original predictions and compare them without truth.
    native_agreement(args,role)
    current = row_args(args)
    base.score(current,role,reference_loader=reference)
    if role == 'target':
        new = summary_metrics(read_json(current.output/'target_confirmation'/d.KIND/'summary.json'),'target',training_F_access=False)
        old = summary_metrics(read_json(args.compact_parent/'prune_refine/target_confirmation'/d.KIND/'summary.json'),'target')
        r17 = summary_metrics(read_json(args.teacher_run/'unroll_unshared/target_confirmation'/d.KIND/'summary.json'),'target')
        write_json(current.output/'assessment.json',dict(status='ANALYZED',arm=args.arm,
            user_acceptance=compact_acceptance(new,old),original_R17_acceptance=target_acceptance(new,r17),
            target_vs_R46=compare(new,old),target_vs_R17=compare(new,r17),target_feedback=False))
