import json
import math
import os
import sys
import time
import numpy as np
import torch
from . import design as d
from .runtime import initialize, save_checkpoint, load_checkpoint, validate_calibration
from .views import ViewLoss
from experiments.pamf_reception_restore.runner import pair_loss
from experiments.pamf_restoration.runner import array_tensor, tensor_array, read_json
from experiments.pamf_restoration.data import write_json


def learning_rate(update):
    if not 0 <= update < 4940:
        raise ValueError('Incremental update outside fixed budget')
    if update < 247:
        return 3e-5*(update+1)/247
    return 3e-5*(.1+.9*(1+math.cos(math.pi*(update-247)/4692))/2)


def train(args):
    if args.row not in d.ROWS:
        raise ValueError('Registered analytic comparison row required')
    torch.manual_seed(d.MODEL_SEED); torch.cuda.manual_seed_all(d.MODEL_SEED)
    model, parent, provenance = initialize(args.parent_run, 'cuda:0')
    parameters = list(model.prior.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=3e-5, weight_decay=1e-4)
    calibration = read_json(args.output/'calibration.json') if args.row == 'analytic_views' else None
    if calibration is not None:
        validate_calibration(calibration)
    view_loss = ViewLoss(calibration['scales']).to('cuda:0') if calibration is not None else None
    root = args.output/args.row; root.mkdir()
    row = root/d.KIND; row.mkdir()
    (root/'reference_controls').symlink_to(args.parent_run/'reference_controls', target_is_directory=True)
    write_json(root/'gpu_ready.json', dict(pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv, GPU=os.environ.get('CUDA_VISIBLE_DEVICES')))
    write_json(root/'resolved_config.json', dict(d.configuration(), candidate=args.row, frozen_parent=parent,
        analytic_training_forwards=view_loss is not None))
    cache = args.parent_action/'cache'
    if args.parent_action.name != d.ACTION:
        raise ValueError('Original R07 complete source observations required')
    clean = np.load(cache/'train/clean.npy', mmap_mode='r', allow_pickle=False)
    roles = np.load(cache/'train/roles.npy', allow_pickle=False)
    if clean.shape != (63000,2,256) or (roles=='L_s').sum()!=6300 or (roles=='U_s').sum()!=56700:
        raise ValueError('Exactly full original L/U without V required')
    rng = np.random.default_rng(d.DATA_SEED); updates = seen_l = seen_u = forwards = 0
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    with (root/'training.jsonl').open('x', encoding='utf-8') as log:
        for epoch in range(1,21):
            received = np.load(cache/'epochs'/f'{epoch:02d}'/'received.npy', mmap_mode='r', allow_pickle=False)
            if received.shape != clean.shape:
                raise ValueError('Complete original epoch observation array required')
            with np.load(cache/'epochs'/f'{epoch:02d}'/'groups.npz',allow_pickle=False) as archive:
                groups={key:archive[key] for key in archive.files}
            order=rng.permutation(63000); model.train(); clipped=0; strata={}
            totals=dict(loss=0.,iq_nmse=0.,log_power=0.,analytic=0.,frequency=0.,polynomial=0.,statistics=0.,response=0.)
            for begin in range(0,len(order),256):
                index=order[begin:begin+256]; x=array_tensor(clean[index],'cuda:0'); r=array_tensor(received[index],'cuda:0')
                output=model(r); paired,metrics=pair_loss(output,x)
                extra,parts=view_loss(output,x) if view_loss is not None else (torch.zeros_like(paired),{})
                forwards+=2 if view_loss is not None else 0
                loss=(paired+extra).mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite paired/analytic objective')
                lr=learning_rate(updates)
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);clipped+=int(norm>1)
                optimizer.step();updates+=1
                seen_l+=int((roles[index]=='L_s').sum());seen_u+=int((roles[index]=='U_s').sum())
                totals['loss']+=float(loss)*len(index);totals['analytic']+=float(extra.sum())
                for key in ('iq_nmse','log_power'):totals[key]+=float(metrics[key].sum())
                for key,value in parts.items():totals[key]+=float(value.sum())
                values=tensor_array(metrics['iq_nmse']);masks={key:roles[index]==key for key in ('L_s','U_s')}
                for key,labels in groups.items():masks.update({key+':'+str(label):labels[index]==label for label in np.unique(labels)})
                for key,mask in masks.items():
                    item=strata.setdefault(key,dict(count=0,sum=0.));item['count']+=int(mask.sum());item['sum']+=float(values[mask].sum())
                if updates==1:
                    save_checkpoint(row/'initial_source_smoke.pt',model,optimizer,epoch,updates,args.row,parent,provenance,calibration)
                    fresh,_=load_checkpoint(row/'initial_source_smoke.pt','cuda:0');model.eval()
                    with torch.no_grad():
                        if not torch.allclose(model(r[:2]),fresh(r[:2]),atol=1e-6,rtol=1e-5):raise ValueError('Actual checkpoint roundtrip differs')
                    del fresh;model.train()
            record=dict(epoch=epoch,updates=updates,losses={key:value/63000 for key,value in totals.items()},
                L_exposures=seen_l,U_exposures=seen_u,learning_rate_last=lr,gradient_clip_fraction=clipped/247,
                analytic_forwards=forwards,strata={key:dict(count=value['count'],mean_iq_nmse=value['sum']/max(value['count'],1)) for key,value in strata.items()},
                seconds=time.perf_counter()-started)
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
            print(json.dumps(dict(event='ANALYTIC_EPOCH_COMPLETE',candidate=args.row,**record)),flush=True)
    if (updates,seen_l,seen_u)!=(4940,126000,1134000):raise ValueError('Full incremental exposure required')
    save_checkpoint(row/'final.pt',model,optimizer,20,updates,args.row,parent,provenance,calibration)
    if any(p.grad is not None for p in model.frozen.parameters()):raise ValueError('Frozen context received parameter gradients')
    write_json(root/'training_complete.json',dict(status='SOURCE_TRAINED',candidate=args.row,epochs=20,updates=updates,
        L_exposures=seen_l,U_exposures=seen_u,TX_labels_read=0,F_outputs_read=0,F_weights_read=0,target_reads=0,
        frozen_context_updates=0,analytic_forwards=forwards,trainable_parameters=sum(p.numel() for p in parameters),
        parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
