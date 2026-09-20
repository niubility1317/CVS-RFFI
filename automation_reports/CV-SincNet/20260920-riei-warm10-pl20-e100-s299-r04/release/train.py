"""STAR-matched source-only CVCNN / RIEI with ordinary hard pseudo labels."""
import argparse
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from Extracter_and_Classifier import Ni_model_2part
from get_dataset_lab_unlab import TestDataset
from source_data import load_source,warm_view
from optimized_training import pseudo_objective
from riei_model import RIEIModel
from riei_losses import mutual_independence_loss, entropy_from_logits

ROOT = Path(__file__).resolve().parent


def tensor(a):
    a = np.ascontiguousarray(a)
    dtype = torch.float32 if a.dtype == np.float32 else torch.long
    assert a.dtype in (np.float32, np.int64)
    return torch.frombuffer(bytearray(a.tobytes()), dtype=dtype).reshape(a.shape)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def append(path, value):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')


def create_model(method, device):
    if method == 'riei':
        return RIEIModel(10, 5).to(device)
    model = Ni_model_2part().to(device)
    # Materialize STAR LazyLinear without updating BN state or training data.
    model.eval()
    with torch.no_grad():
        model.encoder(torch.zeros(2, 2, 2048, device=device))
    model.classifier.linear1.sigma.requires_grad_(False)
    return model


def forward(model, x, method):
    if method == 'riei':
        return model(x)['emitter_logits']
    return model.classifier(model.encoder(x)[:, 128:], stochastic=False)


def make_optimizers(model, method):
    if method == 'cvcnn':
        return [torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.0001)]
    heads = list(model.ec.parameters()) + list(model.rc.parameters())
    return [torch.optim.Adam(heads, lr=.0001), torch.optim.Adam(model.fed.parameters(), lr=.0001)]


def train_step(model, method, optimizers, xl, yl, xu, du, epoch, threshold=.9, feature_norm=.0001, pseudo_start=20):
    # Original RIEI optimized alternating algorithm: clean labeled CE_RX/MI/IE,
    # plus concatenated satellite emitter CE. U supplies only ordinary TX PL.
    assert method=='riei'
    model.train()
    for opt in optimizers:opt.zero_grad(set_to_none=True)
    clean=xl[:,0]
    out=model(xl.flatten(0,1))
    ce_clean=F.cross_entropy(out['emitter_logits'][0::2],yl[:,0],reduction='sum')
    ce_sg=F.cross_entropy(out['emitter_logits'][1::2],yl[:,0],reduction='sum')
    rx_ce=F.cross_entropy(out['receiver_logits'][0::2],yl[:,1],reduction='sum')
    norm=out['z_e'][0::2].square().mean()+out['z_r'][0::2].square().mean()
    pl=ce_clean.new_zeros(());selected=selected_clean=selected_sg=0
    if xu is not None and epoch>=pseudo_start:
        logits=model(xu.flatten(0,1))['emitter_logits']
        pl,meta=pseudo_objective(logits,'self',threshold,True)
        selected=int(meta['used'].sum());selected_clean=int(meta['used'][0::2].sum());selected_sg=int(meta['used'][1::2].sum())
    loss=ce_clean+ce_sg+rx_ce+feature_norm*norm+.65*pl
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite RIEI supervised loss')
    loss.backward()
    for opt in optimizers:opt.step()
    for head in (model.ec,model.rc):head.requires_grad_(False)
    out=model(clean)
    mi=mutual_independence_loss(out['z_e'],out['z_r'],reduction='sum')
    ie=entropy_from_logits(out['cross_emitter_logits'],reduction='sum')+entropy_from_logits(out['cross_receiver_logits'],reduction='sum')
    norm_dis=out['z_e'].square().mean()+out['z_r'].square().mean()
    dis=1.2*mi-1.2*ie+feature_norm*norm_dis
    if not torch.isfinite(dis):raise FloatingPointError('nonfinite RIEI disentanglement loss')
    optimizers[1].zero_grad(set_to_none=True);dis.backward();optimizers[1].step()
    for head in (model.ec,model.rc):head.requires_grad_(True)
    return dict(loss=float(loss.detach()),ce_clean=float(ce_clean.detach()),ce_sg=float(ce_sg.detach()),
        weighted_ce_clean=float(ce_clean.detach()),weighted_ce_sg=float(ce_sg.detach()),supervised_ce_weight=1.,
        rx_ce=float(rx_ce.detach()),feature_norm=float(norm.detach()),mi=float(mi.detach()),ie=float(ie.detach()),
        disentangle_loss=float(dis.detach()),pseudo_ce=float(pl.detach()),weighted_pseudo_ce=.65*float(pl.detach()),
        pseudo_active=xu is not None and epoch>=pseudo_start,selected=selected,selected_clean=selected_clean,selected_sg=selected_sg,
        labeled_views=2*len(xl),unlabeled_views=0 if xu is None else 2*len(xu),gradient_clip_active=False)


@torch.no_grad()
def predict(model, method, x, device):
    model.eval()
    outputs=[]
    for (xb,) in DataLoader(TensorDataset(tensor(x)),batch_size=50,shuffle=False):
        outputs.extend(forward(model,xb.to(device),method).softmax(1).cpu().tolist())
    return np.asarray(outputs,dtype=np.float32)


def score(probability_path, truth):
    # Scorer reads fixed predictions back before joining labels.
    probabilities=np.load(probability_path)
    predictions=probabilities.argmax(1)
    y=truth[:,0]
    assert len(y)==len(predictions)
    cm=np.zeros((10,10),dtype=np.int64)
    np.add.at(cm,(y,predictions),1)
    denom=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*np.diag(cm),denom,out=np.zeros(10),where=denom>0)
    return dict(n=len(y),accuracy=float((predictions==y).mean()),macro_f1=float(f1.mean()),confusion=cm.tolist())


def evaluate(model,method,arrays,device,out):
    out.mkdir(parents=True,exist_ok=False)
    for name,(x,y) in arrays.items():
        p=predict(model,method,x,device)
        np.save(out/(name+'_probabilities.npy'),p)
        np.save(out/(name+'_ids.npy'),y[:,2:4] if y.shape[1]>=4 else np.arange(len(y)))
    result={}
    for name,(x,y) in arrays.items():
        np.save(out/(name+'_truth.npy'),y[:,0])
        result[name]=score(out/(name+'_probabilities.npy'),y)
    write_json(out/'metrics.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',choices=['cvcnn','riei'],required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--use-pl',type=int,choices=[0,1],required=True)
    parser.add_argument('--data-root',required=True)
    args=parser.parse_args()
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    write_json(out/'resolved_config.json',dict(manifest,method=args.method,use_pl=bool(args.use_pl),pid=os.getpid(),cwd=str(ROOT)))
    random.seed(299);np.random.seed(299);torch.manual_seed(299);torch.cuda.manual_seed_all(299)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda:0')
    xl,yl,xv,yv,xu,yu=load_source(args.data_root,out/'splits',bool(args.use_pl))
    assert set(yl[:,2])==set([0,1,2,3,5])
    if xu is not None:assert 4 not in set(yu[:,2]) and (yu[:,0]==-1).all()
    warm=xl.copy();warm[:,1]=warm_view(xl[:,0])
    arrays={'source_clean':(xv,yv)}
    sg={rx:np.load(Path(args.data_root)/f'rx_{rx}_sg_x.npy',mmap_mode='r') for rx in [0,1,2,3,5]}
    arrays['source_sg']=(np.stack([sg[int(y[2])][int(y[3])] for y in yv]).astype(np.float32),yv)
    for view in ['clean','sg']:
        arrays['target_'+view]=TestDataset(10,data_root=args.data_root,target_rx=4,test_variant=view)
    label_dataset=TensorDataset(tensor(xl),tensor(yl))
    warm_dataset=TensorDataset(tensor(warm),tensor(yl))
    label_generator=torch.Generator()
    ld=DataLoader(label_dataset,batch_size=64,shuffle=True,generator=label_generator)
    wd=DataLoader(warm_dataset,batch_size=64,shuffle=True,generator=label_generator)
    ud=None
    if args.use_pl:
        ud=DataLoader(TensorDataset(tensor(xu),tensor(yu)),batch_size=256,shuffle=True,generator=torch.Generator())
    write_json(out/'data_summary.json',dict(labeled=len(xl),unlabeled_loaded=0 if xu is None else len(xu),
        validation=len(xv),source_rxs=[0,1,2,3,5],target_rx=4,use_pl=bool(args.use_pl),steps_per_epoch=len(ld),
        no_pl_does_not_index_unlabeled_iq=not bool(args.use_pl)))
    model=create_model(args.method,device)
    optimizers=make_optimizers(model,args.method)
    for epoch in range(1,101):
        started=time.perf_counter();torch.cuda.reset_peak_memory_stats(device)
        totals={};steps=0
        label_generator.manual_seed(299+epoch)
        active_loader=wd if epoch<=10 else ld
        ui=None
        if ud is not None and epoch>=20:
            ud.generator.manual_seed(1299+epoch);ui=iter(ud)
        for step,(bl,by) in enumerate(active_loader):
            bl=bl.to(device);bu=bd=None
            if ui is not None:
                try:bu,bd=next(ui)
                except StopIteration:ui=iter(ud);bu,bd=next(ui)
                bu=bu.to(device);bd=bd[:,1].to(device)
            metrics=train_step(model,args.method,optimizers,bl,by.to(device),bu,bd,epoch,
                               feature_norm=.0001,pseudo_start=20)
            append(out/'batches.jsonl',dict(epoch=epoch,batch=step,**metrics,
                augmentation='moderate_Loo_sigma1' if epoch<=10 else 'original_fixed_Loo_sigma3',
                loo_sigma=1. if epoch<=10 else 3.))
            for key,value in metrics.items():totals[key]=totals.get(key,0)+float(value)
            steps+=1
        torch.cuda.synchronize(device)
        row=dict(epoch=epoch,steps=steps,train_seconds=time.perf_counter()-started,lr=.0001,
                 peak_memory_bytes=torch.cuda.max_memory_allocated(device),
                 averages={k:v/steps for k,v in totals.items()},
                 selected=int(totals['selected']),unlabeled_views=int(totals['unlabeled_views']))
        torch.save(dict(epoch=epoch,model_state_dict=model.state_dict(),
                        optimizer_states=[o.state_dict() for o in optimizers]),out/'latest.pth')
        if epoch%10==0:
            t=time.perf_counter()
            row['evaluation']=evaluate(model,args.method,arrays,device,out/'evaluations'/f'epoch_{epoch:03d}')
            row['evaluation_seconds']=time.perf_counter()-t
        append(out/'metrics_epoch.jsonl',row)
        print(json.dumps(row),flush=True)
    final={}
    for name,state in [('student',model)]:
        assert state is not None
        final[name]=evaluate(state,args.method,arrays,device,out/'final_last_epoch'/name)
    write_json(out/'COMPLETE.json',dict(status='ARTIFACTS_COMPLETE',epoch=100,method=args.method,use_pl=bool(args.use_pl),
                                      primary='student',metrics=final))


if __name__=='__main__':
    main()
