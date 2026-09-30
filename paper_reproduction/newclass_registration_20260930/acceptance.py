"""Real-IQ acceptance for pinned ISSL and explicit LoRa WiSig port.

Uses a diagnostic split, not a paper accuracy reproduction or CVS benchmark.
No query labels are loaded here. Predictions are saved before independent scoring.
"""
import argparse
import copy
import csv
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.neighbors import KNeighborsClassifier
from runtime import verify_source, load_module, expand_head, channel_spectrogram

def write_json(path, payload):
    Path(path).write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

class Logger:
    def __init__(self, out):
        self.out=out; self.rows=[]
    def log(self, **row):
        self.rows.append(row)
        line=json.dumps(row,ensure_ascii=False,allow_nan=False)
        print(line,flush=True)
        with (self.out/'steps.jsonl').open('a',encoding='utf-8') as f: f.write(line+'\n')
    def finish(self):
        keys=sorted({k for r in self.rows for k in r})
        with (self.out/'steps.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(self.rows)

def norm(x):
    x=np.asarray(x,dtype=np.float32)
    std=float(np.std(x))
    if std<=0 or not np.isfinite(std): raise ValueError('Nonfinite/constant IQ')
    return (x-np.mean(x))/std

def batch_tensor(x, device, crop=None):
    samples=[]
    for a in x:
        a=a.T
        if crop is not None:
            start=random.randint(0,a.shape[-1]-crop); a=a[:,start:start+crop]
        samples.append(norm(a))
    return torch.from_numpy(np.stack(samples)).to(device)

def batches(n, size):
    order=np.random.permutation(n)
    for s in range(0,n-size+1,size): yield order[s:s+size]

def grad_norm(model):
    values=[p.grad.detach().square().sum() for p in model.parameters() if p.grad is not None]
    return float(torch.sqrt(torch.stack(values).sum())) if values else 0.

def predict(model, x, device, size=64):
    model.eval(); output=[]
    with torch.no_grad():
        for s in range(0,len(x),size):
            logits,_=model(batch_tensor(x[s:s+size],device))
            if not torch.isfinite(logits).all(): raise FloatingPointError('Nonfinite inference')
            output.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(output)

def train_ce(model, x, y, epochs, batch, lr, stage, device, log, teacher=None, old=3, kd=None, crop=None):
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,100,.5)
    for epoch in range(epochs):
        start=time.perf_counter(); totals=[]
        for step, ix in enumerate(batches(len(x),batch)):
            model.train(); a=batch_tensor(x[ix],device,crop=crop); target=torch.tensor(y[ix],device=device)
            logits,_=model(a); ce=F.cross_entropy(logits,target)
            distill=None
            if teacher is not None:
                old_logits,_=teacher(a) # author teacher train/BN behavior retained
                distill=kd.MultiClassCrossEntropy(logits[:,:old],old_logits)
            loss=ce if distill is None else .5*ce+.5*distill
            optimizer.zero_grad(); loss.backward(); g=grad_norm(model)
            if not torch.isfinite(loss) or not np.isfinite(g): raise FloatingPointError(stage)
            optimizer.step(); totals.append(float(loss.detach()))
            log.log(stage=stage,epoch=epoch,step=step,loss=float(loss.detach()),ce=float(ce.detach()),
                    kd=None if distill is None else float(distill.detach()),kd_weight=None if distill is None else .5,
                    lr=optimizer.param_groups[0]['lr'],grad_norm=g,source_validation=None,
                    validation_note='diagnostic has no model-selection validation; no query feedback')
        if not totals: raise ValueError('drop_last would yield zero training steps')
        scheduler.step()
        log.log(stage=stage,epoch=epoch,summary=True,loss=float(np.mean(totals)),seconds=time.perf_counter()-start)

def issl(args,x,y,query,out,log,device):
    source=Path(args.issl_source); verify_source(source,'ISSL')
    resnet=load_module(source/'resnet.py','issl_pinned_resnet')
    kd=load_module(source/'loss.py','issl_pinned_loss') # unmodified author CUDA functions
    if device.type!='cuda': raise ValueError('Unmodified ISSL losses require CUDA')
    model=resnet.resnet18(num_classes=3,drop_out=.1).to(device)
    old_mask=y<3
    initial=copy.deepcopy(model.state_dict())
    train_ce(model,x[old_mask],y[old_mask],args.base_epochs,128,.03,'base',device,log)
    base=copy.deepcopy(model)
    torch.save({'state_dict':base.state_dict(),'initialization':'scratch','training_ids_ref':str(args.data)+'.json'},out/'base.pt')
    a=predict(base,query,device)
    # main_self_supervised tuple unpack runtime repair; expansion itself unchanged.
    # Separate official scripts serialize after base eval: no base gradients,
    # key stays in eval while q switches to train inside the SSL loop.
    model=copy.deepcopy(base)
    model,_=expand_head(model,3); key=copy.deepcopy(model)
    ssl_x=np.concatenate((x[old_mask],x)) # upstream loader repeats old train samples
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,100,.8)
    kk=random.randint(1,1000)
    if kk>len(ssl_x): raise ValueError('Original random queue initialization exceeds training set')
    qi=np.random.permutation(len(ssl_x))[:kk]
    queue,_=key(batch_tensor(ssl_x[qi],device,crop=218)); queue=queue.detach()
    # Confirm the source KD issue numerically; do not silently fix the method.
    probe=torch.randn(4,3,device=device,requires_grad=True)
    kd.MultiClassCrossEntropy(probe,torch.randn_like(probe)).backward()
    kd_detached=probe.grad is None
    if not kd_detached: raise AssertionError('Pinned KD semantics changed')
    for epoch in range(args.ssl_epochs):
        start=time.perf_counter(); losses=[]
        for step,ix in enumerate(batches(len(ssl_x),64)):
            model.train(); qx=batch_tensor(ssl_x[ix],device,crop=218); kx=batch_tensor(ssl_x[ix],device,crop=218)
            q,_=model(qx); k,_=key(kx); ki,_=key(qx)
            k=k.detach(); ki=ki.detach()
            loss_old=kd.MultiClassCrossEntropy(q[:,:3],ki[:,:3],20)
            q=q/torch.norm(q,dim=1,keepdim=True); k=k/torch.norm(k,dim=1,keepdim=True)
            loss_new=kd.loss_function2(q,k,queue,5)
            loss=.5*loss_new+.5*loss_old
            # Missing zero_grad is upstream behavior. Keep it explicitly in this variant.
            loss.backward(); g=grad_norm(model)
            if not torch.isfinite(loss) or not np.isfinite(g): raise FloatingPointError('ssl')
            optimizer.step(); queue=torch.cat((queue,k),0)
            if len(queue)>1000: queue=queue[64:,:]
            with torch.no_grad():
                for kp,qp in zip(key.parameters(),model.parameters()): kp.copy_(.99*kp+.01*qp)
            losses.append(float(loss.detach()))
            log.log(stage='ssl',epoch=epoch,step=step,loss=float(loss.detach()),contrastive=float(loss_new.detach()),
                    kd=float(loss_old.detach()),kd_weight=.5,contrastive_weight=.5,kd_student_gradient=False,
                    zero_grad=False,negative_queue_layout='author_reshape_not_transpose',queue_size=len(queue),
                    momentum=.99,temperature_contrastive=5,temperature_kd=20,lr=1e-5,grad_norm=g,source_validation=None)
        scheduler.step()
        log.log(stage='ssl',epoch=epoch,summary=True,loss=float(np.mean(losses)),seconds=time.perf_counter()-start)
    torch.save({'state_dict':model.state_dict(),'parent':'base.pt','stage':'ssl'},out/'ssl.pt')
    train_ce(model,x,y,args.downstream_epochs,128,.03,'downstream_transfer',device,log,crop=218)
    c=predict(model,query,device)
    torch.save({'state_dict':model.state_dict(),'parent':'ssl.pt','stage':'downstream'},out/'downstream.pt')
    # Official standalone incremental baseline branch, initialized from the same base.
    baseline,teacher=expand_head(copy.deepcopy(base),3)
    train_ce(baseline,x,y,args.downstream_epochs,128,1e-4,'incremental_baseline',device,log,teacher,3,kd)
    cb=predict(baseline,query,device)
    torch.save({'state_dict':baseline.state_dict(),'parent':'base.pt','stage':'incremental_baseline'},out/'incremental_baseline.pt')
    return {'A':a,'C':c,'C_incremental_baseline':cb}, {'kd_student_gradient':False,
        'old_only_adaptation_B':None,'B_note':'Original workflow has no old-only adaptation stage',
        'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'parameter_update_verified':any(not torch.equal(v.cpu(),initial[k].cpu()) for k,v in base.state_dict().items() if v.is_floating_point())}

def lora(args,x,y,query,out,log,device):
    source=Path(args.lora_torch_source); verify_source(source,'LoRa_RFFI_Torch')
    official=Path(args.lora_source); verify_source(official,'LoRa_RFFI')
    module=load_module(source/'Openset_RFFI/deep_learning_models.py','lora_third_party_model')
    prep=load_module(official/'Openset_RFFI_TIFS/dataset_preparation.py','lora_official_preprocessing')
    # Numerical parity of author operations at the ORIGINAL 8192-sample length.
    probe=np.random.default_rng(91).normal(size=(2,8192,2))
    original=prep.ChannelIndSpectrogram().channel_ind_spectrogram(probe[...,0]+1j*probe[...,1])
    adapted=channel_spectrogram(probe,256,128)
    parity=float(np.max(np.abs(original-adapted)))
    if not np.allclose(original,adapted,atol=1e-5,rtol=1e-5): raise AssertionError('Preprocess parity')
    # WiSig 256 IQ cannot yield the original adjacent STFT frames.
    augmented=prep.awgn((x[...,0]+1j*x[...,1]).astype(np.complex128),np.arange(20,80))
    spectrum=channel_spectrogram(np.stack((augmented.real,augmented.imag),-1),64,32)
    model=module.FeatureExtractor(input_shape=spectrum.shape[1:]).to(device)
    # Restore author Keras Glorot/zero initialization for conv/dense instead of Torch Kaiming.
    for layer in model.modules():
        if isinstance(layer,(torch.nn.Conv2d,torch.nn.Linear)):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None: torch.nn.init.zeros_(layer.bias)
    optimizer=torch.optim.RMSprop(model.parameters(),lr=.001,alpha=.9,eps=1e-7)
    old=np.flatnonzero(y<3)
    for epoch in range(args.lora_epochs):
        start=time.perf_counter(); values=[]
        for step in range(len(old)//32):
            ai=[]; pi=[]; ni=[]
            for _ in range(32):
                label=int(np.random.choice([0,1,2])); neg=int(np.random.choice([v for v in [0,1,2] if v!=label]))
                ai.append(np.random.choice(np.flatnonzero(y==label))); pi.append(np.random.choice(np.flatnonzero(y==label))); ni.append(np.random.choice(np.flatnonzero(y==neg)))
            tensors=[torch.from_numpy(spectrum[ix]).permute(0,3,1,2).to(device) for ix in (ai,pi,ni)]
            model.train(); embeddings=[model(t) for t in tensors]
            loss=module.triplet_loss(*embeddings,margin=.1)
            optimizer.zero_grad(); loss.backward(); g=grad_norm(model)
            if not torch.isfinite(loss) or not np.isfinite(g): raise FloatingPointError('LoRa')
            optimizer.step(); values.append(float(loss.detach()))
            log.log(stage='lora_triplet',epoch=epoch,step=step,loss=float(loss.detach()),triplet_margin=.1,
                    lr=.001,grad_norm=g,source_validation=None,validation_note='bounded execution diagnostic, fixed epochs; no early-stop model selection')
        log.log(stage='lora_triplet',epoch=epoch,summary=True,loss=float(np.mean(values)),seconds=time.perf_counter()-start)
    torch.save({'state_dict':model.state_dict(),'input_shape':spectrum.shape[1:],'initialization':'scratch'},out/'extractor.pt')
    enrollment=channel_spectrogram(x,64,32)
    qs=channel_spectrogram(query,64,32)
    def features(data):
        model.eval(); result=[]
        with torch.no_grad():
            for start in range(0,len(data),64):
                result.append(model(torch.from_numpy(data[start:start+64]).permute(0,3,1,2).to(device)).cpu().numpy())
        return np.concatenate(result)
    ef=features(enrollment); qf=features(qs)
    old_knn=KNeighborsClassifier(n_neighbors=15,metric='euclidean').fit(ef[y<3],y[y<3])
    full_knn=KNeighborsClassifier(n_neighbors=15,metric='euclidean').fit(ef,y)
    return {'A':old_knn.predict(qf),'C':full_knn.predict(qf)}, {'original_preprocessing_max_error':parity,
        'preprocessing_original_shape':list(original.shape),'wisig_spectrogram_shape':list(spectrum.shape),
        'B_note':'Frozen extractor plus same-domain support: A equals B; no gradient adaptation',
        'port':'third-party Torch, not author Keras; full numerical training parity unverified',
        'optimizer_difference':'Torch RMSprop epsilon placement differs from TF2.1; no bitwise parity claim',
        'parameters':sum(p.numel() for p in model.parameters()),'phase2_trainable_parameters':0}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=['issl','lora'],required=True)
    p.add_argument('--data',required=True); p.add_argument('--output',required=True)
    p.add_argument('--issl-source',default='local_artifacts/external_refs/ISSL_20260930')
    p.add_argument('--lora-source',default='local_artifacts/external_refs/LoRa_RFFI_20260930')
    p.add_argument('--lora-torch-source',default='local_artifacts/external_refs/LoRa_RFFI_Torch_20260930')
    p.add_argument('--base-epochs',type=int,default=1); p.add_argument('--ssl-epochs',type=int,default=1)
    p.add_argument('--downstream-epochs',type=int,default=1); p.add_argument('--lora-epochs',type=int,default=1)
    p.add_argument('--seed',type=int,default=392005); args=p.parse_args()
    for budget in (args.base_epochs,args.ssl_epochs,args.downstream_epochs,args.lora_epochs):
        if budget<1: raise ValueError('Positive epoch budget required')
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=False
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda': torch.cuda.manual_seed_all(args.seed); torch.cuda.reset_peak_memory_stats()
    with np.load(args.data,allow_pickle=False) as data:
        mask=data['train']; x=data['x'][mask]; y=data['y'][mask] # query labels never read
        query=data['x'][~mask]; ids=data['ids'][~mask]
    if set(y.tolist())!=set(range(6)) or x.shape[1:]!=(256,2): raise ValueError('Unexpected diagnostic data')
    config=vars(args)|{'device':str(device),'torch':torch.__version__,'pid':os.getpid(),'cwd':str(Path.cwd()),
                      'claim_scope':'real WiSig execution acceptance only','checkpoint':'scratch',
                      'gpu_name':torch.cuda.get_device_name() if device.type=='cuda' else None}
    write_json(out/'resolved_config.json',config); print(json.dumps(config),flush=True)
    log=Logger(out); start=time.perf_counter()
    predictions,checks=(issl if args.method=='issl' else lora)(args,x,y,query,out,log,device)
    # Predictions are immutable once this artifact is created, before the scorer reads truth.
    with (out/'predictions.npz').open('xb') as f: np.savez_compressed(f,ids=ids,**predictions)
    log.finish()
    checks.update(status='VERIFIED',method=args.method,seconds=time.perf_counter()-start,
                  peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
                  steps=sum(not r.get('summary',False) for r in log.rows),query_truth_used_for_training=False)
    write_json(out/'acceptance.json',checks); print(json.dumps(checks),flush=True)

if __name__=='__main__': main()
