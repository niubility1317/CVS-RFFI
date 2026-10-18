"""Pinned IB-RFF two-stage author-code pilot, with isolated prediction/scoring.

Model, HSIC, RNG order, optimizer, split, and source-only selection follow the
pinned author implementation. This is not a verified reproduction of a paper
table. All steps use a fresh immutable run root. No author checkpoint is read.
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

class NumpyExportUnpickler(pickle.Unpickler):
    """Read NumPy2-created arrays with the author's NumPy1 environment."""
    def find_class(self,module,name):
        if module.startswith('numpy._core.'):
            module='numpy.core.'+module[len('numpy._core.'):]
        return super().find_class(module,name)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write('\n')

def config(runtime):
    return read_json(runtime['candidate'])

def contract(conf):
    rx=conf['rx_author_order']
    if conf['round'] != 0 or len(rx)!=12 or len(set(rx))!=12:
        raise ValueError('Only preregistered author ManySig round0 is implemented')
    return {'dataset':'ManySig','export_run':conf['data_export_run'],
            'representation':conf['representation'],'source_rx':rx[3:],
            'target_rx':rx[:3],'source_days':conf['train_days'],
            'packet_rows':list(range(*conf['packet_rows'])),
            'tx_indices':list(range(6)),'validation_fraction':conf['validation_fraction'],
            'split_seed':conf['split_seed'],'physical_id':'ManySig:TX:RX:day:packet_row'}

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

def modules(runtime):
    sys.path.insert(0,runtime['model_root'])
    from backbones.FAN import FAN
    from utils.hsic import hsic_normalized_cca
    return FAN,hsic_normalized_cca

def model(runtime,conf):
    FAN,_=modules(runtime)
    return FAN(seq_len=256,in_chans=2,patch_size=conf['patch_size'],
               embed_dim=conf['embed_dim'],mlp_ratio=conf['mlp_ratio'],num_classes=6,
               bottleneck_type=conf['bottleneck_type'],use_SAF=True,use_FAF=True)

def read_block(data_root,rx,day):
    path=Path(data_root)/'ManySig/non_equalized'/f'date{day}'/f'rx_{rx}_data.pkl'
    with path.open('rb') as handle:
        payload=NumpyExportUnpickler(handle).load()
    blocks=[]
    for tx in range(6):
        raw=np.asarray(payload['data'][tx])[:100]
        if raw.shape!=(100,256,2):
            raise ValueError(f'Unexpected physical packet shape: {path}: {raw.shape}')
        x=np.transpose(raw,(0,2,1)).copy()
        if not np.isfinite(x).all():
            raise ValueError(f'Invalid IQ in {path}')
        # Preserve the author's float32, per-sample operation order.
        for i in range(len(x)):
            power=np.sum(x[i,0,:]**2+x[i,1,:]**2)/x.shape[2]
            if not np.isfinite(power) or power<=0:
                raise ValueError(f'Invalid power: {path}, TX={tx}, row={i}')
            x[i]=x[i]/np.sqrt(power)
        blocks.append(x)
    return np.concatenate(blocks),np.repeat(np.arange(6),100)

def source_data(runtime,conf):
    # This function never constructs or opens a target path.
    ct=contract(conf)
    physical=read_json(runtime['physical_roles'])
    if physical['rx_author_order']!=conf['rx_author_order'] or physical['source']!=ct['source_rx'] or physical['target']!=ct['target_rx']:
        raise ValueError('Existing export does not match the registered physical roles')
    if physical['asknet_wavemlp']['packet_rows']!=ct['packet_rows']:
        raise ValueError('Packet IDs differ from the reused export')
    xs,ys=[],[]
    for rx in ct['source_rx']:
        for day in ct['source_days']:
            x,y=read_block(runtime['data_root'],rx,day)
            xs.append(x);ys.append(y)
    x=np.concatenate(xs);y=np.concatenate(ys)
    train,val=train_test_split(np.arange(len(x)),test_size=conf['validation_fraction'],random_state=conf['split_seed'])
    if len(train)!=7560 or len(val)!=3240 or np.intersect1d(train,val).size:
        raise ValueError('Author source split count/overlap mismatch')
    return (x[train],y[train]),(x[val],y[val])

def load_checkpoint(path,expected,device):
    # Only state_dicts produced by this scratch run are permitted.
    kwargs={'map_location':device}
    import inspect
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only']=True
    state=torch.load(path,**kwargs)
    if state.get('source_contract')!=expected or state.get('upstream_commit')!='5a03a04da86c4bbf5cf90f41a55c268bf2152203':
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
    return state

def save_checkpoint(path,net,conf,stage,epoch,val_loss):
    torch.save({'state_dict':net.state_dict(),'source_contract':contract(conf),
                'upstream_commit':conf['upstream_commit'],'stage':stage,
                'epoch':epoch,'source_val_loss':val_loss,'model_seed':conf['model_seed']},path)

def smoke(runtime):
    conf=config(runtime);seed_everything(conf['model_seed'])
    train,_=source_data(runtime,conf)
    net=model(runtime,conf).to(runtime['device'])
    x=torch.tensor(train[0][:4],device=runtime['device'])
    y=torch.tensor(train[1][:4],dtype=torch.long,device=runtime['device'])
    hidden,logits=net(x)
    _,hsic=modules(runtime)
    loss=F.nll_loss(F.log_softmax(logits,dim=1),y)+conf['stage2_alpha']*hsic(hidden.flatten(1),x.flatten(1))
    loss.backward()
    if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None):
        raise ValueError('Nonfinite source smoke')
    torch.optim.Adam(net.parameters(),lr=conf['lr']).step()
    out=Path(runtime['run_root'])/'smoke'
    out.mkdir()
    save_checkpoint(out/'source_smoke.pt',net,conf,'source-smoke',0,float(loss.detach().cpu()))
    loaded=model(runtime,conf).to(runtime['device'])
    loaded.load_state_dict(load_checkpoint(out/'source_smoke.pt',contract(conf),runtime['device'])['state_dict'],strict=True)
    net.eval();loaded.eval()
    with torch.no_grad():
        if not torch.equal(net(x)[1],loaded(x)[1]):
            raise ValueError('Source checkpoint roundtrip differs')
    write_json(out/'result.json',{'status':'PASS','source_batch':4,'checkpoint_roundtrip':True,'query_access':False,'fresh_training_will_discard_smoke_weights':True})
    print('SOURCE_CHECKPOINT_NO_QUERY_SMOKE_PASS',flush=True)

def train_stage(runtime,stage):
    conf=config(runtime);seed_everything(conf['model_seed'])
    train,val=source_data(runtime,conf)
    train_loader=DataLoader(TensorDataset(torch.Tensor(train[0]),torch.Tensor(train[1])),batch_size=conf['batch_size'],shuffle=True)
    val_loader=DataLoader(TensorDataset(torch.Tensor(val[0]),torch.Tensor(val[1])),batch_size=conf['batch_size'],shuffle=True)
    net=model(runtime,conf).to(runtime['device'])
    root=Path(runtime['run_root']);out=root/stage
    out.mkdir()
    if stage=='stage2':
        parent=load_checkpoint(root/'stage1/best.pt',contract(conf),runtime['device'])
        if parent['stage']!='stage1' or parent['model_seed']!=conf['model_seed']:
            raise ValueError('Wrong stage1 parent')
        net.load_state_dict(parent['state_dict'],strict=True)
    optimizer=torch.optim.Adam(net.parameters(),lr=conf['lr'],weight_decay=conf['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,**conf['scheduler'])
    _,hsic=modules(runtime)
    alpha=conf[stage+'_alpha']
    write_json(out/'resolved_config.json',{'candidate':conf,'source_contract':contract(conf),'stage':stage,'alpha':alpha,'optimizer_reset':True,'train_count':len(train[0]),'val_count':len(val[0]),'query_access':False})
    best=float('inf');no_improve=0
    with (out/'epochs.jsonl').open('x',encoding='utf-8') as handle:
        for epoch in range(1,conf['epochs_per_stage']+1):
            started=time.monotonic();net.train()
            total_loss=total_ce=total_hsic=0.0;correct=0
            for x,y in train_loader:
                x=x.to(runtime['device']);y=y.long().to(runtime['device'])
                optimizer.zero_grad()
                hidden,logits=net(x)
                output=F.log_softmax(logits,dim=1)
                ce=F.nll_loss(output,y)
                penalty=hsic(hidden.flatten(1),x.flatten(1))
                loss=ce+alpha*penalty
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite source loss')
                loss.backward();optimizer.step()
                n=len(x)
                total_loss+=float(loss.detach().cpu())*n
                total_ce+=float(ce.detach().cpu())*n
                total_hsic+=float(penalty.detach().cpu())*n
                correct+=int(output.argmax(1).eq(y).sum())
            net.eval();v_loss=0.0;v_correct=0
            with torch.no_grad():
                for x,y in val_loader:
                    x=x.to(runtime['device']);y=y.long().to(runtime['device'])
                    _,logits=net(x)
                    v_loss+=float(F.nll_loss(F.log_softmax(logits,dim=1),y).cpu())*len(x)
                    v_correct+=int(logits.argmax(1).eq(y).sum())
            v_loss/=len(val[0]);scheduler.step(v_loss)
            improved=v_loss<best
            if improved:
                best=v_loss;no_improve=0
                save_checkpoint(out/'best.pt',net,conf,stage,epoch,v_loss)
            else:
                no_improve+=1
            record={'stage':stage,'epoch':epoch,'loss':total_loss/len(train[0]),'ce':total_ce/len(train[0]),'hsic':total_hsic/len(train[0]),'train_accuracy_percent':100*correct/len(train[0]),'source_val_loss':v_loss,'source_val_accuracy_percent':100*v_correct/len(val[0]),'lr':optimizer.param_groups[0]['lr'],'improved':improved,'seconds':time.monotonic()-started}
            line=json.dumps(record,allow_nan=False);handle.write(line+'\n');handle.flush();print(line,flush=True)
            if no_improve>=conf['early_stop_patience']:
                break
    save_checkpoint(out/'last.pt',net,conf,stage,epoch,v_loss)
    write_json(out/'training_complete.json',{'status':'TRAINING_COMPLETE','epochs':epoch,'best_source_val_loss':best,'checkpoint':'best.pt','query_access':False})

def prepare_queries(runtime):
    conf=config(runtime);root=Path(runtime['run_root'])
    if not (root/'stage2/training_complete.json').exists():
        raise ValueError('Training must freeze before target export')
    query=root/'query';truth=root/'truth';query.mkdir();truth.mkdir()
    xs,ys,physical=[],[],[]
    for rx in contract(conf)['target_rx']:
        for day in conf['test_days']:
            x,y=read_block(runtime['data_root'],rx,day)
            xs.append(x);ys.append(y)
            physical.extend([{'rx':rx,'day':day,'tx':tx,'row':i} for tx in range(6) for i in range(100)])
    x=np.concatenate(xs);y=np.concatenate(ys)
    order=np.random.RandomState(conf['evaluation_order_seed']).permutation(len(x))
    ids=np.asarray([f'q{i:06d}' for i in range(len(x))])
    np.save(query/'iq.npy',x[order],allow_pickle=False)
    np.save(query/'ids.npy',ids,allow_pickle=False)
    np.savez(truth/'labels.npz',ids=ids,labels=y[order])
    write_json(truth/'physical_ids.json',[physical[int(i)] for i in order])
    write_json(query/'metadata.json',{'count':len(x),'scenario':'clean_CRD','has_truth':False})

def predict(runtime):
    conf=config(runtime);root=Path(runtime['run_root'])
    query=root/'query'
    x=np.load(query/'iq.npy',allow_pickle=False)
    ids=np.load(query/'ids.npy',allow_pickle=False)
    if x.shape!=(3600,2,256) or len(ids)!=len(x) or len(set(ids.tolist()))!=len(ids):
        raise ValueError('Unexpected query inputs')
    net=model(runtime,conf).to(runtime['device'])
    state=load_checkpoint(root/'stage2/best.pt',contract(conf),runtime['device'])
    if state['stage']!='stage2':
        raise ValueError('Wrong final checkpoint')
    net.load_state_dict(state['state_dict'],strict=True);net.eval()
    predictions=[]
    with torch.no_grad():
        for start in range(0,len(x),conf['test_batch_size']):
            _,logits=net(torch.tensor(x[start:start+conf['test_batch_size']],device=runtime['device']))
            predictions.append(logits.argmax(1).cpu().numpy())
    path=root/'predictions.npz'
    with path.open('xb') as handle:
        np.savez(handle,ids=ids,predicted=np.concatenate(predictions))
    write_json(root/'predictions_complete.json',{'status':'PREDICTIONS_COMPLETE','count':len(ids),'file':path.name,'query_truth_access':False,'checkpoint':'stage2/best.pt','selected_epoch':state['epoch']})

def score(root):
    root=Path(root)
    completion=read_json(root/'predictions_complete.json')
    if completion['status']!='PREDICTIONS_COMPLETE':
        raise ValueError('Prediction completion required before truth access')
    p=np.load(root/'predictions.npz',allow_pickle=False)
    if set(p.files)!={'ids','predicted'} or len(p['ids'])!=completion['count']:
        raise ValueError('Invalid fixed predictions')
    truth=np.load(root/'truth/labels.npz',allow_pickle=False)
    if not np.array_equal(p['ids'],truth['ids']):
        raise ValueError('Opaque ID join mismatch')
    from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
    y,yp=truth['labels'],p['predicted']
    physical=read_json(root/'truth/physical_ids.json')
    groups={}
    for rx in sorted(set(r['rx'] for r in physical)):
        for day in [3,4]:
            mask=np.asarray([r['rx']==rx and r['day']==day for r in physical])
            groups[f'{rx}/day{day}']={'count':int(mask.sum()),'accuracy_percent':100*accuracy_score(y[mask],yp[mask])}
    result={'status':'SCORED','count':len(y),'accuracy_percent':100*accuracy_score(y,yp),'macro_f1_percent':100*f1_score(y,yp,labels=list(range(6)),average='macro',zero_division=0),'confusion_matrix':confusion_matrix(y,yp,labels=list(range(6))).tolist(),'by_rx_day':groups,'source_selection_only':True,'paper_effect_verified':False}
    write_json(root/'scores.json',result)
    print(json.dumps(result),flush=True)

def worker(runtime_path):
    runtime=read_json(runtime_path);root=Path(runtime['run_root'])
    write_json(root/'worker_started.json',{'pid':os.getpid(),'started_at':time.time()})
    for step in ['smoke','stage1','stage2','prepare-query','predict','score']:
        argv=[sys.executable,'-u',str(Path(__file__).resolve()),step,'--runtime',str(runtime_path)]
        child=subprocess.Popen(argv,cwd=runtime['release_root'])
        write_json(root/f'{step}_process.json',{'pid':child.pid,'argv':argv,'cwd':runtime['release_root'],'gpu':runtime['gpu']})
        rc=child.wait()
        write_json(root/f'{step}_exit.json',{'returncode':rc,'finished_at':time.time()})
        if rc:
            write_json(root/'failure.json',{'step':step,'returncode':rc,'status':'FAILED_TECHNICAL','prior_outputs_preserved':True})
            raise SystemExit(rc)
    write_json(root/'completion.json',{'status':'SCORED','paper_effect_verified':False,'all_preregistered_predictions_complete':True,'finished_at':time.time()})

def launch(runtime_path):
    runtime=read_json(runtime_path);root=Path(runtime['run_root']);logs=Path(runtime['log_root'])
    if root.exists() or logs.exists():
        raise FileExistsError('Immutable run/log root already exists; reconcile prior launch')
    root.mkdir(parents=True);logs.mkdir(parents=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(runtime['gpu']),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    with (logs/'pipeline.log').open('xb') as handle:
        p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--runtime',str(runtime_path)],stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,cwd=runtime['release_root'],env=env,start_new_session=True)
    write_json(root/'launch.json',{'worker_pid':p.pid,'runtime':str(runtime_path),'log':str(logs/'pipeline.log'),'gpu':runtime['gpu'],'launch_owner':'root-ibrff-20261019'})
    print(json.dumps({'worker_pid':p.pid,'run_root':str(root)}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['launch','worker','smoke','stage1','stage2','prepare-query','predict','score'])
    p.add_argument('--runtime',type=Path,required=True)
    args=p.parse_args();runtime=read_json(args.runtime)
    torch.set_num_threads(4)
    if args.action=='launch':launch(args.runtime)
    elif args.action=='worker':worker(args.runtime)
    elif args.action=='smoke':smoke(runtime)
    elif args.action in ('stage1','stage2'):train_stage(runtime,args.action)
    elif args.action=='prepare-query':prepare_queries(runtime)
    elif args.action=='predict':predict(runtime)
    else:score(runtime['run_root'])

if __name__=='__main__':main()
