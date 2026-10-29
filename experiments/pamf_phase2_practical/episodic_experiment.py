"""Stage-scoped frozen PAMF features and support-only MILKR predictions."""
from pathlib import Path
import time
import numpy as np
from .io import read,write,require,activate
from .episodic_io import load_task,query_complement
from .features import NAMES,branches,fft96,tensor,load_reader,load_G


def split_rows(cap,s):
    registered=np.load(Path(cap)/'stage_rows'/f'N{s["new_count"]:02d}.npy')
    return query_complement(registered,np.asarray(s['support_indices'],dtype=np.int32))


class Cache:
    def __init__(self,rt,cap,seed,path,root,reader,engine=None,restore_root=None):
        self.rt,self.cap,self.seed,self.path,self.root,self.reader=rt,Path(cap),seed,path,Path(root),reader
        self.engine,self.restore_root=engine,Path(restore_root) if restore_root else None
        self.n=read(self.cap/'manifest.json')['received_count'];self.root.mkdir(parents=True,exist_ok=False)
        self.valid=np.zeros(self.n,bool)
        self.arrays={k:np.lib.format.open_memmap(self.root/(k+'.npy'),mode='w+',dtype=np.float32,shape=(self.n,96 if k=='fft' else 160)) for k in NAMES}
        self.default=np.lib.format.open_memmap(self.root/'default.npy',mode='w+',dtype=np.int16,shape=(self.n,))
        if path=='P1':
            if seed==rt['model_seeds'][0]:
                self.restore_root.mkdir(parents=True,exist_ok=False)
                self.restored=np.lib.format.open_memmap(self.restore_root/'received.npy',mode='w+',dtype=np.float32,shape=(self.n,2,256))
                self.restored_valid=np.zeros(self.n,bool)
            else:
                require(read(self.restore_root/'complete.json')['status']=='RESTORED_COMPLETE','DURA cache not complete')
                self.restored=np.load(self.restore_root/'received.npy',mmap_mode='r');self.restored_valid=np.ones(self.n,bool)

    def get(self,rows,raw):
        import torch
        rows=np.asarray(rows,dtype=np.int32);raw=np.asarray(raw)
        require(len(rows)==len(raw),'Feature request alignment mismatch')
        missing=np.flatnonzero(~self.valid[rows]);batch=self.rt.get('feature_batch_size',128)
        with torch.inference_mode():
            for start in range(0,len(missing),batch):
                ix=missing[start:start+batch];positions=rows[ix];received=np.asarray(raw[ix],dtype=np.float32)
                if self.path=='P1':
                    if self.seed==self.rt['model_seeds'][0]:
                        y=self.engine(tensor(received,'cuda:0'))[0];received=np.asarray(y.cpu().tolist(),np.float32)
                        self.restored[positions]=received;self.restored_valid[positions]=True
                    else:received=np.array(self.restored[positions],copy=True)
                values,logits=branches(self.reader,tensor(received,'cuda:0'))
                for k,v in values.items():self.arrays[k][positions]=np.asarray(v.cpu().tolist(),np.float32)
                self.arrays['fft'][positions]=fft96(received);self.default[positions]=logits.argmax(1).cpu().tolist();self.valid[positions]=True
        return {k:np.asarray(v[rows]) for k,v in self.arrays.items()},np.asarray(self.default[rows])

    def finish(self):
        require(self.valid.all(),'Incomplete stage-scoped features')
        for value in self.arrays.values():value.flush()
        self.default.flush()
        if self.path=='P1' and self.seed==self.rt['model_seeds'][0]:
            require(self.restored_valid.all(),'Incomplete DURA cache');self.restored.flush()
            write(self.restore_root/'complete.json',dict(status='RESTORED_COMPLETE',count=self.n,weight_updates=0,query_statistics_updates=0,truth_read=False))
        write(self.root/'complete.json',dict(status='FEATURES_COMPLETE',count=self.n,model_seed=self.seed,input_path=self.path,truth_read=False,stage_scoped=True))


def decide(state,features,classes,batch):
    count=len(features['z_id']);pred=np.empty(count,np.int16);restricted=np.empty(count,np.int16)
    order=np.asarray(sorted(range(len(classes)),key=lambda i:classes[i]));old=np.asarray(sorted(range(6),key=lambda i:classes[i]))
    for start in range(0,count,batch):
        values=state.score(**{k:v[start:start+batch] for k,v in features.items()})
        require(values.shape==(min(batch,count-start),len(classes)) and np.isfinite(values).all(),'Invalid all-class scores')
        pred[start:start+len(values)]=order[values[:,order].argmax(1)];restricted[start:start+len(values)]=old[values[:,old].argmax(1)]
    return pred,restricted


def run(rt,job):
    import torch
    from .milkr.d92_branch_local_ridge import fit_branch_local_ridge
    activate(rt);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cap=Path(rt['data_root'])/job['capsule'];root=Path(job['output']);root.mkdir(parents=True,exist_ok=False);started=time.monotonic();seed=job['seed']
    manifest=read(cap/'manifest.json');specs=[read(p) for p in (cap/'splits').glob('*.json')]
    lookup={(s['k'],s['new_count']):s for s in specs};reader=load_reader(rt,seed,'cuda:0')
    before={k:v.detach().clone() for k,v in reader.state_dict().items()};engine=load_G(rt,'cuda:0') if seed==rt['model_seeds'][0] else None
    restore_root=Path(rt['run_root'])/'DURA'/job['capsule'];rows_out=[]
    def save(path,s,stage,arm,query,pred,restricted=None,audit=None):
        name=f'{path}-{stage}-K{s["k"]}-N{s["new_count"]}-{arm}.npz';payload=dict(query_indices=query,predicted_indices=np.asarray(pred,dtype=np.int16))
        if restricted is not None:payload['old_restricted_indices']=restricted
        np.savez_compressed(root/name,**payload)
        rows_out.append(dict(split_id=s['split_id'],stage=stage,arm=arm,input_path=path,prediction=name,k=s['k'],new_count=s['new_count'],
            registered_classes=s['registered_classes'],query_count=len(query),audit=audit))
    for path in ('P0','P1'):
        cache=Cache(rt,cap,seed,path,root/('features-'+path),reader,engine,restore_root)
        # Complete all old-class A/B tasks before any new-class feature is read.
        for n in rt['new_counts']:
            for k in rt['shots']:
                s=lookup[k,n];query=split_rows(cap,s)
                if n==0:
                    a=load_task(cap,s['split_id'],stage='A');qf,default=cache.get(query,a['query_x'])
                    save(path,s,'A','default_head',query,default);del a
                task=load_task(cap,s['split_id'],stage='B' if n==0 else 'C')
                sf,_=cache.get(s['support_indices'],task['support_x']);qf,_=cache.get(query,task['query_x'])
                for arm in rt['arms']:
                    begin=time.monotonic()
                    state=fit_branch_local_ridge(**sf,support_labels=task['support_y'].astype(np.int64),support_ids=task['support_ids'].tolist(),classes=s['registered_classes'],old_classes=rt['old_classes'],arm=arm)
                    require(list(state.classes)==s['registered_classes'],'Wrong class output order')
                    predicted,restricted=decide(state,qf,s['registered_classes'],rt.get('score_batch_size',256))
                    audit=state.audit_dict();require(audit['optimizer_steps']==0,'Unexpected target optimizer')
                    audit.update(query_rows_used_for_fit=0,source_rows_used_for_fit=0,C_inherits_B_kernel=False,seconds=time.monotonic()-begin)
                    save(path,s,'B' if n==0 else 'C',arm,query,predicted,restricted if n else None,audit)
                    print('PREDICTED',job['id'],path,k,n,arm,len(query),flush=True)
                del task,sf,qf
        cache.finish();del cache
    require(all(torch.equal(before[k],v) for k,v in reader.state_dict().items()),'Frozen F changed')
    expected=2*(len(rt['shots'])+len(rt['shots'])*len(rt['new_counts'])*len(rt['arms']))
    require(len(rows_out)==expected,'Missing predictions')
    write(root/'predictions.json',dict(capsule_id=manifest['capsule_id'],capsule=job['capsule'],model_seed=seed,rows=rows_out))
    write(root/'complete.json',dict(status='PREDICTIONS_COMPLETE',capsule_id=manifest['capsule_id'],rows=len(rows_out),model_seed=seed,
          truth_read=False,query_fit=False,source_weights_unchanged=True,seconds=time.monotonic()-started))
