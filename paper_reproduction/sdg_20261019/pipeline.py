"""Author SDG WiSig 200-epoch recipe; source-only selection, separate truth scorer."""
import argparse,contextlib,io,json,math,os,pickle,re,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().with_name('deps')))
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as h:json.dump(value,h,ensure_ascii=False,indent=2,allow_nan=False);h.write('\n')
def config(rt):return read_json(rt['candidate'])
def contract(c):
    return {key:c[key] for key in ['dataset','rx','source_days','target_days','packet_rows','tx_labels','split_seed','representation','upstream_commit']}
def author(rt):
    sys.path.insert(0,rt['model_root'])
    import main
    return main
def block(rt,c,day):
    path=Path(rt['data_root'])/f'rx_1-1_date{day}.pkl'
    with path.open('rb') as h:payload=pickle.load(h)
    if payload['tx_list']!=c['tx_labels'] or len(payload['data'])!=6:raise ValueError('Author class mapping differs')
    for x in payload['data']:
        if np.asarray(x).shape!=(1000,256,2) or not np.isfinite(x).all():raise ValueError('Wrong author packet shape')
        if not (np.sum(np.asarray(x)**2,axis=(1,2))>0).all():raise ValueError('Zero-power IQ')
    # This is the unchanged author loader, including float64 normalization before
    # torch.Tensor conversion. The single file is read again by that function.
    author(rt)
    from util.get_dataset import load_dataset_from_file
    return load_dataset_from_file(str(path),num_device=6)
def source_indices():
    trainval,test=train_test_split(np.arange(6000),test_size=.1,random_state=2023)
    train,val=train_test_split(trainval,test_size=.3,random_state=2023)
    return train,val,test
def source_data(rt,c):
    x,y=block(rt,c,1);train,val,test=source_indices()
    if [len(train),len(val),len(test)]!=[3780,1620,600]:raise ValueError('Wrong source split')
    return (x[train],y[train]),(x[val],y[val])
def build(rt):return author(rt).MACNN(in_channels=2,channels=8,num_classes=6)
def save(path,net,c,epoch,val_loss):
    torch.save({'state_dict':net.state_dict(),'source_contract':contract(c),'epoch':epoch,'source_val_loss':val_loss,'model_seed':2023},path)
def load(path,c,device):
    # This run starts from scratch; only its own source-selected weights enter.
    import inspect
    kwargs={'map_location':device}
    if 'weights_only' in inspect.signature(torch.load).parameters:kwargs['weights_only']=True
    value=torch.load(path,**kwargs)
    if value['source_contract']!=contract(c) or value['model_seed']!=2023:raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
    return value
def loaders(rt,c,small=False):
    a=author(rt);train,val=source_data(rt,c)
    if small:train=(train[0][:12],train[1][:12])
    conf=argparse.Namespace(main_aug_depth=[2],aux_aug_depth=[1],lambda_con=[1.,100.])
    dl=DataLoader(a.AugDataset(train[0],train[1],a.transforms.ToTensor(),conf.main_aug_depth,conf.aux_aug_depth),batch_size=32,shuffle=True)
    vl=DataLoader(TensorDataset(torch.Tensor(val[0]),torch.Tensor(val[1])),batch_size=16,shuffle=True)
    return dl,vl,conf
def initialize_training(rt):
    a=author(rt);a.setup_seed(2023)
    net=build(rt).to(rt['device'])
    optimizer=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=0)
    loss=(torch.nn.NLLLoss().to(rt['device']),a.SupConLoss().to(rt['device']))
    # The author's torchsummary forward changes RNG and BN state. Retain it.
    a.summary(net,(2,256),device='cuda' if rt['device'].startswith('cuda') else 'cpu')
    return net,optimizer,loss
def smoke(rt):
    c=config(rt);a=author(rt);dl,_,conf=loaders(rt,c,small=True)
    net,opt,loss=initialize_training(rt)
    a.train(net,loss,dl,opt,1,conf)
    if not all(torch.isfinite(v).all() for v in net.state_dict().values()):raise ValueError('Nonfinite source smoke')
    out=Path(rt['run_root'])/'smoke';out.mkdir();save(out/'source.pt',net,c,0,0.)
    copy=build(rt).to(rt['device']);copy.load_state_dict(load(out/'source.pt',c,rt['device'])['state_dict'],strict=True)
    net.eval();copy.eval();x=torch.Tensor(source_data(rt,c)[0][0][:4]).to(rt['device'])
    with torch.no_grad():
        if not torch.equal(net(x)[1],copy(x)[1]):raise ValueError('Source checkpoint roundtrip differs')
    write_json(out/'result.json',{'status':'PASS','real_source_checkpoint_roundtrip':True,'query_access':False,'smoke_weights_discarded_for_training':True})
def train(rt):
    c=config(rt);a=author(rt);dl,vl,conf=loaders(rt,c)
    net,opt,loss=initialize_training(rt)
    out=Path(rt['run_root'])/'train';out.mkdir();best=float('inf')
    write_json(out/'resolved_config.json',{'candidate':c,'source_contract':contract(c),'train_count':3780,'val_count':1620,'source_holdout_count':600,'query_access':False,'author_train_evaluate_unchanged':True,'torchsummary_forward_retained':True,'numpy':np.__version__,'torch':torch.__version__})
    with (out/'epochs.jsonl').open('x',encoding='utf-8') as h:
        for epoch in range(1,201):
            started=time.monotonic();buffer=io.StringIO()
            with contextlib.redirect_stdout(buffer):a.train(net,loss,dl,opt,epoch,conf)
            text=buffer.getvalue();print(text,flush=True)
            match=re.search(r'Loss:\s*(\S+)',text)
            if match is None or not math.isfinite(float(match.group(1).rstrip(','))):raise ValueError('Nonfinite/unreadable source loss')
            val_loss=a.evaluate(net,loss,vl,epoch)
            if not math.isfinite(val_loss):raise ValueError('Nonfinite source validation loss')
            improved=val_loss<best
            if improved:best=val_loss;save(out/'best.pt',net,c,epoch,val_loss)
            record={'epoch':epoch,'source_train_loss':float(match.group(1).rstrip(',')),'source_val_loss':val_loss,'improved':improved,'seconds':time.monotonic()-started}
            h.write(json.dumps(record,allow_nan=False)+'\n');h.flush()
    save(out/'last.pt',net,c,200,val_loss)
    write_json(out/'training_complete.json',{'status':'TRAINING_COMPLETE','epochs':200,'best_source_val_loss':best,'selection':'minimum_source_validation_nll','query_access':False})
def prepare_query(rt):
    c=config(rt);root=Path(rt['run_root'])
    if not (root/'train/training_complete.json').is_file():raise ValueError('Training freeze required')
    query=root/'query';truth=root/'truth';query.mkdir();truth.mkdir();xs=[];ys=[];physical=[]
    for day in [1,2,3,4]:
        x,y=block(rt,c,day)
        rows=source_indices()[2] if day==1 else np.arange(6000)
        xs.append(x[rows]);ys.append(y[rows])
        physical.extend([{'day':day,'rx':'1-1','tx_index':int(i)//1000,'packet_row':int(i)%1000} for i in rows])
    x=np.concatenate(xs);y=np.concatenate(ys);order=np.random.RandomState(20261019).permutation(len(x))
    ids=np.asarray([f'q{i:06d}' for i in range(len(x))])
    np.save(query/'iq.npy',x[order].astype(np.float32),allow_pickle=False);np.save(query/'ids.npy',ids,allow_pickle=False)
    np.savez(truth/'labels.npz',ids=ids,labels=y[order]);write_json(truth/'physical_ids.json',[physical[int(i)] for i in order])
def predict(rt):
    c=config(rt);root=Path(rt['run_root']);x=np.load(root/'query/iq.npy',allow_pickle=False);ids=np.load(root/'query/ids.npy',allow_pickle=False)
    if x.shape!=(18600,2,256) or len(set(ids.tolist()))!=18600:raise ValueError('Wrong opaque query inputs')
    state=load(root/'train/best.pt',c,rt['device']);net=build(rt).to(rt['device']);net.load_state_dict(state['state_dict'],strict=True);net.eval()
    pred=[]
    with torch.no_grad():
        for start in range(0,len(x),16):pred.append(net(torch.Tensor(x[start:start+16]).to(rt['device']))[1].argmax(1).cpu().numpy())
    with (root/'predictions.npz').open('xb') as h:np.savez(h,ids=ids,predicted=np.concatenate(pred))
    write_json(root/'predictions_complete.json',{'status':'PREDICTIONS_COMPLETE','count':len(ids),'selected_epoch':state['epoch'],'query_truth_access':False})
def score(root):
    root=Path(root);marker=read_json(root/'predictions_complete.json')
    if marker['status']!='PREDICTIONS_COMPLETE':raise ValueError('Predictions must complete before truth access')
    p=np.load(root/'predictions.npz',allow_pickle=False);t=np.load(root/'truth/labels.npz',allow_pickle=False)
    if not np.array_equal(p['ids'],t['ids']) or len(p['ids'])!=18600:raise ValueError('Opaque ID join mismatch')
    from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
    physical=read_json(root/'truth/physical_ids.json');days=np.asarray([r['day'] for r in physical]);groups={}
    for day in [1,2,3,4]:
        mask=days==day;y=t['labels'][mask];yp=p['predicted'][mask]
        groups[str(day)]={'count':int(mask.sum()),'accuracy_percent':100*accuracy_score(y,yp),'macro_f1_percent':100*f1_score(y,yp,average='macro',labels=list(range(6)),zero_division=0),'confusion_matrix':confusion_matrix(y,yp,labels=list(range(6))).tolist()}
    target=days!=1
    result={'status':'SCORED','scenario':'author WiSig cross-day single-source DG','by_day':groups,'target_count':int(target.sum()),'target_accuracy_percent':100*accuracy_score(t['labels'][target],p['predicted'][target]),'target_macro_f1_percent':100*f1_score(t['labels'][target],p['predicted'][target],average='macro',labels=list(range(6)),zero_division=0),'paper_effect_verified':False}
    write_json(root/'scores.json',result);print(json.dumps(result),flush=True)
def worker(path):
    rt=read_json(path);root=Path(rt['run_root']);write_json(root/'worker_started.json',{'pid':os.getpid(),'started_at':time.time()})
    for step in ['smoke','train','prepare-query','predict','score']:
        argv=[sys.executable,'-u',str(Path(__file__).resolve()),step,'--runtime',str(path)]
        child=subprocess.Popen(argv,cwd=rt['release_root']);write_json(root/(step+'_process.json'),{'pid':child.pid,'argv':argv,'cwd':rt['release_root'],'gpu':rt['gpu']})
        rc=child.wait();write_json(root/(step+'_exit.json'),{'returncode':rc,'finished_at':time.time()})
        if rc:write_json(root/'failure.json',{'status':'FAILED_TECHNICAL','step':step,'returncode':rc,'prior_outputs_preserved':True});raise SystemExit(rc)
    write_json(root/'completion.json',{'status':'SCORED','paper_effect_verified':False,'all_preregistered_predictions_complete':True})
def launch(path):
    rt=read_json(path);root=Path(rt['run_root']);logs=Path(rt['log_root'])
    if root.exists() or logs.exists():raise FileExistsError('Immutable outputs exist; reconcile before retry')
    root.mkdir(parents=True);logs.mkdir(parents=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(rt['gpu']),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    with (logs/'pipeline.log').open('xb') as h:
        child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--runtime',str(path)],cwd=rt['release_root'],stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    value={'worker_pid':child.pid,'gpu':rt['gpu'],'launch_owner':'root-sdg-20261019','log':str(logs/'pipeline.log')};write_json(root/'launch.json',value);print(json.dumps(value),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','worker','smoke','train','prepare-query','predict','score']);p.add_argument('--runtime',type=Path,required=True);a=p.parse_args();rt=read_json(a.runtime);torch.set_num_threads(4)
    if a.action=='launch':launch(a.runtime)
    elif a.action=='worker':worker(a.runtime)
    elif a.action=='smoke':smoke(rt)
    elif a.action=='train':train(rt)
    elif a.action=='prepare-query':prepare_query(rt)
    elif a.action=='predict':predict(rt)
    else:score(rt['run_root'])
if __name__=='__main__':main()
