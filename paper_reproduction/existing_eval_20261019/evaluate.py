"""Evaluate three frozen source-only historical models, without retraining heads."""
import argparse,importlib,inspect,json,os,pickle,shutil,subprocess,sys,time
from pathlib import Path

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as h:json.dump(value,h,ensure_ascii=False,indent=2,allow_nan=False);h.write('\n')
def bootstrap(rt):
    if rt.get('numpy_root'):sys.path.insert(0,rt['numpy_root'])
    global np,torch
    import numpy as np
    import torch
    torch.set_num_threads(4)
def checked_row(rt,method):
    row=rt['rows'][method];plan=read_json(rt['parent_launch_plan']);roles=read_json(rt['physical_roles'])
    completion=read_json(Path(rt['parent_run_root'])/method/'source_training_complete.json')
    if not plan['source_only'] or completion['target_scored'] or completion['status']!='SOURCE_TRAINING_COMPLETE_AWAITING_SOURCE_REVIEW':raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED')
    if roles['source']!=rt['source_rx'] or roles['target']!=rt['target_rx']:raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
    if row['model_root']!=plan['rows'][method]['cwd']:raise ValueError('Wrong original model source')
    configs=[read_json(path) for path in row['resolved_configs']]
    if method!='difl':
        cfg=configs[0]
        for key,value in {'dataset_name':'ManySig','exp':'CRD','train_date':[1,2],'test_round':0,'seed':2023,'source_receivers':list(range(3,12)),'test_receivers':[0,1,2],'selection':'source validation loss','code_state':'only_train'}.items():
            if cfg.get(key)!=value:raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: '+key)
        commands=plan['rows'][method]['commands']
        if len(commands)!=1 or commands[0][commands[0].index('--code_state')+1]!='only_train':raise ValueError('Historical target use not excluded')
        if roles['asknet_wavemlp']['packet_rows']!=list(range(100)):raise ValueError('Wrong raw packet IDs')
    else:
        expected=[str(Path(rt['data_root'])/'difl'/f'rx_{rx}.npy') for rx in rt['source_rx']]
        for cfg in configs:
            if cfg['source_files']!=expected or cfg['seed']!=300 or cfg['split_seed']!=32 or cfg['epochs']!=100:raise ValueError('DIFL source provenance mismatch')
        student=[c for c in configs if c['method']=='difl'][0]
        if student['teacher_checkpoint']!=str(Path(rt['parent_run_root'])/'difl/teacher/pretrain.pth') or student['code_state']!='only_train':raise ValueError('DIFL teacher provenance mismatch')
        if roles['difl']['packet_rows']!=np.random.RandomState(300).choice(1000,size=200,replace=False).tolist():raise ValueError('Wrong equalized packet IDs')
    if not Path(row['checkpoint']).is_file():raise FileNotFoundError(row['checkpoint'])
    return row,roles,configs
def model(rt,method):
    row,_,configs=checked_row(rt,method);sys.path.insert(0,row['model_root'])
    if method=='difl':
        from Network1 import my_resnet
        net=my_resnet()
    else:
        native=importlib.import_module('main');cfg=argparse.Namespace(**configs[0])
        net=native.build_model(cfg,6) if method=='asknet' else native._build_model_from_conf(cfg,tx_num=6,wavelet_kernel_size=cfg.wavelet_kernel_size)
    kwargs={'map_location':'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:kwargs['weights_only']=True
    state=torch.load(row['checkpoint'],**kwargs)
    if not isinstance(state,dict):raise TypeError('Expected own frozen state_dict')
    net.load_state_dict(state,strict=True);return net.to(rt['device']).eval()
def logits(net,x,method):
    out=net(x)
    if method=='wavemlp':out=out[0]
    elif method=='difl':out=out[1]
    if out.ndim!=2 or out.shape!=(len(x),6) or not torch.isfinite(out).all():raise ValueError('Wrong original-head predictions')
    return out
class NumpyExportUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module.startswith('numpy._core.'):module='numpy.core.'+module[len('numpy._core.'):]
        return super().find_class(module,name)
def raw_source(rt):
    path=Path(rt['data_root'])/'author_layout/ManySig/non_equalized/date1/rx_2-19_data.pkl'
    with path.open('rb') as h:raw=NumpyExportUnpickler(h).load()['data'][0][:8]
    x=raw.transpose(0,2,1).copy()
    for i in range(len(x)):
        power=np.sum(x[i,0]**2+x[i,1]**2)/256
        if power<=0 or not np.isfinite(power):raise ValueError('Invalid source power')
        x[i]/=np.sqrt(power)
    return x
def smoke(rt,method):
    checked_row(rt,method)
    x=np.load(Path(rt['data_root'])/'difl/rx_2-19.npy',allow_pickle=False)[:8].transpose(0,2,1) if method=='difl' else raw_source(rt)
    net=model(rt,method)
    with torch.no_grad():out=logits(net,torch.tensor(x,dtype=torch.float32,device=rt['device']),method)
    write_json(Path(rt['run_root'])/method/'smoke.json',{'status':'PASS','real_frozen_checkpoint_loaded':True,'source_count':len(x),'query_access':False,'default_head':True,'no_parameter_fitting':True})
def prepare(rt,method):
    row,roles,_=checked_row(rt,method);root=Path(rt['run_root'])/method
    if read_json(root/'smoke.json')['status']!='PASS':raise ValueError('Source checkpoint smoke required')
    query=root/'query';truth=root/'truth';query.mkdir();truth.mkdir()
    if method!='difl':
        # Reuse exactly the frozen IB-RFF fold0 first100/raw/day3-4 export.
        for name in ['iq.npy','ids.npy']:shutil.copyfile(Path(rt['raw_query_root'])/name,query/name)
        for name in ['labels.npz','physical_ids.json']:shutil.copyfile(Path(rt['raw_truth_root'])/name,truth/name)
        metadata=read_json(rt['raw_query_metadata'])
        if metadata['count']!=3600 or metadata['scenario']!='clean_CRD':raise ValueError('Wrong shared query export')
    else:
        xs=[];ys=[];physical=[];packet_rows=roles['difl']['packet_rows']
        labels=np.load(Path(rt['data_root'])/'difl/labels.npy',allow_pickle=False).astype(int)-1
        if not np.array_equal(labels,np.repeat(np.arange(6),200)):raise ValueError('Wrong original DIFL label map')
        for rx in rt['target_rx']:
            x=np.load(Path(rt['data_root'])/'difl'/f'rx_{rx}.npy',allow_pickle=False)
            if x.shape!=(1200,256,2) or not np.isfinite(x).all():raise ValueError('Wrong DIFL target IQ')
            xs.append(x.transpose(0,2,1));ys.append(labels)
            physical.extend([{'rx':rx,'day':1,'tx':tx,'row':int(i)} for tx in range(6) for i in packet_rows])
        order=np.random.RandomState(20261019).permutation(3600);ids=np.asarray([f'q{i:06d}' for i in range(3600)])
        np.save(query/'iq.npy',np.concatenate(xs)[order],allow_pickle=False);np.save(query/'ids.npy',ids,allow_pickle=False)
        np.savez(truth/'labels.npz',ids=ids,labels=np.concatenate(ys)[order]);write_json(truth/'physical_ids.json',[physical[int(i)] for i in order])
    write_json(root/'query_prepared.json',{'status':'PREPARED','method':method,'count':3600,'training_frozen_before_this_run':True})
def predict(rt,method):
    root=Path(rt['run_root'])/method
    if read_json(root/'query_prepared.json')['status']!='PREPARED':raise ValueError('Query export missing')
    x=np.load(root/'query/iq.npy',allow_pickle=False);ids=np.load(root/'query/ids.npy',allow_pickle=False)
    if x.shape!=(3600,2,256) or len(set(ids.tolist()))!=3600:raise ValueError('Wrong anonymous query input')
    net=model(rt,method);pred=[]
    with torch.no_grad():
        for start in range(0,3600,32):pred.append(logits(net,torch.tensor(x[start:start+32],dtype=torch.float32,device=rt['device']),method).argmax(1).cpu().numpy())
    with (root/'predictions.npz').open('xb') as h:np.savez(h,ids=ids,predicted=np.concatenate(pred))
    write_json(root/'predictions_complete.json',{'status':'PREDICTIONS_COMPLETE','method':method,'count':3600,'query_truth_access':False,'checkpoint':rt['rows'][method]['checkpoint'],'default_head':True,'no_parameter_fitting':True})
def score(root):
    root=Path(root)
    if read_json(root/'predictions_complete.json')['status']!='PREDICTIONS_COMPLETE':raise ValueError('Predictions must complete before truth access')
    p=np.load(root/'predictions.npz',allow_pickle=False);t=np.load(root/'truth/labels.npz',allow_pickle=False)
    if not np.array_equal(p['ids'],t['ids']) or len(p['ids'])!=3600:raise ValueError('Opaque ID join mismatch')
    from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
    y=t['labels'];yp=p['predicted'];physical=read_json(root/'truth/physical_ids.json');groups={}
    for rx,day in sorted({(r['rx'],r['day']) for r in physical}):
        mask=np.asarray([r['rx']==rx and r['day']==day for r in physical]);groups[f'{rx}/day{day}']={'count':int(mask.sum()),'accuracy_percent':100*accuracy_score(y[mask],yp[mask])}
    result={'status':'SCORED','method':root.name,'count':len(y),'accuracy_percent':100*accuracy_score(y,yp),'macro_f1_percent':100*f1_score(y,yp,labels=list(range(6)),average='macro',zero_division=0),'confusion_matrix':confusion_matrix(y,yp,labels=list(range(6))).tolist(),'by_rx_day':groups,'paper_effect_verified':False}
    write_json(root/'scores.json',result);print(json.dumps(result),flush=True)
def worker(path):
    rt=read_json(path);root=Path(rt['run_root'])
    for method in ['asknet','wavemlp','difl']:
        out=root/method;out.mkdir()
        for step in ['smoke','prepare','predict','score']:
            argv=[sys.executable,'-u',str(Path(__file__).resolve()),step,'--runtime',str(path),'--method',method]
            child=subprocess.Popen(argv,cwd=rt['release_root']);write_json(out/(step+'_process.json'),{'pid':child.pid,'argv':argv,'cwd':rt['release_root'],'gpu':rt['gpu']})
            rc=child.wait();write_json(out/(step+'_exit.json'),{'returncode':rc,'finished_at':time.time()})
            if rc:write_json(root/'failure.json',{'method':method,'step':step,'returncode':rc,'prior_outputs_preserved':True});raise SystemExit(rc)
        write_json(out/'completion.json',{'status':'SCORED','paper_effect_verified':False})
    write_json(root/'completion.json',{'status':'SCORED','methods':['asknet','wavemlp','difl'],'paper_effect_verified':False})
def launch(path):
    rt=read_json(path);root=Path(rt['run_root']);logs=Path(rt['log_root'])
    if root.exists() or logs.exists():raise FileExistsError('Existing output preserved; reconcile prior launch')
    root.mkdir(parents=True);logs.mkdir(parents=True)
    env=dict(os.environ,PYTHONPATH=rt['numpy_root'],CUDA_VISIBLE_DEVICES=str(rt['gpu']),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    with (logs/'pipeline.log').open('xb') as h:
        p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--runtime',str(path)],stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT,cwd=rt['release_root'],env=env,start_new_session=True)
    value={'worker_pid':p.pid,'gpu':rt['gpu'],'launch_owner':'root-existing-eval-20261019','log':str(logs/'pipeline.log')};write_json(root/'launch.json',value);print(json.dumps(value),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','worker','smoke','prepare','predict','score']);p.add_argument('--runtime',required=True,type=Path);p.add_argument('--method',choices=['asknet','wavemlp','difl']);a=p.parse_args();rt=read_json(a.runtime);bootstrap(rt)
    if a.action=='launch':launch(a.runtime)
    elif a.action=='worker':worker(a.runtime)
    elif a.action=='smoke':smoke(rt,a.method)
    elif a.action=='prepare':prepare(rt,a.method)
    elif a.action=='predict':predict(rt,a.method)
    else:score(Path(rt['run_root'])/a.method)
if __name__=='__main__':main()
