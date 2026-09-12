"""Explicit historical-target replay of all frozen source candidates, truth last."""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from cvsrffi.evidence_target_evaluation import SCENES,_IQOnlyTarget,_scene_iq
from cvsrffi.anchored_pipeline import FrozenAnchoredSystem
from cvsrffi.anchored_source import FrozenPartialSystem,load_ground
from cvsrffi.evidence_conditions import received_conditions
from cvsrffi.evidence_observation import extract_evidence
NAMES=('A0','A1','A2','A3','A4','C_angle','C_angle_keep','P1','A5-0','A5-25','A5-50','A5-75','A5-100','A6')
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
def dataset(contract):
    from dataset_wisig import load_wisig_compact_pkl,WiSigCompactDataset,WiSigIndex
    ds=load_wisig_compact_pkl(contract['dataset_id'])
    assert list(map(str,ds['tx_list']))==contract['tx_mapping']
    data=WiSigCompactDataset(ds,out_len=256,crop_mode='center',normalize=True,center=False,equalized=contract['equalized'],rx_keep=contract['target_receivers'],day_keep=contract['actual_target_days'],domain='rx_day',build_index=False)
    data.index=[WiSigIndex(y,rx,day,data.eq_keep[0],sig) for y,rx,day,sig in [list(map(int,k.split(':'))) for k in contract['roles']['target']]]
    assert all(r.rx_i in contract['target_receivers'] and r.rx_i not in contract['source_receivers'] for r in data.index)
    data.role='R_t'
    return _IQOnlyTarget(data)
def predict(args):
    out=Path(args.output)/args.scene;out.mkdir(parents=True,exist_ok=False)
    anchor,payload,contract=load_ground(args.ground,args.contract,args.device)
    systems={};before={}
    for name in NAMES:
        state=torch.load(Path(args.source)/'candidates'/name/'export/frozen_system.pt',map_location='cpu',weights_only=True)
        assert state['class_labels']==contract['tx_mapping']
        actual=anchor.original_model.id_backbone.state_dict()
        assert all(torch.equal(v.cpu(),actual[k].cpu()) for k,v in state['identity_backbone'].items())
        s=(FrozenPartialSystem if name=='P1' else FrozenAnchoredSystem).from_state(state)
        assert s.calibrator is not None
        systems[name]=s;before[name]=s.export_state()['system_identity']
    # Actual checkpoint/source IQ smoke before any target iteration.
    source=torch.load(Path(args.source)/'inputs/L_s.pt',map_location='cpu',weights_only=True)['x'][:6]
    h,s0,valid=anchor.extract(source)
    torch.testing.assert_close(systems['A0'].predict_features(h,s0,received_conditions(source)['quality'],valid)['top_class'],s0.cpu().argmax(-1))
    print('REAL_CHECKPOINT_SOURCE_SMOKE_PASS',flush=True)
    data=dataset(contract);rows={name:[] for name in NAMES};ids_all=[];start=time.time()
    versions={k:v._version for k,v in anchor.original_model.state_dict().items()}
    with torch.no_grad():
        for batch,(x,ids) in enumerate(DataLoader(data,batch_size=128,shuffle=False,num_workers=0)):
            x=_scene_iq(x,ids,args.scene,392005)
            aux=anchor.extract_aux(x);h=aux['feat_joint'];s0=aux['logits'];conditions=received_conditions(x)
            q=conditions['quality'];valid=conditions['valid'];blocks=extract_evidence(aux,'blocks').z.cpu()
            assert torch.isfinite(h).all() and torch.isfinite(s0).all()
            for name,s in systems.items():
                if name=='P1':
                    p=s.head(blocks,s.head.pattern_spec.mask('full',len(x)),q)
                    result=s.calibrator.predict(p['scores'],p['mahalanobis'],p['observed_count'],q,['full']*len(x))
                else:result=s.predict_features(h,s0,q,valid)
                rows[name].append({k:result[k].detach().cpu() for k in ('top_class','confidence','accepted')})
            ids_all.append(ids)
            if batch%100==0:print(json.dumps(dict(scene=args.scene,rows=min((batch+1)*128,len(data)),seconds=time.time()-start)),flush=True)
    ids=torch.cat(ids_all)
    for name,s in systems.items():
        assert s.export_state()['system_identity']==before[name]
        artifact={k:torch.cat([r[k] for r in rows[name]]) for k in ('top_class','confidence','accepted')};artifact['ids']=ids
        assert len(ids)==len(data) and torch.isfinite(artifact['confidence']).all()
        with (out/(name+'.pt')).open('xb') as f:torch.save(artifact,f)
    assert all(v._version==versions[k] for k,v in anchor.original_model.state_dict().items())
    write(out/'seal.json',dict(status='SEALED',scene=args.scene,candidates=list(NAMES),rows=len(data),seed=392005,truth_access=False,seconds=time.time()-start,confirmation=False,interpretation='historical target replay; no fitting or candidate selection'))
    print('SEALED '+args.scene,flush=True)
def count(pred,y,accepted):
    correct=pred==y;a=int(accepted.sum());n=len(y)
    return dict(rows=n,correct=int(correct.sum()),closed_accuracy=float(correct.double().mean()),accepted=a,coverage=a/n,accepted_correct=int((correct&accepted).sum()),accepted_accuracy=float(correct[accepted].double().mean()) if a else None)
def score(args):
    root=Path(args.output)
    seals=[json.loads((root/scene/'seal.json').read_text()) for scene in SCENES]
    assert all(s['status']=='SEALED' and s['candidates']==list(NAMES) for s in seals)
    # Truth is joined only in this independent process after all 56 files seal.
    contract=json.loads(Path(args.contract).read_text());truth={hashlib.sha256(k.encode('ascii')).hexdigest():list(map(int,k.split(':'))) for k in contract['roles']['target']}
    records=[];confusions=[]
    for scene in SCENES:
        for name in NAMES:
            p=torch.load(root/scene/(name+'.pt'),map_location='cpu',weights_only=True)
            assert set(p)=={'ids','top_class','confidence','accepted'}
            ids=[bytes(x.tolist()).hex() for x in p['ids']]
            assert len(ids)==len(set(ids))==len(truth) and set(ids)==set(truth)
            meta=torch.tensor([truth[k][:3] for k in ids]);y=meta[:,0];pred=p['top_class'];accepted=p['accepted']
            assert pred.shape==y.shape and ((pred>=0)&(pred<6)).all() and accepted.dtype==torch.bool
            groups=[('all','all',torch.ones(len(y),dtype=torch.bool))]
            for dim,col in [('TX',0),('RX',1),('day',2)]:groups.extend((dim,str(int(v)),meta[:,col]==v) for v in meta[:,col].unique(sorted=True))
            for rx in meta[:,1].unique(sorted=True):
                for tx in meta[:,0].unique(sorted=True):groups.append(('RX_TX',f'{int(rx)}:{int(tx)}',(meta[:,1]==rx)&(y==tx)))
            for dim,value,m in groups:records.append(dict(candidate=name,scene=scene,group=dim,value=value,**count(pred[m],y[m],accepted[m])))
            confusions.append(dict(candidate=name,scene=scene,matrix=torch.bincount(y*6+pred,minlength=36).reshape(6,6).tolist()))
    write(root/'scores.json',dict(records=records,confusions=confusions,rows_per_scene=len(truth),confirmation=False,truth_after_all_seals=True))
    with (root/'scores.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    print('ALL_SCORED',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['predict','score'],required=True);p.add_argument('--source');p.add_argument('--ground');p.add_argument('--contract',required=True);p.add_argument('--output',required=True);p.add_argument('--scene',choices=SCENES);p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(4)
    (predict if a.stage=='predict' else score)(a)
