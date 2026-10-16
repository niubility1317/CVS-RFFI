"""Frozen real checkpoint smoke first, then all fit-L default-head decisions."""
import os,sys,time
from pathlib import Path
import torch
from . import design as d
from .core import rotate,select_fit,score,qualify

def checkpoint_model():
    from experiments.cvs_crossrx_satmix20 import design as model_design,model as api
    src=Path(d.MODEL_SOURCE);init=d.read(src/'initialization.json');done=d.read(src/'completion.json');frozen=d.read(src/'source_frozen.json');contract=d.read(src/'source_contract.json')
    if not init['scratch_only'] or init['checkpoint_sources'] or init['ancestors'] or init['target_contact'] or init['seed']!=2026092701:raise ValueError('Unmatched checkpoint ancestry')
    if done['epoch']!=20 or done['optimizer_steps']!=4440 or done['target_access'] or done['checkpoint_sources'] or done['head_fit_updates'] or frozen['epoch']!=20 or frozen['selection']!='final_only' or frozen['target_access']:raise ValueError('Fixed final20 source-only checkpoint required')
    if contract['held_RX']!=1 or set(contract['source_receivers'])!=set(d.FIT) or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}:raise ValueError('Model data roles differ')
    ck=torch.load(src/'final_ssdg.pth',map_location='cpu',weights_only=False)
    if ck['args']['seed']!=2026092701 or ck['args']['run_id']!=d.PARENT_RUN or ck['args']['baseline_ckpt'] or ck['args']['teacher_ckpt'] or ck['args']['source_fold_held_rx']!=1 or ck['checkpoint_selection']!='final_only' or ck['a1_ema_successful_updates']!=4440:raise ValueError('Payload provenance differs')
    c=model_design.config(model_design.rows()[0])
    if any(c[k]!=frozen['config'][k] for k in ('arm','model_seed','epochs','training_mechanism','supcon','fold')) or c['arm']!='B00' or not c['training_mechanism']['CROSS_RX'] or c['training_mechanism']['Ag'] or c['training_mechanism']['As']:raise ValueError('Original F+CROSS_RX B00 recipe differs')
    m=api.build(c,'cpu').eval();m.load_state_dict(ck['model'],strict=True);m.requires_grad_(False)
    ref=m.id_backbone.encoder;head=ref.core.id_backbone.cls_head
    if len(ref.sequence_pool.views)!=3 or ref.sequence_pool.modes!=('SIG_TIME',)*3 or ref.response.contract()['explicit_CFO_identity_input'] is not False or head.weight.shape!=(6,160) or head.scale!=30.:raise ValueError('Existing F/NoExplicitCFO/defaultcos boundary differs')
    return m,contract

def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Registered CPU-only diagnostic')
    torch.set_num_threads(2);c=d.config();d.validate(c);out=d.BASE;out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    d.write(out/'active.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
    try:
        m,trained=checkpoint_model();before={k:v.detach().clone() for k,v in m.state_dict().items()}
        with torch.no_grad():
            x=torch.randn(2,2,256,generator=torch.Generator().manual_seed(20261016131));z=m.forward_identity_only(x)['tx_logits']
        if z.shape!=(2,6) or not bool(torch.isfinite(z).all()) or m._supcon_pending is not None or m._supcon_handles:raise ValueError('Real checkpoint no-query smoke failed')
        d.write(out/'real_checkpoint_smoke.json',dict(status='PASS',strict_load=True,checkpoint_role='Frozen own matching F+CROSS_RX B00,not training initialization',synthetic_NN_packets=2,source_IQ_reads=0,target_access=False,optimizer_updates=0))
        print('SMOKE PASS real matching checkpoint;2synthetic packets;no query;continue',flush=True)
        carrier=d.read(d.SOURCE);prepared=d.read(Path(d.PREPARATION)/'preparation_complete.json')
        if prepared['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or prepared['target_access'] or prepared['identity_updates']!=0:raise ValueError('Existing source preparation differs')
        packed=torch.load(Path(d.PREPARATION)/'source_cache.pt',map_location='cpu',weights_only=False)
        for actual in (packed['contract'],prepared['source_contract']):
            if any(actual.get(k)!=v for k,v in carrier.items()) or actual.get('native_role_comparison')!='EXACT_MATCH':raise ValueError('Reused validated source handle mismatch')
        L=select_fit({k:packed['labeled'][k] for k in ('x','y','rx','day','ids')},carrier,trained);del packed
        d.write(out/'fit_L_ids.json',L['ids'])
        d.write(out/'resolved_config.json',dict(config=c,commit=(d.ROOT/'release_commit.txt').read_text().strip(),backend_commit=d.BACKEND_COMMIT,python=sys.executable,torch_version=torch.__version__,checkpoint_provenance='Own scratch E20/fixedfinal20/matching fourRX/source-only;previous verified source audit reused',fit_unique_L=5040,held_RX1_used=False,other_source_cache_roles_and_features_deserialized_but_never_used=True,source_IQ_transform='Only constant phase,never RMS/PA/noise/channel',target_access=False))
        print('INPUT PASS fitL5040;heldRX1/U/V/target unused;frozen default head',flush=True)
        (out/'predictions').mkdir();scores={};power_errors={}
        labels=L['y'].tolist();rx=L['rx'].tolist();day=L['day'].tolist()
        with torch.no_grad():
            for view,angle in d.VIEWS.items():
                predictions=[];logits=[];ce=0.;maximum=0.
                for start_idx in range(0,5040,128):
                    raw=L['x'][start_idx:start_idx+128];x=rotate(raw,angle);y=L['y'][start_idx:start_idx+128]
                    p=raw.square().sum(1).mean(1);q=x.square().sum(1).mean(1)
                    maximum=max(maximum,float(((q-p).abs()/p.clamp_min(1e-12)).max()))
                    z=m.forward_identity_only(x)['tx_logits'].double()
                    if z.shape!=(len(y),6) or not bool(torch.isfinite(z).all()):raise ValueError('Finite complete default-head predictions required')
                    predictions+=z.argmax(1).tolist();logits+=z.tolist();ce+=float(torch.nn.functional.cross_entropy(z,y,reduction='sum'))
                    d.write(out/'progress.json',dict(view=view,source_decisions=len(scores)*5040+len(predictions),total_source_decisions=15120,optimizer_updates=0,target_access=False))
                if maximum>1e-6:raise ValueError('Unit phase power preservation failed')
                value=score(labels,predictions);value['mean_CE']=ce/5040
                for name,values,allowed in (('RX',rx,d.FIT),('TX',labels,range(6)),('day',day,(1,2,3))):
                    value['by_'+name]={str(v):score([y for y,key in zip(labels,values) if key==v],[p for p,key in zip(predictions,values) if key==v]) for v in allowed}
                cells=[]
                for r in d.FIT:
                    for dd in (1,2,3):
                        for tx in range(6):
                            ids=[i for i,(rr,dy,y) in enumerate(zip(rx,day,labels)) if (rr,dy,y)==(r,dd,tx)]
                            cells.append(dict(RX=r,day=dd,TX=tx,**score([labels[i] for i in ids],[predictions[i] for i in ids])))
                value['cells']=cells;scores[view]=value;power_errors[view]=maximum
                d.write(out/'predictions'/(view+'.json'),dict(view=view,angle_rad=angle,labels=labels,predictions=predictions,logits=logits,physical_ids=L['ids'],RX=rx,day=day,role='fitL',head_fit_updates=0,target_access=False))
                print('VIEW '+view+' count5040 accuracy='+str(value['accuracy'])+' power_error='+str(maximum),flush=True)
        if m.training or any(not torch.equal(before[k],v) for k,v in m.state_dict().items()) or any(p.grad is not None for p in m.parameters()) or m.id_backbone.encoder._view_eval_packets!=15122 or m.id_backbone.encoder._view_train_packets!=0 or m._joint_calls or m._joint_As_packets or m._supcon_calls:raise ValueError('Frozen state or actual NN cost differs')
        result=dict(status='ANALYZED_SOURCE_ONLY',scores=scores,qualification=qualify(scores),power_relative_error_max=power_errors,physical_L_packets=5040,source_decisions=15120,synthetic_smoke_NN_packets=2,actual_NN_packets=15122,actual_optimizer_updates=0,head_fit_updates=0,default_head_only=True,model_state_unchanged=True,NoExplicitCFO=True,teacher_forwards=0,As_generations=0,U_used=False,V_used=False,held_RX1_used=False,target_access=False,elapsed_seconds=time.perf_counter()-start)
        d.write(out/'result.json',result);d.write(out/'completion.json',dict(status='ANALYZED_SOURCE_ONLY',decision=result['qualification']['decision'],views=3,source_decisions=15120,actual_NN_packets=15122,optimizer_updates=0,target_access=False,automatic_training=False))
        print('COMPLETE '+str(result['qualification']),flush=True)
    except Exception as error:
        d.write(out/'failure.json',dict(status='FAILED',error=repr(error),no_automatic_retry=True));raise
