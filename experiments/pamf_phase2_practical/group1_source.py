"""Native source objectives plus practical concat and midpoint ordinary PL.

This module never accepts target handles. All target execution is in a separate
process, after the actual source checkpoint and source-only smoke complete.
"""
import copy
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .io import read,write,atomic,activate,require
from .group1 import row,validate_source_config,pl_enabled,metric_probabilities,masked_ce
from .group1_resources import Measure
from . import external_additional as extra
from .vendor_native.comparison_suite import models as nets


EXTRA={'poster_ft','feature_separation_hu2024','issl_author_behavior','lora_triplet_knn'}


def model_for(method,device):
    if method in EXTRA:return extra.NativeModel(method,6,5).to(device)
    return nets.factory(method,6,5).to(device)


def embedding(model,method,x):
    return extra.native_features(model,x) if method in EXTRA else nets.embedding(model,method,x)


def logits(model,method,x):
    return extra.native_logits(model,x) if method in EXTRA else nets.classifier_logits(model,method,x)


def source_data(c,role):
    from experiments.cric_dura_comparison.f_data import SourceDataset
    data=SourceDataset(c['source_cache'],role,read(c['source_contract']))
    require(role!='U_s' or 'y' not in data.metadata,'Hidden U truth found')
    return data


def recipe(profile):
    r=row(profile)
    if profile=='pamf_tri_sig_time_ssl_milkr':
        return dict(schema='group1_current_PAMF_Tri_SIG_TIME_SSL_v1',profile=profile,method='pamf_tri_sig_time_ssl',
            definition='configs/pamf_ssl_trisigtime.json',epochs=50,updates=11100,signature_views=3,
            temporal_permutation=[0,5,3,6,2,1,4,7],L_only_epochs=13,ordinary_PL_epochs=[14,20],
            full_REPL_epochs=[21,50],REPL_weight=6.,PL_threshold=.95,EMA=False,
            satellite_concat=False,source_policy='reuse_exact_original_complete_Tri_SIG_TIME_SSL',
            target_contact=False,U_TX_truth_read=False)
    return dict(schema='group1_source_concat_midPL_v1',profile=profile,method=r['method'],epochs=r['epochs'],
        PL_start_epoch=r['PL_start_epoch'],PL_threshold=.95,PL_weight=1.,U_batch_ratio=2,
        satellite_concat=True,satellite_weight=.68,satellite_on='L_only',
        U_objective='ordinary_clean_source_PL',source_channel='practical_residual_post_sync_noeq',
        source_channel_probabilities=[.40,.35,.25],source_channel_scenes=['practical_low_suburban','practical_mid_urban','practical_low_urban'],
        source_channel_probability_schedule=[(0.,.3),(.2,.6),(.45,.8)],
        source_fc_hz=2462e6,source_fs_hz=25e6,
        metric_temperature=.1,no_added_classifier=True,EMA=False,early_stopping=False,selection='fixed_final_epoch',
        native_supervised_objective_retained=True,target_contact=False,U_TX_truth_read=False)


@torch.no_grad()
def extract(model,method,x,device,batch=128):
    model.eval()
    return torch.cat([embedding(model,method,x[i:i+batch].to(device)).detach().cpu()
                      for i in range(0,len(x),batch)])


def augment(reader,indices,epoch,epochs,device):
    values,info=reader.get(indices.tolist(),epoch,1,(epoch-1)/epochs,epochs)
    # Torch2.1/Numpy2 deployments do not have a usable from_numpy bridge.
    return torch.tensor(values.tolist(),device=device,dtype=torch.float32),info


def supervised(model,method,x,y,rx,*,episode_support=5):
    if method=='feature_separation_hu2024':
        from .vendor_extra.hu.losses import feature_separation_loss
        prepared=extra.augment_iq(extra.preprocess_iq(x))
        return feature_separation_loss(model.network(prepared),y,rx)[0]
    if method=='orthogonal':
        from .vendor_native.paper_reproduction.orthogonal_incremental_sei.losses import base_training_loss
        z=embedding(model,method,x);targets=model.pseudo_targets
        assigned={i:targets[i] for i in range(6)}
        perturb=model._group1_perturbed_targets
        return base_training_loss(z,y,assigned,targets,perturb)[0]
    if method=='protonet':
        z=embedding(model,method,x)
        support=torch.cat([torch.where(y==i)[0][:episode_support] for i in range(6)])
        query=torch.cat([torch.where(y==i)[0][episode_support:] for i in range(6)])
        require(len(query)>0,'Episode query is missing')
        centers=torch.stack([z[support][y[support]==i].mean(0) for i in range(6)])
        return F.cross_entropy(-torch.cdist(z[query],centers).square(),y[query])
    return F.cross_entropy(logits(model,method,x),y)


def metric_loss(model,method,anchor,positive,negative):
    a,p,n=[embedding(model,method,v) for v in (anchor,positive,negative)]
    return F.relu((a-p).square().sum(1)-(a-n).square().sum(1)+.1)


def _train_native(c,profile,seed,out,device):
    from experiments.practical_full_2x2.cache import Reader
    r=row(profile);method=r['method'];spec=recipe(profile);torch.manual_seed(seed)
    np.random.seed(seed%2**32)
    L,U,V=[source_data(c,role) for role in ('L_s','U_s','V')]
    y=torch.tensor(L.metadata['y'].tolist(),dtype=torch.long)
    x=L.x;train_x=extra._awgn(x,torch.Generator().manual_seed(seed)) if method=='lora_triplet_knn' else x
    validation_x=extra._awgn(V.x,torch.Generator().manual_seed(seed+2)) if method=='lora_triplet_knn' else V.x
    rxmap={v:i for i,v in enumerate(read(c['source_contract'])['source_rxs'])}
    rx=torch.tensor([rxmap[int(v)] for v in L.metadata['rx_i']],dtype=torch.long)
    model=model_for(method,device);out.mkdir(parents=True,exist_ok=False)
    write(out/'recipe.json',spec);write(out/'source_contract.json',read(c['source_contract']))
    initial=out/'initial_source_only.pt';torch.save(dict(model=model.state_dict(),recipe=spec),initial)
    model.load_state_dict(torch.load(initial,map_location=device,weights_only=False)['model'],strict=True)
    model.eval()
    with torch.no_grad():
        values=embedding(model,method,x[:2].to(device)) if method=='lora_triplet_knn' else logits(model,method,x[:2].to(device))
    require(len(values)==2 and bool(torch.isfinite(values).all()),'Real scratch checkpoint no-query smoke failed')
    write(out/'source_smoke.json',dict(status='PASS',real_checkpoint=True,source_rows=2,target_rows=0,query_rows=0))
    batch=32 if method=='lora_triplet_knn' else 256 if method=='feature_separation_hu2024' else 128
    lr=.01 if method in ('orthogonal','protonet') else .005 if method=='feature_separation_hu2024' else .03 if method=='issl_author_behavior' else .001
    if method=='lora_triplet_knn':opt=torch.optim.RMSprop(model.parameters(),lr=lr,alpha=.9,eps=1e-7)
    elif method in ('orthogonal','protonet'):opt=torch.optim.SGD(model.parameters(),lr=lr)
    else:opt=torch.optim.Adam(model.parameters(),lr=lr)
    scheduler=torch.optim.lr_scheduler.StepLR(opt,100,.5) if method=='issl_author_behavior' else None
    if method in ('lora_triplet_knn','feature_separation_hu2024'):
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min' if method=='lora_triplet_knn' else 'max',
            factor=.2 if method=='lora_triplet_knn' else .1,patience=10)
    if method=='orthogonal':
        from .vendor_native.paper_reproduction.orthogonal_incremental_sei.pseudo_targets import perturb_pseudo_targets
        model._group1_perturbed_targets=perturb_pseudo_targets(model.pseudo_targets,noise_range=.01,seed=seed)
    gen=torch.Generator().manual_seed(seed);ugen=torch.Generator().manual_seed(seed+1)
    reader=Reader(dict(c,protocol_id='F0'))
    Uorder=torch.randperm(len(U),generator=ugen);upos=0
    steps=seen=accepted=Ls_seen=sat_seen=sat_applied=0
    U_seen_mask=torch.zeros(len(U),dtype=torch.bool);L_seen_mask=torch.zeros(len(L),dtype=torch.bool)
    accepted_hist=torch.zeros(6,dtype=torch.long);center_seconds=0.
    with Measure(out/'A_resources.json',dict(profile=profile,seed=seed,stage='A'),model,optimizers=True,optimizer_instances=[opt]) as probe:
        with (out/'epochs.jsonl').open('x',encoding='utf-8') as log:
            for epoch in range(1,r['epochs']+1):
                epoch_started=time.perf_counter()
                total=0.;count=0;centers=None
                if method=='lora_triplet_knn' and pl_enabled(epoch,r['epochs']):
                    if device!='cpu':torch.cuda.synchronize()
                    center_started=time.perf_counter()
                    z=extract(model,method,x,device)
                    centers=torch.stack([F.normalize(z[y==i],dim=1).mean(0) for i in range(6)]).to(device)
                    if device!='cpu':torch.cuda.synchronize()
                    center_seconds+=time.perf_counter()-center_started
                if method=='protonet':batches=[torch.cat([torch.where(y==i)[0][torch.randperm(int((y==i).sum()),generator=gen)[:10]] for i in range(6)]) for _ in range(50)]
                elif method=='lora_triplet_knn':batches=[extra._triplets(y,32,gen) for _ in range(len(y)//32)]
                else:batches=[v for v in torch.randperm(len(y),generator=gen).split(batch) if len(v)>1]
                model.train()
                epoch_native=epoch_sat=epoch_PL=0.
                for indices in batches:
                    opt.zero_grad(set_to_none=True)
                    if method=='lora_triplet_knn':
                        a,p,n=indices
                        for ix in (a,p,n):L_seen_mask[ix]=True
                        base=metric_loss(model,method,train_x[a].to(device),train_x[p].to(device),train_x[n].to(device)).mean()
                        star=[];infos=[]
                        for ix in (a,p,n):
                            transformed,info=augment(reader,ix,epoch,r['epochs'],device);star.append(transformed);infos.append(info)
                        sat=metric_loss(model,method,*star).mean();Lcount=len(a)
                        sat_seen+=sum(len(v) for v in (a,p,n));sat_applied+=sum(i['applied_count'] for i in infos)
                    else:
                        ix=indices;xx=x[ix].to(device);yy=y[ix].to(device);rr=rx[ix].to(device)
                        L_seen_mask[ix]=True
                        base=supervised(model,method,xx,yy,rr)
                        star,info=augment(reader,ix,epoch,r['epochs'],device)
                        sat=supervised(model,method,star,yy,rr) if method=='protonet' else F.cross_entropy(logits(model,method,star),yy)
                        Lcount=len(ix);sat_seen+=Lcount;sat_applied+=info['applied_count']
                    value=base+.68*sat;pl=value*0
                    if pl_enabled(epoch,r['epochs']):
                        ub=2*Lcount
                        if upos+ub>len(U):Uorder=torch.randperm(len(U),generator=ugen);upos=0
                        ui=Uorder[upos:upos+ub];upos+=len(ui);ux=U.x[ui].to(device)
                        U_seen_mask[ui]=True
                        model.eval()
                        with torch.no_grad():
                            if method=='lora_triplet_knn':prob=metric_probabilities(embedding(model,method,ux),centers)
                            elif method=='protonet':
                                zp=embedding(model,method,x[ix].to(device));cy=y[ix].to(device)
                                cp=torch.stack([zp[cy==i][:5].mean(0) for i in range(6)])
                                prob=(-torch.cdist(embedding(model,method,ux),cp).square()).softmax(1)
                            else:prob=logits(model,method,ux).softmax(1)
                            conf,pseudo=prob.max(1)
                        model.train();mask=conf>=.95;seen+=len(ux);accepted+=int(mask.sum())
                        accepted_hist+=torch.bincount(pseudo[mask].cpu(),minlength=6)
                        if method=='lora_triplet_knn':
                            positives=[];negatives=[]
                            for label in pseudo.cpu().tolist():
                                pi=torch.where(y==label)[0];ni=torch.where(y!=label)[0]
                                positives.append(int(pi[torch.randint(len(pi),(),generator=gen)]))
                                negatives.append(int(ni[torch.randint(len(ni),(),generator=gen)]))
                            pl=(metric_loss(model,method,ux,x[positives].to(device),x[negatives].to(device))*mask).mean()
                        elif method=='protonet':
                            zu=embedding(model,method,ux)
                            pl=masked_ce(-torch.cdist(zu,cp.detach()).square(),pseudo,conf)
                        else:pl=masked_ce(logits(model,method,ux),pseudo,conf)
                        value=value+pl
                    if not bool(torch.isfinite(value)):raise FloatingPointError('Nonfinite native Group1 source loss')
                    value.backward()
                    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                        raise FloatingPointError('Nonfinite native Group1 source gradient')
                    opt.step();steps+=1;Ls_seen+=Lcount;total+=float(value.detach());count+=1
                    epoch_native+=float(base.detach());epoch_sat+=float(sat.detach());epoch_PL+=float(pl.detach())
                validation_value=None
                if method in ('lora_triplet_knn','feature_separation_hu2024'):
                    model.eval();vy=torch.tensor(V.metadata['y'].tolist(),dtype=torch.long)
                    with torch.no_grad():
                        if method=='lora_triplet_knn':
                            vg=torch.Generator().manual_seed(seed+epoch)
                            val=[]
                            for _ in range(len(vy)//32):
                                a,p,n=extra._triplets(vy,32,vg)
                                val.append(float(metric_loss(model,method,validation_x[a].to(device),validation_x[p].to(device),validation_x[n].to(device)).mean()))
                            validation_value=sum(val)/len(val)
                        else:
                            correct=0
                            for vi in torch.arange(len(V)).split(256):
                                correct+=int((logits(model,method,V.x[vi].to(device)).argmax(1).cpu()==vy[vi]).sum())
                            validation_value=correct/len(V)
                    scheduler.step(validation_value)
                elif scheduler:scheduler.step()
                record=dict(epoch=epoch,optimizer_steps=steps,loss=total/count,native_supervised=epoch_native/count,
                    satellite_loss=epoch_sat/count,PL_loss=epoch_PL/count,PL_enabled=pl_enabled(epoch,r['epochs']),
                    PL_seen=seen,PL_accepted=accepted,L_anchor_exposure=Ls_seen,satellite_seen=sat_seen,
                    satellite_actually_applied=sat_applied,U_TX_truth_read=False,target_contact=False,
                    learning_rate=opt.param_groups[0]['lr'],source_V_native_scheduler_metric=validation_value,
                    epoch_wall_seconds=time.perf_counter()-epoch_started,
                    U_unique_exposed=int(U_seen_mask.sum()),L_supervised_unique=int(L_seen_mask.sum()),
                    PL_accepted_class_histogram=accepted_hist.tolist(),metric_center_refresh_seconds=center_seconds)
                log.write(__import__('json').dumps(record)+'\n');log.flush();atomic(out/'progress.json',record)
                print('SOURCE_EPOCH',profile,epoch,r['epochs'],total/count,seen,accepted,flush=True)
        model.eval();native={}
        if method=='protonet':
            z=extract(model,method,x,device)
            model.source_prototypes.copy_(torch.stack([z[y==i].mean(0) for i in range(6)]).to(device));model.prototypes_ready.fill_(True)
        if method=='lora_triplet_knn':native=dict(source_enrollment_features=extract(model,method,x,device),source_enrollment_y=y)
        if method=='issl_author_behavior':native=dict(source_replay_x=x,source_replay_y=y)
        torch.save(dict(schema='group1_A_v1',model=model.state_dict(),method=method,profile=profile,
            seed=seed,recipe=spec,source_contract=read(c['source_contract']),native_state=native,
            ancestors=[],initialization='scratch',target_contact=False),out/'source.pt')
        probe.result.update(actual_epochs=r['epochs'],native_L_update_steps=steps,PL_seen=seen,PL_accepted=accepted,
            L_anchor_exposure=Ls_seen,satellite_seen=sat_seen,satellite_actually_applied=sat_applied,
            augmentation_wait_seconds=reader.wait_seconds,source_roles=dict(L_s=len(L),U_s=len(U),V=len(V)),
            source_RX_count=5,source_old_classes=6,source_true_label_unique=len(L),source_U_true_labels=0)
        probe.result.update(U_unique_exposed=int(U_seen_mask.sum()),L_supervised_unique=int(L_seen_mask.sum()),
            PL_accepted_class_histogram=accepted_hist.tolist(),metric_center_refresh_seconds=center_seconds,L_batch=batch,U_batch=2*batch)


def pamf_epoch_resource(config,model,updates,started):
    if getattr(model,'_group1_last_epoch_updates',None)==updates:return
    now=time.perf_counter();previous=getattr(model,'_group1_epoch_clock',started)
    model._group1_epoch_clock=now;model._group1_last_epoch_updates=updates
    epoch=(updates+config.get('step_offset',0))//222
    record=dict(global_epoch=epoch,local_optimizer_updates=updates,epoch_wall_seconds=now-previous,
        PL_enabled=epoch>=26,PL_seen=model._cd_U_seen,PL_accepted=model._cd_U_accepted,
        satellite_seen=model._cd_sat_seen,satellite_actually_applied=model._cd_sat_applied,
        PL_accepted_class_histogram=getattr(model,'_group1_PL_hist',[0]*6),
        U_TX_truth_read=False,target_contact=False,observed_at=time.time())
    with (Path(config['output_root'])/'group1_epoch_resources.jsonl').open('a',encoding='utf-8') as f:
        f.write(__import__('json').dumps(record)+'\n')


def bind_pamf_trainer(c):
    """Bind the historical BASE trainer that actually executes labeled hooks."""
    import inspect
    import sys
    from experiments.cvs_phase1_overlay import model as overlay
    directory=Path(c['frozen_sources']['BASE'])/'experiments/adv3b02_xuc/code'
    require((directory/'SSDG/train_ssdg.py').is_file(),'Fixed BASE source trainer missing')
    overlay.NATIVE=directory
    sys.path.insert(0,str(directory))
    native=overlay.native_modules()
    training_function=inspect.unwrap(native.train)
    actual=Path(inspect.getsourcefile(training_function)).resolve()
    require(actual==(directory/'SSDG/train_ssdg.py').resolve(),
            'PAMF native trainer was already imported from a different release')
    require('_source_labeled_feature_hook' in inspect.getsource(training_function),
            'PAMF native trainer cannot execute the full supervised objective')
    return native


def _train_pamf(c,seed,out,*,diagnostic=False):
    bind_pamf_trainer(c)
    from experiments.cric_dura_comparison import t1_train,t1_model,f_train
    from experiments.practical_full_2x2 import pamf_model
    original_install=pamf_model.install
    from experiments.pamf_repl_view_shuffle.source import seed_scope
    from types import MethodType
    cfg=copy.deepcopy(c['source_base_config'])
    for key in ('repl_row','repl','hard_pl','view_comparison','algorithm_id','paper_method','parent_source','core_mechanisms'):
        cfg.pop(key,None)
    cfg.update(run_id=c['run_id'],row_id='pamf_pl_milkr-s'+str(seed),model_seed=seed,
        method_id='pamf_t1_e50',supervised_only_epochs=25,
        output_root=str(out),source_contract=c['source_contract'],source_cache=c['source_cache'],
        code_commit=c['code_commit'],regime='matched_sat_plain_pl',ablation='F21',U_use=True,
        U_start_epoch=26,own_EMA=False,native_U_gate=False,plain_PL_weight=1.,plain_PL_threshold=.95,
        satellite_concat=True,satellite_CE_weight=.68,channel_augmentation=True,
        augmentation_cache=c['augmentation_cache'],protocol_id='F0',supervised_architecture=None,
        default_head_joint=True,posthoc_head_fit=False,group1_recipe=recipe('pamf_pl_milkr'))
    def install(model,config):
        model=original_install(model,config);original=model._source_labeled_feature_hook
        def counted(self,**kwargs):
            correction,metrics=original(**kwargs)
            if self._feature_l_steps%222==0:
                atomic(Path(config['output_root'])/'group1_progress.json',dict(
                    local_L_forward_steps=self._feature_l_steps,global_epoch=(self._feature_l_steps+config.get('step_offset',0))//222,
                    PL_seen=self._cd_U_seen,PL_accepted=self._cd_U_accepted,
                    satellite_seen=self._cd_sat_seen,satellite_actually_applied=self._cd_sat_applied,
                    native_As_seen=self._cd_As_seen,PL_accepted_class_histogram=getattr(self,'_group1_PL_hist',[0]*6),U_TX_truth_read=False))
            return correction,metrics
        counted._accepts_labeled_inputs=True
        model._source_labeled_feature_hook=MethodType(counted,model)
        return model
    old_context=f_train.context
    def context(config):
        from experiments.cric_dura_comparison.f_data import FixedEpochLoader
        value=old_context(config)
        U=value['unlabeled_loader'].dataset
        value['unlabeled_loader']=FixedEpochLoader(U,256,config['model_seed'],drop_last=False)
        return value
    old_pl=pamf_model.plain_pl
    def plain_pl(self,device):
        try:batch=next(self._cd_U_iter)
        except StopIteration:
            self._cd_U_iter=iter(self._cd_U_loader);batch=next(self._cd_U_iter)
        require(bool((batch[1]==-1).all()),'U truth exposed')
        x=batch[0].to(device);self.eval()
        with torch.no_grad():confidence,pseudo=self(x).softmax(1).max(1)
        self.train();mask=confidence>=.95
        values=self.forward_identity_only(x,domain_labels=batch[2].to(device))['tx_logits']
        self._cd_U_seen+=len(x);self._cd_U_accepted+=int(mask.sum())
        hist=torch.bincount(pseudo[mask].cpu(),minlength=6).tolist()
        self._group1_PL_hist=[a+b for a,b in zip(getattr(self,'_group1_PL_hist',[0]*6),hist)]
        return masked_ce(values,pseudo,confidence)
    saved=t1_model.install;t1_model.install=install;f_train.context=context;pamf_model.plain_pl=plain_pl
    try:
        with seed_scope(),Measure(out/'A_resources.json',dict(profile='pamf_pl_milkr',seed=seed,stage='A'),optimizers=True) as probe:
            if diagnostic:
                # Four real CPU optimizer updates, source-only, never an initializer.
                # U starts immediately here solely to exercise its actual backward path.
                out.mkdir(parents=True,exist_ok=False)
                smoke=dict(cfg,output_root=str(out/'segment20'),segment_epochs=20,
                    epoch_offset=0,step_offset=0,parent_source=None,U_start_epoch=1)
                f_train.train(smoke,device='cpu',smoke_steps=2)
                value=read(out/'segment20/progress.json')
                require(value['CRIC_calls']==4 and value['PL_seen']==1024,'PAMF actual training hook/PL missing')
                records=[__import__('json').loads(v) for v in (out/'segment20/group1_epoch_resources.jsonl').read_text().splitlines()]
                require(records[-1]['satellite_seen']==512 and records[-1]['satellite_actually_applied']>0,
                        'PAMF actual satellite supervision missing')
                probe.result.update(diagnostic_only=True,query_reads=0,target_reads=0,
                    smoke_weights_not_training_initializers=True,**value)
                return
            t1_train.train(cfg)
            records=[read(p) for p in sorted(out.glob('segment*/group1_progress.json'))]
            require(len(records)==3,'PAMF source exposure counters missing')
            probe.result.update({key:sum(v[key] for v in records) for key in
                ('PL_seen','PL_accepted','satellite_seen','satellite_actually_applied','native_As_seen')})
            probe.result.update(actual_epochs=50,native_L_update_steps=11100,L_batch=128,U_batch=256,
                deployed_F_parameters=232993,PL_start_epoch=26,ordinary_PL_weight=1.,U_TX_truth_read=False)
            probe.result.update(U_unique_exposed=56700,
                PL_accepted_class_histogram=[sum(v['PL_accepted_class_histogram'][i] for v in records) for i in range(6)])
            epochs=[__import__('json').loads(line) for p in sorted(out.glob('segment*/group1_epoch_resources.jsonl')) for line in p.read_text().splitlines()]
            require(len(epochs)==50,'PAMF actual epoch timing records missing')
            probe.result.update(supervised_only_seconds=sum(v['epoch_wall_seconds'] for v in epochs if not v['PL_enabled']),
                joint_PL_seconds=sum(v['epoch_wall_seconds'] for v in epochs if v['PL_enabled']))
    finally:t1_model.install=saved;f_train.context=old_context;pamf_model.plain_pl=old_pl


def load_source(c,profile,seed,device):
    validate_source_config(c);folder=Path(c.get('external_source_root',c['run_root']))/'source'/f'{profile}-s{seed}'
    expected=read(c['source_contract']);r=row(profile)
    if profile=='pamf_tri_sig_time_ssl_milkr':
        from .trisig_source import load
        return load(c,seed,device)
    if profile=='pamf_pl_milkr':
        bind_pamf_trainer(c)
        from experiments.cric_dura_comparison.frozen_inference import load_F
        from experiments.pamf_repl_view_shuffle.source import seed_scope
        with seed_scope():model,cfg=load_F(folder,expected,device)
        require(cfg['group1_recipe']==recipe(profile),'PAMF source recipe mismatch')
        return dict(model=model,reader=model,classes=tuple(range(6)),profile=profile,method='pamf')
    payload=torch.load(folder/'source.pt',map_location='cpu',weights_only=False)
    require(payload['schema']=='group1_A_v1' and payload['recipe']==recipe(profile),'Source recipe mismatch')
    require(payload['source_contract']==expected and payload['seed']==seed and not payload['target_contact'] and payload['ancestors']==[],
            'CHECKPOINT_DATA_CONTRACT_MISMATCH_OR_PROVENANCE')
    model=model_for(r['method'],device);model.load_state_dict(payload['model'],strict=True);model.eval().requires_grad_(False)
    native=payload['native_state'];score=lambda net,x:logits(net,r['method'],x)
    if r['method']=='lora_triplet_knn':
        base=dict(model=model,profile=r['method'],native_state=native,resources={},classes=tuple(range(6)))
        base=extra._metric_state(base,native['source_enrollment_features'],native['source_enrollment_y'],source=True)
        score=base['score_fn']
    return dict(model=model,profile=profile,method=r['method'],classes=tuple(range(6)),
        score_fn=score,features_fn=lambda net,x:embedding(net,r['method'],x),native_state=native,
        base_metadata={},source_contract=expected,resources={},source_path=str(folder),source_checkpoint=str(folder/'source.pt'),
        provenance=dict(recipe=recipe(profile),source_only=True))


def train(c,profile,seed,output,device='cuda:0'):
    validate_source_config(c);out=Path(output)
    if profile=='pamf_tri_sig_time_ssl_milkr':
        from .trisig_source import reuse
        return reuse(c,seed,out,device)
    total=Path(c['run_root'])/'resource_telemetry'/f'{profile}-s{seed}'/'A_end_to_end.json'
    with Measure(total,dict(profile=profile,seed=seed,stage='A_END_TO_END_SOURCE_ONLY',includes_data_and_serialization=True)) as probe:
        activate(c);torch.set_num_threads(c.get('source_cpu_threads',1))
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        r=row(profile);require(r['active'],'Unbound native method cannot be trained')
        if profile=='pamf_pl_milkr':_train_pamf(c,seed,out)
        else:_train_native(c,profile,seed,out,device)
        bundle=load_source(c,profile,seed,device);L=source_data(c,'L_s')
        with torch.no_grad():
            if profile=='pamf_pl_milkr':
                from .features import branches
                _,values=branches(bundle['reader'],L.x[:2].to(device))
            else:values=bundle['score_fn'](bundle['model'],L.x[:2].to(device))
        require(bool(torch.isfinite(values).all()) and len(values)==2,'Final real-checkpoint no-query smoke failed')
        probe.model=bundle['model'];probe.result['training_resources']=str(out/'A_resources.json')
        write(out/'group1_complete.json',dict(status='GROUP1_SOURCE_READY',profile=profile,seed=seed,
            recipe=recipe(profile),real_checkpoint_smoke='PASS',target_reads=0,query_reads=0,
            model_parameters=sum(p.numel() for p in bundle['model'].parameters()),
            source_contract=c['source_contract'],resources='A_resources.json',end_to_end_resources=str(total)))
