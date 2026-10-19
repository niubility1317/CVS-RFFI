"""Zero-update, source-L-only readout of fixed E30 feature adapters.

Compare clean joint160 drift, last observed As fit, independent channel fit,
and zero-output adapter Dropout mismatch. No target reader or fitting path.
"""
import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np

RUN = '20261019-diagnostic-F-dualAdapter-sourceFit-e30-s2026092701-m8-r01'
REFERENCE_RUN = '20261019-phase1-F-dualTeacher-adapter-e20plus10-s2026092701-m6-r02'
REFERENCE_COMMIT = '9a376ab313b3ba42459df32694c1502675e02a59'
PANEL_RUN = '20261019-diagnostic-F-observedChannel-e20-s2026092701-m4-r01'
NAMES = ('TC','INIT_ZERO','NOAS_ADAPTER','S0','S1','S2','S3','TS')
VIEWS = ('raw','observed_last','fresh_mid_d0','fresh_mid_d1','fresh_low_d0','fresh_low_d1')
DROP_SEEDS = (2026101901,2026101902,2026101903,2026101904)
SEED = 2026092701
MAX_NN = 9361
KEYS = ('eq_i','rx_i','day_i','sig_i','base_index')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def module_at(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def bootstrap(runtime):
    rt=read(runtime);ref=Path(rt['reference_release'])
    if (ref/'release_commit.txt').read_text().strip()!=REFERENCE_COMMIT:
        raise ValueError('Fixed r02 reference commit differs')
    module_at('_sourcefit_immutable_r02',ref/'experiments/cvs_dual_teacher_adapter30/runner.py').bootstrap()
    return rt


def checked_panel(selected,base,roles,opaque,classes):
    if (len(selected)!=180 or len({r['physical_id'] for r in selected})!=180
            or Counter((r['TX'],r['RX'],r['day']) for r in selected)!=Counter(
                {(tx,rx,day):2 for tx in range(6) for rx in (1,3,4,6,8) for day in (1,2,3)})):
        raise ValueError('Reuse fixed balanced 180 L panel; no reselection')
    if (list(base.tx_list)!=list(classes) or base.transform is not None or not base.normalize or base.center):
        raise ValueError('Original normalized raw Dataset required')
    legal=set(roles['L_s'])
    for r in selected:
        pid=r['pid']
        if len(pid)!=5 or not 0<=pid[4]<len(base.index):
            raise ValueError('Invalid L physical index before IQ access')
        it=base.index[pid[4]]
        if (tuple(pid)!=(it.eq_i,it.rx_i,it.day_i,it.sig_i,pid[4])
                or opaque(it)!=r['physical_id'] or r['physical_id'] not in legal
                or (it.tx_i,it.rx_i,it.day_i)!=(r['TX'],r['RX'],r['day'])
                or it.eq_i!=1 or it.rx_i not in (1,3,4,6,8) or it.day_i not in (1,2,3)):
            raise ValueError('Source L physical roles differ before IQ access')


def run(runtime,output):
    rt=bootstrap(runtime)
    import torch
    from torch.nn import functional as F
    from experiments.cvs_dual_teacher_adapter30 import design as d,model as dm
    from experiments.cvs_F_nontrue20 import design as pd,model as pm
    from experiments.cvs_F_nontrue20.teacher import load as load_ts
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_equivariant_identity.precision import numerical_context
    from experiments.cvs_fullsource_concat20.channel import Practical,serial
    from experiments.cvs_fullsource_concat20 import design as cd
    from experiments.cvs_receiver_residual_capacity4.control import lock
    from experiments.cvs_multi_state_action.dispatch import capacity
    from cvsrffi.xuc_fusion.native import role_ids_from_native
    from cvsrffi.game_tracking.data import opaque_id
    helper=module_at('_sourcefit_panel',Path(__file__).with_name('cvs_dual_teacher_adapter_source_panel_20261019.py'))
    metrics=module_at('_sourcefit_metrics',Path(__file__).with_name('cvs_dual_teacher_adapter_source_metrics_20261019.py'))
    if (os.geteuid()==0 or d.RUN!=REFERENCE_RUN or Path(d.PROJECT)!=Path(rt['project'])):
        raise ValueError('Ordinary account and fixed r02 runtime required')
    out=Path(output)
    if out!=Path(rt['project'])/'runs'/RUN:
        raise ValueError('One registered immutable output root only')
    out.mkdir(parents=True,exist_ok=False)
    write(out/'execution.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,
        started_at=time.time(),target_access=False,optimizer_updates=0,EMA_updates=0,head_fit_updates=0))
    torch.set_num_threads(2)
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
    if gpu not in (4,6,7):raise ValueError('Registered GPU4/6/7 only')
    with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
        state=capacity()[gpu]
        if len(state['pids']|{os.getpid()})>2 or state['free_mb']<12000:
            raise ValueError('Total-owner cap2 and free12GB required')
        torch.cuda.init()
        if len(capacity()[gpu]['pids']|{os.getpid()})>2:
            raise ValueError('GPU capacity changed during initialization')
    roles=d.read(d.SOURCE)['role_ids']
    if set(roles)!={'L_s','U_s','V'} or any(len(roles[k])!=n for k,n in [('L_s',6300),('U_s',56700),('V',27000)]):
        raise ValueError('Existing full source contract required')
    panelroot=Path(d.PROJECT)/'runs'/PANEL_RUN
    panel=read(panelroot/'metadata.json')
    if (panel['run_id']!=PANEL_RUN or panel['model_seed']!=SEED or panel['target_access']
            or read(panelroot/'complete.json')['status']!='SOURCE_ONLY_FROZEN_READOUT_COMPLETE'):
        raise ValueError('Fixed source180 panel provenance differs')
    selected=panel['selected']
    pc=pd.config(next(r for r in pd.rows() if r['arm']=='B0'))
    args=SimpleNamespace(**read(d.PARENT_SOURCE/'actual_native_args.json'))
    if not args.a1_source_screen_only or args.baseline_ckpt or args.teacher_ckpt or args.use_concat_sat_channel_aug or args.wisig_out_len!=256:
        raise ValueError('Original source-only native Dataset builder required')
    # This scope closes before ANY teacher or student construction.
    with installed(pc) as native:
        ctx=native._build_ssdg_wisig_data(args,torch.device('cpu'))
    if ctx['named_test_loaders'] or role_ids_from_native(ctx)!=roles:
        raise ValueError('Dataset roles differ; target loaders forbidden')
    base=ctx['train_loader'].dataset.base
    checked_panel(selected,base,roles,opaque_id,d.CLASSES)
    raw=[]
    for r in selected:
        x,_,_,meta=base[r['pid'][4]]
        if tuple(int(meta[k]) for k in KEYS)!=tuple(r['pid']) or x.shape!=(2,256) or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()):
            raise ValueError('Actual raw L preprocessing differs')
        raw.append(x)
    raw=torch.stack(raw)
    src=d.BASE/('S3-s'+str(SEED))/'source'
    origins,origin_audit=helper.recover_origins(src/'training_sequence.jsonl.gz',src/'mechanism_steps.jsonl.gz',selected)
    nn=0;results={};arrays={};state_checks={};head_errors={};drop=None
    labels=np.asarray([r['TX'] for r in selected],dtype=np.int64)
    def asnp(t):return np.asarray(t.detach().cpu().tolist(),dtype=np.float64)
    def forward(m,x,joint=True):
        nonlocal nn
        zz=[];jj=[]
        with torch.no_grad():
            for start in range(0,len(x),128):
                xb=x[start:start+128].to('cuda:0')
                val=m.forward_identity_only(xb) if joint else m(xb)
                z=val['tx_logits'];nn+=len(xb)
                if z.shape!=(len(xb),6) or not bool(torch.isfinite(z).all()):raise ValueError('Finite default logits required')
                zz.append(asnp(z))
                if joint:
                    j=val['id_feat_joint']
                    if j.shape!=(len(xb),160) or not bool(torch.isfinite(j).all()):raise ValueError('Actual joint160 required')
                    jj.append(asnp(j))
                    h=dm.classifier(m)
                    expected=h.classify(j)
                    err=float((expected-z).abs().max())
                    if err>2e-5:raise ValueError('Captured actual joint and unchanged default head logits differ')
                    head_errors[current]=max(head_errors.get(current,0.),err)
        return np.concatenate(zz),np.concatenate(jj) if joint else None
    def snapshot(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    with numerical_context(d.FULL_FP32_POLICY):
        # Launcher-first real checkpoint/no-query smoke; immediately continue.
        tc=pm.own_model(pc,'cuda:0');current='TC'
        before=snapshot(tc)
        smoke,_=forward(tc,raw[:1])
        write(out/'source_smoke.json',dict(status='PASS',source_packets=1,checkpoint='B0 FINAL20',target_access=False))
        print('SOURCE_SMOKE_PASS; continuing zero-update readout',flush=True)
        x,channelmeta=helper.build_views(raw,selected,origins,cd,Practical,serial)
        if x.shape!=(180,6,2,256) or channelmeta['generated_packets']!=900:
            raise ValueError('Fixed 180x6 view budget differs')
        write(out/'source_panel.json',dict(selected=selected,origins=origins,origin_audit=origin_audit,channel=channelmeta,
            views=list(VIEWS),source_raw_reads=180,target_access=False))
        for current in NAMES:
            if current=='TC':m=tc
            elif current=='TS':m=load_ts('cuda:0')
            elif current=='INIT_ZERO':
                c=d.config(next(r for r in d.rows() if r['arm']=='S3'))
                parent=dm.parent_payload(c)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(SEED);m=dm.build_structure(c,'cpu')
                s=m.state_dict();s.update(parent['model']);m.load_state_dict(s,strict=True)
                m=m.to('cuda:0').eval().requires_grad_(False)
            else:m=dm.own_model(d.config(next(r for r in d.rows() if r['arm']==current)),'cuda:0')
            if m.training or any(p.requires_grad or p.grad is not None for p in m.parameters()):
                raise ValueError('Readout models must remain frozen')
            initial=before if current=='TC' else snapshot(m)
            logits,joint=forward(m,x.reshape(1080,2,256),current!='TS')
            logits=logits.reshape(180,6,6)
            joint=None if joint is None else joint.reshape(180,6,160)
            if current!='TS':
                h=dm.classifier(m);weight=asnp(h.weight);scale=float(h.scale)
                if weight.shape!=(6,160) or scale!=30:raise ValueError('Original6x160 cosine30 default head required')
            else:weight=None;scale=5.
            results[current]=dict(logits=logits,joint=joint,head=weight,scale=scale,origin='fixed E20' if current in ('TC','INIT_ZERO') else 'fixed E200' if current=='TS' else 'fixed E30')
            arrays[current+'_logits']=logits
            if joint is not None:arrays[current+'_joint']=joint
            if current=='INIT_ZERO':
                equal=float(np.max(np.abs(logits-results['TC']['logits'])))
                equal_joint=float(np.max(np.abs(joint-results['TC']['joint'])))
                if equal>2e-5 or equal_joint>2e-6:raise ValueError('Zero adapter evaluation does not preserve legal parent function')
                # Isolate Dropout: parent remains eval; no hooks, gradients or optimizer.
                layers=[z for z in m.id_backbone.modules() if isinstance(z,torch.nn.Dropout)]
                if len(layers)!=13:raise ValueError('Actual F identity Dropout topology differs')
                dropout_spec=[dict(p=z.p,inplace=z.inplace) for z in layers]
                dz=[];dj=[]
                for seed in DROP_SEEDS:
                    with torch.random.fork_rng(devices=[0]):
                        torch.manual_seed(seed);torch.cuda.manual_seed(seed)
                        for z in layers:z.train()
                        z,j=forward(m,raw)
                        dz.append(z);dj.append(j)
                    m.eval()
                drop=metrics.dropout_metrics(results['TC']['logits'][:,0],joint[:,0],np.stack(dz),np.stack(dj),temperature=4,labels=labels)
                drop.update(seeds=list(DROP_SEEDS),modules=dropout_spec,isolated_dropout_only=True,
                    zero_adapter_eval_logit_max_abs_error=equal,zero_adapter_eval_joint_max_abs_error=equal_joint,
                    causal_claim=False)
                arrays['dropout_logits']=np.stack(dz);arrays['dropout_joint']=np.stack(dj)
            state_checks[current]=all(torch.equal(initial[k],v.detach().cpu()) for k,v in m.state_dict().items())
            if not state_checks[current]:raise ValueError('Registered parameter/persistent buffer changed during readout')
            if any(p.grad is not None for p in m.parameters()):raise ValueError('Readout produced gradients')
            print(json.dumps(dict(model=current,NN_packets=nn,registered_state_unchanged=state_checks[current])),flush=True)
            if current!='TC':del m
            torch.cuda.empty_cache()
        if nn!=MAX_NN:raise ValueError('Exact declared source NN packet budget differs')
    result=metrics.analyze(results,labels,list(VIEWS))
    result.update(run_id=RUN,status='SOURCE_ONLY_ZERO_UPDATE_COMPLETE',views=list(VIEWS),model_seed_count=1,
        NN_packets=nn,generated_channel_packets=900,source_raw_reads=180,optimizer_updates=0,EMA_updates=0,
        head_fit_updates=0,new_V_NN=0,target_access=False,registered_state_unchanged=state_checks,
        head_forward_max_abs_error=head_errors,dropout=drop,
        limitations=['180 seen L; not unseen-RX evidence','observed_last is deterministic namespace replay, not saved IQ byte comparison',
            'NoAs channel observations were teacher shadows, not student As training','Ts historical exact IQ exposure not asserted',
            'isolated Dropout mismatch alone does not establish cause of E30 degradation','No new trained checkpoint or target retest'])
    arrays.update(TX=labels,ids=np.asarray([r['physical_id'] for r in selected]),RX=np.asarray([r['RX'] for r in selected]),day=np.asarray([r['day'] for r in selected]))
    np.savez_compressed(out/'predictions.npz',**arrays)
    for name,entry in results.items():
        rowroot=out/'models'/name;rowroot.mkdir(parents=True)
        write(rowroot/'readout.json',dict(model=name,views=list(VIEWS),origin=entry['origin'],NN_packets=1080,
            correct=[int((entry['logits'][:,vi].argmax(-1)==labels).sum()) for vi in range(6)],n_per_view=180,
            registered_state_unchanged=state_checks[name],target_access=False,optimizer_updates=0))
    write(out/'analysis.json',result)
    write(out/'complete.json',dict(status=result['status'],NN_packets=nn,target_access=False,finished_at=time.time()))
    print(json.dumps(dict(status=result['status'],NN_packets=nn)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.runtime,a.output)


if __name__=='__main__':main()
