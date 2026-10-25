"""Source-only gradient dose and post-DURA weak-noise falsification."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import time

from experiments.cric_dura_comparison.io import read,write
from .restored_source import (SEEDS,SCENES,VIEWS,validate_context,validate_e30,source_channel,
    snapshot,unchanged,validate_prefix_headers)

RUN='20261025-diagnostic-PAMF-DURA-dose-noise-m1-r01'
BASELINE_RUN='20261025-diagnostic-PAMF-DURA-paired-source-m3-r01'
BATCHES=12
RECIPE=dict(raw_threshold=.95,extra_teacher='three_frozen_E30_unanimous_raw_mean_probability',
    extra_max_weight=.25,extra_to_primary_median_gradient_ratio=.25,ramp_new_epochs=2,
    postG_noise_SNR_dB=40.,noise_seed=20261025301,
    noise_max_accuracy_drop_pp=.1,noise_max_prediction_change_percent=.5,
    gradient_min_median_cosine=-.25)


def soft_loss(logits,target,accepted):
    import torch
    if target.requires_grad or target.shape!=logits.shape or accepted.shape!=(len(logits),):
        raise ValueError('Aligned detached raw teacher targets required')
    return (-(target*torch.log_softmax(logits,1)).sum(1)*accepted).mean()


def decide_dose(rows):
    import numpy as np
    b=np.asarray([r['primary_norm'] for r in rows]);e=np.asarray([r['restored_extra_norm'] for r in rows])
    cosine=np.asarray([r['primary_restored_cosine'] for r in rows])
    if len(rows)!=BATCHES or not np.isfinite(np.r_[b,e,cosine]).all() or (b<=0).any() or (e<=0).any():
        raise ValueError('Twelve finite nonzero source gradient measurements required')
    weight=min(RECIPE['extra_max_weight'],RECIPE['extra_to_primary_median_gradient_ratio']*float(np.median(b)/np.median(e)))
    return dict(selected_weight=weight,median_primary_norm=float(np.median(b)),median_extra_norm=float(np.median(e)),
        median_cosine=float(np.median(cosine)),median_weighted_extra_to_primary=float(np.median(weight*e/b)),
        source_gradient_pass=bool(np.median(cosine)>=RECIPE['gradient_min_median_cosine']),
        rule='min(.25,.25*median(primary_gradient_norm)/median(restored_extra_gradient_norm));no target feedback')


def add_noise(x,meta,scene):
    import numpy as np
    import torch
    from leo_practical.channel import stable_seed
    keys=('eq_i','rx_i','day_i','sig_i','base_index')
    values={k:meta[k].detach().cpu().tolist() for k in keys}
    power=x.detach().square().sum(1).mean(1).cpu().tolist();noise=[]
    for i,p in enumerate(power):
        physical=tuple(int(values[k][i]) for k in keys)
        rng=np.random.default_rng(stable_seed(RECIPE['noise_seed'],'PAMF/postG/sourceV/fixed',scene,physical))
        n=rng.normal(size=(2,256));n*=np.sqrt(p/10**(RECIPE['postG_noise_SNR_dB']/10)/np.mean((n*n).sum(0)))
        noise.append(n)
    a=np.ascontiguousarray(noise,np.float32)
    return x+torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape).to(x.device)


def load_sources(c,device):
    import torch
    from experiments.cric_dura_comparison.frozen_inference import reader_from_payload,load_G
    models=[];configs=[];payloads=[];contract=None
    for case in validate_context(c):
        p=Path(case['parent_source']);d=read(p/'complete.json');expected=read(case['expected_source_contract'])
        cp=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
        validate_prefix_headers(case,d,cp,read(d['config']['source_contract']),expected);del cp
        folder=p.parent/'segment30';d=read(folder/'complete.json');cfg=d['config']
        cp=torch.load(folder/'final_ssdg.pth',map_location='cpu',weights_only=False)
        actual=read(cfg['source_contract']);validate_e30(case,d,cp,actual,expected)
        if contract is not None and actual!=contract:raise ValueError('Identical source contracts required')
        contract=actual;models.append(reader_from_payload(cfg,cp,device));configs.append(cfg);payloads.append(cp)
    if (contract['counts']!=dict(L_s=6300,U_s=56700,V=27000)
            or any(q['source_cache']!=configs[0]['source_cache'] for q in configs)):
        raise ValueError('Original F0 source roles required')
    return models,configs,payloads,contract,load_G(c['G'],contract,device)


@contextmanager
def no_native_U():
    from experiments.cric_dura_comparison import t1_model
    old=t1_model.plain_pl
    def zero(model,device):
        import torch
        return torch.zeros((),device=device)
    t1_model.plain_pl=zero
    try:yield
    finally:t1_model.plain_pl=old


def gradient_probe(models,cfg,payload,g,engine,contract,root):
    import torch
    import torch.nn.functional as F
    from experiments.cric_dura_comparison import f_train,t1_model
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cric_dura_comparison.f_data import SourceDataset
    c=deepcopy(cfg);c.update(epoch_offset=30,step_offset=6660)
    torch.manual_seed(SEEDS[0]);torch.cuda.manual_seed_all(SEEDS[0])
    with f_train.class_dimension(c),installed(f_train.original_config(SEEDS[0])) as native:
        a=f_train.arguments(c,'cuda:0');a.input_len=256;a.baseline_ckpt=a.teacher_ckpt=''
        a.from_scratch=a.a1_scratch_only=True
        if a.use_unlabeled or a.use_ema_teacher:raise ValueError('Native U/EMA disabled in probe')
        student=t1_model.install(native.build_baseline_model(a,'cuda:0'),c).to('cuda:0')
    student.load_state_dict(payload['model'],strict=True);before=snapshot(student)
    params=[p for p in student.parameters() if p.requires_grad]
    labeled=SourceDataset(c['source_cache'],'L_s',contract);unlabeled=SourceDataset(c['source_cache'],'U_s',contract)
    gen=lambda:torch.Generator().manual_seed(SEEDS[0])
    ll=iter(torch.utils.data.DataLoader(labeled,batch_size=128,shuffle=True,generator=gen(),num_workers=0,drop_last=True))
    ul=iter(torch.utils.data.DataLoader(unlabeled,batch_size=256,shuffle=True,generator=gen(),num_workers=0))
    def grad(loss):
        values=torch.autograd.grad(loss,params,allow_unused=True)
        result=torch.cat([(torch.zeros_like(p) if v is None else v).reshape(-1) for p,v in zip(params,values)]).detach()
        if not bool(torch.isfinite(result).all()):raise ValueError('Nonfinite source gradient')
        return result
    def cosine(a,b):return float(F.cosine_similarity(a[None],b[None],dim=1)[0])
    rows=[]
    with no_native_U():
        for step in range(1,BATCHES+1):
            lb,ub=next(ll),next(ul);lx,ly,ld=[v.to('cuda:0') for v in lb[:3]]
            ux,ud=ub[0].to('cuda:0'),ub[2].to('cuda:0')
            if not bool((ub[1]==-1).all()) or bool(torch.as_tensor(ub[3]['tx_label_visible']).any()):
                raise ValueError('Source U truth must remain hidden')
            with torch.no_grad():
                probs=torch.stack([m(ux).softmax(1) for m in models]);conf,label=probs.max(-1)
                accepted=(conf>=.95).all(0)&(label==label[:1]).all(0);target=probs.mean(0).detach()
                scene=SCENES[(step-1)%3]
                hx=source_channel(ux,ub[3],scene,contract,namespace='PAMF/restored_dose/U/fixed')
                gx=engine(hx)[0].clone()
            student.train()
            with student._phase1_labeled_feature_context():out=student(lx,return_aux=True,domain_labels=ld)
            ce=F.cross_entropy(out['tx_logits'],ly,label_smoothing=.01)
            correction,metrics=student._source_labeled_feature_hook(model=student,labeled_output=out,labels=ly,
                extra=(None,lb[3]),domains=ld,optimizer_step=step,identity_loss=ce,labeled_count=128,labeled_inputs=lx)
            lgrad=grad(ce+correction)
            # This is the unchanged ordinary self-PL main term, separate from
            # the frozen-ensemble targets used only by the extra B/C branches.
            student.eval()
            with torch.no_grad():own_conf,own_label=student(ux).softmax(1).max(1)
            student.train();q=student.forward_identity_only(ux,domain_labels=ud)['tx_logits']
            ugrad=grad((F.cross_entropy(q,own_label,reduction='none')*(own_conf>=.95)).mean())
            primary=lgrad+ugrad
            extras=[];losses=[]
            for view in (ux,gx):
                loss=sum(soft_loss(student.forward_identity_only(view,domain_labels=ud)['tx_logits'],target,accepted) for _ in range(2))/2
                losses.append(float(loss.detach()));extras.append(grad(loss))
            rows.append(dict(step=step,scene=scene,L_seen=128,U_seen=256,extra_accepted=int(accepted.sum()),
                primary_norm=float(primary.norm()),L_norm=float(lgrad.norm()),raw_U_norm=float(ugrad.norm()),
                raw_extra_norm=float(extras[0].norm()),restored_extra_norm=float(extras[1].norm()),
                primary_raw_cosine=cosine(primary,extras[0]),primary_restored_cosine=cosine(primary,extras[1]),
                raw_restored_cosine=cosine(extras[0],extras[1]),extra_raw_loss=losses[0],extra_restored_loss=losses[1],
                actual_L_forward_packets=int(metrics['T1_L_packets']),As_KL_weight=float(metrics['T1_As_KL_weight'])))
            # L/U train-mode buffers are ephemeral in this no-optimizer probe.
            # Restore them before the next independent gradient measurement.
            with torch.no_grad():
                for name,value in student.named_buffers():value.copy_(before[name])
            print('SOURCE_DOSE',step,BATCHES,flush=True)
    if any(p.grad is not None for p in student.parameters()) or any(not torch.equal(v,before[k]) for k,v in student.state_dict().items()):
        raise ValueError('Source gradient probe changed parameters or retained buffer updates')
    result=dict(rows=rows,decision=decide_dose(rows),L_exposures=BATCHES*128,U_exposures=BATCHES*256,
        model_seed=SEEDS[0],backward_calls=BATCHES*4,optimizer_updates=0,persistent_buffer_updates=0,
        U_truth_read=False,target_reads=0,V_gradient_reads=0)
    write(root/'gradient_complete.json',result);return result


def run(rt):
    if rt['run_id']!=RUN or rt['allowed_gpus']!=[0] or os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
        raise ValueError('Fixed source dose/noise run and GPU0 required')
    from experiments.pamf_repl_completion.runner import activate
    activate(rt)
    import numpy as np
    import torch
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.external_dura import fp32_copy
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    torch.set_num_threads(2);start=time.perf_counter();root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    c=read(rt['context']);validate_context(c)
    baseline_root=Path(rt['project'])/'runs'/BASELINE_RUN
    baseline=read(baseline_root/'complete.json')
    old_context=read(Path(rt['project'])/'releases/20261025-diagnostic-PAMF-DURA-paired-source-release-r01/context.json')
    if (old_context!=c or baseline['status']!='SOURCE_PAIRED_DIAGNOSTIC_COMPLETE' or baseline['model_seeds']!=list(SEEDS)
            or baseline['views']!=list(VIEWS) or baseline['source_V_seen']!=27000 or baseline['target_reads']
            or not baseline['parameters_and_buffers_unchanged']):raise ValueError('Reuse only completed identical source pairs')
    write(root/'pipeline_state.json',dict(status='SOURCE_LOADING',pid=os.getpid(),code_commit=rt['code_commit']))
    models,configs,payloads,contract,g=load_sources(c,'cuda:0');engine=InferenceEngine(g,reuse=True,graph=True,native=False)
    labeled=SourceDataset(configs[0]['source_cache'],'L_s',contract);x=labeled.x[:2].to('cuda:0')
    with torch.inference_mode():
        gx=engine(x)[0].clone()
        if not torch.allclose(g(x),gx,atol=1e-5,rtol=1e-5) or any(m(gx).shape!=(2,6) for m in models):
            raise ValueError('Actual frozen F/G source-L smoke failed')
    del labeled,x,gx
    write(root/'smoke.json',dict(status='PASS',role='L_s',physical_packets=2,F_models=3,G_models=1,target_reads=0,optimizer_updates=0))
    before=[snapshot(m) for m in models];gbefore=snapshot(g)
    gradient=gradient_probe(models,configs[0],payloads[0],g,engine,contract,root);del payloads
    validation=SourceDataset(configs[0]['source_cache'],'V',contract)
    with np.load(baseline_root/'source_V_predictions.npz',allow_pickle=False) as data:
        if not np.array_equal(data['ids'],validation.metadata['ids']):raise ValueError('Identical source V physical order required')
        old_logits=data['logits'].copy();truth=data['y'].copy()
    noisy=np.empty((3,3,27000,6),np.float32);offset=0
    loader=torch.utils.data.DataLoader(validation,batch_size=256,shuffle=False,num_workers=0)
    with torch.inference_mode():
        for i,batch in enumerate(loader):
            x=batch[0].to('cuda:0');n=len(x)
            for k,scene in enumerate(SCENES):
                gx=engine(source_channel(x,batch[3],scene,contract))[0].clone()
                nx=add_noise(gx,batch[3],scene)
                for j,model in enumerate(models):
                    if i==0:
                        fresh=fp32_copy(model(gx));old=old_logits[j,VIEWS.index('G_'+scene),:n]
                        if not np.allclose(fresh,old,atol=1e-5,rtol=1e-5):raise ValueError('Fixed baseline source replay differs')
                    q=model(nx)
                    if q.shape!=(n,6) or not bool(torch.isfinite(q).all()):raise ValueError('Finite frozen source noise readout required')
                    noisy[j,k,offset:offset+n]=fp32_copy(q)
            offset+=n
            if i%5==0 or offset==27000:
                write(root/'progress.json',dict(status='SOURCE_NOISE_VALIDATING',source_V_seen=offset,total=27000,
                    seconds=time.perf_counter()-start,optimizer_updates=0,target_reads=0),exclusive=False)
                print('SOURCE_NOISE',offset,27000,flush=True)
    if offset!=27000 or not unchanged(g,gbefore) or any(not unchanged(m,b) for m,b in zip(models,before)):
        raise ValueError('Incomplete V or frozen F/G state changed')
    rows=[]
    for j,seed in enumerate(SEEDS):
        for k,scene in enumerate(SCENES):
            a=old_logits[j,VIEWS.index('G_'+scene)].argmax(-1);b=noisy[j,k].argmax(-1)
            accuracy_before=100.*float((a==truth).mean());accuracy_after=100.*float((b==truth).mean())
            change=100.*float((a!=b).mean());delta=accuracy_after-accuracy_before
            rows.append(dict(model_seed=seed,scene=scene,seen=27000,before_accuracy=accuracy_before,after_accuracy=accuracy_after,
                delta_pp=delta,prediction_change_percent=change,
                source_noise_pass=bool(delta>=-RECIPE['noise_max_accuracy_drop_pp'] and change<=RECIPE['noise_max_prediction_change_percent'])))
    np.savez(root/'source_noise_predictions.npz',logits=noisy,y=truth,ids=validation.metadata['ids'],
        scenes=np.asarray(SCENES),model_seeds=np.asarray(SEEDS))
    result=dict(status='SOURCE_DOSE_NOISE_COMPLETE',code_commit=rt['code_commit'],recipe=RECIPE,
        gradient=gradient,noise=rows,noise_feasibility_pass=all(r['source_noise_pass'] for r in rows),
        source_V_seen=27000,source_V_gradient_reads=0,U_truth_read=False,target_reads=0,
        F_G_parameter_updates=0,optimizer_updates=0,persistent_buffer_updates=0,
        parameters_and_buffers_unchanged=True,baseline_source_reused=True,seconds=time.perf_counter()-start)
    write(root/'complete.json',result)
    write(root/'pipeline_state.json',dict(status=result['status'],pid=os.getpid(),code_commit=rt['code_commit']),exclusive=False)
    print('SOURCE_DOSE_NOISE_COMPLETE',result['seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch']);p.add_argument('--config',required=True)
    a=p.parse_args();run(read(a.config))
