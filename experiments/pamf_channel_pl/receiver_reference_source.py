"""Visible-L reference construction and no-optimizer matched source gradients."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import time

from experiments.cric_dura_comparison.io import read,write
from .restored_source import validate_context,snapshot,unchanged,SCENES,SEEDS,source_channel
from .restored_dose_source import load_sources,no_native_U
from .receiver_reference import build_references,relation_loss,decide_shared_dose,metadata_long,VARIANTS,BATCHES,TEMPERATURE

RUN='20261025-diagnostic-PAMF-RXReference-dose-m1-r01'


def student_from_parent(cfg,payload):
    from experiments.cric_dura_comparison import f_train,t1_model
    from experiments.cvs_phase1_stack.runtime import installed
    c=deepcopy(cfg);c.update(epoch_offset=30,step_offset=6660)
    with f_train.class_dimension(c),installed(f_train.original_config(SEEDS[0])) as native:
        a=f_train.arguments(c,'cuda:0');a.input_len=256;a.baseline_ckpt=a.teacher_ckpt=''
        a.from_scratch=a.a1_scratch_only=True
        if a.use_unlabeled or a.use_ema_teacher:raise ValueError('Native U/EMA must stay disabled')
        student=t1_model.install(native.build_baseline_model(a,'cuda:0'),c).to('cuda:0')
    student.load_state_dict(payload['model'],strict=True)
    return student,c


def gradients(student,refs,labeled,unlabeled,teachers,engine,contract,root):
    import torch
    from torch.nn import functional as F
    student.train();before=snapshot(student)
    named=[(name,p) for name,p in student.named_parameters() if p.requires_grad];params=[p for _,p in named]
    sinc=student.get_submodule('id_backbone.encoder.core.id_backbone.sinc')
    if sinc.scale_hz!=25000. or sinc.mode!='CENTER_25K':raise ValueError('Original physical Sinc coordinates required')
    rng=lambda:torch.Generator().manual_seed(SEEDS[0])
    ll=iter(torch.utils.data.DataLoader(labeled,batch_size=128,shuffle=True,generator=rng(),num_workers=0,drop_last=True))
    ul=iter(torch.utils.data.DataLoader(unlabeled,batch_size=256,shuffle=True,generator=rng(),num_workers=0,drop_last=True))
    source_rxs=tuple(sorted(contract['source_rxs']));held_rx=source_rxs[0];rows=[]
    def grad(loss):
        values=torch.autograd.grad(loss,params,allow_unused=True);pieces=[];head=0.
        for (name,p),v in zip(named,values):
            value=torch.zeros_like(p) if v is None else v
            if name.endswith('cls_head.weight'):head=float(value.detach().norm())
            if name in ('id_backbone.encoder.core.id_backbone.sinc.low_coordinate','id_backbone.encoder.core.id_backbone.sinc.band_coordinate'):
                value=value/25000.
            pieces.append(value.reshape(-1))
        vector=torch.cat(pieces).detach()
        if not bool(torch.isfinite(vector).all()):raise ValueError('Finite source gradient required')
        return vector,head
    def restore_buffers(state):
        with torch.no_grad():
            for name,value in student.named_buffers():value.copy_(state[name])
    with no_native_U():
        for step in range(1,BATCHES+1):
            lb,ub=next(ll),next(ul);lx,ly,ld=[v.to('cuda:0') for v in lb[:3]]
            ux=ub[0].to('cuda:0');ud=ub[2].to('cuda:0');rx=ub[3]['rx_i'].to('cuda:0')
            if (not bool((ub[1]==-1).all()) or bool(torch.as_tensor(ub[3]['tx_label_visible']).any())
                    or set(rx.tolist())-set(source_rxs)):raise ValueError('Only hidden-label source U allowed')
            k=(step-1)%3;scenes=(SCENES[k],SCENES[(k+1)%3])
            with torch.no_grad():
                probabilities=torch.stack([m(ux).softmax(1) for m in teachers]);confidence,label=probabilities.max(-1)
                consensus=(confidence>=.95).all(0)&(label==label[:1]).all(0)
                accepted=consensus&(rx!=held_rx);target=probabilities.mean(0).detach()
                restored=[engine(source_channel(ux,ub[3],scene,contract,namespace='PAMF/restored_dose/U/fixed'))[0].clone() for scene in scenes]
            student.train()
            with student._phase1_labeled_feature_context():output=student(lx,return_aux=True,domain_labels=ld)
            ce=F.cross_entropy(output['tx_logits'],ly,label_smoothing=.01)
            correction,metrics=student._source_labeled_feature_hook(model=student,labeled_output=output,labels=ly,
                extra=(None,lb[3]),domains=ld,optimizer_step=step,identity_loss=ce,labeled_count=128,labeled_inputs=lx)
            lg,lh=grad(ce+correction)
            student.eval()
            with torch.no_grad():own_conf,own_label=student(ux).softmax(1).max(1)
            student.train();q=student.forward_identity_only(ux,domain_labels=ud)['tx_logits']
            ug,uh=grad((F.cross_entropy(q,own_label,reduction='none')*(own_conf>=.95)).mean());primary=lg+ug
            row=dict(step=step,first_scene=scenes[0],second_scene=scenes[1],held_extra_rx=held_rx,L_seen=128,U_seen=256,
                raw_teacher_accepted=int(consensus.sum()),extra_accepted_after_holdout=int(accepted.sum()),
                held_U_seen=int((rx==held_rx).sum()),primary_norm=float(primary.norm()),L_norm=float(lg.norm()),raw_U_norm=float(ug.norm()),
                primary_head_gradient_L=lh,primary_head_gradient_U=uh,
                actual_L_forward_packets=int(metrics['T1_L_packets']),As_KL_weight=float(metrics['T1_As_KL_weight']))
            temporary={name:value.detach().clone() for name,value in student.named_buffers()}
            cpu_rng=torch.random.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
            for variant,views in zip(VARIANTS,((ux,ux),(restored[0],restored[0]),tuple(restored))):
                restore_buffers(temporary);torch.random.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
                loss=sum(relation_loss(student.forward_identity_only(view,domain_labels=ud)['z_id'],refs,rx,source_rxs,target,accepted) for view in views)/2
                extra,head=grad(loss)
                if head!=0.:raise ValueError('Reference auxiliary must use identity features, not an additional head objective')
                row.update({variant+'_loss':float(loss.detach()),variant+'_norm':float(extra.norm()),
                    variant+'_cosine':float(F.cosine_similarity(primary[None],extra[None],dim=1)[0]),variant+'_head_gradient':head})
            if metrics['T1_L_packets']!=768 or metrics['T1_As_KL_weight']!=0 or lh<=0:
                raise ValueError('Original six-forward L/default-head main changed')
            rows.append(row);restore_buffers(before)
            write(root/'progress.json',dict(status='SOURCE_REFERENCE_GRADIENTS',batches=step,total=BATCHES,
                optimizer_updates=0,target_reads=0,V_reads=0),exclusive=False)
            print('SOURCE_RX_REFERENCE_DOSE',step,BATCHES,flush=True)
    current=dict(student.state_dict());current.update(dict(student.named_buffers()))
    if (any(p.grad is not None for p in student.parameters()) or set(current)!=set(before)
            or any(not torch.equal(value,before[name]) for name,value in current.items())):
        raise ValueError('Gradient probe mutated student parameters or retained buffers')
    return dict(rows=rows,decision=decide_shared_dose(rows),student_model_seed=SEEDS[0],held_extra_rx=held_rx,
        L_exposures=BATCHES*128,U_exposures=BATCHES*256,gradient_calls=BATCHES*5,
        temporary_train_buffers_restored=True,student_parameters_and_registered_buffers_unchanged=True,
        native_U_EMA=False,optimizer_updates=0,source_V_reads=0,U_truth_read=False,target_reads=0,
        source_holdout_scope='Only extra branch excludes first sorted sourceRX;original raw/L parent already saw all sourceRX;not an unseen-base-RX claim')


def run(rt):
    if rt['run_id']!=RUN or rt['allowed_gpus']!=[0] or os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
        raise ValueError('Fixed source reference diagnostic/GPU0 required')
    from experiments.pamf_repl_completion.runner import activate
    activate(rt)
    import torch
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    torch.set_num_threads(2);torch.manual_seed(SEEDS[0]);torch.cuda.manual_seed_all(SEEDS[0])
    start=time.perf_counter();root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    c=read(rt['context']);validate_context(c)
    write(root/'pipeline_state.json',dict(status='SOURCE_LOADING',code_commit=rt['code_commit'],pid=os.getpid()))
    models,configs,payloads,contract,g=load_sources(c,'cuda:0')
    student,cfg=student_from_parent(configs[0],payloads[0]);del payloads
    engine=InferenceEngine(g,reuse=True,graph=True,native=False)
    labeled=SourceDataset(configs[0]['source_cache'],'L_s',contract);x=labeled.x[:2].to('cuda:0')
    student.eval()
    with torch.no_grad():
        gx=engine(x)[0].clone()
        feature=student.forward_identity_only(x,domain_labels=torch.zeros(2,device='cuda:0',dtype=torch.long))['z_id']
        logits=[m(gx) for m in models]
        if (not torch.allclose(g(x),gx,atol=1e-5,rtol=1e-5) or feature.shape!=(2,160)
                or not torch.allclose(feature,models[0].features(x),atol=1e-5,rtol=1e-5)
                or any(q.shape!=(2,6) or not bool(torch.isfinite(q).all()) for q in logits)):
            raise ValueError('Actual own E30/default head feature reference or deployed G smoke differs')
    write(root/'smoke.json',dict(status='PASS',source_role='L_s',physical_packets=2,teacher_F=3,student_E30=1,
        feature_shape=[2,160],own_seed_features_match=True,default_head=True,optimizer_updates=0,target_reads=0))
    before=[snapshot(m) for m in models];gbefore=snapshot(g);features=[[] for _ in models]
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(labeled,batch_size=256,shuffle=False,num_workers=0):
            x=batch[0].to('cuda:0')
            for j,model in enumerate(models):features[j].append(model.features(x).detach().cpu())
    references=[];counts=[]
    # Outside inference_mode: fixed references must remain ordinary detached
    # tensors usable by autograd for student feature gradients.
    for parts in features:
        ref,count=build_references(torch.cat(parts),metadata_long(labeled.metadata['y']),
            metadata_long(labeled.metadata['rx_i']),tuple(sorted(contract['source_rxs'])),6,'L_s')
        references.append(ref);counts.append(count)
    del features
    if any(int(count.sum())!=6300 for count in counts):raise ValueError('Complete source L references required')
    torch.save(dict(schema='pamf_visible_L_receiver_reference_v1',references=torch.stack(references),counts=torch.stack(counts),
        source_contract=contract,source_rxs=sorted(contract['source_rxs']),model_seeds=list(SEEDS),teacher_epoch=30,
        source_cases=c['cases'],temperature=TEMPERATURE,reference_role='L_s',U_reads=0,V_reads=0,target_reads=0),root/'references.pt')
    unlabeled=SourceDataset(configs[0]['source_cache'],'U_s',contract)
    result=gradients(student,references[0].to('cuda:0'),labeled,unlabeled,models,engine,contract,root)
    if not unchanged(g,gbefore) or any(not unchanged(m,b) for m,b in zip(models,before)):
        raise ValueError('Frozen F/G source state changed')
    result.update(status='SOURCE_RX_REFERENCE_DOSE_COMPLETE',code_commit=rt['code_commit'],reference_role='L_s',
        source_L_references_seen=6300,reference_counts=[count.tolist() for count in counts],reference_teacher_seeds=list(SEEDS),
        reference_teacher_epoch=30,reference_F_packet_forwards=3*6300,raw_teacher_F_packet_forwards=3*BATCHES*256,
        student_gradient_F_packet_forwards=BATCHES*(768+2*256+3*2*256),G_packet_forwards=2*BATCHES*256,
        source_L_smoke_packets=2,reference_updates=0,head_fit_updates=0,F_G_parameter_updates=0,
        frozen_F_G_parameters_and_registered_buffers_unchanged=True,temperature=TEMPERATURE,seconds=time.perf_counter()-start,
        scope='Source dose only;not new F training,not target benefit,not U truth accuracy or crossreceiver transfer proof')
    write(root/'gradient_complete.json',result);write(root/'complete.json',result)
    write(root/'pipeline_state.json',dict(status=result['status'],code_commit=rt['code_commit'],pid=os.getpid()),exclusive=False)
    print('SOURCE_RX_REFERENCE_COMPLETE',result['seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch']);p.add_argument('--config',required=True)
    a=p.parse_args();run(read(a.config))
