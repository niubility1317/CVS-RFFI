"""One two-arm actual F probe: native U256, extra U128; no real IQ/CP/updates."""
from copy import deepcopy
import json,random,tempfile,time
from pathlib import Path
import numpy as np
import torch
from . import design as d,model
from experiments.cvs_crossview_supcon41 import model as base
from experiments.cvs_phase1_overlay.model import native_modules

def metadata(n,offset):
    domains=torch.arange(n)%15;rx=torch.tensor(base.supcon.RX_BY_DOMAIN)[domains]
    return domains,dict(eq_i=torch.ones(n,dtype=torch.long),rx_i=rx,day_i=domains%3+1,
        sig_i=torch.arange(n)+offset,base_index=torch.arange(n)+offset)

def arm(row):
    started=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(6007)
    c=d.config(row);m=model.build(c,'cpu').train();before={k:v.clone() for k,v in m.state_dict().items()}
    assert sum(p.numel() for p in m.parameters())==900722 and len(before)==206
    own_ema=deepcopy(m).eval().requires_grad_(False)
    # Synthetic payloads exercise strict deployment keys and the actual fixed
    # teacher constructor. They are not loaded from any checkpoint or dataset.
    payloads={}
    for name,state in [('B01',own_ema.state_dict())]:
        payloads[name]=dict(epoch=20,a1_ema_successful_updates=4440,run_id=d.old.RUN,
            candidate_id=d.reference_config(name)['row_id'],checkpoint_selection='final_only',
            args=dict(baseline_ckpt='',teacher_ckpt=''),ema_model=state)
    rng_before=torch.get_rng_state().clone();bank=model.prepare_frozen_teachers(m,c,payloads=payloads)
    assert torch.equal(rng_before,torch.get_rng_state()) and len(m.state_dict())==206 and sum(p.numel() for p in m.parameters())==900722
    try:model.teacher_bank(own_ema)
    except ValueError:pass
    else:raise AssertionError('Native own EMA inherited external frozen teachers')
    args=d.old.make_args(d.old.config(next(r for r in d.old.rows() if r['arm']=='B01'),20),'cpu')
    native=native_modules();cfg=native.build_aug_base_cfg(args);aug=native.make_augmentor(cfg);native.configure_augmentor_for_epoch(aug,cfg,13,args)
    x=torch.randn((128,2,256),generator=torch.Generator().manual_seed(6008));y=torch.arange(128)%6;domains,meta=metadata(128,2000)
    random.seed(6009);np.random.seed(6009);torch.manual_seed(6009);ordinary=aug(x,labels=y,no_pa=False);ordinary_saved=ordinary.clone();raw_saved=x.clone();m._feature_l_steps=1
    schedule=base.supcon.schedule;base.supcon.schedule=lambda epoch:schedule(int(epoch)+20)
    try:
        first,maps=base.capture_forward(m,ordinary,domains,True);m._supcon_pending=dict(first=maps,second={});m._legacy_actual_L_input=ordinary
        ce=torch.nn.functional.cross_entropy(first,y,label_smoothing=.01)
        delta,_=m._source_labeled_feature_hook(model=m,labeled_output=dict(tx_logits=first),labels=y,extra=(domains,meta),domains=domains,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
    finally:base.supcon.schedule=schedule
    r=m._full_records[-1]
    assert m.id_backbone.encoder._full_train_packets==768 and r['L_forwards']==6 and r['As_packets']==64 and m._shape_NN==256
    u=torch.randn((256,2,256),generator=torch.Generator().manual_seed(6012));du,mu=metadata(256,5000)
    with torch.no_grad():teacher_out=own_ema.forward_identity_only(u,domain_labels=du)
    weak=u+torch.randn_like(u)*.015;student_out=m.forward_identity_only(weak,domain_labels=du)
    pseudo=teacher_out['tx_logits'].argmax(1);base_mask=torch.arange(256).remainder(3)==0;strong_mask=torch.arange(256).remainder(5)!=0;expected_mask=base_mask&strong_mask
    mask,_=m._source_unlabeled_selection_hook(model=m,teacher=own_ema,x_u=u,
        teacher_logits=teacher_out['tx_logits'],student_logits=student_out['tx_logits'],pseudo=pseudo,
        base_mask=base_mask,strong_mask=strong_mask,extra=(du,mu),domains=du,optimizer_step=1)
    assert torch.equal(mask,expected_mask) and len(mask)==256
    pseudo_loss=torch.nn.functional.cross_entropy(student_out['tx_logits'][mask],pseudo[mask])
    probabilities=student_out['tx_logits'].softmax(1);entropy=-(probabilities*student_out['tx_logits'].log_softmax(1)).sum(1).mean()
    original_loss=model.teaching_loss;captured={}
    def teaching(student,teachers,hidden,metadata,which,epoch,policy):
        rng=torch.get_rng_state().clone();value,record=original_loss(student,teachers,hidden,metadata,which,epoch,policy)
        captured.update(student=student,teachers=teachers,loss=value,rng_preserved=torch.equal(rng,torch.get_rng_state()))
        return value,record
    model.teaching_loss=teaching
    before_extra=torch.get_rng_state().clone()
    try:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            m._clean_teacher_config=dict(c,output_root=tmp)
            extra,_=m._source_unlabeled_feature_hook(model=m,teacher=own_ema,
                student_output=student_out,teacher_output=teacher_out,extra=(du,mu),domains=du,
                optimizer_step=1,identity_loss=ce,pseudo_loss=pseudo_loss,entropy_loss=entropy)
            assert (Path(tmp)/'u_quality_physics.jsonl.gz').is_file()
        extra_gradient=torch.autograd.grad(extra,captured['student'],retain_graph=True)[0]
        total=ce+delta+pseudo_loss+.001*entropy+extra;total.backward()
    finally:model.teaching_loss=original_loss
    p=m._clean_teacher_records[-1];head=m.id_backbone.encoder.core.id_backbone.cls_head
    assert captured['rng_preserved'] and torch.equal(before_extra,torch.get_rng_state())
    assert m._clean_teacher_pending is None and m._clean_teacher_packets==128 and p['packets']==128 and p['native_packets']==256
    assert p['original_selected']==int(expected_mask.sum()) and p['extra_original_selected']==int(expected_mask[:128].sum())
    assert p['selected_physical_ids']==[[int(mu[k][i]) for k in ('eq_i','rx_i','day_i','sig_i','base_index')] for i in range(128)]
    assert torch.equal(x,raw_saved) and torch.equal(ordinary,ordinary_saved) and all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
    assert m._supcon_pending is None and not m._supcon_handles and r['ordinary_redraws']==0
    assert head.scale==30. and list(head.weight.shape)==[6,160] and head.weight.grad is not None and float(head.weight.grad.abs().sum())>0
    assert torch.isfinite(total) and all(torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None)
    assert sum(float(v.grad.abs().sum()) for v in m.id_backbone.encoder.core.id_backbone.sinc.parameters() if v.grad is not None)>0
    bank.assert_frozen();b=bank.summary();assert b['forward_NN']=={'B01':128} and not b['registered_in_student']
    assert all(v.grad is None for t in bank.teachers.values() for v in t.parameters()) and all(v.grad is None for v in own_ema.parameters())
    assert m.id_backbone.encoder._full_train_packets==1152 and own_ema.id_backbone.encoder._full_eval_packets==256
    actual_NN=m.id_backbone.encoder._full_train_packets+own_ema.id_backbone.encoder._full_eval_packets+b['total_forward_NN'];assert actual_NN==1536
    if row['arm']=='ZERO_KL':assert float(extra)==0. and float(extra_gradient.abs().sum())==0.
    elif p['positive_q']>0:assert float(extra_gradient.abs().sum())>0.
    sequence={k:r[k] for k in ('step','epoch','total_L','common_RD','common_CE','common_KL','extra_RD','extra_CE','extra_KL','SupCon','SupCon_weight','ordinary_raw_MSE','second_raw_MSE','scene_counts','state_counts')}
    comparable=dict(ordinary=ordinary.detach(),first=first.detach(),native_student=student_out['tx_logits'].detach(),native_teacher=teacher_out['tx_logits'].detach(),extra_student=captured['student'].detach(),rng=torch.get_rng_state(),sequence=sequence,ids=p['selected_physical_ids'])
    return dict(status='PASS',arm=row['arm'],NN=actual_NN,L_student_NN=768,native_U_student_NN=256,native_U_own_EMA_NN=256,
        extra_student_U_NN=128,extra_frozen_teacher_U_NN=128,frozen_teacher_NN=b['forward_NN'],real_IQ=0,CP=0,
        optimizer_updates=0,head_fit_updates=0,target_access=False,wall_seconds=time.perf_counter()-started,
        ordinary_changed_packets=int((ordinary!=x).flatten(1).any(1).sum()),ordinary_inputs_unchanged=True,
        extra_rng_preserved=True,native_full_mask_unchanged=True,extra_subset128_from_native256=True,
        state_keys=206,student_parameters=900722,head_scale=30.,head_shape=[6,160],
        default_head_gradient_norm=float(head.weight.grad.norm()),extra_logit_gradient_norm=float(extra_gradient.norm()),
        all_joint_gradients_finite=True,frozen_teacher_parameters_and_buffers_unchanged=True,
        frozen_teacher_gradients_none=True,external_bank_not_copied_to_native_EMA=True,
        teaching={k:v for k,v in p.items() if k!='selected_physical_ids'},
        selected_physical_IDs_count=len(p['selected_physical_ids']),
        selected_physical_IDs_first=p['selected_physical_ids'][0],
        selected_physical_IDs_last=p['selected_physical_ids'][-1],bank=b),comparable

def run():
    start=time.perf_counter();rows=[];reference=None
    for row in d.rows():
        result,observed=arm(row);rows.append(result)
        if reference is None:reference=observed
        else:
            assert observed['sequence']==reference['sequence'] and observed['ids']==reference['ids']
            assert all(torch.equal(observed[k],reference[k]) for k in ('ordinary','first','native_student','native_teacher','extra_student','rng'))
    actual=sum(r['NN'] for r in rows);assert actual==d.LOCAL_NN==3072
    return dict(status='PASS',NN=actual,rows=rows,real_IQ=0,CP=0,optimizer_updates=0,
        wall_seconds=time.perf_counter()-start,two_arm_original_L_and_native_U_and_extra_view_fingerprints_identical=True,
        deployment_unchanged=True,all_teacher_parameters_frozen=True)
if __name__=='__main__':print(json.dumps(run()))
