import copy,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from . import design as d
from .signals import alignment,packet_alignment,margin
from experiments.cvs_evidence_relation import components,model
from experiments.cvs_evidence_relation.objective import objective,anchor_groups,distribution,RXS,ALPHA

def models(c):
 d.validate(c)
 from experiments.cvs_evidence_relation.provenance import source_proof
 pc=c['parent_config'];p=source_proof(pc);ck=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if ck['epoch']!=50 or ck['candidate_id']!=pc['row_id'] or ck['run_id']!=pc['run_id'] or ck['checkpoint_selection']!='final_only' or ck['a1_ema_successful_updates']!=11100:raise ValueError('Frozen source checkpoint identity differs')
 expected=d.parent.make_args(pc,'cpu');actual=ck['args'];actual=actual.get('args',actual)
 keys=('run_id','candidate_id','output_dir','epochs','label_epochs','pseudo_epochs','use_unlabeled','use_ema_teacher','ema_decay','pseudo_temporal_gate','pseudo_temporal_mode','pseudo_temporal_window','pseudo_temporal_min_conf','pseudo_strong_agreement','pseudo_domain_gate','pseudo_threshold_mode','pseudo_quantile','label_smoothing','lambda_u','lambda_ent','phase1_lr_schedule','phase1_lr_min','use_concat_sat_channel_aug','use_sat_consistency','lambda_sat_cls','lambda_sat_cons','sat_training_mode','from_scratch','baseline_ckpt','teacher_ckpt','lr','weight_decay')
 if any(actual.get(k)!=getattr(expected,k) for k in keys):raise ValueError('Actual parent checkpoint recipe differs')
 student=model.build(pc,'cpu');student.load_state_dict(ck['model'],strict=True)
 teacher=copy.deepcopy(student);teacher.load_state_dict(ck['ema_model'],strict=True);teacher.eval().requires_grad_(False);student.eval()
 return student,teacher

def sample(L,c):
 if len(L['x'])!=6300 or not set(L['rx'].tolist())<=set(RXS):raise ValueError('Exact legal L source pool required')
 gen=torch.Generator().manual_seed(c['anchor_probe_seed']);anchors=[];probes=[]
 for rx in RXS:
  for tx in range(6):
   for day in (1,2,3):
    cell=torch.nonzero((L['rx']==rx)&(L['y']==tx)&(L['day']==day)).flatten()
    if len(cell)!=70:raise ValueError('Source L TX/RX/day cell differs')
    chosen=cell[torch.randperm(70,generator=gen)[:10]];anchors.extend(chosen[:2].tolist());probes.extend(chosen[2:].tolist())
 if len(set(anchors+probes))!=900:raise ValueError('Source anchor/probe overlap')
 gen=torch.Generator().manual_seed(c['probe_order_seed']);probes=torch.tensor(probes)[torch.randperm(720,generator=gen)]
 return torch.tensor(anchors),probes

def teacher_semantics(teacher,parts,bank,rx,y):
 head=model.classifier(teacher);joint=parts['joint'];full_margin=margin(head.classify(joint),y);result={}
 for branch in ('PA','ref'):
  weak=joint-ALPHA*parts[branch];mass=joint.new_zeros(len(rx));fullmass=mass.clone()
  for r in RXS:
   ix=rx==r
   for _,a in anchor_groups(bank,r,True):
    p=distribution(joint[ix],a);q=distribution(weak[ix],a);mask=torch.arange(6).repeat_interleave(5)[None,:]==y[ix,None]
    mass[ix]+=((p-q)*mask).sum(-1)/(ALPHA*4);fullmass[ix]+=(p*mask).sum(-1)/4
  result[branch]=dict(same_TX_signed_response=mass,default_true_margin_increment=(full_margin-margin(head.classify(weak),y))/ALPHA,same_TX_full_relation_mass=fullmass)
 return result

def run(c):
 d.validate(c);out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();torch.set_num_threads(2)
 from experiments.cvs_nocfo_foundation.source import load_source
 from experiments.cvs_nocfo_foundation import design as nd
 from experiments.cvs_phase1_overlay.model import native_modules
 from experiments.cvs_equivariant_identity.precision import numerical_context
 L,U,V,contract=load_source(nd.config(nd.rows()[0]));assert U is None;del V
 anchors,probes=sample(L,c);student,teacher=models(c);saved={k:v.clone() for k,v in student.state_dict().items()};saved_t={k:v.clone() for k,v in teacher.state_dict().items()}
 if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in student.id_backbone.modules()):raise ValueError('Frozen observer may not update identity BN')
 head=model.classifier(student);excluded={id(head.weight),id(head.gain),id(student.id_backbone.encoder.response_gain)}
 named=[(n,p) for n,p in student.named_parameters() if n.startswith('id_backbone.') and p.requires_grad and id(p) not in excluded];params=[p for _,p in named]
 if not params:raise ValueError('No actual identity feature parameters')
 batches=[];packets={};forward=0
 with numerical_context(d.FULL_FP32_POLICY):
  with torch.no_grad():
   bank=teacher.forward_identity_only(L['x'][anchors])['id_feat_joint'].reshape(5,6,6,160);forward+=180
   ot=teacher.forward_identity_only(L['x'][probes]);tp=components.parts(teacher,ot);forward+=720
   semantics=teacher_semantics(teacher,tp,bank,L['rx'][probes],L['y'][probes])
  for view in c['views']:
   student.train(view=='STRONG_NATIVE');torch.manual_seed(c['strong_rng_seed']);records=[]
   for begin in range(0,720,c['batch_size']):
    end=begin+c['batch_size'];ix=probes[begin:end];x=L['x'][ix];rx=L['rx'][ix];y=L['y'][ix]
    if view=='STRONG_NATIVE':x=native_modules()._strong_augment(x,.015)
    os=student.forward_identity_only(x);sp=components.parts(student,os);forward+=len(ix);teacher_parts={k:v[begin:end] for k,v in tp.items()}
    ce=F.cross_entropy(os['tx_logits'],y,label_smoothing=c['label_smoothing']);gc=torch.autograd.grad(ce,params,retain_graph=True,allow_unused=True)
    _,allraw,_=objective(sp,teacher_parts,bank,rx,'ALL_REL');total,crossraw,facts=objective(sp,teacher_parts,bank,rx,'CROSS_RESP')
    losses=dict(ALL_R=allraw['relation'],CROSS_R=crossraw['relation'],CROSS_E=crossraw['response'],CROSS_R_E=total)
    gradients={k:torch.autograd.grad(v,params,retain_graph=True,allow_unused=True) for k,v in losses.items()}
    controls=torch.autograd.grad(total,[head.weight,head.gain,student.id_backbone.encoder.response_gain],retain_graph=True,allow_unused=True)
    if any(g is not None for g in controls):raise ValueError('Auxiliary uses class W or fusion gains')
    from experiments.cvs_evidence_relation.observe import gradient_summary
    batches.append(dict(view=view,begin=begin,count=len(ix),CE=float(ce.detach()),losses={k:float(v.detach()) for k,v in losses.items()},parameter_alignment={k:alignment(gc,g) for k,g in gradients.items()},feature_group_norms=dict(CE=gradient_summary(named,gc),**{k:gradient_summary(named,g) for k,g in gradients.items()}),class_W_aux_gradient=None,PA_gain_aux_gradient=None,ref_gain_aux_gradient=None))
    # Independent per-packet partial derivatives at deployed joint160. Holding
    # the branch increments fixed is explicit; actual parameter gradients above
    # include every branch path and are the primary direction measurement.
    leaves={k:v.detach().clone().requires_grad_(True) for k,v in sp.items()};logits=head.classify(leaves['joint'])
    if not torch.allclose(logits.detach(),os['tx_logits'].detach(),atol=1e-6,rtol=1e-6):raise ValueError('Default-head exact joint classification differs')
    cp=F.cross_entropy(logits,y,reduction='none',label_smoothing=c['label_smoothing']);gcp=torch.autograd.grad(cp.sum(),leaves['joint'],retain_graph=True)[0]
    _,ar,af=objective(leaves,teacher_parts,bank,rx,'ALL_REL');ct,cr,cf=objective(leaves,teacher_parts,bank,rx,'CROSS_RESP')
    derivatives={k:torch.autograd.grad(v*len(ix),leaves['joint'],retain_graph=True)[0] for k,v in dict(ALL_R=ar['relation'],CROSS_R=cr['relation'],CROSS_E=cr['response'],CROSS_R_E=ct).items()}
    record=dict(CE=cp.detach(),student_prediction=logits.detach().argmax(-1),partial_joint_alignment={k:packet_alignment(gcp,g) for k,g in derivatives.items()})
    records.append(record)
   packets[view]=dict(CE=torch.cat([r['CE'] for r in records]),student_prediction=torch.cat([r['student_prediction'] for r in records]),partial_joint_alignment={k:{s:torch.cat([r['partial_joint_alignment'][k][s] for r in records]) for s in ('dot','cosine','nonzero')} for k in losses})
 student.eval()
 if forward!=c['source_forward_packets'] or any(not torch.equal(v,student.state_dict()[k]) for k,v in saved.items()) or any(not torch.equal(v,teacher.state_dict()[k]) for k,v in saved_t.items()) or any(p.grad is not None for m in (student,teacher) for p in m.parameters()):raise ValueError('Frozen source observer changed state or forward budget')
 artifact=dict(ids=[L['ids'][i] for i in probes.tolist()],anchor_ids=[L['ids'][i] for i in anchors.tolist()],truth=L['y'][probes],rx=L['rx'][probes],day=L['day'][probes],teacher_prediction=ot['tx_logits'].argmax(-1),teacher_semantics=semantics,views=packets,batches=batches)
 torch.save(artifact,out/'source_signals.pt');reloaded=torch.load(out/'source_signals.pt',map_location='cpu',weights_only=False)
 if reloaded['ids']!=artifact['ids'] or reloaded['anchor_ids']!=artifact['anchor_ids'] or len(set(reloaded['ids']+reloaded['anchor_ids']))!=900 or reloaded['batches']!=batches:raise ValueError('Saved source signals differ')
 for view in c['views']:
  for k in losses:
   for s in ('dot','cosine','nonzero'):
    if not torch.equal(reloaded['views'][view]['partial_joint_alignment'][k][s],packets[view]['partial_joint_alignment'][k][s]):raise ValueError('Saved source gradient recount differs')
 y=reloaded['truth'];summaries={}
 for view,v in reloaded['views'].items():
  summaries[view]=dict(count=len(y),student_accuracy=float((v['student_prediction']==y).float().mean()),CE=float(v['CE'].mean()),parameter_alignment={k:[b['parameter_alignment'][k] for b in batches if b['view']==view] for k in losses},partial_joint_alignment={k:dict(nonzero_count=int(a['nonzero'].sum()),positive_dot_fraction=float((a['dot']>0).float().mean()),negative_dot_fraction=float((a['dot']<0).float().mean()),mean_cosine=float(a['cosine'][a['nonzero']].mean()) if a['nonzero'].any() else None) for k,a in v['partial_joint_alignment'].items()})
 semantic_summary={k:{s:dict(mean=float(v.mean()),positive_fraction=float((v>0).float().mean()),negative_fraction=float((v<0).float().mean())) for s,v in z.items()} for k,z in reloaded['teacher_semantics'].items()}
 result=dict(status='SOURCE_SIGNAL_DIAGNOSTIC_COMPLETE',config=c,teacher_accuracy=float((reloaded['teacher_prediction']==y).float().mean()),teacher_semantics=semantic_summary,views=summaries,batches=batches,source_count=720,source_forward_packets=forward,source_decision_recount='VERIFIED',model_parameters_buffers_unchanged=True,head_fit_updates=0,optimizer_updates=0,EMA_updates=0,U_IQ_read=False,U_truth_read=False,target_access=False,threshold_fit=False,seconds=time.perf_counter()-started,claim_scope=c['claim_scope'])
 d.write(out/'resolved_config.json',dict(config=c,execution_commit=(d.ROOT/'release_commit.txt').read_text().strip(),checkpoint_contract='EXACT_MATCH'));d.write(out/'analysis.json',result);d.write(out/'completion.json',dict(status=result['status'],row_id=c['row_id'],source_count=720,source_forward_packets=forward,source_decision_recount='VERIFIED'));print(result['status']+' '+c['row_id'],flush=True);return result

def dispatch():
 spec=d.read(d.ROOT/'registration/experiment.json')
 if spec['run_id']!=d.RUN or [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Source signal registration differs')
 d.BASE.mkdir(parents=True,exist_ok=False);results=[]
 for row in d.rows():
  try:results.append(run(d.config(row)))
  except Exception as error:d.write(d.BASE/'failure.json',dict(status='FAILED',row_id=row['row_id'],error=repr(error),no_automatic_retry=True));raise
  d.write(d.BASE/'progress.json',dict(completed=[r['config']['row_id'] for r in results]))
 result=dict(status='ANALYZED',run_id=d.RUN,models=4,source_physical_packets=720,source_count=2880,source_forward_packets=9360,head_fit_updates=0,optimizer_updates=0,EMA_updates=0,U_IQ_read=False,U_truth_read=False,target_access=False,threshold_fit=False,rows=results)
 d.write(d.BASE/'analysis.json',result);d.write(d.BASE/'completion.json',{k:v for k,v in result.items() if k!='rows'});print(json.dumps(dict(status='ANALYZED',models=4)),flush=True)

if __name__=='__main__':dispatch()
