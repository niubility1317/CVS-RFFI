"""Complete V fixed-rule proxy; labels enter only downstream counts."""
import json,time
from pathlib import Path
import torch
from . import design as d
from .provenance import models
from .policy import weak_view,select,loss,entropy
from .source_gates import physical_metadata
from .order import order

def records(pseudo,masks,signals,truth,rx,day):
 out=[]
 for dimension,levels,values in [('overall',[None],None),('RX',[1,3,4,6,8],rx),('day',[1,2,3],day),('TX',range(6),truth)]:
  for value in levels:
   keep=torch.ones(len(truth),dtype=torch.bool) if value is None else values==value
   for name,mask in masks.items():
    selected=mask&keep;n=int(selected.sum());correct=int((selected&(pseudo==truth)).sum())
    out.append(dict(dimension=dimension,stratum=value,pool_count=int(keep.sum()),gate=name,selected=n,correct=correct,precision=correct/n if n else None,coverage=n/int(keep.sum()),CE_sum=float(signals['CE'][selected].sum()),logit_gradient_L2_mean=float(signals['logit_gradient_norm'][selected].mean()) if n else None,entropy_gradient_L2_mean=float(signals['entropy_gradient_norm'][selected].mean()) if n else None))
 return out

def run(c):
 d.validate(c);out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);started=time.perf_counter()
 from experiments.cvs_nocfo_foundation.source import load_source
 from experiments.cvs_nocfo_foundation import design as nd
 from experiments.cvs_phase1_overlay.model import native_modules
 from experiments.cvs_equivariant_identity.precision import numerical_context
 from experiments.cvs_source_label_gate.audit import gate
 L,U,V,contract=load_source(nd.config(nd.rows()[0]));assert U is None;del L
 meta=physical_metadata(V,'V');indices=order(meta,c['order_seed'],'PAIRS');student,teacher,source=models(c['model_seed']);args=d.old.make_args(d.baseline_config(c['model_seed']),'cpu');native=native_modules()
 if args.strong_noise_std!=.015 or args.lambda_u!=.16 or args.lambda_ent!=.01 or args.pseudo_temporal_mode!='batch_neighbor' or not args.pseudo_strong_agreement:raise ValueError('Actual native gate/std/weights differ')
 saved={k:v.clone() for k,v in student.state_dict().items()};saved_t={k:v.clone() for k,v in teacher.state_dict().items()}
 if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in student.id_backbone.modules()):raise ValueError('Source V may not update identity BN')
 student.train();teacher.eval();torch.manual_seed(c['strong_rng_seed']);pred=[];decisions={};signals={};batch_facts=[]
 with numerical_context(d.FULL_FP32_POLICY):
  for begin in range(0,27000,256):
   ix=indices[begin:begin+256];x=V['x'][ix];metadata={k:v[ix] for k,v in meta.items()};domain=torch.searchsorted(torch.tensor([1,3,4,6,8]),V['rx'][ix])*3+V['day'][ix]-1
   with torch.no_grad():
    weak=teacher.forward_identity_only(x)['tx_logits'];weak2=teacher.forward_identity_only(weak_view(x,c['model_seed']+c['weak_view_seed_offset']+begin))['tx_logits'];strong=student.forward_identity_only(native._strong_augment(x,args.strong_noise_std))['tx_logits']
    original_pseudo,original_masks=gate(native,args,weak,strong,(None,metadata),domain)
    pseudo,masked,facts,values=select(weak,weak2,strong,original_masks['base'],True)
    if not torch.equal(original_pseudo,pseudo) or not torch.equal(original_masks['accepted'],masked['original']):raise ValueError('Original hard gate changed in source proxy')
    masked.update(original_confidence=original_masks['confidence'],physical_neighbor=original_masks['temporal'],geometric_neighbor=original_masks['structural'])
    if not masked['geometric_neighbor'].all():raise ValueError('PAIR source proxy lost original physical neighbor')
    # B/D exactly retain native unweighted accepted-mean CE; cap selects cases.
    native_mean=torch.nn.functional.cross_entropy(strong[masked['selected']],pseudo[masked['selected']]) if bool(masked['selected'].any()) else strong.sum()*0
    if not torch.equal(loss(strong,pseudo,masked['selected']),native_mean):raise ValueError('Accepted CE normalization changed')
   replica=strong.detach().clone().requires_grad_(True);ent=entropy(replica);ent_grad=torch.autograd.grad(ent.sum(),replica)[0]
   values['entropy_gradient_norm']=ent_grad.detach().norm(dim=1)
   pred.append(pseudo);batch_facts.append(dict(begin=begin,count=len(ix),**facts))
   for name,value in masked.items():decisions.setdefault(name,[]).append(value.cpu())
   for name,value in values.items():signals.setdefault(name,[]).append(value.cpu())
 student.eval()
 if any(not torch.equal(v,student.state_dict()[k]) for k,v in saved.items()) or any(not torch.equal(v,teacher.state_dict()[k]) for k,v in saved_t.items()) or any(p.grad is not None for m in (student,teacher) for p in m.parameters()):raise ValueError('Fixed source observer mutated/fit model')
 pred=torch.cat(pred);decisions={k:torch.cat(v) for k,v in decisions.items()};signals={k:torch.cat(v) for k,v in signals.items()};truth=V['y'][indices];rx=V['rx'][indices];day=V['day'][indices]
 rr=records(pred,decisions,signals,truth,rx,day)
 # Keep this local artifact in the native tensor format. The remote Torch
 # NumPy bridge and Python NumPy expose different ndarray classes; mixing
 # string IDs and Tensor.numpy() arrays fails __array_function__ dispatch.
 torch.save(dict(ids=[V['ids'][i] for i in indices.tolist()],pseudo=pred,truth=truth,rx=rx,day=day,**decisions,**signals),out/'source_decisions.pt')
 saved_decisions=torch.load(out/'source_decisions.pt',map_location='cpu',weights_only=False)
 pp,yy,rxx,dd=(saved_decisions[k] for k in ('pseudo','truth','rx','day'))
 mm={k:saved_decisions[k] for k in decisions};ss={k:saved_decisions[k] for k in signals}
 if records(pp,mm,ss,yy,rxx,dd)!=rr or len(set(saved_decisions['ids']))!=27000:raise ValueError('Saved V masks/precision recount differs')
 original=decisions['original'];corrected=decisions['correction_admitted'];candidate=decisions['correction_candidate']
 rejected_combinations={str(code):0 for code in range(64)}
 for a,b,e,f,g,h in zip(decisions['original_confidence'].tolist(),decisions['physical_neighbor'].tolist(),decisions['student_agreement'].tolist(),decisions['teacher_high_conf'].tolist(),decisions['weak_high_conf'].tolist(),decisions['weak_stable'].tolist()):rejected_combinations[str(int(not a)+2*int(not b)+4*int(not e)+8*int(not f)+16*int(not g)+32*int(not h))]+=1
 result=dict(status='SOURCE_TRUST_PROXY_COMPLETE',config=c,records=rr,batch_facts=batch_facts,co_rejection_counts=rejected_combinations,source_count=27000,source_forward_packets=81000,teacher_view_count=2,views_correlated=True,model_parameters_buffers_unchanged=True,head_fit_updates=0,optimizer_updates=0,EMA_updates=0,U_IQ_read=False,U_truth_read=False,target_access=False,threshold_fit=False,source_decision_recount='VERIFIED',checkpoint=str(source/'final_ssdg.pth'),candidate_count=int(candidate.sum()),admitted_count=int(corrected.sum()),admitted_correct=int((corrected&(pred==truth)).sum()),original_count=int(original.sum()),original_correct=int((original&(pred==truth)).sum()),seconds=time.perf_counter()-started,limitation='Fixedfinal50 source V proxy; not actual U correctness or proof of earlier teacher reliability. Analytic logits gradients do not replace actual training backbone gradients.')
 d.write(out/'resolved_config.json',dict(config=c,execution_commit=(d.ROOT/'release_commit.txt').read_text().strip(),source_roles='EXACT_MATCH',threshold_fit=False));d.write(out/'analysis.json',result);d.write(out/'completion.json',dict(status='SOURCE_TRUST_PROXY_COMPLETE',row_id=c['row_id'],source_count=27000,source_decision_recount='VERIFIED'));print('SOURCE_TRUST_PROXY_COMPLETE '+c['row_id'],flush=True);return result

def dispatch():
 spec=d.read(d.ROOT/'registration/experiment.json')
 if spec['run_id']!=d.RUN or [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Source proxy registration differs')
 d.BASE.mkdir(parents=True,exist_ok=False);results=[]
 for row in d.rows():
  try:results.append(run(d.config(row)))
  except Exception as error:
   d.write(d.BASE/'failure.json',dict(status='FAILED',row_id=row['row_id'],error=repr(error),no_automatic_retry=True));raise
  d.write(d.BASE/'progress.json',dict(completed=[r['config']['row_id'] for r in results]))
 criteria=d.config(d.rows()[0])['source_criterion'];n=sum(r['admitted_count'] for r in results);correct=sum(r['admitted_correct'] for r in results)
 passed=n>=criteria['pooled_admitted_count_min'] and correct/n>=criteria['pooled_admitted_precision_min'] and all(r['admitted_count']>=criteria['each_seed_admitted_count_min'] and r['admitted_correct']/r['admitted_count']>=criteria['each_seed_admitted_precision_min'] for r in results)
 result=dict(status='ANALYZED',run_id=d.RUN,models=3,source_count=81000,source_forward_packets=243000,admitted_count=n,admitted_correct=correct,admitted_precision=correct/n if n else None,source_criterion_pass=passed,source_only=True,U_truth_read=False,target_access=False,head_fit_updates=0,optimizer_updates=0,threshold_fit=False,rows=results)
 d.write(d.BASE/'analysis.json',result);d.write(d.BASE/'completion.json',dict(status='ANALYZED',models=3,source_count=81000,source_forward_packets=243000,source_criterion_pass=passed,source_only=True,U_truth_read=False,target_access=False,head_fit_updates=0,optimizer_updates=0));print(json.dumps(dict(status='ANALYZED',source_criterion_pass=passed,admitted_count=n,admitted_precision=result['admitted_precision'])),flush=True)
if __name__=='__main__':dispatch()
