import copy,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .metrics import observe_gradients,vector_stats,summary
from experiments.cvs_source_gate_audit.gates import physical_metadata,masks
from experiments.cvs_phase1_overlay.model import native_modules
from experiments.cvs_equivariant_identity.precision import numerical_context
from experiments.cvs_nocfo_foundation.source import load_source
from experiments.cvs_nocfo_foundation import design as nd

def role(native,student,teacher,data,meta,c,args,name,out):
 if name not in ('L_s','V') or len(data['y'])!=c['counts'][name]:raise ValueError('Illegal source role')
 order=torch.randperm(len(data['y']),generator=torch.Generator().manual_seed(c['shuffle_seeds'][name]));torch.manual_seed(c['strong_dropout_seeds'][name])
 student.train();teacher.eval()
 if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in student.id_backbone.modules()):raise ValueError('Observer must not update BN')
 params=[dict(student.named_parameters())[n] for n in d.PARAMETERS]
 predictions=[];values={k:[] for k in ('teacher_confidence','teacher_margin','joint160_cosine')};selections={k:[] for k in d.SCOPES};raw=[];batches=[]
 for begin in range(0,len(order),c['batch']):
  ix=order[begin:begin+c['batch']];x=data['x'][ix];batch_meta={k:v[ix] for k,v in meta.items()}
  domains=(torch.searchsorted(torch.tensor([1,3,4,6,8]),data['rx'][ix])*3+data['day'][ix]-1).long()
  with torch.no_grad():weak=teacher.forward_identity_only(x)
  with torch.set_grad_enabled(name=='L_s'):strong=student.forward_identity_only(native._strong_augment(x,args.strong_noise_std))
  pseudo,selected,detail=masks(native,weak['tx_logits'],strong['tx_logits'].detach(),batch_meta,domains,args)
  if weak['id_feat_joint'].shape!=(len(ix),160) or strong['id_feat_joint'].shape!=(len(ix),160):raise ValueError('Actual deployed feature shape differs')
  # All gates and feature observations are fixed before observer truth is consulted.
  probs=weak['tx_logits'].softmax(1).topk(2,dim=1).values
  values['teacher_confidence'].append(probs[:,0]);values['teacher_margin'].append(probs[:,0]-probs[:,1]);values['joint160_cosine'].append(F.cosine_similarity(weak['id_feat_joint'],strong['id_feat_joint'].detach(),dim=1))
  predictions.append(pseudo)
  for key in d.SCOPES:selections[key].append(selected[key])
  if name=='L_s':
   records={key:observe_gradients(strong['tx_logits'],pseudo,data['y'][ix],selected[key],params,name) for key in d.SCOPES}
   raw.append(dict(begin=begin,count=len(ix),records=records))
   batches.append(dict(begin=begin,count=len(ix),scopes={key:dict(accepted=r['accepted'],accepted_wrong=r['accepted_wrong'],decomposition_max_abs_error=r['decomposition_max_abs_error'],layers=[dict(parameter=p,**vector_stats(r['vectors']['total'][k],r['vectors']['matched_true'][k],r['vectors']['wrong'][k])) for k,p in enumerate(d.PARAMETERS)]) for key,r in records.items()}))
 all_values={k:torch.cat(v) for k,v in values.items()};correct=torch.cat(predictions)==data['y'][order];all_masks={k:torch.cat(v) for k,v in selections.items()}
 scopes={key:summary(all_values,correct,keep) for key,keep in all_masks.items()};global_gradients={}
 if name=='L_s':
  path=out/'L_source_gradient_vectors.pt';torch.save(raw,path);saved=torch.load(path,map_location='cpu',weights_only=False)
  for key in d.SCOPES:
   count=sum(r['records'][key]['accepted'] for r in saved);sums={label:[torch.zeros(p.numel(),dtype=torch.float64) for p in params] for label in ('total','matched_true','correct','wrong')}
   for record in saved:
    r=record['records'][key]
    for label in sums:
     for k in range(len(params)):sums[label][k]+=r['accepted']*r['vectors'][label][k].double()
   for label in sums:sums[label]=[v/max(count,1) for v in sums[label]]
   global_gradients[key]=dict(accepted=count,layers=[dict(parameter=p,**vector_stats(sums['total'][k],sums['matched_true'][k],sums['wrong'][k])) for k,p in enumerate(d.PARAMETERS)])
  # Independently recompute batch summaries from the saved vectors.
  for saved_record,record in zip(saved,batches):
   for key in d.SCOPES:
    r=saved_record['records'][key]
    for k,p in enumerate(d.PARAMETERS):
     if dict(parameter=p,**vector_stats(r['vectors']['total'][k],r['vectors']['matched_true'][k],r['vectors']['wrong'][k]))!=record['scopes'][key]['layers'][k]:raise ValueError('Saved gradient recount differs')
 torch.save(dict(values=all_values,correct=correct,masks=all_masks),out/(name+'_feature_observations.pt'))
 return dict(role=name,count=len(order),forward_batches=2*((len(order)+255)//256),teacher_weak_correct=int(correct.sum()),scopes=scopes,gradients=global_gradients,gradient_batches=batches,gradient_role='L_s only' if name=='L_s' else 'NONE',saved_gradient_recount='VERIFIED' if name=='L_s' else 'not_applicable')

def run(c):
 d.validate(c);out=Path(c['output_root'])
 if out.exists():raise FileExistsError('Existing immutable diagnostic output')
 row=next(r for r in d.rows() if r['row_id']==c['row_id']);api=d.model_api(row);prior=c['source_config'];source=api.source_proof(prior)
 torch.set_num_threads(2);start=time.time();L,U,V,contract=load_source(nd.config(nd.rows()[0]));assert U is None
 args=d.prior(row).make_args(prior,'cpu')
 if args.pseudo_domain_gate or not args.pseudo_temporal_gate or args.pseudo_temporal_mode!='batch_neighbor' or args.pseudo_temporal_window!=2 or args.strong_noise_std!=.015 or args.lambda_u!=.16:raise ValueError('Original source/U recipe differs')
 metadata={name:physical_metadata(data,name) for name,data in [('L_s',L),('V',V)]}
 with numerical_context(d.FULL_FP32_POLICY),torch.random.fork_rng(devices=[]):
  student=api.own_model(prior,'cpu');teacher=copy.deepcopy(student);ck=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
  if ck['ema_model'] is None or ck['epoch']!=50 or ck['a1_ema_successful_updates']!=11100:raise ValueError('Unverified own EMA')
  teacher.load_state_dict(ck['ema_model'],strict=True);teacher.eval().requires_grad_(False)
  student.requires_grad_(False)
  for name,p in student.named_parameters():p.requires_grad_(name in d.PARAMETERS)
  saved={key:value.clone() for key,value in student.state_dict().items()};saved_t={key:value.clone() for key,value in teacher.state_dict().items()}
  out.mkdir(parents=True,exist_ok=False);d.write(out/'resolved_config.json',dict(config=c,execution_commit=(d.ROOT/'release_commit.txt').read_text().strip(),checkpoint_sources=[str(source/'final_ssdg.pth')],source_roles='EXACT_MATCH',default_head_only=True,new_head_fit=False,query_read=False,U_truth_read=False))
  results=[role(native_modules(),student,teacher,data,metadata[name],c,args,name,out) for name,data in [('L_s',L),('V',V)]]
  if any(not torch.equal(v,student.state_dict()[k]) for k,v in saved.items()) or any(not torch.equal(v,teacher.state_dict()[k]) for k,v in saved_t.items()) or any(p.grad is not None for p in student.parameters()):raise ValueError('Frozen observer mutated model or accumulated parameter gradients')
 result=dict(status='ANALYZED',config=c,roles=results,seconds=time.time()-start,model_parameters_buffers_unchanged=True,source_only=True,query_read=False,U_IQ_read=False,U_truth_read=False,head_updates=0,optimizer_steps=0,model_state_updates=0,limitations='Frozen final50 labeled-source proxies only; L in training and V diagnostic; no U correctness or trajectory inference, no DG effect, no fitted head/threshold scan; same physical source data across model seeds are not independent datasets')
 d.write(out/'analysis.json',result);d.write(out/'completion.json',dict(status='ANALYZED',row_id=c['row_id'],source_counts=[6300,27000],source_only=True,query_read=False,U_truth_read=False,head_updates=0,optimizer_steps=0));return result

def dispatch():
 d.BASE.mkdir(parents=True,exist_ok=False);spec=d.read(d.ROOT/'registration/experiment.json')
 if spec['run_id']!=d.RUN or [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Source preregistration differs')
 results=[]
 for row in d.rows():
  try:result=run(d.config(row));results.append(result)
  except Exception as exc:
   d.write(d.BASE/'failure.json',dict(status='FAILED',row_id=row['row_id'],error=str(exc),no_automatic_retry=True));raise
  print(json.dumps(dict(row_id=row['row_id'],status=result['status'],seconds=result['seconds'])),flush=True)
 d.write(d.BASE/'analysis.json',dict(status='ANALYZED',run_id=d.RUN,models=3,rows=results))
 d.write(d.BASE/'completion.json',dict(status='ANALYZED',models=3,source_only=True,query_read=False,U_truth_read=False,head_updates=0,optimizer_steps=0))
if __name__=='__main__':dispatch()
