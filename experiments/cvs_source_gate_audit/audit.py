import copy,json,time
from pathlib import Path
from types import SimpleNamespace
import torch
from . import design as d
from .gates import physical_metadata,masks,expected_neighbor_visibility
from experiments.cvs_feature_multiscale50.predict import source_proof,own_model
from experiments.cvs_feature_multiscale50.model import build
from experiments.cvs_phase1_overlay.model import native_modules
from experiments.cvs_equivariant_identity.precision import numerical_context
from experiments.cvs_nocfo_foundation.source import load_source
from experiments.cvs_nocfo_foundation import design as nd

def role(native,student,teacher,data,meta,c,args,name):
 gen=torch.Generator().manual_seed(c['shuffle_seeds'][name]);order=torch.randperm(len(data['y']),generator=gen)
 torch.manual_seed(c['strong_dropout_seeds'][name]);student.train();teacher.eval()
 if any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in student.id_backbone.modules()):raise ValueError('Source V must not update identity BN')
 saved={k:v.clone() for k,v in student.state_dict().items()};saved_t={k:v.clone() for k,v in teacher.state_dict().items()}
 predictions=[];decisions={};rates={};forward_count=0
 with torch.no_grad():
  for begin in range(0,len(order),256):
   ix=order[begin:begin+256];x=data['x'][ix];batch_meta={k:v[ix] for k,v in meta.items()}
   domains=(torch.searchsorted(torch.tensor([1,3,4,6,8]),data['rx'][ix])*3+data['day'][ix]-1).long()
   weak=teacher.forward_identity_only(x)['tx_logits']
   strong=student.forward_identity_only(native._strong_augment(x,args.strong_noise_std))['tx_logits']
   pseudo,selected,detail=masks(native,weak,strong,batch_meta,domains,args)
   predictions.append(pseudo)
   for key,value in selected.items():decisions.setdefault(key,[]).append(value)
   for key,value in detail.items():rates.setdefault(key,[]).append(value)
   forward_count+=2
 student.eval()
 if any(not torch.equal(v,student.state_dict()[k]) for k,v in saved.items()) or any(not torch.equal(v,teacher.state_dict()[k]) for k,v in saved_t.items()):raise ValueError('Source observer mutated model/buffer')
 predicted=torch.cat(predictions);truth=data['y'][order];correct=predicted==truth
 masks_all={k:torch.cat(v) for k,v in decisions.items()};records=[]
 for dimension,levels in [('overall',[None]),('receiver',[1,3,4,6,8]),('day',[1,2,3]),('source_class',list(range(6)))]:
  for level in levels:
   keep=torch.ones(len(order),dtype=torch.bool) if level is None else data['rx' if dimension=='receiver' else 'day'][order]==level if dimension!='source_class' else truth==level
   for key,selected in masks_all.items():
    selected=selected&keep;n=int(keep.sum());accepted=int(selected.sum());hits=int((selected&correct).sum())
    records.append(dict(role=name,dimension=dimension,stratum=level,gate=key,count=n,accepted=accepted,accepted_correct=hits,coverage=accepted/n,precision=hits/accepted if accepted else None,rejected_correct=int((keep&~selected&correct).sum()),teacher_weak_correct=int(correct[keep].sum()),teacher_weak_accuracy=int(correct[keep].sum())/n))
 return dict(role=name,source_count=len(order),records=records,pass_rates={k:float(torch.cat(v).float().mean()) for k,v in rates.items() if k!='conf'},mean_teacher_confidence=float(torch.cat(rates['conf']).mean()),forward_batches=forward_count,model_parameters_buffers_unchanged=True,identity_student_train_mode=True,BN_updates=0,observer_labels_used_after_masks=True)

def run(c):
 d.validate(c);out=Path(c['output_root'])
 if out.exists():raise FileExistsError('Existing immutable source diagnostic output')
 prior=c['source_config'];source=source_proof(prior)
 torch.set_num_threads(2);start=time.time();L,U,V,contract=load_source(nd.config(nd.rows()[0]));assert U is None
 args=d.old.make_args(prior,'cpu')
 if args.pseudo_domain_gate or not args.pseudo_temporal_gate or args.pseudo_temporal_mode!='batch_neighbor' or args.pseudo_temporal_window!=2 or args.strong_noise_std!=.015 or args.use_mixstyle or args.use_a1_r3:raise ValueError('Frozen historical gate/path recipe differs')
 metadata={name:physical_metadata(data,name) for name,data in [('L_s',L),('V',V)]}
 with numerical_context(d.FULL_FP32_POLICY):
  native=native_modules()
  student=own_model(prior,'cpu');teacher=copy.deepcopy(student)
  ck=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
  if ck['ema_model'] is None or ck['epoch']!=50 or ck['a1_ema_successful_updates']!=11100:raise ValueError('Unknown actual EMA origin')
  teacher.load_state_dict(ck['ema_model'],strict=True);teacher.eval().requires_grad_(False)
  out.mkdir(parents=True,exist_ok=False);d.write(out/'resolved_config.json',dict(config=c,execution_commit=(d.ROOT/'release_commit.txt').read_text().strip(),source_checkpoint=str(source/'final_ssdg.pth'),source_roles='EXACT_MATCH',checkpoint_sources=[str(source/'final_ssdg.pth')],default_head_only=True,new_head_fit=False,query_read=False,U_truth_read=False))
  values=[role(native,student,teacher,data,metadata[name],c,args,name) for name,data in [('L_s',L),('V',V)]]
 result=dict(status='ANALYZED',config=c,records=values,analytic_geometry=[expected_neighbor_visibility(n,s) for n,s in [(6300,70),(27000,300),(56700,630)]],source_only=True,query_read=False,U_IQ_read=False,U_truth_read=False,head_updates=0,optimizer_steps=0,model_state_updates=0,seconds=time.time()-start,limitation='L/V frozen-default proxy accuracy and random-batch geometry; not actual U pseudo accuracy, training intervention, target DG gain or newTX transfer')
 d.write(out/'analysis.json',result);d.write(out/'completion.json',dict(status='ANALYZED',row_id=c['row_id'],source_counts=[6300,27000],source_only=True,target_access=False,U_truth_read=False,head_updates=0,optimizer_steps=0));return result

def dispatch():
 d.BASE.mkdir(parents=True,exist_ok=False)
 spec=d.read(d.ROOT/'registration/experiment.json')
 if spec['run_id']!=d.RUN or [r['config'] for r in spec['rows']]!=[d.config(r) for r in d.rows()]:raise ValueError('Source audit preregistration differs')
 results=[]
 for row in d.rows():
  try:result=run(d.config(row));results.append(result)
  except Exception as exc:
   d.write(d.BASE/'failure.json',dict(status='FAILED',row_id=row['row_id'],error=str(exc),no_automatic_retry=True,source_only=True));raise
  print(json.dumps(dict(row_id=row['row_id'],status=result['status'],seconds=result['seconds'])),flush=True)
 d.write(d.BASE/'analysis.json',dict(status='ANALYZED',run_id=d.RUN,execution_commit=(d.ROOT/'release_commit.txt').read_text().strip(),models=3,rows=results,source_only=True,query_read=False,U_truth_read=False,head_updates=0,optimizer_steps=0))
 d.write(d.BASE/'completion.json',dict(status='ANALYZED',models=3,source_only=True,query_read=False,U_truth_read=False,head_updates=0,optimizer_steps=0))
if __name__=='__main__':dispatch()
