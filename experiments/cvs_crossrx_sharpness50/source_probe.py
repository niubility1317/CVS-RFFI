"""A fixed source prototype; no actual training and no target/V/U inference."""
import copy
import json
import os
import sys
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .objective import objective,select_names
from experiments.cvs_sigtime_heldcredit.coordinate import convert_model,convert_optimizer,PREFIX
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches,physical_ids
from experiments.cvs_sigtime_heldcredit.source_probe import kernel_changes,norm
from experiments.cvs_u_schedule50.counterfactual import simulate,_equal
from experiments.cvs_crossrx_credit50.credit import add_gradients


def run(request):
 d.validate(request);out=Path(d.OUTPUT)
 if out.exists():raise FileExistsError('Preserve existing source prototype, no relaunch')
 out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();torch.set_num_threads(2)
 runtime=dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,device='cpu',GPU_allocations=0)
 d.write(out/'runtime.json',runtime);print('RUNTIME '+json.dumps(runtime),flush=True)
 from experiments.cvs_crossrx_credit50 import design as parent,model as api
 from experiments.cvs_crossrx_credit50.provenance import source_proof
 from experiments.cvs_phase1_stack.runtime import installed
 from experiments.cvs_phase1_stack.fast_execution import configure
 config=parent.config(next(r for r in parent.rows() if r['row_id']==request['source_row']))
 source=source_proof(config)
 raw=api.build(config,torch.device('cpu'))
 origins={k:getattr(raw.get_submodule(PREFIX),k).detach().clone() for k in ('low_hz_','band_hz_')}
 ck=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if (ck['epoch']!=50 or ck['run_id']!=parent.RUN or ck['candidate_id']!=config['row_id']
     or ck['checkpoint_selection']!='final_only' or ck['a1_ema_successful_updates']!=11100):
  raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: fixed BASE final50')
 raw.load_state_dict(ck['model'],strict=True);raw.eval()
 raw_named=list(raw.named_parameters());model=convert_model(raw,origins,mode='CENTER_25K')
 opt=convert_optimizer(ck['optimizer'],raw_named,model,mode='CENTER_25K')
 with fresh_parameter_caches(model),torch.no_grad():logits=model(torch.zeros(2,2,256))
 if logits.shape!=(2,6) or not torch.isfinite(logits).all():raise ValueError('Real-checkpoint no-query smoke failed')
 print('SMOKE PASS strict own BASE+fixedCENTER25K;query_read=false',flush=True)
 with installed(config) as native:
  context=native._build_ssdg_wisig_data(configure(parent.make_args(config,'cpu')),torch.device('cpu'))
  from cvsrffi.xuc_fusion.native import role_ids_from_native
  if context['named_test_loaders'] or role_ids_from_native(context)!=parent.read(parent.SOURCE)['role_ids']:
   raise ValueError('Original source-only role IDs differ')
 dataset=context['train_loader'].dataset
 if len(dataset)!=6300 or getattr(dataset,'hide',True):raise ValueError('Visible source L6300 required')
 metadata=[];cells={}
 for i,it in enumerate(dataset.index):
  metadata.append(dict(y=int(it.tx_i),rx=int(it.rx_i),day=int(it.day_i),
   id=(int(it.eq_i),int(it.rx_i),int(it.day_i),int(it.sig_i),int(dataset.selected[i]))))
  cells.setdefault((int(it.tx_i),int(it.rx_i),int(it.day_i)),[]).append(i)
 physical_ids([v['id'] for v in metadata],6300)
 if len(cells)!=90 or any(len(v)!=70 for v in cells.values()):raise ValueError('Original 90x70 L cells differ')
 keys=sorted(cells);cohorts={o:[cells[k][o] for k in keys] for o in d.ORDINALS}
 held=[cells[k][o] for k in keys for o in d.HELD_ORDINALS]
 selected=sorted(set(held).union(*(set(v) for v in cohorts.values())));packets={}
 if len(selected)!=540 or len(held)!=180:raise ValueError('Fixed distinct source packet counts differ')
 for i in selected:
  x,y,_,meta=dataset[i]
  if (tuple(int(meta[k]) for k in ('eq_i','rx_i','day_i','sig_i','base_index'))!=metadata[i]['id']
      or int(y)!=metadata[i]['y']):raise ValueError('Metadata/source packet identity differs')
  packets[i]=x
 def batch(indices):
  return (torch.stack([packets[i] for i in indices]),torch.tensor([metadata[i]['y'] for i in indices]),
   torch.tensor([metadata[i]['rx'] for i in indices]),[metadata[i]['id'] for i in indices])
 hx,hy,hr,hids=batch(held);parameters=[p for g in opt.param_groups for p in g['params']]
 names=[n for n,p in model.named_parameters()];selected_names=select_names(model)
 if len(parameters)!=188 or len(selected_names)!=17:raise ValueError('Actual feature/optimizer parameter traversal differs')
 head=api.classifier(model);wnames=[n for n,p in model.named_parameters() if p is head.weight]
 if len(wnames)!=1 or head.weight.shape!=(6,160):raise ValueError('Original default6x160 W differs')
 default_w=wnames[0];wi=names.index(default_w)
 gain=[i for i,n in enumerate(names) if n.endswith('sequence_pool.gain_raw')]
 if len(gain)!=1:raise ValueError('Original gain differs')
 before=copy.deepcopy(model.state_dict());optimizer_before=copy.deepcopy(opt.state_dict())
 records=[];forwards=2;temporary=0
 for ordinal in d.ORDINALS:
  x,y,rx,ids=batch(cohorts[ordinal])
  with fresh_parameter_caches(model):logits=model(x)
  base_loss=F.cross_entropy(logits,y)
  base=torch.autograd.grad(base_loss,parameters,retain_graph=True,allow_unused=True)
  if base[wi] is None or not bool(base[wi].abs().sum()>0):raise ValueError('Main default head must remain jointly trainable')
  outputs={};gradients={}
  for mode in d.MODES:
   result=objective(model,logits,y,rx,hx,hy,hr,ids,hids,selected_names,default_w,mode)
   gradient=torch.autograd.grad(result['loss'],parameters,retain_graph=True,allow_unused=True)
   if gradient[wi] is not None or gradient[gain[0]] is not None:raise ValueError('Auxiliary must not train W/gain')
   if any(g is not None for n,g in zip(names,gradient) if n not in selected_names):raise ValueError('Auxiliary escaped declared features')
   summary=result['summary'];summary['auxiliary_gradient_norm']=norm(gradient)
   summary['gradient_norm_by_layer']={layer:norm([g for n,g in zip(names,gradient) if token in n])
    for layer,token in [('Sinc','.sinc.'),('time_fuse','.time_fuse.'),('t1','.t1.'),('t2','.t2.'),('SIG_TIME','.sequence_pool.')]}
   outputs[mode]=summary;gradients[mode]=gradient;forwards+=summary['extra_IQ_forwards']
  cross=gradients['CROSS_GAP'];global_grad=gradients['GLOBAL_GAP']
  difference=tuple(None if a is None and b is None else -b if a is None else a if b is None else a-b for a,b in zip(cross,global_grad))
  relative=norm(difference)/max(norm(global_grad),1e-12)
  proposal=simulate(opt,base,add_gradients(base,cross),names=names);temporary+=2
  physical=kernel_changes(model,names,proposal)
  record=dict(ordinal=ordinal,current_L_CE=float(base_loss),modes=outputs,
              CROSS_GLOBAL_gradient_relative_difference=relative,physical_Sinc=physical,
              actual_AdamW_increment=proposal['summary'])
  records.append(record);forwards+=90;d.write(out/f'ordinal{ordinal}.json',record)
  print('SOURCE_SHARPNESS '+json.dumps(dict(ordinal=ordinal,relative_difference=relative,
    losses={mode:outputs[mode].get('loss') for mode in d.MODES},physical=physical)),flush=True)
 if (not _equal(before,model.state_dict()) or not _equal(optimizer_before,opt.state_dict())
     or any(p.grad is not None for p in model.parameters())):raise ValueError('Prototype modified live model/optimizer/gradients')
 # The detailed gate is resolved from actual objective summaries by a fixed validator below.
 from .gate import source_gate
 gate=source_gate(records)
 if forwards!=14762 or temporary!=8:raise ValueError('Fixed source execution counts differ')
 summary=dict(status='SOURCE_SCREEN_PASSED' if gate['passed'] else 'SOURCE_SCREEN_NOT_PASSED',run_id=d.RUN,
  analysis_code_commit=request['analysis_code_commit'],source_falsification=d.GATE,gate_result=gate,
  checkpoint_source=dict(run_id=parent.RUN,row_id=config['row_id'],epoch=50,seed=2026092701,
   source_roles='EXACT_MATCH',ancestors=[],selection='fixed BASE control/final50; no target ranking'),
  source_IQ_packet_reads=540,source_metadata_packets=6300,anchor_packets=360,held_L_packets=180,
  network_IQ_forwards=forwards,temporary_AdamW_steps=temporary,kernel_syntheses=12,
  actual_optimizer_updates=0,head_fit_updates=0,EMA_updates=0,U_IQ_reads=0,V_IQ_reads=0,
  U_truth_read=False,target_access=False,default_head_joint_main=True,selected_parameter_tensors=17,
  live_parameters_buffers_optimizer_gradients_unchanged=True,previously_trained_source_L=True,
  current_anchor_held_packets_disjoint=True,runtime=runtime,records=records,wall_seconds=time.perf_counter()-started,
  limitation='Fixed final50 eval-mode L-only source availability. No training efficacy, novel-method proof, unseenRX/never-trained holdout or contemporaneous U-containing step. Future new training uses separate E20 plan.')
 d.write(out/'summary.json',summary);print('SOURCE_COMPLETE '+json.dumps(gate),flush=True)
 return summary
