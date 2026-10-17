"""Strict full-parent joint fine tuning;10 new epochs after fixed20."""
from experiments.cvs_feature_sequence50.source import *
def _train(c,design=None,model_api=None,observer=None,parent=None):
    if design is not None:
        validate,make_args,read,write,SOURCE,FULL_FP32_POLICY,ROOT=(getattr(design,k) for k in ("validate","make_args","read","write","SOURCE","FULL_FP32_POLICY","ROOT"))
    else:
        from . import design as design
        validate,make_args,read,write,SOURCE,FULL_FP32_POLICY,ROOT=(getattr(design,k) for k in ("validate","make_args","read","write","SOURCE","FULL_FP32_POLICY","ROOT"))
    if model_api is None:
        from . import model as model_api
    validate(c);out=Path(c['output_root'])
    if c['baseline_reuse']:raise ValueError('Reference control cannot be retrained')
    if out.exists():raise FileExistsError(out)
    device=torch.device('cuda:0');torch.set_num_threads(2)
    a=configure(make_args(c));expected=read(SOURCE);state={};started=time.time()
    epochs,label_epochs,pseudo_epochs,updates=source_budget(a,design)
    recipe=c.get('training_recipe','historicaloldgate+EMA+cos')
    with numerical_context(FULL_FP32_POLICY),installed(c) as n,incremental_native_writer(n),native_default_head_only(n) as diagnostic_guard:
        from cvsrffi.xuc_fusion.native import role_ids_from_native
        from scripts.train_daot_rc4_baseline import resolved_config
        build_data=n._build_ssdg_wisig_data;build_model=n.build_baseline_model;writer=n._write_ssdg_epoch_telemetry;detach=n._detach_log_mapping
        def checked_data(*args,**kw):
            ctx=build_data(*args,**kw)
            if ctx['named_test_loaders'] or role_ids_from_native(ctx)!=expected['role_ids']:raise ValueError('Physical source roles/target construction mismatch')
            if len(ctx['unlabeled_loader'])!=222:raise ValueError('Unexpected U-based step count')
            source_domain_count=model_api.source_domain_count
            if ctx['num_domains']!=source_domain_count(a):raise ValueError('Historical RX/day source model dimensions differ')
            if observer is not None and callable(getattr(observer,'data',None)):observer.data(n,ctx,out)
            state['ctx']=ctx
            contract_hook=getattr(observer,'source_contract',None)
            contract=contract_hook(expected,ctx) if callable(contract_hook) else dict(expected,native_role_comparison='EXACT_MATCH')
            write(out/'source_contract.json',contract)
            write(out/'resolved_native_args.json',clean(resolved_config(a)));write(out/'actual_native_args.json',clean(resolved_config(a)))
            return ctx
        def checked_model(args,dev):
            m=build_model(args,dev)
            from experiments.cvs_source_identity_readout.input_policy import apply
            install=model_api.install
            m._identity_input_policy=apply(m);m=install(m,c).to(dev)
            if actual_flags()!=FULL_FP32_POLICY:raise ValueError('Numerical policy changed inside native trainer')
            m.load_state_dict(parent['model'],strict=True)
            p=out/'initial_smoke.pt';torch.save(dict(model=m.state_dict(),scratch_only=False,parent=c['parent_source']),p)
            m.load_state_dict(torch.load(p,map_location=dev,weights_only=False)['model'],strict=True);m.eval()
            with torch.no_grad():z=m(torch.zeros(2,2,256,device=dev))
            if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Initial checkpoint smoke failed')
            state['model']=m
            if observer is not None:observer.initialize(m,c,out)
            write(out/'resolved_config.json',dict(c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,
                commit=(ROOT/'release_commit.txt').read_text().strip(),torch_version=torch.__version__,native_fitted_probes_disabled=diagnostic_guard['active'],
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),total_parameters=sum(p.numel() for p in m.parameters()),
                hardware=torch.cuda.get_device_name(dev),backend_flags=actual_flags(),native_args_ref=str(out/'resolved_native_args.json'),execution_policy=POLICY,config=c,actual_feature_extractor=m._feature_learning_report,L_context_steps=m._feature_l_steps,mask_counts=m._feature_mask_counts,identity_input_policy=m._identity_input_policy,baseline_recipe=recipe,label_epochs=label_epochs,pseudo_epochs=pseudo_epochs))
            write(out/'initialization.json',dict(scratch_only=False,checkpoint_sources=[str(Path(c['parent_source'])/'final_ssdg.pth')],ancestors=[c['parent_source']],source_roles='EXACT_MATCH',target_contact=False,teacher_origin='strict own parent EMA',seed=c['model_seed'],optimizer_reset=True,parent_epochs=20,incremental_epochs=10,cumulative_epochs=30))
            print('SMOKE PASS strict matching parent and default head; query_read=false',flush=True)
            return m
        def step_log(values):
            result=detach(values)
            if 'train/loss' in result:
                state['steps']=state.get('steps',0)+1
                if 'step_file' not in state:state['step_file']=gzip.open(out/'step_metrics.jsonl.gz','xt',encoding='utf-8',compresslevel=3)
                metrics=batch_clean(result)
                if state['steps']>label_epochs*222 and float(metrics.get('train/pseudo_truth_available',0.) or 0.)!=0:raise ValueError('Hidden U TX became available during pseudo training')
                state['step_file'].write(json.dumps(dict(step=state['steps'],epoch=(state['steps']-1)//222+1,metrics=metrics),allow_nan=False)+'\n')
                if state['steps']>(epochs-1)*222:state['final_L_CE_sum']=state.get('final_L_CE_sum',0.)+float(metrics['train/loss_tx_labeled'])
                if result.get('train/skipped_nonfinite_loss',0) or result.get('train/skipped_nonfinite_grad',0):
                    state['step_file'].flush()
                    raise FloatingPointError('Native nonfinite loss/gradient: preserve artifacts; no automatic retry')
            return result
        def telemetry(cp,jp,rows):
            writer(cp,jp,rows)
            if observer is not None:observer.epoch(state["model"],state["ctx"],rows,out,device)
            # Native complete telemetry is retained; scalar-only companion is
            # rewritten once per epoch, with nonfinite/missing values as null.
            compact=[{k:clean(v) for k,v in row.items() if not isinstance(v,(list,dict,tuple))} for row in rows]
            with (out/'epoch_metrics.jsonl').open('w',encoding='utf-8') as f:
                for row in compact:f.write(json.dumps(row,allow_nan=False)+'\n')
            fields=sorted({k for r in compact for k in r})
            with (out/'epoch_metrics.csv').open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(compact)
        n._build_ssdg_wisig_data=checked_data;n.build_baseline_model=checked_model;n._write_ssdg_epoch_telemetry=telemetry;n._detach_log_mapping=step_log
        old_load=n.load_checkpoint
        def cached_parent(path,device):
            from .checkpoint_policy import load
            return load(path,device,a.baseline_ckpt,out/'final_ssdg.pth',parent,old_load)
        n.load_checkpoint=cached_parent
        try:
            if observer is None:code=n.train(a)
            else:
                with observer.native_scope(n):code=n.train(a)
            if code:raise RuntimeError('Native train returned '+str(code))
        finally:
            n.load_checkpoint=old_load
            n._build_ssdg_wisig_data=build_data;n.build_baseline_model=build_model;n._write_ssdg_epoch_telemetry=writer;n._detach_log_mapping=detach
            if 'step_file' in state:state['step_file'].close()
        if diagnostic_guard['disabled_calls']!=1 or diagnostic_guard['head_fit_updates']!=0:raise ValueError('Native fitted classifier closure was not disabled')
        payload=torch.load(out/'final_ssdg.pth',map_location=device,weights_only=False)
        if payload['epoch']!=10 or payload['args']['baseline_ckpt']!=a.baseline_ckpt or payload['args']['teacher_ckpt'] or payload['checkpoint_selection']!='final_only':raise ValueError('Final E20+10 checkpoint contract mismatch')
        if not a.use_ema_teacher and payload.get('ema_model') is not None:raise ValueError('Disabled EMA was created or retained')
        require_budget(state.get('steps',0),payload['a1_ema_successful_updates'],updates)
        m=state['model'];m.load_state_dict(payload['model'],strict=True)
        write(out/'resolved_native_args.json',clean(resolved_config(SimpleNamespace(**payload['args']))));write(out/'actual_native_args.json',clean(resolved_config(SimpleNamespace(**payload['args']))))
        if m._feature_l_steps!=updates:raise ValueError('Real L context did not match optimizer budget')
        metrics=getattr(design,'source_validation',source_validation)(m,state['ctx']['val_loader'],device)
        write(out/'completion.json',dict(status='SOURCE_TRAINED',epoch=epochs,checkpoint=str(out/'final_ssdg.pth'),
            elapsed_seconds=time.time()-started,final_source_metrics=metrics,target_access=False,target_evaluated=False,native_fitted_probes_disabled=True,native_disabled_probe_calls=diagnostic_guard['disabled_calls'],head_fit_updates=0,
            checkpoint_sources=[a.baseline_ckpt],config=c,cumulative_epochs=30,cumulative_optimizer_steps=6660,parent_epochs=20,parent_optimizer_steps=4440,logged_steps=state.get('steps',0),optimizer_steps=payload['a1_ema_successful_updates'],feature_extractor=m._feature_learning_report,L_context_steps=m._feature_l_steps,mask_counts=m._feature_mask_counts,identity_input_policy=m._identity_input_policy,baseline_recipe=recipe,label_epochs=label_epochs,pseudo_epochs=pseudo_epochs,final_training_L_CE=state['final_L_CE_sum']/222,ancestors=[c['parent_source']],initialization='E20plus10/strictparent/ownEMA/optimizer_reset'))
        write(out/'source_frozen.json',dict(status='SOURCE_FROZEN',config=c,checkpoint=str(out/'final_ssdg.pth'),epoch=epochs,selection='final_only',target_access=False))


from contextlib import contextmanager
from experiments.cvs_fullsource_concat20.source import RecordingLoader
from experiments.cvs_rdrop_sinc_units.optimizer import native_scope
from experiments.cvs_fullsource_concat20 import supcon
from . import design as d,model as api
class Observer:
 def __init__(self,c,parent):self.c=c;self.parent=parent;self.optimizer=None;self.out=Path(c['output_root'])
 def data(self,native,ctx,out):
  Path(out).mkdir(parents=True,exist_ok=True)
  if ctx['num_domains']!=15:raise ValueError('All five RX/day groups required')
  from .sampler import IsolatedLoader,RepartitionLoader
  self.sampler=RepartitionLoader(ctx['train_loader'],self.c['batch_mode'],seed=d.REPARTITION_SEED,loader_seed=d.L_LOADER_SEED,output=Path(out)/'sampler_pools.jsonl.gz')
  ctx['train_loader']=self.sampler;ctx['unlabeled_loader']=IsolatedLoader(ctx['unlabeled_loader'],d.U_LOADER_SEED)
  self.sequence=gzip.open(Path(out)/'training_sequence.jsonl.gz','xt',encoding='utf8',compresslevel=3)
  for key,role in (('train_loader','L'),('unlabeled_loader','U')):ctx[key]=RecordingLoader(ctx[key],role,self.sequence)
 def initialize(self,m,c,out):
  self.model=m;self.out=Path(out)
  if any(not torch.equal(v.detach().cpu(),self.parent['model'][k]) for k,v in m.state_dict().items()):raise ValueError('Strict parent state changed before joint training')
  d.write(self.out/'actual_modules.json',{k:str(v.__file__) for k,v in sys.modules.items() if k in ('SSDG.train_ssdg','DataAugmentation','leo_practical.channel','leo_practical.residual')})
  if m._full_channel is not None:d.write(self.out/'actual_channel.json',m._full_channel.audit())
 def optimizer_created(self,opt):
  if opt.state:raise ValueError('Incremental optimizer must start empty')
  d.write(self.out/'optimizer_initialization.json',dict(reset=True,state_entries=0,base_lr=opt.param_groups[0]['lr'],weight_decay=opt.param_groups[0]['weight_decay'],schedule='native cosine10',parameter_groups=len(opt.param_groups)))
 @contextmanager
 def native_scope(self,native):
  original=native.deepcopy;aug=native.configure_augmentor_for_epoch;schedule=supcon.schedule;terminal=native._resolve_phase1_terminal_status
  def terminal_status(**kwargs):
   kwargs['mechanism_gates_required']=False
   return terminal(**kwargs)
  native._resolve_phase1_terminal_status=terminal_status
  d.write(self.out/'terminal_policy.json',dict(REJECTED_EXTRA_GATE='Historical P0/P1 mechanism promotion gates do not apply to fixed budget default-head source training',only_mechanism_promotion_disabled=True,protocol_export_nonfinite_and_prediction_failures_preserved=True))
  def copied(v,*a,**kw):
   result=original(v,*a,**kw)
   if v is getattr(self,'model',None):
    result.load_state_dict(self.parent['ema_model'],strict=True)
    if any(not torch.equal(x.detach().cpu(),self.parent['ema_model'][k]) for k,x in result.state_dict().items()):raise ValueError('Own EMA restore differs')
    self.teacher=result;e=result.id_backbone.encoder;self.teacher_offset=dict(train=e._full_train_packets,eval=e._full_eval_packets)
    d.write(self.out/'EMA_initialization.json',dict(strict_own_parent=True,parent=self.c['parent_source'],extra_teacher=False,segment_counters_reset=True,temporal_cache_reset=True))
   return result
  native.deepcopy=copied;parent_args=SimpleNamespace(**self.parent['args'])
  native.configure_augmentor_for_epoch=lambda obj,cfg,epoch,args:aug(obj,cfg,13,parent_args)
  supcon.schedule=lambda epoch:schedule(int(epoch)+20)
  try:
   with native_scope(self,'CENTER_UNITS'):
    original_clip=torch.nn.utils.clip_grad_norm_
    def clip(parameters,max_norm,*args,**kw):
     from .core import consume
     return original_clip(consume(self.model,parameters,max_norm),max_norm,*args,**kw)
    torch.nn.utils.clip_grad_norm_=clip
    try:yield
    finally:torch.nn.utils.clip_grad_norm_=original_clip
  finally:
   native.deepcopy=original;native.configure_augmentor_for_epoch=aug;supcon.schedule=schedule;native._resolve_phase1_terminal_status=terminal
   if hasattr(self,'sequence'):self.sequence.close()
   if hasattr(self,'sampler'):self.sampler.close()
 def epoch(self,m,ctx,rows,out,device):
  ep=len(rows);n=ep*222;
  if m._sam_steps!=n or m._sam_pending is not None or m._sam_recording or m._sam_draws or m._sam_vjp_calls!=2*n or m._sam_replay_packets!=512*n:raise ValueError('L replay/update budget mismatch')
  full=self.c['arm']=='FULL_AS';ag=self.c.get('new_Ag',False);sat=self.c.get('As',False)
  if m._feature_l_steps!=n or m._full_calls!=n or m._full_L_packets!=n*512 or m._supcon_calls!=n or m._supcon_applied!=n or self.optimizer._sinc_physical_clip_calls!=n or self.optimizer._sinc_unit_steps!=n or m._supcon_handles or m._supcon_pending is not None:raise ValueError('Incremental F/SupCon/forwards/update budget differs')
  if sat and m._full_channel.packets!=n*(128 if full else 64):raise ValueError('All satellite slots must be supervised')
  if self.sampler.emitted_batches!=n or self.sampler.pools!=n//6 or self.sampler.input_occurrences!=n*128:raise ValueError('All222updates/37exact pools per epoch required')
  if ag and m._clock.packets!=n*64:raise ValueError('All Ag slots must be generated')
  d.write(out/'reference_progress.json',dict(epoch=ep,cumulative_epoch=20+ep,steps=n,cumulative_steps=4440+n,L_packets=m._full_L_packets,Ag_packets=m._clock.packets if ag else 0,As_packets=m._full_channel.packets if sat else 0,source_RX=list(d.FIT),source_RX_holdout=None,U_truth_read=False,target_access=False))
 def finish(self):
  done=d.read(self.out/'completion.json');m=self.model;cp=torch.load(self.out/'final_ssdg.pth',map_location='cpu',weights_only=False)
  if done['epoch']!=10 or done['optimizer_steps']!=2220 or done['cumulative_epochs']!=30 or done['cumulative_optimizer_steps']!=6660 or cp['a1_ema_successful_updates']!=2220 or cp.get('ema_model') is None or done['final_source_metrics']['source_val_count']!=27000:raise ValueError('Fixed cumulative E30 required')
  W='id_backbone.encoder.core.id_backbone.cls_head.weight'
  if cp['model'][W].shape!=(6,160) or torch.equal(self.parent['model'][W],cp['model'][W]) or len(m._full_records)!=2220:raise ValueError('Default head/body must jointly train for incremental budget')
  for name,records in (('reference_steps',m._full_records),('unit_steps',self.optimizer._sinc_unit_lr_records),('L_replay_steps',m._sam_records)):
   with gzip.open(self.out/(name+'.jsonl.gz'),'xt',encoding='utf8',compresslevel=3) as f:
    for r in records:f.write(json.dumps(dict(r,cumulative_step=4440+r['step']),allow_nan=False)+'\n')
  e=m.id_backbone.encoder;t=self.teacher.id_backbone.encoder
  result=dict(update_mode=self.c['update_mode'],extra_supervised_VJP_calls=m._sam_vjp_calls,replay_calls=m._sam_steps,perturbed_calls=sum(r['radius']>0 for r in m._sam_records),replay_L_forward_packets=m._sam_replay_packets,base_L_forward_packets=m._full_L_packets,status='FULL_SOURCE_E30_TRAINED',config=self.c,actual_steps=2220,parent_steps=4440,cumulative_steps=6660,parent_epochs=20,new_epochs=10,cumulative_epochs=30,L_forward_packets=m._full_L_packets+m._sam_replay_packets,student_train_forward_packets=e._full_train_packets,student_eval_forward_packets=e._full_eval_packets,teacher_train_forward_packets=t._full_train_packets-self.teacher_offset['train'],teacher_eval_forward_packets=t._full_eval_packets-self.teacher_offset['eval'],teacher_inherited_counter_offset=self.teacher_offset,SupCon_applied=m._supcon_applied,SupCon_layer_forward_terms=sum(len(r['SupCon_details']) for r in m._full_records),As_training_observations=0 if m._full_channel is None else m._full_channel.packets,Ag_training_observations=0 if getattr(m,'_clock',None) is None else m._clock.packets,batch_repartition=self.sampler.summary(),channel=None if m._full_channel is None else m._full_channel.summary(),clock=None if getattr(m,'_clock',None) is None else m._clock.summary(),coverage=[dict(TX=k[0],RX=k[1],scene=k[2],state=k[3],lock=k[4],count=n) for k,n in sorted(m._full_coverage.items())],seen_V=done['final_source_metrics'],source_RX_holdout=None,head_fit_updates=0,U_truth_read=False,target_access=False,elapsed_seconds=done['elapsed_seconds'],optimizer_reset=True,own_parent_EMA_restored=True)
  n=sys.modules['SSDG.train_ssdg'];a=SimpleNamespace(**cp['args'])
  d.write(self.out/'ordinary_augmentation_actual.json',dict(use_aug=a.use_aug,base_config=n.build_aug_base_cfg(a),parent_schedule_epoch=13,args={k:v for k,v in cp['args'].items() if k.startswith('aug_')},same_generated_IQ_each_RD_pair=True,ordinary_redraws_in_repeat=0,practical_input_is_raw=True))
  d.write(self.out/'training_result.json',result);return result

def train(c):
 d.validate(c);p=d.parent_proof(c['arm']);parent=d.validate_parent_checkpoint(torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False),c)
 observer=Observer(c,parent);_train(c,design=d,model_api=api,observer=observer,parent=parent);return observer.finish()
