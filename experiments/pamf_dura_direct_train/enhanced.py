"""One-stage E30 physical restoration plus source-F-guided bounded IQ enhancement."""
import copy
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from . import f_guided
from .iq_enhancement import ShapeEnhancer,head_basis,amplified_reference
base=f_guided.base
source_objective=f_guided.source_objective
teacher_reference=f_guided.teacher_reference
source_ids=f_guided.source_ids
RUN='20261031-phase1-DURA-IQenhance-E30-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.enhanced'
GUIDANCE_METRIC='F_amplified_reference_cosine_error'


def learning_rate(update):
    if not 0<=update<7410:raise ValueError('Fixed complete E30 budget required')
    if update<247:return .001*(update+1)/247
    return .00002+(.001-.00002)*.5*(1+math.cos(math.pi*(update-247)/(7410-247-1)))


def objective(student,reader,received,clean,action,variance,probes,reference):
    _,terms,physical=source_objective(student,received,clean,action,variance,probes,return_restored=True)
    enhanced=student.enhancer(received,physical)
    guide=1-F.cosine_similarity(reader.features(enhanced),reference,dim=1,eps=1e-8)
    iq=(enhanced-clean).square().mean((1,2))/clean.square().mean((1,2)).clamp_min(1e-4)
    loss=terms['iq_nmse']+.1*terms['prior_nmse']+.05*terms['spectral_error']+.05*terms['action_probe_nmse']+iq+guide
    return loss,dict(terms,enhanced_iq_nmse=iq,**{GUIDANCE_METRIC:guide}),enhanced


def smoke(args,p):
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    clean,roles=base.source_training(p);source_ids(p,roles)
    take=np.flatnonzero(roles=='L_s')[:2];folder=p['cache']/'epochs/01'
    values={key:base.tensor(np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False)[take],'cpu')
            for key in ('received','kernel','coefficients','variance')}
    x=base.tensor(clean[take],'cpu');student=base.scratch_model(2026092701)
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    frozen={k:v.detach().clone() for k,v in reader.state_dict().items()}
    with torch.no_grad():
        original=reader.features(x)
        reference=amplified_reference(original,head_basis(reader))
        if not torch.equal(reader.classify_features(original).argmax(1),reader.classify_features(reference).argmax(1)):
            raise ValueError('Amplified source reference changed original head choices')
    generator=torch.Generator().manual_seed(2026103004)
    probes=[torch.complex(torch.randn(2,256,generator=generator),torch.randn(2,256,generator=generator)) for _ in range(2)]
    loss,terms,enhanced=objective(student,reader,values['received'],x,
        base.supervised_action(values['kernel'],values['coefficients']),values['variance'],probes,reference)
    physical_parameters=[v for k,v in student.named_parameters() if not k.startswith('enhancer.')]
    disconnected=torch.autograd.grad(terms[GUIDANCE_METRIC].mean(),physical_parameters,allow_unused=True,retain_graph=True)
    if any(v is not None for v in disconnected):raise ValueError('F guidance reached physical restoration parameters')
    loss.mean().backward()
    for group in (physical_parameters,list(student.enhancer.parameters())):
        gradients=[v.grad for v in group if v.grad is not None]
        if not gradients or not all(torch.isfinite(v).all() for v in gradients) or not any(v.abs().sum()>0 for v in gradients):
            raise ValueError('Both source parameter groups require finite nonzero gradients')
    if any(v.grad is not None for v in reader.parameters()) or any(not torch.equal(v,frozen[k]) for k,v in reader.state_dict().items()):
        raise ValueError('Source guidance changed frozen F')
    # Nonzero head state checks that the deployed optimized path includes the head.
    with torch.no_grad():student.enhancer.net[-1].weight.fill_(.01)
    student.eval().requires_grad_(False)
    with torch.no_grad():
        physical=student(values['received'])
        expected=student.enhancer(values['received'],physical)
        if torch.equal(expected,physical):raise ValueError('Nonzero smoke enhancement must change IQ')
    checkpoint=p['output']/'smoke-scratch_joint.pt';save_model(checkpoint,student,'scratch_joint',0,0,args.commit)
    loaded=load_model(checkpoint,'cpu',arm='scratch_joint',commit=args.commit,epoch=0)
    with torch.no_grad():
        torch.testing.assert_close(loaded.enhancer(values['received'],loaded(values['received'])),expected,atol=0,rtol=0)
        optimized=EnhancedPipeline(loaded,reader,True)(values['received'])
        torch.testing.assert_close(optimized[0],expected,atol=3e-5,rtol=3e-5)
        if optimized[3].shape!=(2,6) or not bool(optimized[-1]):raise ValueError('Actual original-head optimized smoke failed')
    base.write_json(p['output']/'smoke.json',dict(status='PASS',real_Ls=2,TX_labels_read=0,target_reads=0,
        training_F_access=True,F_unchanged=True,F_updates=0,head_fit_updates=0,source_physical_roles_match_F=True,
        physical_F_gradients_absent=True,both_G_parameter_groups_have_gradients=True,strict_reload=True,
        nonzero_enhancement_in_optimized_path=True,source_initialization_checkpoint_loads=0))
    print(json.dumps(dict(event='REAL_SOURCE_ENHANCEMENT_NOQUERY_SMOKE_PASS')),flush=True)


class EnhancedPipeline:
    def __init__(self,model,reader,enhance=True):
        self.model,self.reader,self.enhance=model,reader,enhance
        if any(m.training or any(p.requires_grad for p in m.parameters()) for m in (model,reader) if m is not None):
            raise ValueError('Only frozen evaluation modules may enter prediction')

    def __call__(self,x):
        from experiments.pamf_stage2_inference.program import outputs
        physical,variance,_,_,valid=outputs(self.model,x,reuse=True)
        iq=self.model.enhancer(x,physical) if self.enhance else physical
        valid=valid & torch.isfinite(iq).all()
        if self.reader is None:return iq,valid
        features=self.reader.features(iq);logits=self.reader.classify_features(features)
        return iq,variance.mean(-1),features,logits,valid & torch.isfinite(features).all() & torch.isfinite(logits).all()


class EnhancedEngine:
    def __init__(self,model,reader=None,enhance=True):
        from experiments.pamf_r46_cuda_graph.graph import GraphCache
        self.call=GraphCache(EnhancedPipeline(model,reader,enhance))

    @torch.inference_mode()
    def __call__(self,x):
        from experiments.pamf_r46_cuda_graph.graph import validate_iq
        validate_iq(x);result=self.call(x)
        if not bool(result[-1]):raise FloatingPointError('Nonfinite enhanced inference output')
        return result[:-1]

    def setup(self):return self.call.setup()


@torch.no_grad()
def predict(args,p,role):
    from . import evaluation as ev
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    done=base.read_json(p['output']/args.arm/'training_complete.json')
    if (done['epoch'],done['updates'],done['target_reads'],done['F_unchanged'])!=(30,7410,0,True):
        raise ValueError('Complete frozen legal E30 training required')
    folder,ids,_,_=ev.reference(p,role)
    model=load_model(p['output']/args.arm/'final.pt','cuda:0',arm=args.arm,commit=args.commit)
    reader=load_reader(p['reader'],p['source_cache'],'cuda:0').eval().requires_grad_(False)
    # Two readouts of ONE trained model, one shared physical forward per batch.
    from experiments.pamf_r46_cuda_graph.graph import GraphCache
    def pipeline(x):
        from experiments.pamf_stage2_inference.program import outputs
        physical,variance,_,_,valid=outputs(model,x,reuse=True)
        enhanced=model.enhancer(x,physical)
        fp=reader.features(physical);fe=reader.features(enhanced)
        lp=reader.classify_features(fp);le=reader.classify_features(fe)
        raw=reader(x) if role=='source' else torch.zeros_like(le)
        return physical,enhanced,lp,le,raw,valid & torch.isfinite(enhanced).all() & torch.isfinite(lp).all() & torch.isfinite(le).all() & torch.isfinite(raw).all()
    call=GraphCache(pipeline)
    folders={}
    for arm in ('scratch_joint','physical'):
        output=p['output']/arm/(role+'_prediction');output.mkdir(parents=True)
        folders[arm]=output
    started=time.perf_counter();timings={}
    for view in base.VIEWS:
        received=np.load(folder/(view+'.npy'),mmap_mode='r',allow_pickle=False)
        if received.shape!=(len(ids),2,256) or received.dtype!=np.float32:raise ValueError('Original fixed FP32 view required')
        arrays={}
        for arm,output in folders.items():
            shapes=dict(iq=(len(ids),2,256),logits=(len(ids),6))
            if role=='source':shapes['raw_logits']=(len(ids),6)
            arrays[arm]={k:np.lib.format.open_memmap(output/(view+'.'+k+'.npy'),mode='w+',dtype=np.float32,shape=s) for k,s in shapes.items()}
        start=time.perf_counter()
        for offset in range(0,len(ids),128):
            end=min(offset+128,len(ids));x=base.tensor(received[offset:end],'cuda:0')
            physical,enhanced,lp,le,raw,valid=call(x)
            if not bool(valid):raise FloatingPointError('Nonfinite paired prediction')
            for arm,iq,scores in [('physical',physical,lp),('scratch_joint',enhanced,le)]:
                arrays[arm]['iq'][offset:end]=base.numpy_values(iq)
                arrays[arm]['logits'][offset:end]=base.numpy_values(scores)
                if role=='source':arrays[arm]['raw_logits'][offset:end]=base.numpy_values(raw)
        for fields in arrays.values():
            for array in fields.values():array.flush()
        timings[view]=time.perf_counter()-start
        print(json.dumps(dict(event='PAIRED_FROZEN_VIEW_COMPLETE',role=role,view=view,count=len(ids))),flush=True)
    for arm,output in folders.items():
        np.save(output/'physical_ids.npy',ids,allow_pickle=False)
        base.write_json(output/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=list(base.VIEWS),
            role=role,arm=arm,F_run=base.F_RUN,F_seed=2026092701,original_default_head=True,TX_truth_reads=0,
            state_updates=0,head_fit_updates=0,code_commit=args.commit,seconds=time.perf_counter()-started,
            per_view_seconds=timings,setup=call.setup(),paired_shared_physical_forward=True))


def score(args,p,role):
    from . import evaluation as ev
    # Validate BOTH complete seven-view readouts before either may connect truth.
    _,ids,_,_=ev.reference(p,role)
    for arm in ('scratch_joint','physical'):
        ev.complete_arrays(p['output']/arm/(role+'_prediction'),ids,arm=arm,role=role,commit=args.commit)
    for arm in ('scratch_joint','physical'):
        paired=copy.copy(args);paired.arm=arm;ev.score(paired,p,role)


@torch.no_grad()
def profile(args,p):
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    default,_=base.parent_model(p,'cuda:0');default.eval().requires_grad_(False)
    model=load_model(p['output']/args.arm/'final.pt','cuda:0',arm=args.arm,commit=args.commit)
    reader=load_reader(p['reader'],p['source_cache'],'cuda:0').eval().requires_grad_(False)
    values=base.tensor(np.load(p['cache']/'validation/practical_low_urban.npy',mmap_mode='r',allow_pickle=False)[:128],'cuda:0')
    records={}
    for scope in ('G','G_plus_F'):
        f=reader if scope=='G_plus_F' else None
        engines=dict(DURA=InferenceEngine(default,f,reuse=True,graph=True,native=False),
            physical=EnhancedEngine(model,f,False),enhanced=EnhancedEngine(model,f,True))
        for batch in (1,128):
            times={k:[] for k in engines}
            for repeat in range(3):
                for name in list(engines)[::1 if repeat%2==0 else -1]:
                    engine=engines[name]
                    for _ in range(3):engine(values[:batch])
                    torch.cuda.synchronize()
                    for _ in range(16):
                        start=time.perf_counter();engine(values[:batch]);torch.cuda.synchronize()
                        times[name].append((time.perf_counter()-start)*1000)
            records[scope+'_B'+str(batch)]={k:dict(median_ms=float(np.median(v)),p95_ms=float(np.quantile(v,.95)),
                throughput_samples_s=float(batch*1000/np.mean(v)),iterations=48) for k,v in times.items()}
        del engines;gc.collect();torch.cuda.empty_cache()
    base.write_json(p['output']/args.arm/'matched_profile.json',dict(status='PROFILE_COMPLETE',cases=records,
        parameters=sum(v.numel() for v in model.parameters()),enhancer_parameters=sum(v.numel() for v in model.enhancer.parameters()),
        parameter_bytes=sum(v.numel()*v.element_size() for v in model.parameters()),
        checkpoint_bytes=(p['output']/args.arm/'final.pt').stat().st_size,shared_GPU_load_not_isolated=True))


def worker(args,p):
    try:
        train(args,p);gc.collect();torch.cuda.empty_cache();profile(args,p)
        for role in ('source','target'):
            predict(args,p,role)
            subprocess.run(base.command(args,'score',args.arm,role),env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
        base.write_json(p['output']/args.arm/'complete.json',dict(status='ANALYZED',all_fourteen_predictions_before_truth=True,target_feedback=False))
    except BaseException as error:
        import traceback
        row=p['output']/args.arm
        if row.exists():base.write_json(row/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True))
        raise


def configure():
    f_guided.configure()
    previous_configuration=base.configuration;previous_command=base.command;previous_scratch=base.scratch_model
    def scratch(seed):
        model=previous_scratch(seed)
        state=torch.get_rng_state()
        torch.manual_seed(2026103107);model.enhancer=ShapeEnhancer();torch.set_rng_state(state)
        return model
    def configuration():
        value=previous_configuration()
        value.update(run=RUN,recipe='physical_plus_bounded_IQ_enhancement_E30',epochs=30,updates=7410,
            L_exposures=189000,U_exposures=1701000,new_cumulative_epochs=30,parameters=267881,
            corruption_epochs=20,corruption_schedule='epochs01-20 then01-10;30fresh source shuffles;20 unique corruption caches',
            source_freeze='fixedfinalE30/7410;noearlystop/source-or-target-ranking',training_stages=1,
            F_feature_weight=1.,enhanced_IQ_weight=1.,F_reference='normalize(u+QheadQheadT*u);factor2 original centered-head projection',
            F_guidance_metric=GUIDANCE_METRIC,F_gradient_destination='enhanceronly;physicaldetached;two independentlyclipped parameter groups;one joint AdamWstep',
            enhancement_RMS_limit=.05,enhancement_relative_energy_limit=.0025,complex_gain_phase_projection_removed=True,
            objective='physicalnoVar4losses+enhancedIQNMSE+cos(Fenhanced,amplifiedFclean)',
            readouts=['physical','scratch_joint'],readout_meaning='two frozen readouts of one training;all2x7 predictions perrole before truth',
            hard_pseudo_TX_labels=False,max_total_visible_GPU_processes=3,allowed_GPUs=[4,6,7],
            host_numpy_conversion='DLPack_owned_FP32',claim_scope='recognition-oriented IQ,not exact physical TX reconstruction or certified recovery upper bound')
        return value
    def command(args,mode,arm=None,role=None):
        return [MODULE if v==f_guided.MODULE else v for v in previous_command(args,mode,arm,role)]
    base.RUN=RUN;base.scratch_model=scratch;base.configuration=configuration;base.command=command
    base.save_model=save_model;base.load_model=load_model;base.smoke=smoke;base.train=train
    base.learning_rate=learning_rate;base.worker=worker
    # runner.main imports the scorer after configure; replace only this release.
    from . import evaluation as ev
    ev.load_model=load_model;ev.configuration=configuration
    original_score=ev.score
    def combined(args,p,role):
        ev.score=original_score
        try:score(args,p,role)
        finally:ev.score=combined
    ev.score=combined


def save_model(path,model,arm,epoch,updates,commit):
    from experiments.pamf_r65_four_route.runtime import validate_aliases
    validate_aliases(model)
    payload=dict(schema='dura_source_F_guided_v1',arm=arm,epoch=epoch,updates=updates,code_commit=commit,
        config=base.configuration(),source_contract='original7/63/30',parent_run=None,parent_arm=None,
        initialization='scratch_no_checkpoint',training_TX_labels_read=0,training_F_access=True,
        V_training=False,target_training=False,F_teacher=teacher_reference(),F_updates=0,model=model.state_dict())
    with path.open('xb') as stream:torch.save(payload,stream)

def load_model(path,device,*,arm,commit,epoch=30):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    expected=dict(schema='dura_source_F_guided_v1',arm=arm,epoch=epoch,updates=epoch*247,code_commit=commit,
        config=base.configuration(),source_contract='original7/63/30',parent_run=None,parent_arm=None,
        initialization='scratch_no_checkpoint',training_TX_labels_read=0,training_F_access=True,
        V_training=False,target_training=False,F_teacher=teacher_reference(),F_updates=0)
    if any(payload.get(key)!=value for key,value in expected.items()):
        raise ValueError('Authorized source F-guided checkpoint metadata required')
    from experiments.pamf_r65_four_route.runtime import validate_aliases
    model=base.scratch_model(2026092701)
    validate_aliases(model,payload['model']);model.load_state_dict(payload['model'],strict=True)
    return model.to(device).eval().requires_grad_(False)

def train(args,p):
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    started_total=time.perf_counter();row=p['output']/args.arm;row.mkdir()
    base.write_json(row/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,
        physical_GPU=os.environ['CUDA_VISIBLE_DEVICES'],commit=args.commit,training_regime='source_F_guided'))
    clean,roles=base.source_training(p);ids=source_ids(p,roles)
    reader=load_reader(p['reader'],p['source_cache'],'cuda:0').eval().requires_grad_(False)
    frozen_state={key:value.detach().cpu().clone() for key,value in reader.state_dict().items()}
    cache=row/'source_feature_cache';cache.mkdir();cache_started=time.perf_counter()
    targets=np.lib.format.open_memmap(cache/'features.npy',mode='w+',dtype=np.float32,shape=(63000,160))
    with torch.no_grad():
        for offset in range(0,63000,256):
            end=min(offset+256,63000);features=reader.features(base.tensor(clean[offset:end],'cuda:0'))
            if features.shape!=(end-offset,160) or not torch.isfinite(features).all():raise ValueError('Invalid source F feature cache')
            targets[offset:end]=base.numpy_values(features)
    targets.flush();np.save(cache/'physical_ids.npy',ids,allow_pickle=False)
    cache_seconds=time.perf_counter()-cache_started
    base.write_json(cache/'complete.json',dict(status='SOURCE_F_FEATURES_COMPLETE',count=63000,dimension=160,
        numeric_bytes=63000*160*4,TX_labels_read=0,V_reads=0,target_reads=0,F_updates=0,F_teacher=teacher_reference(),seconds=cache_seconds))
    target_features=base.tensor(targets,'cuda:0');del targets
    # F loading/cache construction must not change the matched G/probe RNG.
    torch.manual_seed(2026092701);torch.cuda.manual_seed_all(2026092701)
    student=base.scratch_model(2026092701).to('cuda:0')
    before={key:value.detach().cpu().clone() for key,value in student.state_dict().items()}
    physical=[p for name,p in student.named_parameters() if not name.startswith('enhancer.')]
    enhancement=list(student.enhancer.parameters())
    optimizer=torch.optim.AdamW([{'params':physical},{'params':enhancement}],lr=1e-3,weight_decay=1e-4)
    student.head_basis=head_basis(reader)
    target_features=amplified_reference(target_features,student.head_basis)
    base.write_json(row/'resolved_config.json',base.configuration())
    base.write_json(row/'initialization.json',student.initialization_report)
    preparation_seconds=time.perf_counter()-started_total
    rng=np.random.default_rng(2026103004);updates=seen_l=seen_u=0;started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (row/'training.jsonl').open('x',encoding='utf-8') as stream:
        for epoch in range(1,31):
            folder=p['cache']/'epochs'/f'{((epoch-1)%20)+1:02d}'
            arrays={key:np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in ('received','kernel','coefficients','variance')}
            totals=dict.fromkeys(('loss','iq_nmse','prior_nmse','spectral_error','action_probe_nmse','log_variance_mse','enhanced_iq_nmse',GUIDANCE_METRIC),0.)
            clipped=enhancer_clipped=0;order=rng.permutation(63000)
            for offset in range(0,63000,256):
                take=order[offset:offset+256];x=base.tensor(clean[take],'cuda:0')
                values={key:base.tensor(value[take],'cuda:0') for key,value in arrays.items()}
                action=base.supervised_action(values['kernel'],values['coefficients'])
                probes=[torch.complex(torch.randn(len(take),256,device='cuda:0'),torch.randn(len(take),256,device='cuda:0')) for _ in range(2)]
                reference=target_features[torch.tensor(take.tolist(),device='cuda:0',dtype=torch.long)]
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=base.learning_rate(updates)
                loss,terms,_=objective(student,reader,values['received'],x,action,values['variance'],probes,reference)
                if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite source guided objective')
                loss.mean().backward();norm=torch.nn.utils.clip_grad_norm_(physical,1.,error_if_nonfinite=True)
                head_norm=torch.nn.utils.clip_grad_norm_(enhancement,1.,error_if_nonfinite=True)
                enhancer_clipped+=int(head_norm>1)
                clipped+=int(norm>1);optimizer.step();updates+=1
                seen_l+=int((roles[take]=='L_s').sum());seen_u+=int((roles[take]=='U_s').sum())
                for key,value in dict(loss=loss,**terms).items():totals[key]+=float(value.detach().sum())
            record=dict(epoch=epoch,updates=updates,losses={key:value/63000 for key,value in totals.items()},
                L_exposures=seen_l,U_exposures=seen_u,clip_fraction=clipped/247,enhancer_clip_fraction=enhancer_clipped/247,seconds=time.perf_counter()-started,
                F_inputs=seen_l+seen_u,F_clean_cache_inputs=63000,F_updates=0,TX_labels_read=0,initialization_checkpoint_loads=0)
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
            print(json.dumps(dict(event='SOURCE_GUIDED_EPOCH_COMPLETE',**record)),flush=True)
    if (updates,seen_l,seen_u)!=(7410,189000,1701000):raise ValueError('Incomplete guided source budget')
    if any(v.grad is not None for v in reader.parameters()) or any(not torch.equal(v.detach().cpu(),frozen_state[k]) for k,v in reader.state_dict().items()):
        raise ValueError('Training changed frozen F')
    changed=sum(not torch.equal(v.detach().cpu(),before[k]) for k,v in student.state_dict().items())
    if not changed:raise ValueError('Source G did not train')
    save_model(row/'final.pt',student,args.arm,30,updates,args.commit)
    base.write_json(row/'training_complete.json',dict(status='SOURCE_TRAINED',epoch=30,updates=7410,arm=args.arm,
        L_exposures=seen_l,U_exposures=seen_u,training_TX_labels_read=0,training_F_access=True,V_reads=0,target_reads=0,
        F_updates=0,F_unchanged=True,F_clean_teacher_inputs=63000,F_restored_training_inputs=1890000,F_teacher=teacher_reference(),
        initialization_checkpoint_loads=0,G_ancestor_epochs=0,cumulative_G_epochs=30,F_reused_teacher_epochs=50,enhancement_optimizer_steps=updates,physical_optimizer_steps=updates,
        head_fit_updates=0,parameters=sum(p.numel() for p in student.parameters()),changed_state_tensors=changed,seconds=time.perf_counter()-started,
        source_preparation_seconds=preparation_seconds,feature_cache_seconds=cache_seconds,
        total_G_training_pipeline_seconds=time.perf_counter()-started_total,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        training_regime='authorized_source_F_guided_no_direct_TX_labels'))


if __name__=='__main__':configure();base.main()
