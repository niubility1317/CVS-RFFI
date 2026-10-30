"""Single-stage E30 full-width coarse with compact action/priors and cached IQ hints."""
import gc
import time
import numpy as np
import torch
from . import coarse_hint as hint
from . import prior_aux_free as queue
from experiments.pamf_dura_efficiency.model import scratch_model as compact_scratch
from experiments.pamf_reception_restore.models import ComplexTFUNet
base=hint.base
RUN='20261031-phase1-DURA-wideCoarse-E30-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.wide_coarse'
CACHE_RUN='20261031-phase1-DURA-coarse-IQhint-E30-s2026092701-r01'
CACHE_COMMIT='64b8d453e4ead7b8242163bef67dc3db964d5a1f'
PARAMETERS=1466771
PREDECESSOR_MODULE='experiments.pamf_dura_direct_train.strong_phase'
PREDECESSOR_COMMIT='84c5f08b3f334781cad4cb96f9e22bd33d9000eb'
PREDECESSOR_RELEASE='20261031-DURA-strongPhase-E30-r01'


def scratch_model(seed):
    model=compact_scratch(seed)
    # Preserve the compact action/prior states and the later probe RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model.frozen.coarse=ComplexTFUNet()
    model.initialization_report.update(parameters=PARAMETERS,coarse_parameters=1305958,
        architecture='Fulloriginalcoarsewidths;compactaction/twopriortunfold unchanged',
        coarse_spectral_widths=[24,48,72,96],coarse_temporal_widths=[32,64,96,128],
        copied_teacher_parameters=0,coarse_checkpoint_loads=0)
    if sum(p.numel() for p in model.parameters())!=PARAMETERS:raise ValueError('Fixed full-coarse architecture required')
    return model.train().requires_grad_(True)


def validate_cache_metadata(record,producer_launch,producer_plan,producer_done):
    expected=dict(status='SOURCE_COARSE_IQ_CACHE_COMPLETE',teacher_run=hint.TEACHER_RUN,
        teacher_commit=hint.TEACHER_COMMIT,teacher_component='frozen.coarse',
        teacher_used_parameters=1305958,teacher_loaded_parameters=4665073,
        teacher_component_epochs=20,teacher_bundle_component_epochs=60,
        epochs=list(range(1,11)),count_per_epoch=63000,teacher_inputs=630000,numeric_bytes=1290240000,
        teacher_updates=0,teacher_unchanged=True,F_reads=0,TX_labels_read=0,V_reads=0,target_reads=0)
    if any(record.get(k)!=v for k,v in expected.items()):raise ValueError('Original legal frozen source IQ cache required')
    if producer_launch.get('commit')!=CACHE_COMMIT or producer_plan.get('source_contract')!='original7/63/30':
        raise ValueError('Matched original cache producer required')
    expected_done=dict(status='SOURCE_TRAINED',epoch=30,updates=7410,training_F_access=False,
        training_TX_labels_read=0,V_reads=0,target_reads=0,teacher_updates=0)
    if any(producer_done.get(k)!=v for k,v in expected_done.items()):raise ValueError('Complete source-only cache producer required')


def cached_teacher(args,p,row=None):
    started=time.perf_counter();producer=args.root/'runs'/CACHE_RUN
    folder=producer/'scratch_joint/coarse_teacher_cache';original=base.read_json(folder/'complete.json')
    validate_cache_metadata(original,base.read_json(producer/'launch.json'),base.read_json(producer/'plan.json'),
        base.read_json(producer/'scratch_joint/training_complete.json'))
    ids=np.load(p['cache']/'train/physical_ids.npy',allow_pickle=False)
    if ids.shape!=(63000,) or not np.array_equal(ids,np.load(folder/'physical_ids.npy',allow_pickle=False)):
        raise ValueError('Original source IQ cache physical order required')
    record=dict(status='SOURCE_COARSE_IQ_CACHE_REUSED',source_cache_run=CACHE_RUN,source_cache_commit=CACHE_COMMIT,
        source_cache_path=str(folder),source_cache_metadata=original,source_physical_ids_match=True,
        teacher_inputs=0,teacher_checkpoint_loads=0,teacher_updates=0,numeric_bytes=original['numeric_bytes'],
        teacher_component_epochs=20,teacher_bundle_component_epochs=60,
        historic_cache_teacher_inputs=630000,historic_cache_seconds=original['seconds'],
        F_reads=0,TX_labels_read=0,V_reads=0,target_reads=0,seconds=time.perf_counter()-started)
    if row is not None:base.write_json(row/'teacher_cache_reuse.json',record)
    return folder,record


def smoke(args,p):
    clean,roles=base.source_training(p);take=np.flatnonzero(roles=='L_s')[:2]
    folder,_=cached_teacher(args,p)
    reference=base.tensor(np.load(folder/'epoch01.npy',mmap_mode='r',allow_pickle=False)[take],'cpu')
    cache=p['cache']/'epochs/01'
    values={k:base.tensor(np.load(cache/(k+'.npy'),mmap_mode='r',allow_pickle=False)[take],'cpu')
        for k in ('received','kernel','coefficients','variance')}
    x=base.tensor(clean[take],'cpu');student=scratch_model(2026092701)
    generator=torch.Generator().manual_seed(2026103004)
    probes=[torch.complex(torch.randn(2,256,generator=generator),torch.randn(2,256,generator=generator)) for _ in range(2)]
    loss,_=hint.objective(student,values['received'],x,base.supervised_action(values['kernel'],values['coefficients']),
        values['variance'],probes,reference,1.)
    loss.mean().backward();gradients=[p.grad for p in student.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(v).all() for v in gradients) or not any(v.abs().sum()>0 for v in gradients):
        raise ValueError('Finite nonzero full-coarse source gradients required')
    if not any(p.grad is not None and p.grad.abs().sum()>0 for p in student.frozen.coarse.parameters()):
        raise ValueError('New full coarse must receive source gradients')
    student.eval()
    with torch.no_grad():expected=student(values['received']).clone()
    checkpoint=p['output']/'smoke-scratch_joint.pt';base.save_model(checkpoint,student,'scratch_joint',0,0,args.commit)
    fresh=base.load_model(checkpoint,'cpu',arm='scratch_joint',commit=args.commit,epoch=0)
    with torch.no_grad():torch.testing.assert_close(fresh(values['received']),expected,atol=0,rtol=0)
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    with torch.no_grad():scores=reader(expected)
    if scores.shape!=(2,6) or not torch.isfinite(scores).all():raise ValueError('Original F no-query smoke failed')
    base.write_json(p['output']/'smoke.json',dict(status='PASS',real_Ls=2,student_scratch=True,
        parameters=PARAMETERS,new_coarse_gradient_nonzero=True,strict_reload=True,source_initialization_checkpoint_loads=0,
        source_teacher_forwards=0,source_teacher_checkpoint_loads=0,cached_source_IQ_hint=True,
        TX_labels_read=0,target_reads=0,training_F_access=False,F_updates=0,head_fit_updates=0))
    print('{"event":"REAL_WIDE_COARSE_CACHED_HINT_NOQUERY_SMOKE_PASS"}',flush=True)


def configure():
    hint.configure();previous_configuration=base.configuration;previous_command=base.command
    def configuration():
        value=previous_configuration()
        value.update(run=RUN,recipe='scratch_E30_fullwidthcoarse_cachedIQhint',parameters=PARAMETERS,
            coarse_parameters=1305958,compact_coarse_parameters=106138,
            coarse_spectral_widths=[24,48,72,96],coarse_temporal_widths=[32,64,96,128],
            architecture_change='Onlycoarsewidths;compactaction/priorstatesandRNGpreserved;nocoarsestatecopied',
            teacher_cache_reused_from=CACHE_RUN,teacher_cache_inputs=0,
            historic_teacher_cache_inputs=630000,historic_teacher_component_epochs=20,historic_teacher_bundle_component_epochs=60,
            training_F_access=False,source_F_informed_method_research=True,
            allowed_GPUs=[6],source_rationale='Compactcoarse106138params vsoriginal1305958;matchedhintstudentcoarsecleanNMSE~.056vsoriginalteacher~.024;capacityhypothesis not causal proof',
            comparison_control=CACHE_RUN,compare_also_default266951=True,
            evaluation_resource_queue=dict(predecessor_module=PREDECESSOR_MODULE,predecessor_commit=PREDECESSOR_COMMIT,
                wait_until_actual_worker_and_scorer_exit=True))
        return value
    def command(args,mode,arm=None,role=None):
        return [MODULE if v==hint.MODULE else v for v in previous_command(args,mode,arm,role)]
    def train(args,p):
        hint.train(args,p);gc.collect();torch.cuda.empty_cache();queue.wait_evaluation(args,p)
    queue.PREDECESSOR_MODULE=PREDECESSOR_MODULE;queue.PREDECESSOR_COMMIT=PREDECESSOR_COMMIT
    queue.PREDECESSOR_RELEASE=PREDECESSOR_RELEASE
    base.RUN=RUN;base.configuration=configuration;base.command=command;base.scratch_model=scratch_model
    hint.scratch_model=scratch_model;hint.prepare_teacher_cache=cached_teacher
    base.smoke=smoke;base.train=train


if __name__=='__main__':configure();base.main()
