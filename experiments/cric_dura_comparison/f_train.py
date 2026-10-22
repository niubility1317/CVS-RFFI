"""Run the frozen native trainer with contract-bound loaders and CRIC hooks."""
from contextlib import contextmanager
from copy import copy
import gzip
import json
import os
from pathlib import Path
import time
import torch
from .io import read, write
from .f_data import context
from .f_model import install


def original_config(seed):
    from experiments.cvs_f_supcon50 import design
    return design.config(dict(row_id='CROSS_RX-s%d' % seed, arm='CROSS_RX', model_seed=seed))


def arguments(c, device):
    from experiments.cvs_f_supcon50 import design
    a = design.make_args(original_config(c['model_seed']), device)
    a.run_id, a.candidate_id, a.output_dir = c['run_id'], c['row_id'], c['output_root']
    a.wisig_pkl = ''  # Data construction is replaced; no dataset access in worker.
    a.wisig_train_rxs = ','.join(map(str, c['source_rxs']))
    a.wisig_train_days = ','.join(map(str, c['source_days']))
    a.num_classes = c['num_classes']
    a.num_domains = len(c['source_rxs']) * len(c['source_days'])
    a.wisig_test_rxs = ''
    a.wisig_test_days = ''
    a.epochs, a.label_epochs, a.pseudo_epochs = 50, 33, 17
    # The native U mechanism is retained only for the original complete method.
    # Matched ordinary PL is implemented explicitly in the labeled-step hook.
    if c['regime'] != 'native_cric' or c.get('ablation') == 'F15':
        a.use_unlabeled = a.use_ema_teacher = False
        a.lambda_u = a.lambda_ent = 0.
        a.label_epochs, a.pseudo_epochs = 50, 0
    for ablation, key, value in [('F11', 'use_ema_teacher', False),
        ('F12', 'pseudo_temporal_gate', False), ('F13', 'pseudo_strong_agreement', False),
        ('F14', 'lambda_ent', 0.), ('F15', 'label_smoothing', 0.)]:
        if c.get('ablation') == ablation:
            setattr(a, key, value)
    a.a1_gradient_snapshot = True
    a.source_validation_reuse = True
    a.use_concat_sat_channel_aug = False
    a.lambda_sat_cls = a.lambda_sat_cons = 0.
    a.sat_training_mode = 'disabled'
    a.use_sat_consistency = False
    if c.get('method_id')=='pamf_t1_e50':
        a.epochs=c.get('segment_epochs',20)
        a.label_epochs=13 if not c.get('step_offset',0) else 0
        a.pseudo_epochs=a.epochs-a.label_epochs
        a.use_unlabeled=a.use_ema_teacher=c['regime']=='native_cric' and c.get('ablation') not in ('F15','F20','F21')
        a.lambda_u=.16 if a.use_unlabeled else 0.
        a.lambda_ent=.01 if a.use_unlabeled and c.get('ablation')!='F14' else 0.
        if c.get('ablation')=='F11':a.use_ema_teacher=False
        if not a.use_unlabeled:a.label_epochs,a.pseudo_epochs=a.epochs,0
        if c.get('ablation')=='F15':a.label_smoothing=0.
        a.from_scratch=not bool(c.get('parent_source'))
        a.a1_scratch_only=a.from_scratch
        a.baseline_ckpt=str(Path(c['parent_source'])/'final_ssdg.pth') if c.get('parent_source') else ''
        a.teacher_ckpt=''
        a.lambda_dom=0.
    return a


@contextmanager
def class_dimension(c):
    """Set the native scratch cosine output size before any training forward."""
    import experiments.cvs_phase1_stack.runtime as runtime
    original = runtime.Phase1IdentityAdapter
    def construct(a):
        dims = copy(a)
        dims.num_classes = 6
        return original(dims)
    runtime.Phase1IdentityAdapter = construct
    try:
        yield
    finally:
        runtime.Phase1IdentityAdapter = original


def train(c, device='cuda:0', smoke_steps=None):
    if any(c.get(k) for k in ('target_truth', 'truth', 'views_root', 'tests', 'dataset_path', 'checkpoint', 'teacher_checkpoint')):
        raise ValueError('Only source cache and scratch model permitted')
    out = Path(c['output_root'])
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_phase1_stack.fast_execution import incremental_native_writer, batch_clean
    from .f_optimizer import native_scope
    ctx = context(c)
    a = arguments(c, device)
    is_t1=c.get('method_id')=='pamf_t1_e50'
    parent=None
    if c.get('parent_source'):
        previous=read(Path(c['parent_source'])/'complete.json')
        if not is_t1 or previous['source_contract']!=c['source_contract'] or previous['config']['model_seed']!=c['model_seed'] or previous['config']['ablation']!=c.get('ablation'):
            raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
        parent=torch.load(Path(c['parent_source'])/'final_ssdg.pth',map_location='cpu',weights_only=False)
        if previous['target_access'] or parent.get('ema_model') is None and a.use_ema_teacher:raise ValueError('Own segment EMA provenance missing')
    if smoke_steps is not None:
        if device != 'cpu' or smoke_steps != 2:
            raise ValueError('Only the bounded local CPU integration smoke may shorten execution')
        a.epochs, a.label_epochs, a.pseudo_epochs = 2, 1, 1 if a.use_unlabeled else 0
        if not a.use_unlabeled:
            a.label_epochs = 2
        ctx['train_loader'].steps = 2
        if ctx['unlabeled_loader']:
            ctx['unlabeled_loader'].steps = 2
    torch.manual_seed(c['model_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(c['model_seed'])
    owner = type('OptimizerOwner', (), {})()
    owner.optimizer = None
    state = dict(steps=0)
    start = time.perf_counter()
    with class_dimension(c), installed(original_config(c['model_seed'])) as native:
        saved = {name: getattr(native, name) for name in ('_build_ssdg_wisig_data', 'build_baseline_model', '_detach_log_mapping', '_write_ssdg_epoch_telemetry', '_evaluate_zid_leakage_probes', '_muse_epoch_pairs')}
        saved.update({name:getattr(native,name) for name in ('deepcopy','configure_augmentor_for_epoch','load_checkpoint','_resolve_phase1_terminal_status')})
        if is_t1:
            def copied(value,*args,**kwargs):
                result=saved['deepcopy'](value,*args,**kwargs)
                if value is getattr(owner,'model',None) and parent is not None and a.use_ema_teacher:
                    result.load_state_dict(parent['ema_model'],strict=True)
                return result
            native.deepcopy=copied
            def own_load(path,dev):
                if a.baseline_ckpt and str(path)==a.baseline_ckpt:return parent
                if Path(path).parent!=out:raise ValueError('Unexpected external checkpoint read')
                return saved['load_checkpoint'](path,dev)
            native.load_checkpoint=own_load
            if c.get('step_offset',0):
                from types import SimpleNamespace
                prior=SimpleNamespace(**parent['args']);prior.epochs=20;prior.label_epochs=13;prior.pseudo_epochs=7
                native.configure_augmentor_for_epoch=lambda obj,cfg,epoch,args:saved['configure_augmentor_for_epoch'](obj,cfg,13,prior)
            def terminal(**kwargs):
                kwargs['mechanism_gates_required']=False
                return saved['_resolve_phase1_terminal_status'](**kwargs)
            native._resolve_phase1_terminal_status=terminal
        def pairs(labeled, unlabeled, **kwargs):
            # Fixed L has exactly 222 steps already. A supervised row must not
            # open a dummy U dataset just to supply the old U-derived length.
            for batch in labeled:
                yield batch, None
        native._muse_epoch_pairs = pairs
        def data(*args, **kwargs):
            write(out / 'source_contract.json', ctx['source_contract'])
            return ctx
        def model(args, dev):
            installer=install
            if is_t1:
                from .t1_model import install as installer
            value = installer(saved['build_baseline_model'](args, dev), c).to(dev)
            if parent is not None:value.load_state_dict(parent['model'],strict=True)
            if c['regime'] == 'matched_sat_plain_pl' or c.get('ablation')=='F21':
                value._cd_U_loader = ctx['unlabeled_loader']
                value._cd_U_iter = iter(ctx['unlabeled_loader'])
            owner.model = value
            # Real scratch checkpoint round trip, using source IQ and no query.
            path = out / 'initial_smoke.pt'
            torch.save(dict(model=value.state_dict(), config=c, ancestors=[]), path)
            value.load_state_dict(torch.load(path, map_location=dev, weights_only=False)['model'], strict=True)
            value.eval()
            with torch.no_grad():
                logits = value(ctx['train_loader'].dataset.x[:2].to(dev))
            if logits.shape != (2, c['num_classes']) or not bool(torch.isfinite(logits).all()):
                raise ValueError('Actual-source checkpoint smoke failed')
            write(out / 'smoke.json', dict(status='PASS', source_samples=2, query_reads=0,
                TX_labels_U_read=0, parameters=sum(p.numel() for p in value.parameters()), default_head_classes=c['num_classes'],
                source_IQ_binding={role:loader.dataset.iq_binding for role,loader in
                    (('L_s',ctx['train_loader']),('U_s',ctx['unlabeled_loader']),('V',ctx['val_loader'])) if loader is not None}))
            value.train()
            return value
        metrics = gzip.open(out / 'step_metrics.jsonl.gz', 'xt', encoding='utf-8', compresslevel=3)
        def detach(values):
            result = saved['_detach_log_mapping'](values)
            if 'train/loss' in result:
                state['steps'] += 1
                compact = batch_clean(result)
                if compact.get('train/skipped_nonfinite_loss', 0) or compact.get('train/skipped_nonfinite_grad', 0):
                    raise FloatingPointError('Nonfinite native update')
                if float(compact.get('train/pseudo_truth_available', 0) or 0) != 0:
                    raise ValueError('U TX truth exposed')
                metrics.write(json.dumps(dict(step=state['steps'], metrics=compact), allow_nan=False) + '\n')
            return result
        def telemetry(cp, jp, rows):
            saved['_write_ssdg_epoch_telemetry'](cp, jp, rows)
            m = owner.model
            write(out / 'progress.json', dict(epoch=len(rows), updates=state['steps'],
                cumulative_epoch=len(rows)+c.get('epoch_offset',0),cumulative_updates=state['steps']+c.get('step_offset',0),
                CRIC_calls=m._cd_calls, extra_L=m._cd_extra_L,
                PL_seen=m._cd_U_seen, PL_accepted=m._cd_U_accepted,
                L_actual=ctx['train_loader'].actual_examples,
                U_actual=ctx['unlabeled_loader'].actual_examples if ctx['unlabeled_loader'] else 0,
                seconds=time.perf_counter() - start), exclusive=False)
            print(json.dumps(dict(event='CRIC_EPOCH_COMPLETE', epoch=len(rows), updates=state['steps'])), flush=True)
        native._build_ssdg_wisig_data = data
        native.build_baseline_model = model
        native._detach_log_mapping = detach
        native._write_ssdg_epoch_telemetry = telemetry
        native._evaluate_zid_leakage_probes = lambda *a, **kw: dict(status='DISABLED_DEFAULT_HEAD_ONLY', head_fit_updates=0)
        try:
            with native_scope(owner, 'CENTER_UNITS'):
                status = native.train(a)
            if status:
                raise RuntimeError('Native source trainer failed')
        finally:
            for name, value in saved.items():
                setattr(native, name, value)
            metrics.close()
    value = torch.load(out / 'final_ssdg.pth', map_location='cpu', weights_only=False)
    expected_steps, expected_epoch = (4, 2) if smoke_steps is not None else (a.epochs*222, a.epochs)
    if state['steps'] != expected_steps or owner.model._cd_calls != expected_steps or value['epoch'] != expected_epoch or value['a1_ema_successful_updates'] != expected_steps:
        raise ValueError('Complete fixed F budget missing')
    write(out / 'complete.json', dict(status='LOCAL_SYNTHETIC_SMOKE_ONLY' if smoke_steps is not None else 'SOURCE_FROZEN', epoch=expected_epoch, updates=expected_steps,
        model_calls=owner.model._cd_calls, source_contract=c['source_contract'], config=c,
        source_roles=ctx['source_contract']['counts'], checkpoint='final_ssdg.pth',
        initialization='own_previous_segment' if parent is not None else 'scratch', ancestors=[c['parent_source']] if parent is not None else [], head_fit_updates=0, U_truth_read=False,
        target_access=False, seconds=time.perf_counter() - start))
