"""Bounded aggregate checks; real parent-CP smoke remains the launcher's first step."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import torch
from . import design as d, source, optimizer


def _reject(call):
    try:
        call()
    except ValueError:
        return 1
    raise AssertionError('Invalid fixed configuration or optimizer scope was accepted')


def _canonical(value):
    if isinstance(value,float) and math.isnan(value):return '<native NaN sentinel>'
    if isinstance(value,dict):return {k:_canonical(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_canonical(v) for v in value]
    return value


def configuration_checks():
    # Four material mutations exercise each registered row without
    # expanding the matrix or evaluating any experiment outcome.
    fixed = {
        'NOAS_ADAPTER': (True,False,'raw',1.,0.,.25,0.),
        'S1': (True,True,'raw',.5,.5,0.,0.),
        'S2': (True,True,'raw',.5,.5,.25,0.),
        'S3': (True,True,'raw',.5,.5,.25,.25),
    }
    negatives=0
    for row in d.rows():
        c=d.config(row);d.validate(c);arm=c['arm']
        actual=(c['adapter']['enabled'],c['As'],c['first_view'],c['loss']['clean_CE_weight'],
            c['loss']['satellite_CE_weight'],c['loss']['clean_KD_weight'],c['loss']['satellite_KD_weight'])
        if (actual!=fixed[arm] or c['default_head_trainable'] or c['original_F_trainable']
                or c['adapter']['parameters']!=4256 or c['adapter']['state_tensors']!=8
                or (c['epochs'],c['updates'],c['parent_epochs'],c['parent_updates'],
                    c['cumulative_epochs'],c['cumulative_updates'])!=(10,2220,20,4440,30,6660)):
            raise AssertionError('Fixed four-row raw-IQ/head/loss/continuation matrix differs')
        bad=deepcopy(c)
        if arm=='NOAS_ADAPTER':bad['As']=True
        elif arm=='S1':bad['original_F_trainable']=True
        elif arm=='S2':bad['parent_updates']=2220
        else:bad['loss']['satellite_KD_weight']=.26
        negatives+=_reject(lambda bad=bad:d.validate(bad))
    return dict(status='PASS',rows=list(fixed),protocol_negatives=negatives,
        fixed_parent20_segment10_cumulative30=True,default_head_frozen_all_rows=True)


def native_argument_checks():
    from experiments.cvs_F_nontrue20 import design as original
    parent_args=vars(original.make_args(d.parent_config(),'cpu'))
    changed={}
    for row in d.rows():
        c=d.config(row)
        # Calling both public paths proves design's forwarding reaches source
        # without the source entry calling design.make_args recursively.
        actual=vars(source.make_args(c,'cpu'))
        forwarded=vars(d.make_args(c,'cpu'))
        if _canonical(actual)!=_canonical(forwarded):
            raise AssertionError('Public design/source native argument interfaces differ')
        expected=dict(parent_args)
        expected.update(run_id=c['run_id'],candidate_id=c['row_id'],output_dir=c['output_root'],
            epochs=10,label_epochs=0,pseudo_epochs=10,
            baseline_ckpt=str(Path(c['parent_source'])/'final_ssdg.pth'),
            from_scratch=False,freeze_backbone=False,a1_scratch_only=False,lr=.0002,weight_decay=.0001,
            phase1_lr_schedule='cosine',phase1_lr_min=.000001,a1_ema_startup_average=False)
        if _canonical(actual)!=_canonical(expected):
            keys=sorted(k for k in set(actual)|set(expected)
                        if _canonical(actual.get(k))!=_canonical(expected.get(k)))
            raise AssertionError('Unexpected change from native B0 arguments: '+','.join(keys))
        if (actual['a1_scratch_only'] or actual['from_scratch'] or not actual['baseline_ckpt']
                or actual['teacher_ckpt'] or not actual['use_unlabeled'] or not actual['use_ema_teacher']
                or not actual['use_aug'] or actual['use_concat_sat_channel_aug']
                or actual['pseudo_temporal_mode']!='batch_neighbor'
                or actual['batch_size']!=128 or actual['label_smoothing']!=.01):
            raise AssertionError('Original native U/EMA/augmentation/visible-L recipe differs')
        changed[c['arm']]=sorted(k for k in actual
            if _canonical(actual[k])!=_canonical(parent_args.get(k)))
    from .observer import continuation_terminal_status
    example=dict(mechanism_gates_required=True,finite=True,predictions_legal=True)
    result=continuation_terminal_status(lambda **kw:kw,**example)
    if result!=dict(example,mechanism_gates_required=False) or example['mechanism_gates_required'] is not True:
        raise AssertionError('Only obsolete full-mechanism promotion scope may be disabled')
    return dict(status='PASS',rows=len(d.rows()),design_forwarding_nonrecursive=True,
        inherited_scratch_only_disabled=True,legacy_terminal_scope_adapted=True,
        original_native_arguments_preserved_except=changed,
        native_epochs=10,label_epochs=0,pseudo_epochs=10,parent_epochs=20,
        cumulative_epochs=30,optimizer_reset=True,schedule='newcosine10',
        U='original hiddenTX batch_neighbor/strong/ownEMA',CP_reads=0,IQ_reads=0)


class CoordinateSinc(torch.nn.Module):
    """Parameter-only fixture; never performs an IQ or model forward."""
    def __init__(self):
        super().__init__()
        self.low_coordinate=torch.nn.Parameter(torch.ones(24,1))
        self.band_coordinate=torch.nn.Parameter(torch.ones(24,1))
        self.sample_rate=25000000.;self.scale_hz=25000.;self.mode='CENTER_25K'
        self.register_buffer('low_origin_hz',torch.ones(24,1)*100000.)
        self.register_buffer('band_origin_hz',torch.ones(24,1)*10000.)


def _parameter_fixture(adapter):
    model=torch.nn.Module();model.id_backbone=torch.nn.Module()
    model.id_backbone.encoder=torch.nn.Module();model.id_backbone.encoder.core=torch.nn.Module()
    body=model.id_backbone.encoder.core.id_backbone=torch.nn.Module()
    body.sinc=CoordinateSinc();body.body=torch.nn.Parameter(torch.ones(3))
    body.cls_head=torch.nn.Module()
    body.cls_head.weight=torch.nn.Parameter(torch.ones(6,160),requires_grad=False)
    body.cls_head.gain=torch.nn.Parameter(torch.ones(1),requires_grad=False)
    model._dta_original_parameter_names=tuple(dict(model.named_parameters()))
    if adapter:
        model.requires_grad_(False)
        model.id_backbone.encoder.raw_iq_adapter=torch.nn.Module()
        model.id_backbone.encoder.raw_iq_adapter.fixture=torch.nn.Parameter(torch.ones(4256))
        model._dta_adapter_names=('id_backbone.encoder.raw_iq_adapter.fixture',)
    return model


def optimizer_checks():
    negatives=0;reports={}
    for arm in ('S1',):
        c=d.config(next(row for row in d.rows() if row['arm']==arm))
        model=_parameter_fixture(c['adapter']['enabled'])
        params=[p for p in model.parameters() if p.requires_grad]
        options=dict(lr=c['optimizer']['lr'],weight_decay=c['optimizer']['weight_decay'])
        owner=SimpleNamespace(model=model,c=c,optimizer=None,optimizer_created=lambda opt:None)
        with optimizer.native_scope(owner):
            opt=torch.optim.AdamW(params,**options)
        head=[p for name,p in model.named_parameters() if '.cls_head.' in name]
        selected={id(p) for group in opt.param_groups for p in group['params']}
        if (opt.state or any(p.requires_grad or p.grad is not None or id(p) in selected for p in head)
                or selected!={id(p) for p in params}):
            raise AssertionError('Reset optimizer must exclude every frozen classifier parameter')
        if (len(opt.param_groups)!=1 or sum(p.numel() for p in params)!=4256
                or any(p.requires_grad for p in model.get_submodule(optimizer.PREFIX[:-1]).parameters())):
            raise AssertionError('Only raw-IQ 4256 parameters may enter ordinary AdamW')
        negatives+=_reject(lambda:optimizer.specification(model,params[:-1],c,options))
        head[-1].requires_grad_(True)
        bad_params=[p for p in model.parameters() if p.requires_grad]
        negatives+=_reject(lambda:optimizer.specification(model,bad_params,c,options))
        head[-1].requires_grad_(False)
        reports[arm]=dict(mode=opt._sinc_unit_policy['mode'],head_excluded=True,
            reset_state_entries=len(opt.state),trainable_parameters=sum(p.numel() for p in params))
    return dict(status='PASS',modes=reports,protocol_negatives=negatives,
                model_forwards=0,optimizer_updates=0,CP_reads=0,IQ_reads=0)


def run():
    from . import checks_model, checks_distillation, adapters_checks
    configuration=configuration_checks()
    arguments=native_argument_checks()
    model=checks_model.run()
    adapters=adapters_checks.run()
    distillation=checks_distillation.run()
    opt=optimizer_checks()
    results=dict(configuration=configuration,native_arguments=arguments,
                 model=model,adapters=adapters,distillation=distillation,optimizer=opt)
    if any(result['status']!='PASS' for result in results.values()):
        raise AssertionError('A constituent check did not pass')
    return dict(status='PASS',**results,
        protocol_negatives=sum(result.get('protocol_negatives',0) for result in results.values()),
        checkpoints_loaded=0,real_IQ_packets=0,remote_actions=0,real_training_updates=0,
        real_parent_checkpoint_smoke='launcher first step; PASS immediately continues native train')


if __name__=='__main__':
    print(json.dumps(run(),ensure_ascii=False,allow_nan=False))
