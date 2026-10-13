"""Direct default-classifier prediction; no source extraction or head fitting.

Target truth is opened only by the independent scorer after completion.
"""
import json
import math
from pathlib import Path
import time
from . import design as d

COUNT = d.QUERY_COUNT


def source_proof(c):
    d.validate(c)
    if c['arm'] == 'LEGACY50':
        from experiments.cvs_frequency_context50.predict import source_proof as parent_proof
        return parent_proof(c['source_config'])
    p = Path(c['output_root'])
    done = d.read(p / 'completion.json')
    resolved = d.read(p / 'resolved_config.json')
    init = d.read(p / 'initialization.json')
    frozen = d.read(p / 'source_frozen.json')
    args = d.read(p / 'resolved_native_args.json')
    if (done.get('status') != 'SOURCE_TRAINED' or done.get('config') != c
            or done.get('epoch') != c['epochs'] or done.get('logged_steps') != c['updates']
            or done.get('optimizer_steps') != c['updates'] or done.get('target_access') is not False
            or done.get('target_evaluated') is not False or done.get('checkpoint_sources') != []):
        raise ValueError('Source budget or role differs')
    if (init.get('scratch_only') is not True or init.get('checkpoint_sources') != []
            or init.get('ancestors') != [] or init.get('target_contact') is not False
            or init.get('seed') != c['model_seed'] or init.get('source_roles') != 'EXACT_MATCH'):
        raise ValueError('Scratch provenance differs')
    actual = dict(c['pa_cross_delay'], head='unchanged cosine6x160', pseudo_EMA='historical unmodified', separate_head_fit=False)
    if (any(resolved.get(k) != v for k, v in c.items())
            or resolved.get('identity_input_policy', {}).get('explicit_CFO_identity_input') is not False
            or resolved.get('actual_pa_cross_delay') != actual):
        raise ValueError('Resolved PA cross-delay recipe differs')
    if (args.get('from_scratch') is not True or args.get('baseline_ckpt') != ''
            or args.get('teacher_ckpt') != '' or args.get('label_epochs') != d.LABEL_EPOCHS
            or args.get('pseudo_epochs') != d.PSEUDO_EPOCHS or args.get('epochs') != d.EPOCHS
            or any(args.get(k) != v for k, v in d.LEGACY_FIELDS.items())):
        raise ValueError('Historical pseudo/EMA/cos execution differs')
    expected_freeze = dict(status='SOURCE_FROZEN', config=c, epoch=c['epochs'], updates=c['updates'],
                           checkpoint=str(p / 'final_ssdg.pth'), selection='final_only', target_access=False)
    if frozen != expected_freeze:
        raise ValueError('Final source freeze differs')
    from experiments.cvs_phase1_stack.recover import verify_contract
    verify_contract(d.read(p / 'source_contract.json'), d.read(d.SOURCE))
    return p


def own_model(c, device):
    import torch
    from types import SimpleNamespace
    d.validate(c)
    if c['arm'] == 'LEGACY50':
        from experiments.cvs_frequency_context50.predict import own_model as parent_model
        return parent_model(c['source_config'], device)
    p = source_proof(c)
    from experiments.cvs_phase1_overlay.model import native_modules
    native_modules()  # Resolve native pickled argument classes before loading.
    checkpoint = torch.load(p / 'final_ssdg.pth', map_location='cpu', weights_only=False)
    args = checkpoint['args']
    if (checkpoint.get('epoch') != c['epochs'] or checkpoint.get('a1_ema_successful_updates') != c['updates']
            or checkpoint.get('run_id') != d.RUN or checkpoint.get('candidate_id') != c['row_id']
            or checkpoint.get('checkpoint_selection') != 'final_only'
            or args.get('baseline_ckpt') != '' or args.get('teacher_ckpt') != ''
            or args.get('from_scratch') is not True):
        raise ValueError('Own final PA cross-delay checkpoint differs')
    from .model import installed
    with installed(c, training=False) as native:
        if getattr(native, '_source_feature_auxiliary_hook', None) is not None:
            raise ValueError('Prediction must not install auxiliary training hooks')
        model = native.build_baseline_model(SimpleNamespace(**checkpoint['baseline_args']), torch.device(device))
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.eval().requires_grad_(False)


def baseline_prediction_root(c):
    return Path(d.PROJECT) / 'runs' / d.BASELINE_EVAL_RUN / c['row_id'] / 'prediction'


def baseline_original(c, ids):
    import numpy as np
    if c['arm'] != 'LEGACY50':
        raise ValueError('Only the frozen baseline can reuse old original-head predictions')
    p = baseline_prediction_root(c)
    done = d.read(p / 'complete.json')
    source = c['source_config']
    if (done.get('status') != 'PREDICTIONS_COMPLETE' or done.get('config') != source
            or done.get('count') != COUNT or done.get('views') != list(d.VIEWS)
            or done.get('truth_read') is not False or done.get('query_fit') is not False):
        raise ValueError('Baseline original predictions are incomplete or mismatched')
    expected = dict(status='VERIFIED', config=source, checkpoint=str(Path(source['output_root']) / 'final_ssdg.pth'),
                    own_final50=True, ancestors=[], truth_read=False, query_fit=False, source_roles='EXACT_MATCH')
    if d.read(p / 'provenance.json') != expected:
        raise ValueError('Baseline original prediction provenance differs')
    with np.load(p / 'predictions.npz', allow_pickle=False) as saved:
        if set(saved.files) != {'ids', 'clean'} or not np.array_equal(saved['ids'], ids):
            raise ValueError('Baseline original prediction IDs differ')
        value = saved['clean'].copy()
    validate_decisions(value)
    return value


def validate_decisions(value):
    if (value.shape != (COUNT,) or value.dtype.kind not in 'iu'
            or value.min() < 0 or value.max() >= len(d.CLASSES)):
        raise ValueError('Invalid fixed six-class decisions')


def checkpoint_path(c):
    source = c['source_config'] if c['arm'] == 'LEGACY50' else c
    return str(Path(source['output_root']) / 'final_ssdg.pth')


def frozen_metadata(c):
 return dict(status='DEFAULT_MODEL_FROZEN',config=c,readouts=list(d.READOUTS),checkpoint=checkpoint_path(c),head_type='ResidualPhysicalHead',head_shape=[6,160],head_scale=30.,original_default_W_retained=True,joint_source_training=True,separate_head_fit=False,separate_head_updates=0,backbone_updates_after_freeze=0,all_model_parameters_frozen=True,query_fit=False,truth_read=False)

def validate_default_freeze(c,value):
 if value!=frozen_metadata(c):raise ValueError('Default model freeze or separate-head policy differs')
 return value

def provenance(c,source):
 return dict(status='VERIFIED',config=c,checkpoint=checkpoint_path(c),source_root=str(source),source_roles='EXACT_MATCH',ancestors=[],own_final50=True,original_default_head=True,separate_head_fit=False,model_frozen_before_target_IQ=True,readouts=list(d.READOUTS),truth_read=False,query_fit=False,Phase2_claim=False,baseline_original_reused=c['arm']=='LEGACY50',baseline_prediction_root=str(baseline_prediction_root(c)) if c['arm']=='LEGACY50' else None)

def completion(c):
 return dict(status='PREDICTIONS_COMPLETE',config=c,count=COUNT,views=list(d.VIEWS),readouts=list(d.READOUTS),truth_read=False,query_fit=False,model_frozen_before_target_IQ=True,separate_head_fit=False,separate_head_updates=0,baseline_original_reused=c['arm']=='LEGACY50')

def pipeline():
 from experiments.cvs_default_head.pipeline import DefaultPipeline
 from experiments.cvs_frequency_context50.predict import observation_metadata
 return DefaultPipeline(d,source_proof,own_model,baseline_original,observation_metadata)
def predict(row):return pipeline().predict(row)
def preflight(row):return pipeline().preflight(row)
