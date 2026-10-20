"""Same-run original50 source ancestry and fixed default-head checkpoints."""
from pathlib import Path
from types import SimpleNamespace
from . import design as d
from .baseline import assert_recipe
from experiments.cvs_evidence_responsibility_train20.provenance import RECIPE_FIELDS

def validate_args(actual,c):
 actual=actual.get('args',actual);assert_recipe(SimpleNamespace(**actual));expected=d.make_args(c,'cpu')
 if any(actual.get(k)!=getattr(expected,k) for k in RECIPE_FIELDS) or actual.get('seed')!=c['model_seed'] or not actual.get('a1_source_screen_only') or actual.get('sat_training_mode')!='disabled' or actual.get('eval_sat_channel'):raise ValueError('Actual original50/no-satellite native recipe differs')

def source_proof(c):
 d.validate(c);out=Path(c['output_root']);done=d.read(out/'local_source_complete.json');original=d.read(out/'completion.json');initial=d.read(out/'initialization.json');freeze=d.read(out/'source_frozen.json');aux=d.read(out/'supcon_training_observation.json');sinc=d.read(out/'sinc_optimizer_observation.json')
 expected=dict(status='SOURCE_FROZEN',config=c,epoch=50,optimizer_steps=11100,head_fit_updates=0,source_roles='EXACT_MATCH',initialization='scratch',ancestors=[],U_TX_truth_hidden=True,satellite_training=False,default_head_joint=True,target_access=False)
 if any(done.get(k)!=v for k,v in expected.items()) or any(original.get(k)!=v for k,v in dict(status='SOURCE_TRAINED',config=c,initialization='scratch',ancestors=[],checkpoint_sources=[],target_access=False,target_evaluated=False,head_fit_updates=0,logged_steps=11100,optimizer_steps=11100).items()):raise ValueError('SOURCE_PROVENANCE_UNVERIFIED')
 if initial!=dict(scratch_only=True,checkpoint_sources=[],ancestors=[],source_roles='EXACT_MATCH',target_contact=False,teacher_origin='own scratch student only',seed=c['model_seed']) or freeze.get('config')!=c or freeze.get('epoch')!=50 or freeze.get('selection')!='final_only' or freeze.get('target_access') is not False:raise ValueError('Own scratch student/EMA and final50 required')
 expected_aux=dict(status='VERIFIED',mode='CROSS_RX',calls=11100,applied_calls=9990,actual_layer_forward_objectives=39960,extra_model_forwards=0,head_fit_updates=0,U_truth_read=False)
 if any(aux.get(k)!=v for k,v in expected_aux.items()) or sinc.get('mode')!='CENTER_UNITS' or sinc.get('steps')!=11100 or sinc.get('physical_clip_calls')!=11100 or done['actual_source_packet_reads']['U_s']<963900:raise ValueError('Original feature/Sinc/U budget incomplete')
 aux_new=d.read(out/'decoupled_training_observation.json')
 if any(aux_new.get(k)!=v for k,v in dict(status='VERIFIED',treatment=c['treatment'],calls=11100,applied_calls=9990,actual_layer_forward_objectives=39960,extra_model_forwards=0,extra_L_packets=0,extra_optimizer_updates=0,extra_parameters=0,head_fit_updates=0,default_head_direct_aux_gradient=False,U_truth_read=False,temperature=.2,layers=['time_down','t2.pool']).items()) or aux_new['valid_anchors']!=aux['valid_anchors']:raise ValueError('Actual decoupled contrast execution differs')
 g=done['sourceV']['geometry']
 if g['head_fit_updates'] or g['prototype_classifier_decisions'] or g['count']!=27000 or g['class_names']!=list(d.CLASSES):raise ValueError('Source descriptive geometry differs')
 validate_args(d.read(out/'actual_native_args.json'),c);return done

def validate_checkpoint(checkpoint,c):
 if any(checkpoint.get(k)!=v for k,v in dict(run_id=d.RUN,candidate_id=c['row_id'],epoch=50,checkpoint_selection='final_only',a1_ema_successful_updates=11100).items()) or not checkpoint.get('ema_model'):raise ValueError('Own native final50 checkpoint/EMA required')
 validate_args(checkpoint['args'],c)
