from pathlib import Path
from . import design as d
from experiments.cvs_evidence_responsibility_train20 import provenance as original

def validate_args(actual,c):
    d.validate(c);actual=actual.get('args',actual);expected=d.make_args(c,'cpu')
    for key in original.RECIPE_FIELDS:
        if actual.get(key)!=getattr(expected,key):raise ValueError('Actual formal all-source recipe differs: '+key)
    if actual.get('source_fold_held_rx') is not None:raise ValueError('Four-RX source-fold training cannot initialize the all-five-RX experiment')

def validate_evidence(c,done,initial,freeze,resolved,args,contract,expected):
    d.validate(c);p=Path(c['output_root'])
    if (done.get('status')!='SOURCE_TRAINED' or done.get('config')!=c or done.get('epoch')!=20
            or any(done.get(key)!=4440 for key in ('logged_steps','optimizer_steps','L_context_steps'))
            or done.get('target_access') is not False or done.get('target_evaluated') is not False
            or done.get('checkpoint_sources')!=[] or done.get('ancestors')!=[] or done.get('initialization')!='scratch'
            or done.get('baseline_recipe')!=c['training_recipe'] or done.get('label_epochs')!=13 or done.get('pseudo_epochs')!=7):
        raise ValueError('Own full-source scratch20 provenance differs')
    if initial!=dict(scratch_only=True,checkpoint_sources=[],ancestors=[],source_roles='EXACT_MATCH',target_contact=False,teacher_origin='own scratch student only',seed=c['model_seed']):
        raise ValueError('Fresh all-source student/teacher origin required')
    if freeze!=dict(status='SOURCE_FROZEN',config=c,checkpoint=str(p/'final_ssdg.pth'),epoch=20,selection='final_only',target_access=False) or resolved.get('config')!=c:
        raise ValueError('Own final20 source freeze required')
    if (done.get('native_fitted_probes_disabled') is not True or done.get('native_disabled_probe_calls')!=1
            or done.get('head_fit_updates')!=0 or resolved.get('native_fitted_probes_disabled') is not True):
        raise ValueError('Separate fitted heads must remain disabled')
    if any(value.get('identity_input_policy',{}).get('explicit_CFO_identity_input') is not False for value in (done,resolved)):
        raise ValueError('NoExplicitCFO policy differs')
    report=done.get('feature_extractor',{});original.report_validator(report,c)
    placement=report.get('t2_noise_placement',{});mode=c['arm']
    if (resolved.get('actual_feature_extractor')!=report or placement.get('mode')!=mode
            or placement.get('main_t2_dropout_p')!=(0. if mode=='NONE' else .1)
            or placement.get('signature_tap')!=('t2.drop input after original pool' if mode=='CLEAN_SIG' else 'original t2 output')
            or placement.get('signature_gradient_detached') is not False
            or any(placement.get(key)!=0 for key in ('extra_IQ_forwards','extra_rng_draws','extra_parameters'))):
        raise ValueError('Actual branch-specific noise placement differs')
    if done.get('mask_counts') is None or any(done['mask_counts'].values()):raise ValueError('No auxiliary feature masking permitted')
    if done.get('final_source_metrics',{}).get('source_val_count')!=27000:raise ValueError('Full all-five-RX V required')
    validate_args(args,c)
    from experiments.cvs_phase1_stack.recover import verify_contract
    verify_contract(contract,expected)
    return p
def source_proof(c):
    d.validate(c);p=Path(c['output_root'])
    return validate_evidence(c,d.read(p/'completion.json'),d.read(p/'initialization.json'),d.read(p/'source_frozen.json'),
        d.read(p/'resolved_config.json'),d.read(p/'actual_native_args.json'),d.read(p/'source_contract.json'),d.read(d.SOURCE))
def validate_checkpoint(checkpoint,c):
    if (checkpoint.get('run_id')!=d.RUN or checkpoint.get('candidate_id')!=c['row_id'] or checkpoint.get('epoch')!=20
            or checkpoint.get('checkpoint_selection')!='final_only' or checkpoint.get('a1_ema_successful_updates')!=4440
            or not checkpoint.get('ema_model')):raise ValueError('Own final20 student/EMA checkpoint required')
    validate_args(checkpoint.get('args',{}),c)
def own_model(c,device):
    import torch
    from . import model
    p=source_proof(c);checkpoint=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
    validate_checkpoint(checkpoint,c);result=model.build(c,device);result.load_state_dict(checkpoint['model'],strict=True)
    return result.eval().requires_grad_(False)
