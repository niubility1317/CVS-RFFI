"""Source-only audit of executed concat CE; never consumes target metrics."""
import math

FAMILIES=('riei','drift','ibrff','sdg')
SCENES={'clean_duplicate','leo_clear_weak','leo_low_elev_weak','leo_rain_weak'}


def audit_stage(c,stage,records):
    issues=[]
    required=dict(satellite_train_augmentation=True,concat_sat_ce_only=True,lambda_sat_cls=.68,
                  lambda_sat_cons=0.,sat_ce_start=1,default_head_joint_training=True,separate_head_training=False)
    issues += ['config:'+k for k,v in required.items() if c.get(k)!=v]
    if not records:return dict(status='FAIL',issues=issues+['empty_stage_log'])
    previous=dict(updates=0,physical_source_exposure_total=0,satellite_samples_total=0,
                  satellite_applied_samples_total=0,pl_seen_total=0,pl_accepted_total=0)
    errors=[];scenes=set();first_applied=None
    for index,r in enumerate(records):
        for key in ('loss','identity_ce','satellite_ce','native_penalty','pl_loss'):
            if not isinstance(r.get(key),(int,float)) or not math.isfinite(r[key]):issues.append('nonfinite:'+key)
        if issues and any(s.startswith('nonfinite') for s in issues):continue
        expected=r['identity_ce']+.68*r['satellite_ce']+r['native_penalty']+c['pl_weight']*r['pl_loss']
        error=abs(r['loss']-expected);errors.append(error)
        if error>max(2e-4,3e-6*abs(r['loss'])):issues.append('joint_loss_does_not_equal_CE_plus_.68sat_plus_native_plus_PL')
        if r.get('satellite_weight')!=.68 or r.get('concat_sat_ce_only') is not True:issues.append('logged_recipe')
        if r.get('target_access') is not False or r.get('pseudo_label_truth_access') is not False:issues.append('source_permissions')
        if r.get('U_s_use') is not c['U_s_use']:issues.append('U_flag')
        for key,before in previous.items():
            if r.get(key,-1)<before:issues.append('nonmonotone:'+key)
        exposure=r['physical_source_exposure_total'];sat=r['satellite_samples_total'];applied=r['satellite_applied_samples_total']
        if exposure!=sat or not 0<=applied<=sat:issues.append('concat_exposure_count')
        if r['identity_ce']<0 or r['satellite_ce']<=0:issues.append('invalid_CE')
        if not c['U_s_use'] and (r['pl_seen_total'] or r['pl_accepted_total'] or r['pl_loss']):issues.append('supervised_U_access')
        if not 0<=r['pl_accepted_total']<=r['pl_seen_total']:issues.append('PL_exposure')
        if applied and first_applied is None:first_applied=index+1
        scenes.update(r['satellite_scenarios'])
        previous={k:r[k] for k in previous}
    last=records[-1]
    comparisons={'updates':'updates','physical_source_exposure_total':'labeled_exposure',
        'satellite_samples_total':'satellite_examples','satellite_applied_samples_total':'satellite_applied_examples',
        'pl_seen_total':'pl_seen','pl_accepted_total':'pl_accepted','optimizer_steps_total':'optimizer_steps'}
    issues += ['freeze_total:'+key for key,other in comparisons.items() if last.get(key)!=stage.get(other)]
    if len(records)!=stage['log_intervals']:issues.append('stage_record_count')
    if not scenes<=SCENES:issues.append('unexpected_scene')
    if last['satellite_applied_samples_total']<=0:issues.append('no_actual_channel_examples')
    if c['U_s_use'] and last['pl_seen_total']<=0:issues.append('PL_never_iterated')
    return dict(status='FAIL' if issues else 'PASS',issues=sorted(set(issues)),stage=stage['stage'],
        records=len(records),updates=last['updates'],labeled_exposure=last['physical_source_exposure_total'],
        satellite_examples=last['satellite_samples_total'],applied_examples=last['satellite_applied_samples_total'],
        applied_fraction=last['satellite_applied_samples_total']/last['satellite_samples_total'],
        first_channel_interval=first_applied,first_satellite_CE=records[0]['satellite_ce'],
        last_satellite_CE=last['satellite_ce'],max_loss_reconstruction_error=max(errors,default=0.),
        scenes=sorted(scenes),pl_seen=last['pl_seen_total'],pl_accepted=last['pl_accepted_total'])


def audit_selected(c,cost):
    t=cost.get('selected_training_telemetry',{});issues=[]
    if not 0<t.get('satellite_applied_examples',0)<=t.get('satellite_examples',0):issues.append('selected_checkpoint_has_no_channel_training')
    for key in ('satellite_examples','satellite_applied_examples','pl_seen','pl_accepted'):
        if not 0<=t.get(key,-1)<=cost.get(key,-1):issues.append('selected_counter:'+key)
    if c['U_s_use'] and t.get('pl_seen',0)<=0:issues.append('selected_checkpoint_has_no_U')
    if not c['U_s_use'] and (t.get('pl_seen',0) or t.get('pl_accepted',0)):issues.append('selected_supervised_has_U')
    return dict(status='FAIL' if issues else 'PASS',issues=issues,selected_epoch=cost.get('selected_epoch'),
        selected_telemetry=t)
