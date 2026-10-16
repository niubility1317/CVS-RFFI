"""Reconstruct actual ordinary-augmentation schedules; no IQ/model execution."""
import ast,json
from pathlib import Path
from types import SimpleNamespace
from typing import Any,Dict

ROOT=Path(__file__).resolve().parents[2]
LOCAL=ROOT/'local_artifacts/F_crossrx_satmix_confirm20_20261016'


def main():
    actual=json.loads((LOCAL/'actual_callable_audit.json').read_text(encoding='utf8'))
    snapshot=json.loads((LOCAL/'latest.json').read_text(encoding='utf8'))
    args=snapshot['workers'][0]['native_args']
    assert args['seed']==2026092702 and args['epochs']==20 and args['use_aug'] and not args['use_concat_sat_channel_aug'] and args['late_stable_start']==0
    schedule=actual['sources']['configure_augmentor']
    functions={n.name:ast.get_source_segment(schedule,n) for n in ast.parse(schedule).body if isinstance(n,ast.FunctionDef)}
    scope=dict(Any=Any,Dict=Dict)
    for name in ('ramp_value','build_aug_base_cfg','configure_augmentor_for_epoch'):exec(compile(functions[name],name,'exec'),scope)
    namespace=SimpleNamespace(**args)
    base=scope['build_aug_base_cfg'](namespace)
    old=json.loads((ROOT/'local_artifacts/F_crossrx_satmix20_20261016/final/B00-RX1-s2026092701/source/ordinary_augmentation_actual.json').read_text(encoding='utf8'))
    assert base==dict(old['base_config'],seed=2026092702)
    epochs=[]
    for epoch in range(1,21):
        aug=SimpleNamespace(dac=SimpleNamespace(),pa=SimpleNamespace())
        state=scope['configure_augmentor_for_epoch'](aug,base,epoch,namespace)
        row=dict(epoch=epoch,scale=state['scale'],probabilities={k:getattr(aug,k) for k in base if k.startswith('p_')},max_time_shift=aug.max_time_shift,cfo_max_cycles_per_sample=aug.cfo_max,cfo_max_Hz=25000000*aug.cfo_max,phase_noise_sigma_max_rad=aug.phase_noise_sigma_max,dc_offset_max=aug.dc_offset_max,taper_alpha=[aug.taper_alpha_min,aug.taper_alpha_max],taper_max_edge_samples=int(aug.taper_alpha_max*255/2),mp_delay_max=aug.mp_delay_max,pa_scaled_parameters=vars(aug.pa))
        epochs.append(row)
    aug_code=actual['sources']['augmentation'];tree=ast.parse(aug_code)
    pa=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PaParams')
    pa_defaults={n.target.id:ast.literal_eval(n.value) for n in pa.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name)}
    operators=[
        dict(order=1,name='DC',target='whole batch gate,per-packet I/Q offsets',probability='0.30*scale',range='each I/Q U[-0.02*scale,+0.02*scale]',label_dependence=False,boundary='constant offset,not a new receiver'),
        dict(order=2,name='Tukey edge taper',target='whole batch gate and shared alpha',probability='0.25*scale',range='alpha U[0.02*scale,0.10*scale]',label_dependence=False,boundary='edge=int(alpha*255/2);edge<=1 returns ones;E1-7 always identity,late at most4samples/edge'),
        dict(order=3,name='circular time shift',target='whole batch gate,per-packet shift',probability='0.35*scale',range='integer U[-round(32*scale),+round(32*scale)]',label_dependence=False,boundary='mod256 gather creates wrap seam;max3..11samples;seam outside reference80:160,but content alignment changes'),
        dict(order=4,name='constant amplitude',target='whole batch gate,per-packet gain',probability='0.45*scale',range='U[0.9,1.1],range not ramped',label_dependence=False,boundary='no boundary insertion;no IQ RMS normalization at augmentation exit or native backbone entrance'),
        dict(order=5,name='phase rotation',target='whole batch gate,per-packet phase',probability='0.45*scale',range='U[0,2*pi],range not ramped',label_dependence=False,boundary='whole packet rotation'),
        dict(order=6,name='CFO',target='whole batch gate,per-packet linear phase',probability='0.35*scale',range='U[-0.0004*scale,+0.0004*scale]cycles/sample;1000..3500Hz max',label_dependence=False,boundary='no seam;NoExplicitCFO masks identity coordinate27,internal packet compensation preserved'),
        dict(order=7,name='Wiener phase noise',target='whole batch gate,per-packet sigma',probability='0.30*scale',range='sigma U[0,0.006*scale]rad/sample increment',label_dependence=False,boundary='cumsum increments,starts with random first increment'),
        dict(order=8,name='ordinary multipath',target='whole batch gate,shared sampled taps/delays',probability='0.18*scale',range='2..4complex taps;integer delay0..round(4*scale),max0..1 in this schedule',label_dependence=False,boundary='left reflect pad then conv1d,256output;not cascaded into practical'),
        dict(order=9,name='ordinary AWGN',target='whole batch gate,per-packet SNR',probability='0.40*scale',range='U[20,36]dB,range not ramped;power from signal after preceding operations',label_dependence=False,boundary='complex white noise;not cascaded into practical'),
        dict(order=10,name='DAC gate',target='per-packet mask',probability='0*scale=0',range='disabled',label_dependence=False,boundary='no normal DAC quantization/clipping chain executed'),
        dict(order=11,name='PA gate',target='per-packet mask,subchain operation gates shared in selected subset',probability='0.14*scale=0.014..0.049',range='s=sqrt(U[0,1]);PA polynomial/FIR/AMPM/IQ maxima partly scaled;Rapp sat0.6..1.1 and p2..5 not ramped',label_dependence='normal call passes visible L labels;class_signature=false disables label-conditioned signature',boundary='polynomial zero-pad memory<=2;FIR uses circular roll<=3;Rapp nonlinear compression;unvalidated identity-preserving operation'),
    ]
    value=dict(status='ACTUAL_CODE_AND_ARGS_AUDITED',execution_commit='d71531d53c1d9ec46e3915963e05bff96760e80e',scope='Shared old ordinary L augmentation in qualified seed01 and active confirmation02;not a new Ag efficacy experiment',actual_callable_refs={k:v.rsplit('/releases/',1)[-1] for k,v in actual['actual_callable_files'].items()},base_config=base,resolved_controls={k:args.get(k,'absent;getattr(-1) applies') for k in ('use_aug','aug_enable_pa_normal','aug_enable_class_signature','aug_scale_min','aug_scale_max','aug_warmup_epochs','aug_ramp_epochs','aug_ramp_curve','late_stable_start','late_aug_min_scale','use_concat_sat_channel_aug')},operators=operators,epoch_schedule=epochs,pa_unramped_defaults=pa_defaults,normalization=dict(loader='normalize=True;pad/crop before RMS',ordinary_exit='nan_to_num only,no IQ RMS',native_backbone_entrance='pad_crop_iq then filters,not input RMS',reference='per-packet coherent mean energy and relative transfer normalization;does not establish whole-network gain invariance'),input_evidence=dict(completed_seed01_ordinary_raw_MSE_min=old['input_MSE_min'],completed_seed01_ordinary_raw_MSE_max=old['input_MSE_max'],same_generated_IQ_per_RDrop_pair=True,practical_input='raw x_l,not augmented x_l_main'),limitations=['Configured probabilities and reconstructed schedule,not realized random event counts','PA identity preservation is not proved;its inherited common use is not newAg evidence','Circular boundary is artificial;safe placement does not prove identity preservation','No IQ or model calls in this audit;no identity-performance claim'],source_IQ_reads=0,target_access=False,model_forwards=0,training_updates=0,does_not_change_live_run=True)
    (LOCAL/'ordinary_augmentation_audit.json').write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(status=value['status'],operators=len(operators),epochs=len(epochs),scale_first=epochs[0]['scale'],scale_last=epochs[-1]['scale'],nontrivial_taper_first_possible_epoch=next(v['epoch'] for v in epochs if v['taper_max_edge_samples']>1),source_IQ_reads=0,model_forwards=0,identity_efficacy_proved=False)))
    return value


if __name__=='__main__':main()
