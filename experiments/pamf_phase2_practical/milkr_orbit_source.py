"""Offline source-only C4 encoder, fixed rank64, source-calibrated K1 rule.

The new feature is tested once on a fixed small target corner matrix. Source
validation chooses only the kernel rule, without a retrospective gain gate.
"""
from pathlib import Path
import time
import numpy as np
from .io import read,write,require
from .milkr_screen import initialize,subset
from .milkr.orbit_cca import OrbitCCA
from .milkr import nested_fisher,signature_kernel
from .milkr.support_calibrated import LAMBDAS
from .milkr_identity_source import source_train_indices,scores


def query_scores(rt,state,pair,held):
    if rt['kernel_method'] in ('prior_fusion','confidence_prior','native_skip','time_orbit','episodic_metric','ridge_prior','residual_skip','center_prior','phase8_prior','orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble','rf_class_ensemble'):
        from .milkr.prior_fusion import fuse
        raw=state.evaluate(subset(pair[0],held));restored=state.evaluate(subset(pair[1],held));fused,_=fuse(raw,restored)
        return {0:raw[0],1:restored[0],2:fused}
    return {path:state.score(subset(features,held)) for path,features in enumerate(pair)}


def evaluate_frames(rt,frames,fit_K1,fit_many):
    grid=[]
    for config in signature_kernel.CONFIGS:
        for ridge in LAMBDAS:
            rule=dict(config=list(config),ridge=ridge,outer_folds=3,outer_strengths=[0.,.25,.5,.75],ridge_reference_classes=6);rows=[]
            if rt['kernel_method']=='registered_scale':rule['bandwidth_scope']='registered'
            if rt['kernel_method']=='local_anchor':rule['local_anchor_policy']=rt['local_anchor_policy']
            if rt['kernel_method']=='teacher_kernel':rule['teacher_weight']=rt['teacher_weight']
            if rt['kernel_method'] in ('prior_fusion','confidence_prior','native_skip','time_orbit','episodic_metric','ridge_prior','residual_skip','center_prior','phase8_prior','orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble','rf_class_ensemble'):rule.update(prior_weight=rt['prior_weight'],fusion_policy=rt['fusion_policy'])
            if rt['kernel_method'] in ('confidence_prior','native_skip','time_orbit','episodic_metric','ridge_prior','residual_skip','center_prior','phase8_prior','orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble','rf_class_ensemble'):rule['confidence_rule']=rt['confidence_rule']
            if rt['kernel_method']=='center_prior':rule['support_center_strength']=rt['support_center_strength']
            if rt['kernel_method'] in ('orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble','rf_class_ensemble'):rule['orbit_ensemble_weight']=rt['orbit_ensemble_weight']
            if rt['kernel_method']=='density_orbit_ensemble':rule['kernel_density_rule']=rt['kernel_density_rule']
            if rt['kernel_method'] in ('kernel_gate_ensemble','rf_class_ensemble'):rule['rf_prior_gate_rule']=rt['rf_prior_gate_rule']
            if rt['kernel_method']=='rf_class_ensemble':rule['rf_class_prior_rule']=rt['rf_class_prior_rule']
            for frame in frames:
                support=np.arange(6)*52
                for sp in (0,1):
                    state=fit_K1(subset(frame['mapped'][sp],support),np.arange(6),[frame['ids'][i] for i in support],old_count=4,fixed=rule,conditional_loo=False)
                    for qp,prediction in query_scores(rt,state,frame['mapped'],frame['held']).items():rows.append(dict(rx=frame['rx'],k=1,S=sp,Q=qp,**scores(np.repeat(np.arange(6),32),prediction.argmax(1))))
            objective=[r['H'] for r in rows if rt['kernel_method'] not in ('prior_fusion','confidence_prior','native_skip','time_orbit','episodic_metric','ridge_prior','residual_skip','center_prior','phase8_prior','orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble','rf_class_ensemble') or r['Q'] in (0,2)]
            grid.append(dict(K1_rule=rule,source_H=float(np.mean(objective)),rows=rows))
    best=max(enumerate(grid),key=lambda item:(item[1]['source_H'],item[1]['K1_rule']['ridge'],-item[0]))[1]
    validation=list(best['rows'])
    for frame in frames:
        support=np.concatenate([np.arange(c*52,c*52+20) for c in range(6)])
        for sp in (0,1):
            state=fit_many(subset(frame['mapped'][sp],support),np.repeat(np.arange(6),20),[frame['ids'][i] for i in support],old_count=4,fixed=best['K1_rule'])
            for qp,prediction in query_scores(rt,state,frame['mapped'],frame['held']).items():validation.append(dict(rx=frame['rx'],k=20,S=sp,Q=qp,**scores(np.repeat(np.arange(6),32),prediction.argmax(1))))
        print('ORBIT_SOURCE_VALIDATION',frame['rx'],flush=True)
    return best,validation,grid


def calibrate(rt):
    from .milkr_cv_screen import extract_for_method
    from experiments.cric_dura_comparison.f_data import SourceDataset
    started=time.perf_counter();reader,engine=initialize(rt)
    data=SourceDataset(rt['source_cache'],'L_s',read(rt['source_contract']));m=data.metadata;banks=[[],[]]
    distribution=rt['kernel_method']=='phase_kernel'
    if distribution:
        from .milkr.phase_encoder import PhaseEncoder
        from .milkr.phase_kernel import fit as phase_fit
    encoder_type=PhaseEncoder if distribution else OrbitCCA
    fit_K1=phase_fit if distribution else signature_kernel.fit
    fit_many=phase_fit if distribution else nested_fisher.fit
    if rt['kernel_method']=='local_anchor':
        from .milkr.local_anchor import fit as local_fit,POLICY
        require(rt.get('local_anchor_policy')==POLICY,'Frozen local-anchor policy differs')
        fit_K1=local_fit;fit_many=local_fit
    if rt['kernel_method']=='teacher_kernel':
        from .teacher_features import TeacherBridge
        from .milkr.teacher_kernel import fit as teacher_fit,WEIGHT
        require(rt.get('teacher_weight')==WEIGHT,'Frozen teacher kernel weight differs')
        encoder_type=TeacherBridge;fit_K1=teacher_fit;fit_many=teacher_fit
    if rt['kernel_method'] in ('prior_fusion','confidence_prior','native_skip','time_orbit','episodic_metric','ridge_prior','residual_skip','center_prior','phase8_prior','orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble','rf_class_ensemble'):
        from .teacher_features import TeacherBridge
        from .milkr.prior_fusion import fit as prior_fit,PRIOR_WEIGHT,FUSION_POLICY
        require(rt.get('prior_weight')==PRIOR_WEIGHT and rt.get('fusion_policy')==FUSION_POLICY,'Frozen prior/fusion settings differ')
        if rt['kernel_method'] in ('confidence_prior','native_skip','time_orbit','episodic_metric','ridge_prior','residual_skip','center_prior','phase8_prior','orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble','rf_class_ensemble'):
            from .milkr.support_prior_gate import RULE
            if rt['kernel_method'] in ('ridge_prior','residual_skip','center_prior','phase8_prior','orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble','rf_class_ensemble'):
                from .milkr.support_ridge_prior import RULE
            if rt['kernel_method']=='class_orbit_ensemble':
                from .milkr.support_class_prior import RULE
            require(rt.get('confidence_rule')==RULE,'Frozen support-confidence rule differs')
        from .teacher_features import bridge_type
        encoder_type=bridge_type(rt['kernel_method']);fit_K1=prior_fit;fit_many=prior_fit
    if rt['kernel_method'] in ('orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble','rf_class_ensemble'):
        from .milkr.orbit_ensemble import fit as ensemble_fit
        require(rt.get('orbit_ensemble_weight')==.5,'Fixed equal orbit kernels required')
        if rt['kernel_method']=='class_orbit_ensemble':require(rt.get('class_prior_projection')=='simplex','Fixed class prior simplex projection required')
        fit_K1=ensemble_fit;fit_many=ensemble_fit
    if rt['kernel_method']=='density_orbit_ensemble':
        from .milkr.kernel_density import RULE as DENSITY_RULE
        require(rt.get('kernel_density_rule')==DENSITY_RULE,'Wrong frozen support kernel density rule')
    if rt['kernel_method'] in ('kernel_gate_ensemble','rf_class_ensemble'):
        from .milkr.support_kernel_gate import RULE as RF_RULE
        require(rt.get('rf_prior_gate_rule')==RF_RULE and rt.get('rf_gate_ridge_fraction')==.1,'Wrong RF prior evidence rule')
    if rt['kernel_method']=='rf_class_ensemble':
        from .milkr.support_rf_class_prior import RULE as RF_CLASS_RULE
        require(rt.get('rf_class_prior_rule')==RF_CLASS_RULE,'Wrong frozen RF class prior rule')
    require(rt['source_cache_epochs']==[91,92] and rt['orbit_quarters']==[0,1,2,3] and rt['orbit_rank']==64,'Wrong preregistered orbit configuration')
    if rt['kernel_method']=='episodic_metric':require(rt.get('metric_steps')==600 and rt.get('metric_gain_bounds')==[.25,4.],'Wrong source episodic metric budget')
    if rt['kernel_method'] in ('ridge_prior','residual_skip','center_prior','phase8_prior','orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble','rf_class_ensemble'):require(rt.get('calibration_ridge_fraction')==.1,'Wrong fixed support calibration ridge')
    if rt['kernel_method'] in ('phase8_prior','orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble','rf_class_ensemble'):require(rt.get('RF_phase_eighths')==list(range(8)),'Fixed C8 RF orbit required')
    if rt['kernel_method']=='center_prior':require(rt.get('support_center_strength')==.5,'Fixed half RF centering required')
    if rt['kernel_method']=='time_orbit':require(rt.get('time_shifts')==[0,64,128,192],'Fixed four cyclic shifts required')
    if rt['kernel_method']=='residual_skip':require(rt.get('residual_skip_weight')==.5 and rt.get('FFT_complement') is False and rt.get('frequency_embedding_complement') is False,'Fixed invariant-block CCA-only and other-block half-complement required')
    if rt['kernel_method']=='native_skip':require(rt.get('native_skip_weight')==.5,'Fixed equal native/CCA energy required')
    if rt['kernel_method']=='registered_scale':require(rt.get('bandwidth_scope')=='registered','Registered support scale required')
    for epoch in rt['source_cache_epochs']:
        folder=Path(rt['augmentation_cache'])/'F0'/f'stage1-epoch{epoch:03d}-p80'
        require(read(folder/'complete.json')['status']=='L_AUGMENTATION_READY','Missing source augmentation')
        require(np.array_equal(np.load(folder/'ids.npy'),m['ids']),'Source physical alignment differs')
        iq=np.asarray(np.load(folder/'x.npy',mmap_mode='r'))
        for path,g in enumerate((None,engine)):banks[path].append(extract_for_method(rt,reader,iq,'cuda:0',g)[0])
        print('ORBIT_SOURCE_EPOCH',epoch,len(iq),flush=True)
    features=[{key:np.concatenate([part[key] for part in parts]) for key in parts[0]} for parts in banks]
    rx_all=np.tile(m['rx_i'],2);source_y=np.tile(m['y'],2);ids_all=np.tile(m['ids'],2);frames=[];control_frames=[];folds=[];metric_training=[]
    for index,rx in enumerate(sorted(np.unique(m['rx_i']).tolist())):
        held_tx=[index,index+1];old_tx=[c for c in range(6) if c not in held_tx]
        train=source_train_indices(rx_all,source_y,rx,held_tx)
        training_options=dict(labels=source_y[train],receivers=rx_all[train],physical_ids=ids_all[train],seed=rt['source_proxy_seed']+rx*10000+303) if rt['kernel_method']=='episodic_metric' else {}
        encoder=encoder_type.fit(*[subset(f,train) for f in features],**training_options);rng=np.random.default_rng(rt['source_proxy_seed']+rx*100)
        if rt['kernel_method']=='episodic_metric':metric_training.append(dict(held_RX=int(rx),**encoder.cca.training_summary))
        groups=[rng.permutation(np.flatnonzero((m['rx_i']==rx)&(m['y']==c)))[:52] for c in old_tx+held_tx]
        require(all(len(g)==52 for g in groups),'Insufficient source holdout count')
        positions=np.concatenate(groups);held=np.concatenate([np.arange(c*52+20,(c+1)*52) for c in range(6)])
        options=dict(old_columns=old_tx) if rt['kernel_method'] in ('teacher_kernel','prior_fusion','confidence_prior','native_skip','time_orbit','episodic_metric','ridge_prior','residual_skip','center_prior','phase8_prior','orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble','rf_class_ensemble') else {}
        mapped=[encoder.transform(subset(f,positions),restored=bool(path),**options) for path,f in enumerate(features)]
        frames.append(dict(rx=int(rx),mapped=mapped,ids=[str(m['ids'][int(p)]) for p in positions],held=held))
        if rt['kernel_method']=='native_skip':
            from .teacher_features import TeacherBridge
            control=TeacherBridge.fit(*[subset(f,train) for f in features])
            control_mapped=[control.transform(subset(f,positions),restored=bool(path),**options) for path,f in enumerate(features)]
            control_frames.append(dict(rx=int(rx),mapped=control_mapped,ids=[str(m['ids'][int(p)]) for p in positions],held=held))
        folds.append(dict(receiver=int(rx),held_TX_indices=held_tx,train_TX_indices=old_tx,training_observations=len(train),eval_epoch=91))
        print('ORBIT_SOURCE_FOLD',rx,flush=True)
    best,validation,grid=evaluate_frames(rt,frames,fit_K1,fit_many)
    eligible=True;control_evidence=None
    if rt['kernel_method'] in ('native_skip','time_orbit','episodic_metric'):
        if rt['kernel_method']=='native_skip':
            control_best,control_validation,_=evaluate_frames(rt,control_frames,fit_K1,fit_many)
        else:
            control=read(rt['source_control_ref'])
            require(control['status']=='SOURCE_FROZEN' and control['feature_method']=='confidence_prior' and control['source_primary_query_paths']==[0,2], 'Wrong source-only temporal control')
            require(all(control[key]==0 for key in ('target_reads','query_reads','U_truth_reads','F_updates','G_updates')), 'Control is not source-only frozen')
            control_best=dict(K1_rule=control['K1_rule']);control_validation=control['source_summary']
        primary=lambda rows:np.mean([r['H'] for r in rows if r['Q'] in (0,2)])
        gain=100*float(primary(validation)-primary(control_validation));raw_gains=[]
        for k in (1,20):
            for sp in (0,1):
                mean=lambda rows:np.mean([r['H'] for r in rows if r['k']==k and r['S']==sp and r['Q']==0])
                raw_gains.append(dict(k=k,S=sp,delta_pp=100*float(mean(validation)-mean(control_validation))))
        eligible=bool(gain>=.5 and min(r['delta_pp'] for r in raw_gains)>=-.25)
        control_evidence=dict(K1_rule=control_best['K1_rule'],source_summary=control_validation,primary_H_gain_pp=gain,raw_K_S_gains=raw_gains,
            threshold=dict(primary_H_gain_pp=.5,each_raw_K_S_min_pp=-.25),pass_gate=eligible)
    path=None;residual_summary=None
    if eligible:
        training_options=dict(labels=source_y,receivers=rx_all,physical_ids=ids_all,seed=rt['source_proxy_seed']+303) if rt['kernel_method']=='episodic_metric' else {}
        final=encoder_type.fit(*features,**training_options);path=Path(rt['run_root'])/('source_phase_cca.npz' if distribution else 'source_orbit_cca.npz');final.save(path);loaded=encoder_type.load(path)
        if rt['kernel_method']=='residual_skip':
            residual_summary=dict(schema='source_cca_complement_skip_v1',weight=.5,
                parameters=sum(a.size for p in final.cca.cca.parameters for a in p)+sum(a.size for p in final.cca.parameters if p is not None for a in p),
                FFT_complement=False,frequency_embedding_complement=False,FFT_rule='original_CCA_only',frequency_embedding_rule='original_CCA_only',
                native_orthogonality_norms=[float(np.linalg.norm(np.concatenate((p[2],p[3]),axis=1).T@native[1])) for p,native in zip(final.cca.cca.parameters,final.cca.parameters) if native is not None],
                source_examples_retained=0)
        if rt['kernel_method']=='episodic_metric':metric_training.append(dict(held_RX=None,**final.cca.training_summary))
        for sp in (0,1):
            small=subset(features[sp],slice(0,12))
            require(all(np.array_equal(value,loaded.transform(small,restored=bool(sp))[key]) for key,value in final.transform(small,restored=bool(sp)).items()),'Orbit encoder reload differs')
    write(Path(rt['run_root'])/'source_selection.json',dict(status='SOURCE_FROZEN',K1_rule=best['K1_rule'],target_eligible=eligible,
        scientific_verdict=('SOURCE_FEATURE_GATE_PASS' if eligible else 'SOURCE_FEATURE_GATE_FAIL') if rt['kernel_method'] in ('native_skip','time_orbit','episodic_metric') else 'FIXED_FEATURE_CORNER_FALSIFICATION_NO_SOURCE_GAIN_GATE',source_control=control_evidence,source_metric_training=metric_training,RF_phase_eighths=rt.get('RF_phase_eighths',[0,2,4,6]),RF_kernel_orbits=([4,8] if rt['kernel_method'] in ('orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble','rf_class_ensemble') else [8] if rt['kernel_method']=='phase8_prior' else [4]),head_phase_quarters=[0,1,2,3],time_shifts=rt.get('time_shifts',[0]),source_control_ref=rt.get('source_control_ref'),encoder_checkpoint_created=bool(path),selected_rank=64,orbit_quarters=[0,1,2,3],source_K1_grid=grid,source_summary=validation,
        source_RX_TX_folds=folds,source_residual_skip=residual_summary,source_physical_rows=len(m['ids']),source_paired_observations=len(features[0]['z_id']),
        source_phase_pairs=len(features[0]['z_id'])*(4 if distribution else 1),feature_method=rt['kernel_method'],source_held_new_head_columns_removed_before_softmax=rt['kernel_method'] in ('teacher_kernel','prior_fusion','confidence_prior','native_skip','time_orbit','episodic_metric','ridge_prior','residual_skip','center_prior','phase8_prior','orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble','rf_class_ensemble'),
        source_primary_query_paths=[0,2] if rt['kernel_method'] in ('prior_fusion','confidence_prior','native_skip','time_orbit','episodic_metric','ridge_prior','residual_skip','center_prior','phase8_prior','orbit_ensemble','class_orbit_ensemble','density_orbit_ensemble','kernel_gate_ensemble','rf_class_ensemble') else [0,1],
        target_reads=0,query_reads=0,U_truth_reads=0,F_updates=0,G_updates=0,seconds=time.perf_counter()-started,source_numerical_smoke='PASS',
        scope='source old4/new2; encoder fit excludes held RX/TX but F saw all source6; fixed RFphaseorbit/rank64 not targetselected; no source examples/statistics used by target worker'))
