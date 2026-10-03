"""Source-only architecture protocol and complete measured-log reconciliation."""
import copy,csv,io,json,math
import pytest
from experiments.cvs_crosspath_relation_identity.model import VARIANTS,readout_contract
from experiments.cvs_crosspath_relation_identity.source import validate_config
from experiments.cvs_crosspath_relation_identity.dispatch import CANDIDATES,SEEDS,select_source_candidate
from experiments.cvs_crosspath_relation_identity.collect import audit_logs
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY

def config(variant):
    return dict(method='cvs_crosspath_relation_identity',variant=variant,epochs=200,batch_size=128,lr=.0002,
                lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,
                extra_losses=[],selection='fixed_last_epoch',split_seed=392005,
                readout=readout_contract(variant),numerical_policy=FULL_FP32_POLICY.copy(),model_seed=2026092701,
                dataset='/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl',
                source_contract='/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json')

@pytest.mark.parametrize('variant',VARIANTS)
def test_original_ce_training_contract(variant):
    c=config(variant)
    assert validate_config(c)==c


def test_baseline_source_record_without_readout_uses_its_own_validator(tmp_path):
    from experiments.cvs_crosspath_relation_identity import dispatch
    from experiments.cvs_neural_residual_identity.model import neural_contract

    seed=2026092701;variant='neural_residual_shallow';method='cvs_neural_residual_identity'
    resolved=config(VARIANTS[0]);resolved.pop('readout')
    architecture=neural_contract(variant)
    resolved.update(method=method,variant=variant,model_seed=seed,neural=architecture,
        neural_actual=architecture,neural_residual_active=True,classifier_scale=30.,
        total_parameters=220987,steps_per_epoch=50,source_counts={'L_s':6300,'U_s':56700,'V':27000},
        U_s_use='unused',target_access=False,precision='float32',gradient_clipping=None,
        optimizer='AdamW+CosineAnnealingLR',loader_seed=seed,output_root=str(tmp_path),
        source_contract=dispatch.PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json',
        dataset=dispatch.PROJECT+'/Dataset_WigSig/ManySig.pkl',backend_flags=FULL_FP32_POLICY.copy())
    initial=dict(status='SCRATCH',scratch_only=True,checkpoint=None,ancestors=[],checkpoint_sources=[],
        target_access=False,target_contact=False,model_seed=seed,physical_roles='EXACT_MATCH',selection='fixed_last_epoch')
    contract=dict(physical_roles='EXACT_MATCH',dataset_path=resolved['dataset'],
        classes=['14-10','14-7','20-15','20-19','6-15','8-20'],equalized=1,out_len=256,normalize=True)
    metrics=dict(source_val_count=27000,source_val_accuracy=.9,source_val_worst_rx=.9,
        source_val_rx_accuracy={rx:.9 for rx in ('1','3','4','6','8')})
    complete=dict(status='SOURCE_TRAINED',epoch=200,steps=10000,target_access=False,target_evaluated=False,
        final_source_metrics=metrics,backend_flags=FULL_FP32_POLICY.copy())
    profile=dict(total_parameters=220987,gradient_used_parameters=220987,conv_linear_macs_per_sample=100)
    for name,value in [('completion.json',complete),('resolved_config.json',resolved),
                       ('initialization.json',initial),('source_contract.json',contract),('resource_profile.json',profile)]:
        (tmp_path/name).write_text(json.dumps(value),encoding='utf-8')
    assert 'readout' not in resolved
    record=dispatch.read_source_record(dict(variant=variant,model_seed=seed,source_output=str(tmp_path)),contract,method)
    assert record['variant']==variant and record['parameters']==220987
    assert record['accuracy']==.9 and record['source_output']==str(tmp_path)


@pytest.mark.parametrize('key,value',[
    ('checkpoint','old.pt'),('resume','old.pt'),('teacher','old.pt'),('initial_checkpoint','old.pt'),
    ('target_inputs','query.npz'),('target_truth','truth.json'),('p1_truth','truth.json'),('p1_capsule','capsule'),
    ('epochs',199),('batch_size',256),('lr',.001),('lr_min',0.),('weight_decay',0.),('drop_last',True),
    ('augmentation',True),('domain_backbone',True),('extra_losses',['aux']),('selection','best_epoch'),('split_seed',1),
    ('checkpoint_sources',['anchor.pt']),('ancestors',['old.pt']),('teacher_checkpoint','old.pt'),
    ('ema_checkpoint','old.pt'),('target_access',True),('target_metrics',{'accuracy':1.}),
    ('model_seed',1),('model_seed',2026092701.0),('source_contract','other_roles.json'),('dataset','other_data.pkl')])
def test_source_contract_rejects_changed_training_or_inheritance(key,value):
    c=config(VARIANTS[0]);c[key]=value
    with pytest.raises(ValueError):validate_config(c)

@pytest.mark.parametrize('corruption',['architecture','precision','unknown_variant'])
def test_invalid_architecture_or_numerics(corruption):
    c=config(VARIANTS[0])
    if corruption=='architecture':c['readout']['relation_pairs_per_lag']=64
    elif corruption=='precision':c['numerical_policy']['cudnn_allow_tf32']=True
    else:c['variant']='unregistered'
    with pytest.raises(ValueError):validate_config(c)

def records(winner):
    return [dict(variant=v,seed=s,accuracy=.99 if v==winner else .98,
                 worst_rx=.98 if v==winner else .96,parameters=233275 if v==winner else 1,macs=200 if v==winner else 1)
            for v in CANDIDATES for s in SEEDS]

@pytest.mark.parametrize('winner',CANDIDATES)
def test_fixed_source_ranking_beats_cost_and_handles_control(winner):
    chosen=select_source_candidate(records(winner))
    assert chosen['selected_variant']==winner
    assert chosen['new_candidate_selected']==(winner in VARIANTS)
    assert chosen['target_access'] is False and chosen['target_score_used'] is False

@pytest.mark.parametrize('kind',['missing','duplicate_seed','extra'])
def test_source_selection_rejects_incomplete_matrix(kind):
    rows=records(VARIANTS[0])
    if kind=='missing':rows.pop()
    elif kind=='duplicate_seed':rows[-1]['seed']=rows[-2]['seed']
    else:rows.append(copy.deepcopy(rows[-1]))
    with pytest.raises(ValueError):select_source_candidate(rows)


@pytest.mark.parametrize('field,value', [('accuracy',float('nan')),('worst_rx',1.1),
                                      ('macs',float('inf')),('parameters',0)])
def test_source_selection_rejects_invalid_measured_inputs(field,value):
    rows=records(VARIANTS[0]);rows[0][field]=value
    with pytest.raises(ValueError):select_source_candidate(rows)


def test_prepare_builds_only_registered_scratch_source_rows(tmp_path,monkeypatch):
    from experiments.cvs_crosspath_relation_identity import prepare,dispatch
    source=tmp_path/'experiments/cvs_neural_residual_identity/configs/experiment_spec.json'
    source.parent.mkdir(parents=True)
    sample=config(VARIANTS[0])
    source.write_text(json.dumps(dict(code={},permissions={},checkpoint={},execution={},metrics_plan={},
        data=dict(contract_ref=sample['source_contract'],dataset=sample['dataset']))),encoding='utf-8')
    monkeypatch.setattr(prepare,'ROOT',tmp_path)
    monkeypatch.setattr(prepare.subprocess,'check_output',lambda *a,**k:'a'*40+'\n')
    prepare.main()
    directory=tmp_path/'experiments/cvs_crosspath_relation_identity/configs'
    spec=json.loads((directory/'experiment_spec.json').read_text(encoding='utf-8'))
    launch=json.loads((directory/'launch_spec.json').read_text(encoding='utf-8'))
    assert len(spec['rows'])==len(launch['rows'])==8
    assert len(launch['source_controls'])==8
    assert spec['checkpoint']['initialization']=='scratch_only' and spec['checkpoint']['sources']==[]
    assert spec['test_completion_plan']['total_rows']==48 and spec['test_completion_plan']['reused_rows']==44
    assert spec['test_completion_plan']['candidate_universe']==list(CANDIDATES)
    assert 'attention' not in json.dumps(spec).lower()
    assert json.loads((directory/'local_parameter_counts.json').read_text())=={v:233275 for v in VARIANTS}
    for row in launch['rows']:
        path=directory/(row['row_id']+'.json');cfg=json.loads(path.read_text())
        assert validate_config(cfg)==cfg and cfg['readout']==readout_contract(row['variant'])
        row['source_config']=str(path)
    assert dispatch.validate_spec(launch)==launch
    with pytest.raises(FileExistsError):prepare.main()
    launch['target_truth']='forbidden.json'
    with pytest.raises(ValueError):dispatch.validate_spec(launch)


def preflight_fixture(publisher):
    cfg=dict(project=publisher.PROJECT,run=publisher.RUN,release=publisher.RELEASE,archive=publisher.RELEASE+'.tar.gz')
    return cfg,dict(identity=dict(user='szu2070436088',host='dell-DSS8440'),
        paths=[publisher.PROJECT+'/releases/'+publisher.RELEASE,publisher.PROJECT+'/releases/'+cfg['archive'],
               publisher.PROJECT+'/runs/'+publisher.RUN,publisher.PROJECT+'/logs/'+publisher.RUN],
        existing=[],disk_free_bytes=10*1024**3,gpu='0, synthetic-gpu-uuid, 24000\n',apps='')


def test_publisher_preflight_parses_ssh_bytes_and_compiles_scripts(monkeypatch):
    from experiments.cvs_crosspath_relation_identity import publish
    cfg,data=preflight_fixture(publish)
    monkeypatch.setattr(publish,'ssh',lambda script:json.dumps(data).encode('utf-8'))
    assert publish.preflight(cfg)==data
    compile(publish.PREFLIGHT.replace('c=CONFIG','c='+repr(cfg),1),'preflight','exec')
    compile(publish.REMOTE.replace('c=CONFIG','c='+repr(cfg),1),'submit','exec')
    compile(publish.INSPECT.replace('PROJECT',repr(publish.PROJECT)).replace('RELEASE',repr(publish.RELEASE)).replace('RUN',repr(publish.RUN)),'inspect','exec')


@pytest.mark.parametrize('issue',['release','archive','run','log','identity','disk','gpu','unverified_paths'])
def test_publisher_refuses_existing_or_unverified_pretransfer_state(monkeypatch,issue):
    from experiments.cvs_crosspath_relation_identity import publish
    cfg,data=preflight_fixture(publish)
    if issue in ('release','archive','run','log'):data['existing']=[data['paths'][('release','archive','run','log').index(issue)]]
    elif issue=='identity':data['identity']['user']='wrong-user'
    elif issue=='disk':data['disk_free_bytes']=0
    elif issue=='gpu':data['gpu']=''
    else:data['paths']=[]
    monkeypatch.setattr(publish,'ssh',lambda script:json.dumps(data).encode('utf-8'))
    with pytest.raises((ValueError,FileExistsError)):publish.preflight(cfg)


@pytest.mark.parametrize('blocked',[False,True])
def test_publisher_checks_remote_preflight_before_scp(tmp_path,monkeypatch,blocked):
    from experiments.cvs_crosspath_relation_identity import publish
    _,data=preflight_fixture(publish);events=[];oid='b'*40
    def git(command,**kwargs):
        if 'rev-parse' in command:return oid+'\n'
        if 'branch' in command:return 'synthetic-branch\n'
        if 'ls-remote' in command:return oid+'\trefs/heads/synthetic-branch\n'
        if 'ls-files' in command or 'status' in command:return ''
        raise AssertionError(command)
    def ssh(script):
        if 'existing=' in script:
            events.append('preflight')
            if blocked:data['existing']=[data['paths'][1]]
            return json.dumps(data).encode('utf-8')
        events.append('submit')
        return b'{"status":"SUBMITTED","synthetic":true}'
    monkeypatch.setattr(publish.subprocess,'check_output',git)
    monkeypatch.setattr(publish.subprocess,'run',lambda command,**kwargs:events.append(command[0]))
    monkeypatch.setattr(publish,'ssh',ssh)
    if blocked:
        with pytest.raises(FileExistsError):publish.publish(tmp_path)
        assert events==['preflight']
    else:
        publish.publish(tmp_path)
        assert events==['preflight','scp','submit']
        assert json.loads((tmp_path/'preflight.json').read_text())==data

def artifacts(epochs,steps):
    compact=[{k:v for k,v in e.items() if not isinstance(v,dict)} for e in epochs]
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(compact[0]));writer.writeheader();writer.writerows(compact)
    return (epochs,steps,stream.getvalue(),'RESOLVED_CONFIG {}\n'+'\n'.join('EPOCH '+json.dumps(e) for e in epochs)), '\n'.join(json.dumps(r) for r in compact)

@pytest.fixture(scope='module',params=VARIANTS)
def logs(request):
    variant=request.param;npar=233275
    flags=dict(ce_weight=1.,augmentation_active=False,domain_backbone_active=False,pseudo_labels_active=False,
               extra_losses_active=False,learned_readout_active=True,gradient_used_parameters=npar,
               alignment_gradient_used_parameters=0,alignment_gradient_norm=None,cudnn_allow_tf32=False,
               mixture_gradient_used_parameters=2,mix_gradient3=.001,mix_gradient5=-.002,mix_gradient_norm=.00223606797749979,
               readout_gradient_norm=.01,readout_gradient_used_parameters=npar-220987)
    lift=dict(active=True,raw_parameters=[0.,0.],coefficients=[0.,0.],records=[dict(block='behavior.0.conv',packets=28,
        complex_terms=12,actual_envelope_lag=4,actual_phase_lag=4,input_formula_max_abs_error=0.,input_abs_max=3.,
        degree3_relative_input_change_mean=.1,degree5_relative_input_change_mean=.2)])
    diag=dict(alignment_strength=0.,adaptive_input=lift,normalization=dict(blocks=6,records=[dict(block=p+'.'+str(i),
        eligible_packets=28,relative_energy_fraction_max_error=1e-7,relative_energy_fraction_variance_mean=.02)
        for p in ('time','behavior') for i in range(3)]),learned_readout=dict(active=True,records=[dict(block='crosspath.readout',
        packets=28,relative_output_change_mean=.1,projection_norm=1.,compress_norm=1.,
        relation_norm_by_lag=[.4,.5,.6],output_dimension=320,relation_linear_rank=16,relation_pairs_per_lag=56,
        per_projected_channel_normalization=variant=='crosspath_coherence',
        floor_records=[dict(path=p,lag=lag,floor_fraction=0.) for lag in (0,4,8) for p in ('time','behavior')])]))
    steps=[];epochs=[]
    for epoch in range(1,201):
        for batch in range(50):
            steps.append(dict(epoch=epoch,step=(epoch-1)*50+batch+1,clean_ce=.1,total_loss=.1,
                learning_rate=.0002,gradient_norm=2.,source_samples=128 if batch<49 else 28,satellite_samples=0,
                mix_raw3_before=0.,mix_raw5_before=0.,mix_raw3_after=0.,mix_raw5_after=0.,
                mix_coefficient3_before=0.,mix_coefficient5_before=0.,mix_coefficient3_after=0.,mix_coefficient5_after=0.,
                alignment_strength_before=0.,alignment_strength_after=0.,**flags))
        epochs.append(dict(epoch=epoch,optimizer_steps=50,optimizer_steps_total=epoch*50,source_sample_exposure=6300,
            satellite_sample_exposure=0,clean_ce=.1,total_loss=.1,learning_rate=.0002,gradient_norm=2.,
            source_val_count=27000,source_val_accuracy=.9,source_val_rx_accuracy={r:.9 for r in ('1','3','4','6','8')},
            source_val_worst_rx=.9,mix_raw3=0.,mix_raw5=0.,mix_coefficient3=0.,mix_coefficient5=0.,
            alignment_strength=0.,readout_diagnostics=copy.deepcopy(diag),elapsed_seconds=1.,peak_cuda_allocated_bytes=1234,**flags))
    return epochs,steps,npar,variant

def test_all10000_steps_and200epochs_match(logs):
    epochs,steps,npar,variant=logs;data,compact=artifacts(epochs,steps)
    assert audit_logs(*data,npar,0.,compact,expected_phase_lag=4,expected_readout_contract=readout_contract(variant))['steps']==10000


def test_frozen_public_relation_diagnostics_use_30_packets(logs):
    from experiments.cvs_crosspath_relation_identity.collect import validate_relation_diagnostics
    epochs,_,_,variant=logs
    diag=copy.deepcopy(epochs[0]['readout_diagnostics']['learned_readout'])
    diag['records'][0]['packets']=30
    assert validate_relation_diagnostics(diag,30,readout_contract(variant))['relation_pairs_per_lag']==56
    with pytest.raises(ValueError):validate_relation_diagnostics(diag,28,readout_contract(variant))


def test_source_analysis_pairs_complete_e200_and_actual_relation_telemetry(logs):
    from experiments.cvs_crosspath_relation_identity import analyze
    epochs,_,_,_=logs;rows=[]
    for variant in CANDIDATES:
        for index,seed in enumerate(sorted(SEEDS)):
            measured=copy.deepcopy(epochs)
            for epoch in measured:
                epoch['source_val_ce']=.2+.01*index
                epoch['source_val_accuracy']=.901 if variant=='crosspath_coherence' else .9
                epoch['readout_diagnostics']['learned_readout']['records'][0]['per_projected_channel_normalization']=variant=='crosspath_coherence'
            measured[0]['source_val_accuracy']=.999  # Descriptive only: no epoch reselection.
            rows.append(dict(resolved=dict(variant=variant,model_seed=seed),epochs=measured,log_audit=dict(steps=10000)))
    original=copy.deepcopy(rows)
    report=analyze.summarize_source_telemetry(rows)
    assert rows==original
    assert len(report['source_curve_summary'])==800
    assert len(report['readout_gradient_epochs'])==1600
    assert len(report['readout_last_epoch'])==8 and len(report['readout_last_epoch_summary'])==2
    assert all(r['CE_gap_mean']==pytest.approx(.115) for r in report['source_learning_summary'])
    assert all(r['epoch']==200 and r['score_delta_pp']==pytest.approx(.05) and r['V_delta_pp']==pytest.approx(.1)
               for r in report['readout_pairwise_by_seed'])
    assert report['scope']['epoch_reselection'] is False and report['scope']['target_results_read'] is False
    text=analyze.telemetry_report(report)
    assert 'attention' not in text and '28个样本不能代表全源分布' in text
    assert '架构上界' in text and '不是中心化协方差' in text
    for variant in VARIANTS:
        new=next(r for r in rows if r['resolved']['variant']==variant)
        new['epochs'][0]['readout_diagnostics']['learned_readout']['records'][0]['relation_norm_by_lag'][0]=1.1
        with pytest.raises(ValueError):analyze.summarize_source_telemetry(rows)
        new['epochs'][0]['readout_diagnostics']['learned_readout']['records'][0]['relation_norm_by_lag'][0]=.4

@pytest.mark.parametrize('kind',['missing_step','extra_loss','precision','wrong_exposure','wrong_average','stdout',
    'readout_gradient','readout_paths','relation_nan','relation_norm','projection','compress','wrong_rank',
    'wrong_pairs','wrong_normalization','floor_nan','floor_count','old_gate_jump','compact'])
def test_measured_execution_corruption_rejected(logs,kind):
    epochs,steps,npar,variant=copy.deepcopy(logs)
    diag=epochs[73]['readout_diagnostics']['learned_readout']
    if kind=='missing_step':steps.pop(17)
    elif kind=='extra_loss':steps[500]['extra_losses_active']=True
    elif kind=='precision':steps[500]['cudnn_allow_tf32']=True
    elif kind=='wrong_exposure':steps[49]['source_samples']=128
    elif kind=='wrong_average':steps[500]['clean_ce']=steps[500]['total_loss']=.2
    elif kind=='readout_gradient':steps[500]['readout_gradient_used_parameters']=1
    elif kind=='readout_paths':diag['records'][0]['block']='frequency.readout'
    elif kind=='relation_nan':diag['records'][0]['relation_norm_by_lag'][0]=float('nan')
    elif kind=='relation_norm':diag['records'][0]['relation_norm_by_lag'][0]=1.1
    elif kind=='projection':diag['records'][0]['projection_norm']=-1
    elif kind=='compress':diag['records'][0]['compress_norm']=-1
    elif kind=='wrong_rank':diag['records'][0]['relation_linear_rank']=320
    elif kind=='wrong_pairs':diag['records'][0]['relation_pairs_per_lag']=64
    elif kind=='wrong_normalization':diag['records'][0]['per_projected_channel_normalization']=not diag['records'][0]['per_projected_channel_normalization']
    elif kind=='floor_nan':diag['records'][0]['floor_records'][0]['floor_fraction']=float('nan')
    elif kind=='floor_count':diag['records'][0]['floor_records'].pop()
    elif kind=='old_gate_jump':steps[500]['mix_raw3_before']=.1
    data,compact=artifacts(epochs,steps)
    if kind=='stdout':data=(*data[:-1],data[-1].replace('"gradient_norm": 2.0','"gradient_norm": 3.0',1))
    elif kind=='compact':compact=compact.replace('"clean_ce": 0.1','"clean_ce": 0.2',1)
    with pytest.raises(ValueError):audit_logs(*data,npar,0.,compact,expected_phase_lag=4,expected_readout_contract=readout_contract(variant))
