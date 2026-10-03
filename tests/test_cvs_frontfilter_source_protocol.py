"""Source-only architecture protocol and complete measured-log reconciliation."""
import copy,csv,io,json,math
import pytest
from experiments.cvs_frontfilter_identity.model import VARIANTS,filter_contract
from experiments.cvs_frontfilter_identity.source import validate_config
from experiments.cvs_frontfilter_identity.dispatch import CANDIDATES,SEEDS,select_source_candidate
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY

def config(variant):
    return dict(method='cvs_frontfilter_identity',variant=variant,epochs=200,batch_size=128,lr=.0002,
                lr_min=1e-6,weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,
                extra_losses=[],selection='fixed_last_epoch',split_seed=392005,
                frontfilter=filter_contract(variant),numerical_policy=FULL_FP32_POLICY.copy(),model_seed=2026092701,
                dataset='/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl',
                source_contract='/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json')

@pytest.mark.parametrize('variant',VARIANTS)
def test_original_ce_training_contract(variant):
    c=config(variant)
    assert validate_config(c)==c


def test_baseline_source_record_without_frontfilter_uses_its_own_validator(tmp_path):
    from experiments.cvs_frontfilter_identity import dispatch
    from experiments.cvs_neural_residual_identity.model import neural_contract

    seed=2026092701;variant='neural_residual_shallow';method='cvs_neural_residual_identity'
    resolved=config(VARIANTS[0]);resolved.pop('frontfilter')
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
    assert 'frontfilter' not in resolved
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
    if corruption=='architecture':c['frontfilter']['new_trainable_parameters']=999
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
    from experiments.cvs_frontfilter_identity import prepare,dispatch
    source=tmp_path/'experiments/cvs_neural_residual_identity/configs/experiment_spec.json'
    source.parent.mkdir(parents=True)
    sample=config(VARIANTS[0])
    source.write_text(json.dumps(dict(code={},permissions={},checkpoint={},execution={},metrics_plan={},
        data=dict(contract_ref=sample['source_contract'],dataset=sample['dataset']))),encoding='utf-8')
    monkeypatch.setattr(prepare,'ROOT',tmp_path)
    monkeypatch.setattr(prepare.subprocess,'check_output',lambda *a,**k:'a'*40+'\n')
    prepare.main()
    directory=tmp_path/'experiments/cvs_frontfilter_identity/configs'
    spec=json.loads((directory/'experiment_spec.json').read_text(encoding='utf-8'))
    launch=json.loads((directory/'launch_spec.json').read_text(encoding='utf-8'))
    assert len(spec['rows'])==len(launch['rows'])==8
    assert len(launch['source_controls'])==8
    assert spec['checkpoint']['initialization']=='scratch_only' and spec['checkpoint']['sources']==[]
    assert spec['test_completion_plan']['total_rows']==48 and spec['test_completion_plan']['reused_rows']==44
    assert spec['test_completion_plan']['candidate_universe']==list(CANDIDATES)
    assert 'attention' not in json.dumps(spec).lower()
    assert json.loads((directory/'local_parameter_counts.json').read_text())=={v:220987+filter_contract(v)['new_trainable_parameters'] for v in VARIANTS}
    for row in launch['rows']:
        path=directory/(row['row_id']+'.json');cfg=json.loads(path.read_text())
        assert validate_config(cfg)==cfg and cfg['frontfilter']==filter_contract(row['variant'])
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
    from experiments.cvs_frontfilter_identity import publish
    cfg,data=preflight_fixture(publish)
    monkeypatch.setattr(publish,'ssh',lambda script:json.dumps(data).encode('utf-8'))
    assert publish.preflight(cfg)==data
    compile(publish.PREFLIGHT.replace('c=CONFIG','c='+repr(cfg),1),'preflight','exec')
    compile(publish.REMOTE.replace('c=CONFIG','c='+repr(cfg),1),'submit','exec')
    compile(publish.INSPECT.replace('PROJECT',repr(publish.PROJECT)).replace('RELEASE',repr(publish.RELEASE)).replace('RUN',repr(publish.RUN)),'inspect','exec')


@pytest.mark.parametrize('issue',['release','archive','run','log','identity','disk','gpu','unverified_paths'])
def test_publisher_refuses_existing_or_unverified_pretransfer_state(monkeypatch,issue):
    from experiments.cvs_frontfilter_identity import publish
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
    from experiments.cvs_frontfilter_identity import publish
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
