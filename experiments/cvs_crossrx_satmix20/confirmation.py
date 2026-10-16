"""Bounded independent-seed confirmation of the qualified source recipe."""
from pathlib import Path
import math

PROFILE='confirmation_seed02'
RUN='20261016-phase1-F-crossRX-satmix25-confirm-sourceRX1-e20-s2026092702-m2-r01'
RELEASE='cvs_F_crossRX_satmix25_confirm_source_e20_20261016_r01'
OWNER='codex/root/F-crossRX-satmix25-confirm-20261016'
SEED=2026092702
FIRST='20261016-phase1-F-crossRX-satmix25-sourceRX1-e20-s2026092701-m2-r01'
LOCAL='local_artifacts/F_crossrx_satmix_confirm20_20261016'


def activate():
    from . import design as d
    d.prior.SEED=SEED
    d.SEED=SEED
    d.RUN=RUN;d.RELEASE=RELEASE;d.OWNER=OWNER
    d.BASE=Path(d.PROJECT)/'runs'/RUN
    return d


def checks():
    from . import design as d
    before=d.config(d.rows()[0]);args_before=vars(d.make_args(before,'cpu')).copy()
    d=activate()
    assert d.base_model_config()['model_seed']==SEED
    result=[];negative=0
    for row in d.rows():
        c=d.config(row);args=vars(d.make_args(c,'cpu'))
        assert c['model_seed']==SEED and args['seed']==SEED
        assert c['run_id']==RUN and args['run_id']==RUN and args['candidate_id']==row['row_id']
        assert c['output_root']==args['output_dir'] and RUN in c['output_root'] and FIRST not in c['output_root']
        assert c['training_mechanism']==dict(before['training_mechanism'],As=row['arm']=='B01')
        assert c['channel']==before['channel'] and c['supcon']==before['supcon'] and c['fold']==before['fold']
        assert c['source_selection']==before['source_selection'] and not args['baseline_ckpt'] and not args['teacher_ckpt']
        # Two unused native feasibility defaults are NaN in both recipes.
        # Treat those identical sentinels as equal without changing execution.
        diff={k for k,v in args.items() if v!=args_before.get(k) and not (isinstance(v,float) and math.isnan(v) and isinstance(args_before.get(k),float) and math.isnan(args_before[k]))}
        assert diff=={'seed','run_id','candidate_id','output_dir'}, sorted(diff)
        for path,value in ((('model_seed',),2026092701),(('epochs',),50),(('test_views',),['clean']),(('training_mechanism','raw_weight'),.25)):
            from copy import deepcopy
            bad=deepcopy(c);target=bad
            for key in path[:-1]:target=target[key]
            target[path[-1]]=value
            try:d.validate(bad)
            except ValueError:negative+=1
            else:raise AssertionError('Unregistered confirmation configuration accepted')
        result.append(dict(row_id=row['row_id'],model_seed=args['seed'],native_argument_changes=sorted(diff),no_checkpoint_initialization=True,same_data_channel_loss_U_budget=True))
    assert activate().rows()==d.rows()
    # Unsupported immutable runtime profiles must fail before dependencies or SSH.
    import json,tempfile
    from . import runner,publish
    with tempfile.TemporaryDirectory() as temporary:
        fake=Path(temporary)/'experiments/cvs_crossrx_satmix20/runner.py'
        fake.parent.mkdir(parents=True)
        (Path(temporary)/'runtime.json').write_text(json.dumps(dict(AF='unused',profile='unregistered')),encoding='utf8')
        original=runner.__file__;runner.__file__=str(fake)
        try:
            try:runner.bootstrap()
            except ValueError:negative+=1
            else:raise AssertionError('Unregistered worker profile accepted')
        finally:runner.__file__=original
    try:publish.main(profile='unregistered')
    except ValueError:negative+=1
    else:raise AssertionError('Unregistered publish profile accepted')
    value=dict(status='PASS',profile=PROFILE,rows=result,negative_cases=negative,source_IQ_reads=0,target_access=False,model_forwards=0,training_updates=0)
    d.write(d.ROOT/LOCAL/'local_validation.json',value)
    print(value)
    return value


def prepare():
    d=activate()
    first=d.read(d.ROOT/'automation_reports/CV-SincNet'/FIRST/'source_analysis.json')
    assert first['qualification']['qualified'] and first['source_only'] and not first['target_access'] and not first['joint_goal_complete']
    from .prepare_record import record
    c=record()
    c.update(display_name='F＋跨RX同TX：固定75/25 practical独立seed02确认',description='Unchanged qualified75/25 source recipe, independent model seed2026092702; paired scratchE20 B00/B01, no new IQ/loss/U mechanism or target.',aliases=['F CROSS_RX satmix25 seed02 confirmation'],tags=[*c['tags'],'independent_seed_confirmation'],comparison_group_id='phase1-F-CROSS_RX-satmix25-source-Ag-As')
    c['authorization']='Active joint Ag/As research, original single-seed source criteria passed. Minimal independent seed02 confirmation, E20 per model unchanged; no target or recipe/ratio search.'
    c['metrics_plan']['dimensions']=[v.replace('2026092701',str(SEED)) for v in c['metrics_plan']['dimensions']]
    for row in c['rows']:row['purpose']='independent_seed_source_confirmation'
    c['notes']=[v.replace('First fixed source-only pair','Independent fixed source-only pair') for v in c['notes']]
    c['notes']+=[
        'Confirms '+FIRST+' after complete source-only PASS, not a retry, inherited model, or target-selected source. parent_run_ids remains empty because both models start scratch.',
        'Immutable runtime.json profile=confirmation_seed02 binds both dispatcher and every worker before configuration/model import; underlying timing and F design seed also2026092702. Native seed alone changes; split/channel/receiver/eval seeds remain fixed.',
        'Original lowurban>=+5pp,urbanmean>=+3pp,clean/eachsuburban/seenV>=-.5pp and healthy full budget thresholds all required independently. Failure means no promotion and no seed03/RX/ratio rescue. No automatic full-source or target work.',
        'Previous pair cost separately:40 model epochs/8880updates/9373358 actualNNpackets. Confirmation adds40/8880; no continuation or external teacher. Report aggregate with all seeds including negative confirmation.',
    ]
    d.write(d.ROOT/LOCAL/'spec.json',c)
    print(d.ROOT/LOCAL/'spec.json')
    return c


if __name__=='__main__':
    import sys
    if sys.argv[1:]==['--check']:checks()
    elif sys.argv[1:]==['--prepare']:prepare()
    elif sys.argv[1:]==['--publish']:
        activate()
        from .publish import main
        main(profile=PROFILE)
    else:raise ValueError('Use exactly --check, --prepare or --publish')
