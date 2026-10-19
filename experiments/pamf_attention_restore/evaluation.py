from types import SimpleNamespace
from . import design as d
from .runtime import load_checkpoint
from experiments.pamf_endpoint_restore.evaluation import validate_references,qualify_source
from experiments.pamf_fullresolution_restore.evaluation import qualify_source as qualify_pair
from experiments.pamf_unrolled_restore.evaluation import selection_rank
from experiments.pamf_uncertainty_restore.evaluation import VIEWS,predict as original_predict,score as original_score
from experiments.pamf_native_formal.runner import assess_goal
from experiments.pamf_restoration.runner import read_json
from experiments.pamf_restoration.data import write_json


def validate_control(root):
    if (root.name!=d.CONTROL_RUN or read_json(root/'launch.json').get('commit')!=d.CONTROL_COMMIT):
        raise ValueError('Original matching R23 source control required')
    meta=read_json(root/'complete.json');row=root/d.CONTROL_CANDIDATE
    if any(meta.get(k)!=v for k,v in dict(status='ANALYZED',source_only=True,target_rows=[],training_TX_F_access=False).items()):
        raise ValueError('Completed source-only R23 root required')
    complete=read_json(row/'complete.json')
    if any(complete.get(k)!=v for k,v in dict(status='ANALYZED',role='source',all_predictions_before_truth=True,training_TX_F_access=False).items()):
        raise ValueError('Completed truth-last source control required')
    expected=dict(status='SOURCE_TRAINED',epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        prior_updates=4940,coarse_updates=0,action_estimator_updates=0,TX_labels_read=0,F_outputs_read=0,
        F_weights_read=0,target_reads=0,prior_initialization='scratch',inherited_prior_epochs=0)
    if any(read_json(row/'training_complete.json').get(k)!=v for k,v in expected.items()):
        raise ValueError('Fixed fullLU scratch prior-only R23 training contract required')
    cfg=read_json(row/'resolved_config.json')
    if (any(cfg.get(k)!=v for k,v in dict(schema='joint_coarse_v1',candidate=d.CONTROL_CANDIDATE,
            model_seed=d.MODEL_SEED,data_seed=d.DATA_SEED,training_F_access=False,target_training=False,steps=3,CG_total=6).items())
            or cfg.get('parameter_groups',{}).get('prior')!=2507526):
        raise ValueError('Original source control configuration differs')
    summary=read_json(row/d.KIND/'summary.json')
    expected=dict(status='ANALYZED',role='source',count=1800,F_seed=d.MODEL_SEED,original_default_head=True,
        training_TX_labels_read=0,training_F_access=False,target_access=False,all_predictions_before_truth=True,head_fit_updates=0)
    if any(summary.get(k)!=v for k,v in expected.items()) or not (row/d.KIND/'final.pt').is_file():
        raise ValueError('Original F50 source control summary and whole checkpoint required')
    return summary


def row_args(args):return SimpleNamespace(**dict(vars(args),output=args.output/args.row,kind=d.KIND))


def prepare_target(args):
    value=read_json(args.output/'source_decision.json');selected=value.get('selected')
    if (value['status']!='SOURCE_DECISION_FROZEN' or not value['qualified'] or selected not in d.ROWS
            or value['target_rows']!=[selected] or args.row!=selected):
        raise ValueError('Only source-qualified selected fullresolution row may access target')
    root=args.output/args.row;root.mkdir(exist_ok=True)
    write_json(root/'source_decision.json',dict(value,target_candidates=[d.KIND]))


def predict(args,role):
    if role=='target':prepare_target(args)
    elif args.row not in d.ROWS:raise ValueError('Registered source row required')
    def loader(path,device):
        model,payload=load_checkpoint(path,device)
        if payload['candidate']!=args.row:raise ValueError('Wrong fullresolution checkpoint row')
        return model,payload
    original_predict(row_args(args),role,checkpoint_loader=loader)


def score(args,role):
    original_score(row_args(args),role)
    if role=='target':
        summary=read_json(args.output/args.row/'target_confirmation'/d.KIND/'summary.json')
        write_json(args.output/args.row/'goal_assessment.json',assess_goal(summary))


def select_source(candidates,single,native,fft,unrolled,alternating,control):
    if set(candidates)!=set(d.ROWS):raise ValueError('Both registered source rows must finish')
    checks={}
    for name,summary in candidates.items():
        five=qualify_source(single,summary,native,fft,unrolled,alternating)
        paired=qualify_pair(control,summary,native)
        checks[name]=dict(five,qualified=bool(five['qualified'] and paired['qualified']),
            waveform_A=bool(five['waveform_A'] and paired['waveform_A']),
            source_identity_B=bool(five['source_identity_B'] and paired['source_identity_B']),vs_reused_R23_control=paired)
    ranking=sorted([name for name in d.ROWS if checks[name]['qualified']],key=lambda name:selection_rank(name,candidates[name]))
    selected=ranking[0] if ranking else None
    return dict(status='SOURCE_DECISION_FROZEN',qualified=bool(selected),selected=selected,checks=checks,
        eligible_ranking=ranking,target_rows=[selected] if selected else [],training_TX_F_access=False,
        identity_uses_source_V_labels=True,target_reads_before_decision=0,target_feedback=False,
        reused_source_control=d.CONTROL_RUN,thresholds=d.configuration()['source_thresholds'],selection=d.configuration()['source_selection'])
