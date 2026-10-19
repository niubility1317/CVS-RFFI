from types import SimpleNamespace
from . import design as d
from .runtime import load_checkpoint
from experiments.pamf_unrolled_restore.evaluation import (
    validate_references as validate_old_references,qualify_source as old_qualify,selection_rank)
from experiments.pamf_fullresolution_restore.evaluation import qualify_source as shared_qualify
from experiments.pamf_uncertainty_restore.evaluation import VIEWS,predict as original_predict,score as original_score
from experiments.pamf_native_formal.runner import assess_goal
from experiments.pamf_restoration.runner import read_json
from experiments.pamf_restoration.data import write_json


def validate_references(single,fft,unrolled):
    validate_old_references(single,fft)
    if (unrolled.name!=d.UNROLLED or
            read_json(unrolled/'launch.json')['commit']!=d.UNROLLED_COMMIT):
        raise ValueError('Fixed R17 source reference required')
    config=read_json(unrolled/'resolved_config.json')
    decision=read_json(unrolled/'source_decision.json')
    if (config['run']!=d.UNROLLED or config['training_F_access'] or config['target_training']
            or not decision['qualified'] or decision['selected']!=d.REFERENCE_CANDIDATE
            or decision['training_TX_F_access'] or decision['target_reads_before_decision']!=0
            or decision['target_feedback']):
        raise ValueError('R17 must have source-only training and source-only frozen selection')
    row=unrolled/d.REFERENCE_CANDIDATE
    train=read_json(row/'training_complete.json');summary=read_json(row/d.KIND/'summary.json')
    expected=dict(status='SOURCE_TRAINED',epochs=20,updates=4940,L_exposures=126000,
        U_exposures=1134000,TX_labels_read=0,F_outputs_read=0,F_weights_read=0,
        target_reads=0,frozen_context_updates=0,prior_initialization='scratch',inherited_prior_epochs=0)
    if (any(train.get(k)!=v for k,v in expected.items()) or
            summary['status']!='ANALYZED' or summary['role']!='source' or summary['count']!=1800
            or summary['target_access'] or summary['training_TX_labels_read']!=0
            or summary['training_F_access'] or not summary['all_predictions_before_truth']):
        raise ValueError('Legal complete source-only R17 reference required')
    # Do not open R17 target summaries, predictions, goal assessments or truth.
    return summary


def row_args(args):
    return SimpleNamespace(**dict(vars(args),output=args.output/args.row,kind=d.KIND))


def prepare_target(args):
    value=read_json(args.output/'source_decision.json');selected=value.get('selected')
    if (value['status']!='SOURCE_DECISION_FROZEN' or not value['qualified']
            or selected not in d.ROWS or value['target_rows']!=[selected] or args.row!=selected):
        raise ValueError('Only source-selected qualified alternating candidate may read target inputs')
    root=args.output/args.row;root.mkdir(exist_ok=True)
    write_json(root/'source_decision.json',dict(value,target_candidates=[d.KIND]))


def predict(args,role):
    if role=='target':
        prepare_target(args)
    elif args.row not in d.ROWS:
        raise ValueError('Registered alternating source row required')
    def loader(path,device):
        model,payload=load_checkpoint(path,device)
        if payload['candidate']!=args.row:
            raise ValueError('Wrong alternating checkpoint row')
        return model,payload
    original_predict(row_args(args),role,checkpoint_loader=loader)


def score(args,role):
    original_score(row_args(args),role)
    if role=='target':
        summary=read_json(args.output/args.row/'target_confirmation'/d.KIND/'summary.json')
        write_json(args.output/args.row/'goal_assessment.json',assess_goal(summary))


def qualify_source(single,candidate,native,fft,unrolled):
    previous=old_qualify(single,candidate,native,fft)
    latest=shared_qualify(unrolled,candidate,native)
    return dict(qualified=bool(previous['qualified'] and latest['qualified']),
        waveform_A=bool(previous['waveform_A'] and latest['waveform_A']),
        source_identity_B=bool(previous['source_identity_B'] and latest['source_identity_B']),
        vs_previous=previous,vs_R17_native=latest)


def select_source(candidates,single,native,fft,unrolled):
    if set(candidates)!=set(d.ROWS):
        raise ValueError('Both preregistered alternating controls must finish source scoring')
    checks={name:qualify_source(single,candidates[name],native,fft,unrolled) for name in d.ROWS}
    eligible=[name for name in d.ROWS if checks[name]['qualified']]
    ranking=sorted(eligible,key=lambda name:selection_rank(name,candidates[name]))
    selected=ranking[0] if ranking else None
    return dict(status='SOURCE_DECISION_FROZEN',qualified=bool(selected),selected=selected,
        checks=checks,eligible_ranking=ranking,target_rows=[selected] if selected else [],
        training_TX_F_access=False,identity_uses_source_V_labels=True,target_reads_before_decision=0,
        target_feedback=False,thresholds=d.configuration()['source_thresholds'],
        selection=d.configuration()['source_selection'],reused_formal_reference=d.UNROLLED)
