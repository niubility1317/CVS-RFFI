from types import SimpleNamespace
import math
import os
from . import design as d,response_evaluation as response
from .runtime import load_checkpoint
from experiments.pamf_uncertainty_restore.evaluation import predict as original_predict,score as original_score
from experiments.pamf_fullresolution_restore.evaluation import qualify_source as qualify_pair
from experiments.pamf_native_formal.runner import assess_goal
from experiments.pamf_restoration.runner import read_json
from experiments.pamf_restoration.data import write_json
from experiments.pamf_fcrossrx_readout.source_diagnostic import source_reference


def validate_reference(root):
    if root.name!=d.UNROLLED or read_json(root/'launch.json')['commit']!=d.UNROLLED_COMMIT:
        raise ValueError('Original source-selected R17 reference required')
    record=read_json(root/d.REFERENCE_CANDIDATE/'training_complete.json')
    expected=dict(status='SOURCE_TRAINED',epochs=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        TX_labels_read=0,F_outputs_read=0,F_weights_read=0,target_reads=0,prior_initialization='scratch',inherited_prior_epochs=0)
    if any(record.get(k)!=v for k,v in expected.items()):raise ValueError('Original matching fullLU source R17 training required')
    summary=read_json(root/d.REFERENCE_CANDIDATE/d.KIND/'summary.json')
    if summary['role']!='source' or summary['target_access'] or not summary['all_predictions_before_truth']:
        raise ValueError('Source evidence only,not formal ranking')
    return summary


def row_args(args):return SimpleNamespace(**dict(vars(args),output=args.output/args.row,kind=d.KIND))


def prepare_target(args):
    value=read_json(args.output/'source_decision.json')
    if (value['status']!='SOURCE_DECISION_FROZEN' or not value['qualified'] or value['selected']!=d.CANDIDATE
            or value['target_rows']!=[d.CANDIDATE] or args.row!=d.CANDIDATE):
        raise ValueError('Only source-qualified response candidate may access target')
    write_json(args.output/args.row/'source_decision.json',dict(value,target_candidates=[d.KIND]))


def predict(args,role):
    if role=='target':prepare_target(args)
    elif args.row not in d.ROWS:raise ValueError('Registered source response row required')
    def loader(path,device):
        model,payload=load_checkpoint(path,device)
        if payload['candidate']!=args.row:raise ValueError('Wrong response checkpoint row')
        return model,payload
    original_predict(row_args(args),role,checkpoint_loader=loader)
    if role=='source':response.predict(args)


def score(args,role):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU truth-side scoring required')
    if role=='source':
        ids,_,_=source_reference(args.parent_action/'cache/validation',args.source_cache)
        response.load_predictions(args.output/args.row/'source_response',ids,args.row)
    original_score(row_args(args),role)
    if role=='source':response.score(args)
    else:
        root=args.output/args.row
        write_json(root/'goal_assessment.json',assess_goal(read_json(root/'target_confirmation'/d.KIND/'summary.json')))


def response_comparison(control,candidate):
    means={};counts={}
    for name,summary in [('control',control),('candidate',candidate)]:
        if (summary.get('status')!='ANALYZED' or summary.get('run')!=d.RUN or summary.get('count')!=1800
                or not summary.get('all_predictions_before_truth') or summary.get('target_access')
                or summary.get('training_TX_F_access') or summary.get('source_V_state_updates')!=0):
            raise ValueError('Complete source-only response summary required')
        total=0.;count=0;per={}
        for direction in d.DIRECTIONS[:-1]:
            records=[r for r in summary['records'] if r['view']=='practical_low_urban' and r['direction']==direction
                     and r['dimension']=='overall' and r['value']=='all']
            if len(records)!=1:raise ValueError('Every primary response direction required')
            record=records[0];n=record['observable_count'];value=record['observable_oracle_response_error_mean']
            if not isinstance(n,int) or n<0 or (n>0 and (value is None or not math.isfinite(value) or value<0)):
                raise ValueError('Finite source response statistics required')
            if n:total+=n*value
            count+=n;per[direction]=n
        means[name]=total/count if count else None;counts[name]=per
    if counts['candidate']!=counts['control']:raise ValueError('Matched source response eligibility differs')
    ratio=means['candidate']/means['control'] if means['control'] is not None and means['control']>0 else (1. if means['candidate']==means['control']==0 else None)
    enough=all(n>=100 for n in counts['candidate'].values())
    return dict(qualified=bool(enough and ratio is not None and ratio<=.95),control_error=means['control'],candidate_error=means['candidate'],ratio=ratio,
        counts=counts,threshold=.95,source_only=True,TX_fidelity_unproven=True)


def select_source(candidates,responses,reference):
    if set(candidates)!=set(d.ROWS) or set(responses)!=set(d.ROWS):raise ValueError('Both matched rows must complete all source scoring')
    comparison=qualify_pair(candidates[d.ROWS[0]],candidates[d.CANDIDATE],reference)
    mechanism=response_comparison(responses[d.ROWS[0]],responses[d.CANDIDATE])
    qualified=bool(comparison['qualified'] and mechanism['qualified'])
    return dict(status='SOURCE_DECISION_FROZEN',qualified=qualified,selected=d.CANDIDATE if qualified else None,
        checks=dict(waveform_and_identity=comparison,local_response=mechanism),target_rows=[d.CANDIDATE] if qualified else [],
        training_TX_F_access=False,identity_uses_source_V_labels=True,target_reads_before_decision=0,target_feedback=False,
        selection=d.configuration()['source_selection'],thresholds=d.configuration()['source_thresholds'])
