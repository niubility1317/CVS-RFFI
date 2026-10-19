from types import SimpleNamespace
from . import design as d
from .runtime import load_checkpoint
from experiments.pamf_endpoint_restore.evaluation import validate_references,qualify_source
from experiments.pamf_unrolled_restore.evaluation import selection_rank
from experiments.pamf_uncertainty_restore.evaluation import VIEWS,predict as original_predict,score as original_score
from experiments.pamf_native_formal.runner import assess_goal
from experiments.pamf_restoration.runner import read_json
from experiments.pamf_restoration.data import write_json


def row_args(args):return SimpleNamespace(**dict(vars(args),output=args.output/args.row,kind=d.KIND))


def prepare_target(args):
    value=read_json(args.output/'source_decision.json');selected=value.get('selected')
    if (value['status']!='SOURCE_DECISION_FROZEN' or not value['qualified'] or selected not in d.ROWS
            or value['target_rows']!=[selected] or args.row!=selected):
        raise ValueError('Only source-qualified selected analytic row may access target')
    root=args.output/args.row;root.mkdir(exist_ok=True)
    write_json(root/'source_decision.json',dict(value,target_candidates=[d.KIND]))


def predict(args,role):
    if role=='target':prepare_target(args)
    elif args.row not in d.ROWS:raise ValueError('Registered source row required')
    def loader(path,device):
        model,payload=load_checkpoint(path,device)
        if payload['candidate']!=args.row:raise ValueError('Wrong analytic checkpoint row')
        return model,payload
    original_predict(row_args(args),role,checkpoint_loader=loader)


def score(args,role):
    original_score(row_args(args),role)
    if role=='target':
        summary=read_json(args.output/args.row/'target_confirmation'/d.KIND/'summary.json')
        write_json(args.output/args.row/'goal_assessment.json',assess_goal(summary))


def select_source(candidates,single,native,fft,unrolled,alternating):
    if set(candidates)!=set(d.ROWS):raise ValueError('Both registered source rows must finish')
    checks={name:qualify_source(single,summary,native,fft,unrolled,alternating) for name,summary in candidates.items()}
    ranking=sorted([name for name in d.ROWS if checks[name]['qualified']],key=lambda name:selection_rank(name,candidates[name]))
    selected=ranking[0] if ranking else None
    return dict(status='SOURCE_DECISION_FROZEN',qualified=bool(selected),selected=selected,checks=checks,
        eligible_ranking=ranking,target_rows=[selected] if selected else [],training_TX_F_access=False,
        identity_uses_source_V_labels=True,target_reads_before_decision=0,target_feedback=False,
        thresholds=d.configuration()['source_thresholds'],selection=d.configuration()['source_selection'])
