import math
from types import SimpleNamespace
from . import design as d
from .runtime import load_checkpoint
from experiments.pamf_iqonly_restore.runtime import load_checkpoint as load_single
from experiments.pamf_fullresolution_restore.evaluation import qualify_source as shared_qualify
from experiments.pamf_uncertainty_restore.evaluation import VIEWS,predict as original_predict,score as original_score
from experiments.pamf_native_formal.runner import assess_goal
from experiments.pamf_restoration.runner import read_json
from experiments.pamf_restoration.data import write_json

def validate_references(single,fft):
    for root,run,commit in ((single,d.SINGLE,d.SINGLE_COMMIT),(fft,d.FFT,d.FFT_COMMIT)):
        if root.name!=run or read_json(root/'launch.json')['commit']!=commit:raise ValueError('Fixed original source reference required')
        done=read_json(root/'complete.json')
        if done['status']!='ANALYZED' or not done['source_only'] or done['target_rows'] or done['training_TX_F_access']:
            raise ValueError('Source references must remain complete source-only with zero training TX/F')

def row_args(args):return SimpleNamespace(**dict(vars(args),output=args.output/args.row,kind=d.KIND))

def prepare_target(args):
    value=read_json(args.output/'source_decision.json');selected=value.get('selected')
    if value['status']!='SOURCE_DECISION_FROZEN' or not value['qualified'] or selected not in d.ROWS or value['target_rows']!=[selected,d.CONTROL] or args.row not in value['target_rows']:
        raise ValueError('Source-selected qualified unrolled candidate required before target IO')
    root=args.output/args.row;root.mkdir(exist_ok=True)
    write_json(root/'source_decision.json',dict(value,target_candidates=[d.KIND]))

def predict(args,role):
    if role=='target':prepare_target(args)
    elif args.row not in d.ROWS:raise ValueError('Single-shot reference is target-only')
    def loader(path,device):
        if args.row==d.CONTROL:
            validate_references(args.single_run,args.fft_run);model,payload=load_single(args.single_run/'unet_control'/d.KIND/'final.pt',device);candidate='unet_control'
        else:model,payload=load_checkpoint(path,device);candidate=args.row
        if payload['candidate']!=candidate:raise ValueError('Wrong unrolled/source-control checkpoint')
        return model,payload
    original_predict(row_args(args),role,checkpoint_loader=loader)

def score(args,role):
    if args.row==d.CONTROL and role!='target':raise ValueError('Single-shot source score is reused')
    original_score(row_args(args),role)
    if role=='target':write_json(args.output/args.row/'goal_assessment.json',assess_goal(read_json(args.output/args.row/'target_confirmation'/d.KIND/'summary.json')))

def qualify_source(single,candidate,native,fft):
    first=shared_qualify(single,candidate,native);second=shared_qualify(fft,candidate,native)
    return dict(qualified=bool(first['qualified'] and second['qualified']),waveform_A=bool(first['waveform_A'] and second['waveform_A']),
        source_identity_B=bool(first['source_identity_B'] and second['source_identity_B']),
        vs_single_native=first,vs_fft_native=second)

def selection_rank(name,summary):
    def values(view,dimension):
        rows=[r for r in summary['records'] if r['view']==view and r['readout']=='restored' and r['dimension']==dimension]
        result={r['value']:r['accuracy'] for r in rows}
        if len(rows)!=len(result) or not rows or any(not math.isfinite(x) or not 0<=x<=100 for x in result.values()):raise ValueError('Unique finite source rank groups required')
        return result
    low=values('practical_low_urban','overall')['all'];mid=values('practical_mid_urban','overall')['all']
    tx=values('practical_low_urban','TX');rx=values('practical_low_urban','RX')
    if len(tx)!=6 or len(rx)!=5:raise ValueError('All six TX and five source RX required for fixed ranking')
    return (-(low+mid)/2,-min(tx.values()),-min(rx.values()),name)

def select_source(candidates,single,native,fft):
    if set(candidates)!=set(d.ROWS):raise ValueError('Both preregistered candidates must finish source scoring')
    checks={name:qualify_source(single,candidates[name],native,fft) for name in d.ROWS}
    eligible=[name for name in d.ROWS if checks[name]['qualified']]
    ranking=sorted(eligible,key=lambda name:selection_rank(name,candidates[name]));selected=ranking[0] if ranking else None
    return dict(status='SOURCE_DECISION_FROZEN',qualified=bool(selected),selected=selected,checks=checks,eligible_ranking=ranking,
        target_rows=[selected,d.CONTROL] if selected else [],training_TX_F_access=False,identity_uses_source_V_labels=True,
        target_reads_before_decision=0,target_feedback=False,thresholds=d.configuration()['source_thresholds'],selection=d.configuration()['source_selection'])
