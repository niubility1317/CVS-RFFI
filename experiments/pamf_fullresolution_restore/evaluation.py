import math
from types import SimpleNamespace
from . import design as d
from .runtime import load_checkpoint
from experiments.pamf_uncertainty_restore.evaluation import VIEWS,predict as original_predict,score as original_score
from experiments.pamf_native_formal.runner import assess_goal
from experiments.pamf_fcrossrx_readout.frozen import REFERENCE_RUN
from experiments.pamf_restoration.runner import read_json
from experiments.pamf_restoration.data import write_json


def row_args(args):return SimpleNamespace(**dict(vars(args),output=args.output/args.row,kind=d.KIND))


def prepare_target(args):
    value=read_json(args.output/'source_decision.json')
    if value['status']!='SOURCE_DECISION_FROZEN' or not value['qualified'] or args.row not in value['target_rows']:
        raise ValueError('TCN primary source qualification required before target IO')
    write_json(args.output/args.row/'source_decision.json',dict(value,target_candidates=[d.KIND]))


def predict(args,role):
    if role=='target':prepare_target(args)
    def loader(path,device):
        model,payload=load_checkpoint(path,device)
        if payload['candidate']!=args.row:raise ValueError('Checkpoint belongs to another fullresolution row')
        return model,payload
    original_predict(row_args(args),role,checkpoint_loader=loader)


def score(args,role):
    original_score(row_args(args),role)
    if role=='target':
        root=args.output/args.row;write_json(root/'goal_assessment.json',assess_goal(read_json(root/'target_confirmation'/d.KIND/'summary.json')))


def qualify_source(control,candidate,reference):
    for value in (control,candidate,reference):
        required=dict(status='ANALYZED',kind=d.KIND,role='source',count=1800,F_run=REFERENCE_RUN,F_seed=d.MODEL_SEED,
            original_default_head=True,all_predictions_before_truth=True,training_TX_labels_read=0,training_F_access=False,head_fit_updates=0,target_access=False)
        if any(value.get(key)!=item for key,item in required.items()) or list(value['views'])!=list(VIEWS):
            raise ValueError('Complete matching source/originalF evidence required')
    def accuracy(value,view,dimension='overall'):
        rows=[r for r in value['records'] if r['view']==view and r['dimension']==dimension and r['readout']=='restored']
        result={r['value']:r['accuracy'] for r in rows}
        if len(result)!=len(rows) or not result or any(not math.isfinite(x) or not 0<=x<=100 for x in result.values()):
            raise ValueError('Unique finite source groups required')
        return result
    def mean(value):
        numbers=[value['views'][view]['restored']['waveform']['iq_nmse'] for view in VIEWS[1:]]
        if any(not math.isfinite(x) or x<0 for x in numbers):raise ValueError('Finite nonnegative source errors required')
        return sum(numbers)/6
    mc,mt,mr=mean(control),mean(candidate),mean(reference);clean=candidate['views']['clean']['restored']['waveform']['iq_nmse']
    if not math.isfinite(clean) or clean<0:raise ValueError('Finite clean NMSE required')
    gains={view:dict(vs_control_pp=accuracy(candidate,view)['all']-accuracy(control,view)['all'],
        vs_reference_pp=accuracy(candidate,view)['all']-accuracy(reference,view)['all']) for view in ('practical_low_urban','practical_mid_urban')}
    row=next(r for r in candidate['records'] if r['view']=='clean' and r['dimension']=='overall' and r['readout']=='restored')
    original=row['baseline_accuracy']
    if not math.isfinite(original) or not 0<=original<=100:raise ValueError('Original source clean baseline required')
    clean_drop=original-accuracy(candidate,'clean')['all'];new=accuracy(candidate,'practical_low_urban','TX');oldc=accuracy(control,'practical_low_urban','TX');oldr=accuracy(reference,'practical_low_urban','TX')
    if set(new)!={str(i) for i in range(6)} or set(new)!=set(oldc) or set(new)!=set(oldr):raise ValueError('Every source TX required')
    worst_tx_c=min(new[key]-oldc[key] for key in new);worst_tx_r=min(new[key]-oldr[key] for key in new)
    other_c=min(accuracy(candidate,view)['all']-accuracy(control,view)['all'] for view in VIEWS[1:] if view!='practical_low_urban')
    other_r=min(accuracy(candidate,view)['all']-accuracy(reference,view)['all'] for view in VIEWS[1:] if view!='practical_low_urban')
    wave=mt<=1.05*mc and mt<=1.05*mr and clean<=.02
    identity=(all(x['vs_control_pp']>=.5 and x['vs_reference_pp']>=.5 for x in gains.values()) and clean_drop<=.1
        and worst_tx_c>=-1. and worst_tx_r>=-1. and other_c>=-.5 and other_r>=-.5)
    qualified=bool(wave and identity)
    return dict(status='SOURCE_DECISION_FROZEN',qualified=qualified,primary='tcn_prior',control='unet_control',waveform_A=bool(wave),source_identity_B=bool(identity),
        mean_control=mc,mean_TCN=mt,mean_reference=mr,TCN_clean_NMSE=clean,identity_gains=gains,clean_drop_vs_original_pp=clean_drop,
        worst_lowurban_TX_vs_control_pp=worst_tx_c,worst_lowurban_TX_vs_reference_pp=worst_tx_r,worst_other_scene_vs_control_pp=other_c,worst_other_scene_vs_reference_pp=other_r,
        target_rows=list(d.ROWS) if qualified else [],training_TX_F_access=False,identity_B_uses_source_V_labels=True,target_reads_before_decision=0,target_feedback=False,thresholds=d.configuration()['source_thresholds'])
