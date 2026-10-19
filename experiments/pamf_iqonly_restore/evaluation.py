from types import SimpleNamespace
from . import design as d
from .runtime import load_checkpoint
from experiments.pamf_fullresolution_restore.runtime import load_checkpoint as load_fft
from experiments.pamf_fullresolution_restore.evaluation import qualify_source as shared_qualify
from experiments.pamf_uncertainty_restore.evaluation import VIEWS,predict as original_predict,score as original_score
from experiments.pamf_native_formal.runner import assess_goal
from experiments.pamf_restoration.runner import read_json
from experiments.pamf_restoration.data import write_json

def validate_pair(root):
    if root.name!=d.PAIRED or read_json(root/'launch.json')['commit']!=d.PAIRED_COMMIT:
        raise ValueError('Original fixed R15 paired source-only run required')
    done=read_json(root/'complete.json')
    if done['status']!='ANALYZED' or not done['source_only'] or done['target_rows'] or done['training_TX_F_access']:
        raise ValueError('R15 must remain complete source-only with zero training TX/F')

def row_args(args):return SimpleNamespace(**dict(vars(args),output=args.output/args.row,kind=d.KIND))

def prepare_target(args):
    value=read_json(args.output/'source_decision.json')
    if value['status']!='SOURCE_DECISION_FROZEN' or not value['qualified'] or value['primary']!=d.PRIMARY or value['target_rows']!=list(d.TARGET_ROWS) or args.row not in d.TARGET_ROWS:
        raise ValueError('Fixed UNet IQ-only primary qualification required before target IO')
    root=args.output/args.row;root.mkdir(exist_ok=True)
    write_json(root/'source_decision.json',dict(value,target_candidates=[d.KIND]))

def predict(args,role):
    if role=='target':prepare_target(args)
    elif args.row not in d.ROWS:raise ValueError('FFT reference is target-only,not a new source training row')
    def loader(path,device):
        if args.row==d.FFT_CONTROL:
            validate_pair(args.paired_run);model,payload=load_fft(args.paired_run/d.PRIMARY/d.KIND/'final.pt',device);candidate=d.PRIMARY
        else:model,payload=load_checkpoint(path,device);candidate=args.row
        if payload['candidate']!=candidate:raise ValueError('Checkpoint belongs to a different architecture')
        return model,payload
    original_predict(row_args(args),role,checkpoint_loader=loader)

def score(args,role):
    if args.row==d.FFT_CONTROL and role!='target':raise ValueError('FFT reference source score is reused,not recomputed')
    original_score(row_args(args),role)
    if role=='target':write_json(args.output/args.row/'goal_assessment.json',assess_goal(read_json(args.output/args.row/'target_confirmation'/d.KIND/'summary.json')))

def qualify_source(control,candidate,reference):
    value=shared_qualify(control,candidate,reference)
    value.update(primary=d.PRIMARY,control=d.PAIRED+'/'+d.PRIMARY,secondary_source_only='tcn_prior',
        target_rows=list(d.TARGET_ROWS) if value['qualified'] else [],thresholds=d.configuration()['source_thresholds'])
    value['mean_IQ']=value.pop('mean_TCN');value['IQ_clean_NMSE']=value.pop('TCN_clean_NMSE')
    return value
