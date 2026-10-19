import numpy as np
import torch
from torch import nn
from types import SimpleNamespace
from . import design as d
from .runtime import load_checkpoint
from experiments.pamf_iterative_restore.evaluation import validate_references,qualify_source
from experiments.pamf_unrolled_restore.evaluation import selection_rank
from experiments.pamf_uncertainty_restore.evaluation import VIEWS,predict as original_predict,score as original_score
from experiments.pamf_fcrossrx_readout.frozen import load_reader
from experiments.pamf_native_formal.runner import assess_goal
from experiments.pamf_restoration.runner import read_json,array_tensor,tensor_array
from experiments.pamf_restoration.data import write_json


class SampledReader(nn.Module):
    """Worker-owned independent noise; no fitted or query-derived state."""
    def __init__(self,model,device,seed=d.PRIMARY_SEED):
        super().__init__();self.model=model;self.generator=torch.Generator(device=device).manual_seed(seed)
    def training_outputs(self,received):
        noise=torch.randn(received.shape,dtype=received.dtype,device=received.device,generator=self.generator)
        return self.model.restore(received,noise)
    def forward(self,received):return self.training_outputs(received)[0]


def row_args(args):return SimpleNamespace(**dict(vars(args),output=args.output/args.row,kind=d.KIND))


def prepare_target(args):
    value=read_json(args.output/'source_decision.json');selected=value.get('selected')
    if (value['status']!='SOURCE_DECISION_FROZEN' or not value['qualified'] or selected not in d.ROWS
            or value['target_rows']!=[selected] or args.row!=selected):
        raise ValueError('Only source-qualified selected flow may access target')
    root=args.output/args.row;root.mkdir(exist_ok=True)
    write_json(root/'source_decision.json',dict(value,target_candidates=[d.KIND]))


def stability(logits):
    if logits.shape!=(3,128,6) or not np.isfinite(logits).all():raise ValueError('All three complete finite source draw logits required')
    decisions=logits.argmax(-1)
    return float(100*np.any(decisions!=decisions[0],axis=0).mean())


@torch.no_grad()
def predict_stability(args):
    folder=args.output/args.row/'sampling_stability';folder.mkdir()
    received=np.load(args.parent_action/'cache/validation'/f'{d.configuration()["stability_view"]}.npy',allow_pickle=False)
    if received.shape!=(1800,2,256):raise ValueError('Original fixed source panel required')
    r=array_tensor(received[:128],'cuda:0')
    model,payload=load_checkpoint(args.output/args.row/d.KIND/'final.pt','cuda:0')
    if payload['candidate']!=args.row:raise ValueError('Source stability checkpoint row differs')
    model.requires_grad_(False)
    reader=load_reader(args.reader_folder,args.source_cache,'cuda:0').requires_grad_(False).eval()
    arrays=[]
    for seed in d.STABILITY_SEEDS:
        generator=torch.Generator(device='cuda:0').manual_seed(seed)
        noise=torch.randn(r.shape,dtype=r.dtype,device=r.device,generator=generator)
        output=model(r,noise);features=reader.features(output);arrays.append(tensor_array(reader.classify_features(features)))
    logits=np.stack(arrays);percent=stability(logits);np.save(folder/'logits.npy',logits,allow_pickle=False)
    write_json(folder/'complete.json',dict(status='SOURCE_STABILITY_PREDICTIONS_COMPLETE',role='source',count=128,
        seeds=list(d.STABILITY_SEEDS),view=d.configuration()['stability_view'],disagreement_percent=percent,
        TX_truth_reads=0,state_updates=0,head_fit_updates=0,full_sampling_steps=d.SAMPLING_STEPS,primary_changed=False))


def validate_stability(folder):
    meta=read_json(folder/'complete.json')
    expected=dict(status='SOURCE_STABILITY_PREDICTIONS_COMPLETE',role='source',count=128,seeds=list(d.STABILITY_SEEDS),
        view=d.configuration()['stability_view'],TX_truth_reads=0,state_updates=0,head_fit_updates=0,
        full_sampling_steps=d.SAMPLING_STEPS,primary_changed=False)
    if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('Registered source-only complete sampling diagnostics required')
    percent=stability(np.load(folder/'logits.npy',allow_pickle=False))
    if abs(percent-meta['disagreement_percent'])>1e-9:raise ValueError('Sampling diagnostic arithmetic differs')
    return dict(meta,qualified=percent<=d.configuration()['stability_max_percent'])


def predict(args,role):
    if role=='target':prepare_target(args)
    elif args.row not in d.ROWS:raise ValueError('Registered flow source row required')
    def loader(path,device):
        model,payload=load_checkpoint(path,device)
        if payload['candidate']!=args.row:raise ValueError('Wrong flow checkpoint row')
        return SampledReader(model,device),payload
    original_predict(row_args(args),role,checkpoint_loader=loader)
    if role=='source':predict_stability(args)


def score(args,role):
    if role=='source':sample=validate_stability(args.output/args.row/'sampling_stability')
    original_score(row_args(args),role)
    if role=='target':
        summary=read_json(args.output/args.row/'target_confirmation'/d.KIND/'summary.json')
        write_json(args.output/args.row/'goal_assessment.json',assess_goal(summary))
    else:
        path=args.output/args.row/d.KIND/'summary.json';summary=read_json(path)
        write_json(path.with_name('flow_source_summary.json'),dict(summary,sampling_stability=sample))


def select_source(candidates,single,native,fft,unrolled):
    if set(candidates)!=set(d.ROWS):raise ValueError('Both registered flow starts must finish')
    checks={}
    for name,summary in candidates.items():
        check=qualify_source(single,summary,native,fft,unrolled);sample=summary['sampling_stability']
        if (sample['role']!='source' or sample['count']!=128 or sample['seeds']!=list(d.STABILITY_SEEDS)
                or sample['TX_truth_reads'] or sample['primary_changed']):raise ValueError('Legal fixed source sampling evidence required')
        stable=bool(np.isfinite(sample['disagreement_percent']) and 0<=sample['disagreement_percent']<=d.configuration()['stability_max_percent'])
        checks[name]=dict(check,qualified=bool(check['qualified'] and stable),sampling_stable=stable,sampling_stability=sample)
    ranking=sorted([n for n in d.ROWS if checks[n]['qualified']],key=lambda n:selection_rank(n,candidates[n]))
    selected=ranking[0] if ranking else None
    return dict(status='SOURCE_DECISION_FROZEN',qualified=bool(selected),selected=selected,checks=checks,eligible_ranking=ranking,
        target_rows=[selected] if selected else [],training_TX_F_access=False,identity_uses_source_V_labels=True,
        target_reads_before_decision=0,target_feedback=False,thresholds=d.configuration()['source_thresholds'],
        sampling_threshold_percent=d.configuration()['stability_max_percent'],selection=d.configuration()['source_selection'],reused_formal_reference=d.UNROLLED)
