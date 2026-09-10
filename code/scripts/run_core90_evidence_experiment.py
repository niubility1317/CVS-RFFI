"""Staged CORE90 experiment: scratch H0, frozen H1--H4, source freeze, truth-last.

One coordinator owns one new run root. H5 and joint retraining are intentionally
not scheduled: they require the source evidence specified in the design report.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import traceback

import torch
from torch.utils.data import Dataset

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from SSDG.train_ssdg import build_arg_parser,_build_ssdg_wisig_data
from cvsrffi.core90_evidence_profile import core90_arguments,load_profile
from cvsrffi.evidence_head_training import validate_data_contract
from cvsrffi.evidence_pipeline import fit_frozen_head,export_deployment,verify_checkpoint_contract
from post_stage_common import build_baseline_model

VARIANTS=('H0','H1','H2','H3','H4')
CONTROLS=('cosine','diagonal_gaussian','linear_ridge')


def write_json(path,value,*,exclusive=False):
    with Path(path).open('x' if exclusive else 'w',encoding='utf-8',newline='\n') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)


def physical_id(record):
    return ':'.join(str(int(getattr(record,key))) for key in ('tx_i','rx_i','day_i','sig_i'))


class TargetRows(Dataset):
    """Builder-side target rows; prediction module strips labels before inference."""
    def __init__(self,ctx,contract):
        self.index=[]; self.rows=[]; seen=set()
        receivers=set(contract['target_receivers'])
        for name,loader in sorted(ctx['named_test_loaders'].items()):
            dataset=loader.dataset
            for i,record in enumerate(dataset.index):
                key=physical_id(record)
                if int(record.rx_i) in receivers and key not in seen:
                    seen.add(key); self.index.append(record); self.rows.append((dataset,i))
        if seen!=set(contract['roles']['target']):
            raise ValueError('target builder differs from the frozen physical contract')
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        dataset,j=self.rows[i]
        return dataset[j]


def build_context(plan,root):
    args=build_arg_parser().parse_args(core90_arguments(plan['dataset'],str(root/'data_contract.json'),str(root/'H0'),
        seed=plan['seed'],device='cpu',source_rxs=plan['source_rxs'],source_days=plan['source_days'],
        target_rxs=plan['target_rxs'],target_days=plan['heldout_days']))
    # Extraction resource choices do not alter sample membership or the H0 trainer.
    args.num_workers=0;args.eval_batch_size=64
    return args,_build_ssdg_wisig_data(args,torch.device('cpu'))


def make_contract(ctx,args,plan):
    roles={role:[physical_id(r) for r in ctx[key].dataset.index]
           for role,key in (('L_s','train_loader'),('U_s','unlabeled_loader'),('V','val_loader'))}
    receivers=[int(x) for x in plan['target_rxs'].split(',')]
    target={physical_id(r) for loader in ctx['named_test_loaders'].values()
            for r in loader.dataset.index if int(r.rx_i) in receivers}
    roles['target']=sorted(target)
    return dict(dataset_id=ctx['dataset_id'],source_receivers=[int(x) for x in plan['source_rxs'].split(',')],
        target_receivers=receivers,roles=roles,split_seed=plan['seed'],equalized=int(args.wisig_equalized),
        source_days=[int(x) for x in plan['source_days'].split(',')],
        actual_target_days=sorted({int(key.split(':')[2]) for key in target}),
        tx_mapping=list(ctx['class_id_to_tx']),ratios={'L_s':.07,'U_s':.63,'V':.30},
        source_selection='final_only',checkpoint_initialization='from_scratch',upstream=[])


def _load_plan(root):return json.loads((root/'plan.json').read_text(encoding='utf-8'))


def prepare(root):
    plan=_load_plan(root);args,ctx=build_context(plan,root)
    contract=make_contract(ctx,args,plan)
    write_json(root/'data_contract.json',contract,exclusive=True)
    validate_data_contract(ctx,root/'data_contract.json')
    # Store only L_s for the later frozen fit, never U_s labels or target IQ.
    chunks=[];labels=[]
    for batch in ctx['probe_train_loader']:
        chunks.append(batch[0].cpu());labels.append(batch[1].cpu())
    source={'x':torch.cat(chunks),'y':torch.cat(labels),'physical_ids':contract['roles']['L_s']}
    torch.save(source,root/'source_training.pt')
    summary={'dataset_id':contract['dataset_id'],'counts':{k:len(v) for k,v in contract['roles'].items()},
             'source_receivers':contract['source_receivers'],'target_receivers':contract['target_receivers'],
             'source_days':contract['source_days'],'actual_target_days':contract['actual_target_days'],
             'seed':plan['seed'],'checkpoint_initialization':'from_scratch','target_truth_used':False}
    write_json(root/'data_summary.json',summary,exclusive=True)
    print('[DATA-CONTRACT] '+json.dumps(summary),flush=True)


def scratch_smoke(root):
    """One actual L_s checkpoint save/reload/forward before the formal scratch run."""
    plan=_load_plan(root)
    source=torch.load(root/'source_training.pt',map_location='cpu',weights_only=True)
    torch.manual_seed(plan['seed'])
    args=SimpleNamespace(num_classes=int(source['y'].max())+1,num_domains=15,model_size='M',model_variant='lite_d',
        branch_ablation='no_dac',domain_branch_ablation='no_stats',dataset='wisig',input_len=256,
        sample_rate_hz=25e6,evidence_config='')
    model=build_baseline_model(args,torch.device('cuda:0'));model.eval()
    with torch.no_grad():before=model(source['x'][:4].cuda())
    torch.save({'model':model.state_dict(),'baseline_args':vars(args)},root/'scratch_smoke.pt')
    payload=torch.load(root/'scratch_smoke.pt',map_location='cuda:0',weights_only=True)
    restored=build_baseline_model(SimpleNamespace(**payload['baseline_args']),torch.device('cuda:0'))
    restored.load_state_dict(payload['model'],strict=True);restored.eval()
    with torch.no_grad():after=restored(source['x'][:4].cuda())
    torch.testing.assert_close(before,after)
    if not torch.isfinite(after).all():raise FloatingPointError('scratch smoke nonfinite')
    print('[SCRATCH-CHECKPOINT-SMOKE] PASS; this checkpoint is NOT a training initialization',flush=True)


def source_worker(root,variant):
    from cvsrffi.evidence_source_evaluation import evaluate_source
    plan=_load_plan(root);_,ctx=build_context(plan,root)
    contract=validate_data_contract(ctx,root/'data_contract.json')
    payload=torch.load(root/'H0'/'ground_checkpoint.pt',map_location='cpu',weights_only=True)
    verify_checkpoint_contract(payload,contract)
    if variant=='H0':
        model=build_baseline_model(SimpleNamespace(**payload['baseline_args']),torch.device('cuda:0'))
        model.load_state_dict(payload['model'],strict=True)
    else:
        output=root/variant;output.mkdir(exist_ok=False)
        source=torch.load(root/'source_training.pt',map_location='cpu',weights_only=True)
        config=load_profile()['variant_configs'][variant]
        started=time.monotonic()
        model,trained=fit_frozen_head(payload,source,contract,config,epochs=plan['head_epochs'],
                                     batch_size=plan['head_batch_size'],device='cuda:0',seed=plan['seed'])
        torch.save(trained,output/'ground_checkpoint.pt')
        export_deployment(model,SimpleNamespace(**trained['baseline_args']),output/'deployment.pt')
        write_json(output/'activation.json',{'activation':trained['activation'],'history':trained['history'],
            'fit_seconds':time.monotonic()-started,'seed':plan['seed'],'backbone_frozen':True},exclusive=True)
    summary=evaluate_source(model,ctx['probe_train_loader'],ctx['val_loader'],torch.device('cuda:0'),root/variant/'source')
    print('[SOURCE-FROZEN] '+json.dumps({'variant':variant,'count_train':summary['count_train'],
          'count_val':summary['count_val'],'target_access':False}),flush=True)


class FrozenControl(torch.nn.Module):
    def __init__(self,backbone,head):
        super().__init__();self.backbone=backbone
        self.register_buffer('weight',head['weight']);self.register_buffer('bias',head['bias'])
        self.normalize_input=head['normalize_input'];self.num_classes=len(self.bias)
    def forward(self,x,**kwargs):
        out=self.backbone(x,return_aux=True)
        z=out['z_id']
        if self.normalize_input:z=torch.nn.functional.normalize(z,dim=-1)
        out['tx_logits']=z@self.weight.T+self.bias
        out.pop('evidence',None)
        return out


def target_worker(root,variant):
    from cvsrffi.evidence_target_evaluation import seal_target,opaque_record_id
    from cvsrffi.evidence_decision import ConditionAwareCalibrator
    # This file is created only after ALL source fits and calibrations completed.
    frozen=json.loads((root/'all_source_frozen.json').read_text(encoding='utf-8'))
    if frozen['variants']!=list(VARIANTS):raise ValueError('incomplete frozen comparison')
    plan=_load_plan(root);_,ctx=build_context(plan,root)
    contract=validate_data_contract(ctx,root/'data_contract.json')
    dataset=TargetRows(ctx,contract)
    base='H0' if variant in CONTROLS else variant
    bundle=torch.load(root/base/'deployment.pt',map_location='cuda:0',weights_only=True)
    model=build_baseline_model(SimpleNamespace(**bundle['baseline_args']),torch.device('cuda:0'))
    model.load_state_dict(bundle['model'],strict=True);model.eval()
    calibrator=None
    if variant in CONTROLS:
        heads=torch.load(root/'H0'/'source'/'source_readouts.pt',map_location='cuda:0',weights_only=True)
        model=FrozenControl(model,heads['controls'][variant]).cuda().eval()
    else:
        path=root/variant/'source'/'source_calibrator.json'
        if path.is_file():calibrator=ConditionAwareCalibrator.from_state_dict(json.loads(path.read_text(encoding='utf-8')))
    seal_target(model,dataset,torch.device('cuda:0'),root/'target_predictions'/variant,
                seed=plan['seed'],batch_size=plan['eval_batch_size'],calibrator=calibrator)
    print('[TARGET-SEALED] '+variant,flush=True)


def score_all(root):
    from cvsrffi.evidence_target_evaluation import score_target,opaque_record_id
    for variant in (*VARIANTS,*CONTROLS):
        seal=json.loads((root/'target_predictions'/variant/'seal.json').read_text(encoding='utf-8'))
        if seal['status']!='SEALED' or set(seal['scenes'])!={'clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak'}:
            raise ValueError('all candidates and scenes must be sealed before truth connection')
    plan=_load_plan(root);_,ctx=build_context(plan,root)
    contract=validate_data_contract(ctx,root/'data_contract.json');dataset=TargetRows(ctx,contract)
    # This is the first scorer-side label connection, after ALL predictions exist.
    truth={'rows':{opaque_record_id(r):{'label':int(r.tx_i),'receiver':int(r.rx_i),'day':int(r.day_i)} for r in dataset.index}}
    write_json(root/'target_truth.json',truth,exclusive=True)
    metrics={}
    (root/'metrics').mkdir(exist_ok=False)
    for variant in (*VARIANTS,*CONTROLS):
        metrics[variant]=score_target(root/'target_predictions'/variant,root/'target_truth.json',root/'metrics'/f'{variant}.json')
    paired={}
    for scene in ('clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak'):
        base=torch.load(root/'target_predictions'/'H0'/f'{scene}.pt',map_location='cpu',weights_only=True)
        ids=[bytes(row.tolist()).hex() for row in base['ids']]
        labels=torch.tensor([truth['rows'][key]['label'] for key in ids])
        correct=base['scores'].argmax(-1)==labels
        def margin(scores):
            chosen=scores.gather(1,labels[:,None]).squeeze(1)
            others=scores.clone();others.scatter_(1,labels[:,None],-torch.inf)
            return chosen-others.max(-1).values
        base_margin=margin(base['scores']);paired[scene]={}
        for variant in (*VARIANTS[1:],*CONTROLS):
            current=torch.load(root/'target_predictions'/variant/f'{scene}.pt',map_location='cpu',weights_only=True)
            current_ids=[bytes(row.tolist()).hex() for row in current['ids']]
            lookup={key:i for i,key in enumerate(current_ids)}
            if set(lookup)!=set(ids):raise ValueError('paired comparison physical IDs differ')
            scores=current['scores'][[lookup[key] for key in ids]]
            now=scores.argmax(-1)==labels
            paired[scene][variant]={'rescue_count':int((~correct & now).sum()),'harm_count':int((correct & ~now).sum()),
                'prediction_change_count':int((scores.argmax(-1)!=base['scores'].argmax(-1)).sum()),
                'raw_score_delta_mean':float((scores-base['scores']).mean()),
                'raw_true_class_margin_delta_mean':float((margin(scores)-base_margin).mean()),
                'raw_scale_warning':'logit scales differ; rescue/harm are decision comparisons'}
    write_json(root/'comparison.json',{'seed':plan['seed'],'results':metrics,'target_feedback':False,
               'paired_to_H0':paired,'scientific_promotion_claim':False,'H5':'DEFERRED_PENDING_SOURCE_EVIDENCE'},exclusive=True)


def child(root,phase,variant,gpu):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--root',str(root),'--phase',phase]
    if variant:cmd+=['--variant',variant]
    log=root/'logs'/f'{phase}_{variant or "main"}.log'
    with log.open('x',encoding='utf-8') as handle:
        process=subprocess.Popen(cmd,cwd=str(CODE.parent),env=env,stdout=handle,stderr=subprocess.STDOUT)
    with (root/'commands.jsonl').open('a',encoding='utf-8') as handle:
        handle.write(json.dumps({'pid':process.pid,'gpu':str(gpu),'argv':cmd,'cwd':str(CODE.parent),'log':str(log)})+'\n')
    return process


def wait_processes(processes):
    failures=[]
    for label,process in processes:
        code=process.wait()
        if code:failures.append((label,code))
    if failures:raise RuntimeError('stage subprocess failure; artifacts preserved: '+repr(failures))


def coordinate(root,plan):
    root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir()
    write_json(root/'plan.json',plan,exclusive=True)
    def status(stage,**extra):
        write_json(root/'status.json',{'stage':stage,'owner_pid':os.getpid(),'time':time.time(),**extra})
    try:
        status('PREPARING')
        wait_processes([('prepare',child(root,'prepare',None,plan['train_gpu']))])
        wait_processes([('smoke',child(root,'smoke',None,plan['train_gpu']))])
        status('H0_TRAINING')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(plan['train_gpu']),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        command=[sys.executable,'-u',str(CODE/'scripts'/'run_core90_evidence.py'),'train','--dataset',plan['dataset'],
            '--contract',str(root/'data_contract.json'),'--output',str(root/'H0'),'--variant','H0','--seed',str(plan['seed']),
            '--source-rxs',plan['source_rxs'],'--source-days',plan['source_days'],'--target-rxs',plan['target_rxs'],
            '--target-days',plan['heldout_days'],'--external-final-eval']
        with (root/'logs'/'H0_train.log').open('x',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=str(CODE.parent),env=env,stdout=log,stderr=subprocess.STDOUT)
        with (root/'commands.jsonl').open('a',encoding='utf-8') as log:
            log.write(json.dumps({'pid':process.pid,'gpu':str(plan['train_gpu']),'argv':command,'cwd':str(CODE.parent)})+'\n')
        wait_processes([('H0',process)])
        if not (root/'H0'/'deployment.pt').is_file():raise FileNotFoundError('H0 deployment export')
        status('FROZEN_SOURCE_FITS')
        wait_processes([(v,child(root,'source',v,plan['head_gpus'][i])) for i,v in enumerate(VARIANTS)])
        write_json(root/'all_source_frozen.json',{'variants':list(VARIANTS),'controls':list(CONTROLS),
                'seed':plan['seed'],'target_access':False,'H5':'DEFERRED_PENDING_SOURCE_EVIDENCE'},exclusive=True)
        status('TARGET_PREDICTIONS')
        # At most five new processes; user permits exceeding two total jobs/GPU.
        candidates=(*VARIANTS,*CONTROLS)
        for start in range(0,len(candidates),5):
            wait_processes([(v,child(root,'target',v,plan['head_gpus'][i])) for i,v in enumerate(candidates[start:start+5])])
        status('ALL_PREDICTIONS_SEALED')
        wait_processes([('score',child(root,'score',None,plan['train_gpu']))])
        status('ARTIFACTS_COMPLETE',scientific_promotion_claim=False)
    except Exception:
        status('TECHNICAL_FAILURE',traceback=traceback.format_exc(),artifacts_preserved=True)
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--phase',choices=('coordinate','prepare','smoke','source','target','score'),default='coordinate')
    parser.add_argument('--variant',choices=(*VARIANTS,*CONTROLS))
    parser.add_argument('--dataset')
    parser.add_argument('--seed',type=int,default=392005)
    parser.add_argument('--train-gpu',type=int,default=2)
    parser.add_argument('--head-gpus',default='0,1,2,3,4')
    args=parser.parse_args(argv);root=Path(args.root).resolve()
    if args.phase=='coordinate':
        if not args.dataset or not Path(args.dataset).is_file():raise FileNotFoundError('actual dataset required')
        gpus=[int(x) for x in args.head_gpus.split(',')]
        if len(gpus)!=5:raise ValueError('five explicit source worker GPU bindings required')
        coordinate(root,dict(dataset=str(Path(args.dataset).resolve()),seed=args.seed,train_gpu=args.train_gpu,head_gpus=gpus,
            source_rxs='1,3,4,6,8',source_days='1,2,3',target_rxs='0,2,5,7,9,10,11',heldout_days='0',
            h0_epochs=200,head_epochs=20,head_batch_size=64,eval_batch_size=64,
            initialization='scratch_H0_then_same_frozen_backbone',state_semantics='receiver_proxy_not_physical_excitation',
            per_gpu_two_process_limit_overridden_by_user=True,H5='DEFERRED_PENDING_SOURCE_EVIDENCE'))
    elif args.phase=='prepare':prepare(root)
    elif args.phase=='smoke':scratch_smoke(root)
    elif args.phase=='source':source_worker(root,args.variant)
    elif args.phase=='target':target_worker(root,args.variant)
    else:score_all(root)
    return 0


if __name__=='__main__':raise SystemExit(main())
