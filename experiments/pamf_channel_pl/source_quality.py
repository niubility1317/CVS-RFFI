"""Fixed source-V falsification of pseudo labels; never fits or queries a model."""
from pathlib import Path
import math
import time

from experiments.cric_dura_comparison.io import read,write
from .policy import RECIPE,raw_predictions,channel_gate

RULE = dict(min_transfer_precision_percent=99.,max_precision_drop_pp=.1,
            min_transfer_fraction=.1,zero_transfer_fails=True)


def validate_prefix_headers(case,done,payload,actual_contract,expected_contract):
    """Check the actual E20 scratch provenance, including legacy F0 flags."""
    pc=done['config'];expected=case['source_config'];args=payload['args']
    if (done.get('status')!='SOURCE_FROZEN' or done.get('epoch')!=20 or done.get('updates')!=4440
            or done.get('target_access') is not False or done.get('U_truth_read') is not False
            or done.get('head_fit_updates')!=0 or done.get('initialization')!='scratch'
            or done.get('ancestors')!=[] or pc.get('parent_source')
            or actual_contract!=expected_contract):
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH:complete source-only scratch E20 required')
    if (pc['model_seed']!=case['model_seed'] or pc['num_classes']!=len(expected_contract['classes'])
            or any(pc.get(k)!=expected.get(k) for k in ('ablation','model_seed','num_classes','run_id','row_id'))
            or pc['ablation']!='F21' or payload.get('run_id')!=pc['run_id']
            or payload.get('candidate_id')!=pc['row_id'] or payload.get('epoch')!=20
            or payload.get('a1_ema_successful_updates')!=4440):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED:matching ordinary-PL E20 required')
    # The old F0 metadata says own_EMA=true. Actual F21 payload and code disable
    # native U/EMA; the explicit ordinary-PL hook already operated in E14-E20.
    if (args.get('baseline_ckpt')!='' or args.get('teacher_ckpt')!=''
            or args.get('use_unlabeled') is not False or args.get('use_ema_teacher') is not False):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED:foreign/native teacher or inherited state')


def scientific_rule(values):
    n,k,a,b=(values[x] for x in ('candidate','candidate_correct','transfer','transfer_correct'))
    if any(type(x) is not int for x in (n,k,a,b)) or not 0<=b<=a<=n or not 0<=k<=n or b>k:
        raise ValueError('Aligned source candidate and transfer counts required')
    candidate_precision=100.*k/n if n else None
    transfer_precision=100.*b/a if a else None
    fraction=a/n if n else 0.
    drop=candidate_precision-transfer_precision if a and n else None
    passed=bool(a and transfer_precision>=RULE['min_transfer_precision_percent']
        and drop<=RULE['max_precision_drop_pp'] and fraction>=RULE['min_transfer_fraction'])
    return dict(rule=dict(RULE),pass_=passed,**{'pass':passed},candidate_precision=candidate_precision,
        transfer_precision=transfer_precision,precision_drop_pp=drop,transfer_fraction=fraction)


def run(case,output):
    import torch
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.frozen_inference import reader_from_payload
    from experiments.cric_dura_comparison.t1_practical import Practical
    from .objective import subset_metadata
    root=Path(output);root.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    if any(case.get(k) for k in ('truth','target_truth','target_inputs','dataset_path','views_root','tests')):
        raise ValueError('Source diagnostic rejects query and target inputs')
    folder=Path(case['parent_source']);done=read(folder/'complete.json');c=done['config']
    expected=read(case['expected_source_contract']);actual=read(c['source_contract'])
    payload=torch.load(folder/'final_ssdg.pth',map_location='cpu',weights_only=False)
    validate_prefix_headers(case,done,payload,actual,expected)
    if c['source_rxs']!=actual['source_rxs'] or c['source_days']!=actual['source_days']:
        raise ValueError('Checkpoint source RX/day differs')
    model=reader_from_payload(c,payload,'cuda:0');del payload
    buffers={k:v.detach().clone() for k,v in model.named_buffers()}
    labeled=SourceDataset(c['source_cache'],'L_s',actual)
    with torch.no_grad():smoke=model(labeled.x[:2].to('cuda:0'))
    if smoke.shape!=(2,c['num_classes']) or not bool(torch.isfinite(smoke).all()):
        raise ValueError('Real-checkpoint no-query source smoke failed')
    del labeled,smoke
    write(root/'smoke.json',dict(status='PASS',source_role='L_s',packets=2,updates=0,query_reads=0,
        default_head=True,source_contract=case['expected_source_contract']))
    validation=SourceDataset(c['source_cache'],'V',actual)
    loader=torch.utils.data.DataLoader(validation,batch_size=256,shuffle=False,num_workers=0)
    simulator=Practical(dict(c,channel_pl=dict(RECIPE)))
    values=dict(seen=0,raw_accepted=0,raw_accepted_correct=0,candidate=0,candidate_correct=0,
        transfer=0,transfer_correct=0,channel_teacher_views=0)
    strata={}
    for i,batch in enumerate(loader):
        x,y=batch[0].to('cuda:0'),batch[1].to('cuda:0')
        step=4441+i;epoch=(step-1)//222+1
        with torch.no_grad():
            logits=model(x)
            if logits.shape!=(len(x),c['num_classes']):raise ValueError('Default class dimension differs')
            _,pseudo,accepted=raw_predictions(logits)
            physical=(torch.arange(len(x),device=x.device)+step).remainder(2).bool()
            selected=torch.nonzero(accepted&physical,as_tuple=False).flatten()
            valid_indices=torch.empty(0,dtype=torch.long,device=x.device)
            if len(selected):
                channel=simulator(x[selected],subset_metadata(batch[3],selected,len(x)),epoch,step)
                if not bool(torch.isfinite(channel).all()):raise ValueError('Source simulation not finite')
                q=model(channel);valid=channel_gate(pseudo,accepted,selected,q)
                valid_indices=selected[valid]
        values['seen']+=len(x);values['raw_accepted']+=int(accepted.sum())
        values['raw_accepted_correct']+=int((accepted&(pseudo==y)).sum())
        values['candidate']+=len(selected);values['candidate_correct']+=int((pseudo[selected]==y[selected]).sum())
        values['transfer']+=len(valid_indices);values['transfer_correct']+=int((pseudo[valid_indices]==y[valid_indices]).sum())
        values['channel_teacher_views']+=len(selected)
        # Source-only strata diagnose receiver/class concentration. They never
        # alter the threshold, budget, recipe or candidate order.
        for dimension,z in [('rx',batch[3]['rx_i'].to(x.device)),('day',batch[3]['day_i'].to(x.device)),('class',y)]:
            for label in z.unique().tolist():
                m=z==label;key=dimension+':'+str(label)
                s=strata.setdefault(key,dict(candidate=0,candidate_correct=0,transfer=0,transfer_correct=0))
                candidate=selected[m[selected]];transfer=valid_indices[m[valid_indices]]
                s['candidate']+=len(candidate);s['candidate_correct']+=int((pseudo[candidate]==y[candidate]).sum())
                s['transfer']+=len(transfer);s['transfer_correct']+=int((pseudo[transfer]==y[transfer]).sum())
        if i%10==0 or i+1==len(loader):
            write(root/'progress.json',dict(status='SOURCE_VALIDATING',batches=i+1,batches_total=len(loader),
                **values,seconds=time.perf_counter()-start),exclusive=False)
    if values['seen']!=actual['counts']['V']:raise ValueError('Complete declared source V required')
    if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError('Source diagnostic updated trainable parameters')
    if any(not torch.equal(v,buffers[k]) for k,v in model.named_buffers()):
        raise ValueError('Source diagnostic changed model buffers')
    result=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',row_id=case['row_id'],protocol=case['protocol'],
        model_seed=case['model_seed'],epoch=20,checkpoint=str(folder/'final_ssdg.pth'),
        actual_prefix_native_U=False,actual_prefix_EMA=False,ordinary_PL_start_epoch=14,
        recipe=dict(RECIPE),counts=values,scientific_rule=scientific_rule(values),strata=strata,
        source_channel_coverage=dict(simulator.coverage),model_updates=0,head_fit_updates=0,
        source_V_training_reads=0,source_V_diagnostic_reads=values['seen'],U_truth_read=False,target_reads=0,
        query_reads=0,G_reads=0,seconds=time.perf_counter()-start,
        interpretation='Source-V precision proxy only; not U-truth correctness or target performance.')
    write(root/'complete.json',result)
    print({k:result[k] for k in ('status','row_id','seconds','scientific_rule')},flush=True)
    return result
