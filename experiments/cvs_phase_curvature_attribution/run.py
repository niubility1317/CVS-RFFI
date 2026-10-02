"""Whole-V frozen coefficient ablations; no training, target or selection edits."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
import torch
import torch.nn.functional as F

CONDITIONS=('all_on','curvature_all_off',*(f'curvature_off_{i}' for i in range(6)),'input_mix_off')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

@contextmanager
def intervention(model,condition):
    if condition not in CONDITIONS:raise ValueError('Unregistered source intervention')
    parameters=[*model.memory_parameters(),model.core.behavior[0].mix_raw]
    saved=[p.detach().clone() for p in parameters]
    try:
        with torch.no_grad():
            if condition=='curvature_all_off':
                for p in parameters[:6]:p.zero_()
            elif condition.startswith('curvature_off_'):parameters[int(condition.rsplit('_',1)[1])].zero_()
            elif condition=='input_mix_off':parameters[-1].zero_()
        yield
    finally:
        with torch.no_grad():
            for p,value in zip(parameters,saved):p.copy_(value)

@torch.no_grad()
def evaluate(model,loader,device):
    model.eval();predictions=[];truth=[];rx=[];ids=[];loss=0.;count=0
    start=time.perf_counter()
    for batch in loader:
        x=batch['iq'].to(device);y=batch['label'].to(device);logits=model(x)
        if not torch.isfinite(logits).all():raise ValueError('Nonfinite source logits')
        loss+=float(F.cross_entropy(logits,y,reduction='sum'));count+=len(y)
        predictions.extend(logits.argmax(1).cpu().tolist());truth.extend(batch['label'].tolist());rx.extend(batch['receiver'].tolist())
        ids.extend(item['sample_id'] for item in batch['meta'])
    if count!=27000 or len(set(ids))!=count or set(rx)!={1,3,4,6,8}:raise ValueError('Incomplete source V roles')
    correct=[int(p==y) for p,y in zip(predictions,truth)]
    rates={str(r):sum(c for c,x in zip(correct,rx) if x==r)/sum(x==r for x in rx) for r in sorted(set(rx))}
    return dict(count=count,accuracy=sum(correct)/count,ce=loss/count,rx_accuracy=rates,worst_rx=min(rates.values()),seconds=time.perf_counter()-start),dict(ids=ids,predictions=predictions,truth=truth,receiver=rx,correct=correct)

def execute(config):
    if set(config)!={'source_release','source_commit','source_output','variant','model_seed','output_root','conditions','role','launch_owner'} or config['conditions']!=list(CONDITIONS) or config['role']!='V':
        raise ValueError('Only preregistered source V attribution config allowed')
    release=Path(config['source_release'])
    if (release/'release_commit.txt').read_text().strip()!=config['source_commit']:raise ValueError('Frozen source code version mismatch')
    sys.path[:0]=[str(release),str(release/'code')]
    from experiments.cvs_phase_curvature_identity.dispatch import read_source_record
    from experiments.cvs_phase_curvature_identity.model import build
    from experiments.cvs_identity_ce.source import source_args
    from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags
    from baselines.common.practical_source import build_contract_split
    from baselines.common.cvs_data import make_cvs_loader
    source=Path(config['source_output']);resolved=read(source/'resolved_config.json')
    expected=read(resolved['source_contract'])
    record=read_source_record(config,expected,'cvs_phase_curvature_identity')
    out=Path(config['output_root']);out.mkdir(parents=True,exist_ok=False)
    device=torch.device('cuda:0');torch.set_num_threads(2)
    with numerical_context(resolved['numerical_policy']):
        # Source metadata and full physical roles are checked before any weight load.
        payload=torch.load(source/'last.pt',map_location='cpu',weights_only=False)
        contract=read(source/'source_contract.json');initial=read(source/'initialization.json')
        expected_payload=dict(epoch=200,selection='fixed_last_epoch',method='cvs_phase_curvature_identity',variant=config['variant'],config=resolved,source_contract=contract,initialization=initial,classes=contract['classes'],num_classes=6)
        if any(payload.get(k)!=v for k,v in expected_payload.items()):raise ValueError('Checkpoint payload differs from source provenance')
        model=build(config['variant']).to(device);model.load_state_dict(payload['model'],strict=True);model.eval()
        del payload
        with torch.no_grad():
            probe=torch.zeros(2,2,256,device=device);probe[1,0,:]=1.
            if not torch.isfinite(model(probe)).all():raise ValueError('Checkpoint no-query smoke failed')
        frozen={k:v.detach().clone() for k,v in model.state_dict().items()}
        for p in model.parameters():p.requires_grad_(False)
        args=source_args(dict(resolved,output_root=str(out/'source_loader')))
        split=build_contract_split(args)
        if split.test or split.named_tests or split.split_info['counts']!={'L_s':6300,'U_s':56700,'V':27000}:raise ValueError('Source-only roles changed')
        loader=make_cvs_loader(split.val,batch_size=256,shuffle=False,num_workers=0,device=device,drop_last=False)
        runtime=dict(config,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,backend_flags=actual_flags(),hardware=torch.cuda.get_device_name(device),torch_version=torch.__version__,source_record=record,target_access=False,optimizer_updates=0,source_selection_changed=False)
        write(out/'resolved_config.json',runtime);print('RESOLVED_CONFIG '+json.dumps(runtime),flush=True)
        baseline=None;results=[]
        for condition in CONDITIONS:
            with intervention(model,condition):
                metrics,packet=evaluate(model,loader,device)
            if any(not torch.equal(v,model.state_dict()[k]) for k,v in frozen.items()):raise ValueError('Frozen model changed after intervention')
            if baseline is None:
                if abs(metrics['accuracy']-record['accuracy'])>1e-12 or abs(metrics['worst_rx']-record['worst_rx'])>1e-12:raise ValueError('All-on frozen V does not reproduce E200 source metrics')
                baseline=packet
            if any(packet[k]!=baseline[k] for k in ('ids','truth','receiver')):raise ValueError('Source pairing changed between interventions')
            metrics.update(condition=condition,accuracy_delta_pp_vs_all_on=100*(metrics['accuracy']-sum(baseline['correct'])/len(baseline['correct'])),
                all_on_correct_intervention_wrong=sum(a and not b for a,b in zip(baseline['correct'],packet['correct'])),
                all_on_wrong_intervention_correct=sum(not a and b for a,b in zip(baseline['correct'],packet['correct'])),
                prediction_changes=sum(a!=b for a,b in zip(baseline['predictions'],packet['predictions'])))
            write(out/(condition+'_source_predictions.json'),packet);results.append(metrics)
            print('CONDITION '+json.dumps(metrics),flush=True)
        done=dict(status='SOURCE_ATTRIBUTION_COMPLETE',variant=config['variant'],model_seed=config['model_seed'],rows=results,source_v_count=27000,conditions=list(CONDITIONS),target_access=False,optimizer_updates=0,source_selection_changed=False,model_restored=True,
            original_curvature_raw=[float(p) for p in model.memory_parameters()],original_input_mix_raw=model.core.behavior[0].mix_raw.cpu().tolist(),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
            claim='Within-checkpoint source V intervention; not scratch retraining causal effect, not unseen-RX generalization, not unique TX hardware attribution.')
        write(out/'completion.json',done);print(json.dumps(done),flush=True)
    return done

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);execute(read(p.parse_args().config))
