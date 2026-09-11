"""Bounded real-H0/source technical acceptance, never a scientific source run."""
import argparse,hashlib,io,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from cvsrffi.anchored_source import load_ground,file_digest
from cvsrffi.anchored_cache import CacheIdentity,FeatureRows,SCENES,paired_scene_schedule
from cvsrffi.anchored_geometry import AnchoredMetricHead
from cvsrffi.anchored_fit import FitConfig,fit_expert
from cvsrffi.anchored_reporting import save_json,profile_stages
from cvsrffi.evidence_conditions import received_conditions
from cvsrffi.evidence_target_evaluation import _scene_iq


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('ground','contract','source','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='cpu');args=p.parse_args()
    torch.set_num_threads(4);out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    anchor,payload,contract=load_ground(args.ground,args.contract,args.device)
    source=torch.load(args.source,map_location='cpu',weights_only=True)
    if set(source['physical_ids'])!=set(contract['roles']['L_s']):raise ValueError('source role mismatch')
    selected=[];groups={}
    for i,pid in enumerate(source['physical_ids']):groups.setdefault(tuple(map(int,pid.split(':')[:3])),[]).append(i)
    for group in sorted(groups):selected.extend(groups[group][:2])
    ids=[source['physical_ids'][i] for i in selected];x=source['x'][selected];labels=source['y'][selected]
    receiver=torch.tensor([int(p.split(':')[1]) for p in ids]);day=torch.tensor([int(p.split(':')[2]) for p in ids])
    before={k:v.detach().cpu().clone() for k,v in anchor.original_model.state_dict().items()}
    calls=[];hook=anchor.original_model.dom_backbone.register_forward_hook(lambda *unused:calls.append(1))
    sample_index=[i for y in labels.unique(sorted=True).tolist() for i in torch.where(labels==y)[0][:3].tolist()]
    sample=x[sample_index];opaque=torch.tensor([list(hashlib.sha256(ids[i].encode()).digest()) for i in sample_index],dtype=torch.uint8)
    parity=[]
    for scene in ('clean',*SCENES):
        received=_scene_iq(sample,opaque,scene,392005);calls.clear()
        h,s0,valid=anchor.extract(received);identity_calls=len(calls)
        with torch.no_grad():
            full=anchor.original_model(received.to(args.device),y_tx=None,return_aux=True)['tx_logits']
            fast=anchor.original_model(received.to(args.device),y_tx=None,return_aux=False)
            metric=AnchoredMetricHead(anchor.w0,anchor.tau0).to(args.device)(h)
        for value in (full,fast,metric):torch.testing.assert_close(s0,value,atol=1e-5,rtol=1e-5)
        unique=(s0==s0.amax(-1,keepdim=True)).sum(-1)==1
        assert torch.equal(s0[unique].argmax(-1),full[unique].argmax(-1)) and identity_calls==0
        parity.append(dict(scene=scene,rows=len(received),source_class_counts={str(int(y)):int((labels[sample_index]==y).sum()) for y in labels.unique()},max_abs_full=float((s0-full).abs().max()),max_abs_fast=float((s0-fast).abs().max()),
                           max_abs_metric_identity=float((s0-metric).abs().max()),domain_calls_identity=identity_calls,
                           unique_argmax_equal=True,near_ties=int((s0.topk(2).values.diff().abs()<1e-5).sum()),valid_rows=int(valid.sum())))
    hook.remove()
    schedule=paired_scene_schedule(ids,labels,receiver,day,392005);received=[];view_ids=[];pids=[];ys=[];rxs=[];days=[]
    for i,pid in enumerate(ids):
        key=torch.tensor([list(hashlib.sha256(pid.encode()).digest())],dtype=torch.uint8)
        for scene in ('clean',schedule[pid]):
            received.append(_scene_iq(x[i:i+1],key,scene,392005));view_ids.append(scene);pids.append(pid)
            ys.append(int(labels[i]));rxs.append(int(receiver[i]));days.append(int(day[i]))
    received=torch.cat(received);parts=[anchor.extract(batch) for batch in received.split(64)]
    h=torch.cat([part[0].cpu() for part in parts]);s0=torch.cat([part[1].cpu() for part in parts]);valid=torch.cat([part[2].cpu() for part in parts])
    identity=CacheIdentity(file_digest(args.ground),file_digest(args.contract),'received_iq_v1',{'version':'paired_leo_v1','seed':392005},392005,'L_s')
    rows=FeatureRows(h,s0,received_conditions(received)['quality'],torch.ones_like(h,dtype=torch.bool),torch.tensor(ys),tuple(pids),tuple(view_ids),
                     torch.tensor(rxs),torch.tensor(days),valid,identity).validate()
    out.mkdir(parents=True)
    fit=fit_expert(rows,(1,3,4,6),FitConfig(epochs=2),392005,out/'technical_head_fit',w0=anchor.w0.cpu(),tau0=anchor.tau0,validation_rx=(8,))
    assert fit.history[-1]['optimizer_steps']==2 and fit.state['a'].abs().sum()>0
    assert all(torch.equal(value,anchor.original_model.state_dict()[key].cpu()) for key,value in before.items())
    report=dict(status='REAL_H0_TECHNICAL_VERIFIED',scope='Bounded implementation test only: 180 legal L_s physical records, 2 head steps; not SOURCE_ANALYZED.',
                checkpoint_sha256=identity.checkpoint_sha256,contract_sha256=identity.data_contract_sha256,
                lineage=payload['checkpoint_lineage'],parity=parity,backbone_state_unchanged=True,physical_count=len(ids),cache_rows=len(rows),
                head_optimizer_steps=fit.history[-1]['optimizer_steps'],a=fit.state['a'].tolist(),stop_status=fit.stop_status,
                data_roles_read=['L_s'],V_used=False,target_used=False,source_training_budget_completed=False)
    save_json(out/'real_h0_technical.json',report)
    from cvsrffi.anchored_geometry import load_angle_head
    from cvsrffi.anchored_pipeline import FrozenAnchoredSystem
    from cvsrffi.anchored_fusion import mix_log_probs
    from cvsrffi.evidence_observation import extract_evidence
    head=load_angle_head(fit.state)
    sh,ss0,sv=anchor.extract(sample);sh=sh.cpu();ss0=ss0.cpu();sg=head(sh);sq=received_conditions(sample)['quality']
    system=FrozenAnchoredSystem(anchor,fit.state,fixed_action=2,architecture=payload['baseline_args'],class_labels=contract['tx_mapping'])
    logmix=mix_log_probs(ss0,sg,.5)
    profile=profile_stages({'IQ_augmentation':lambda:_scene_iq(sample,opaque,'leo_clear_weak',392005),
                           'identity_backbone':lambda:anchor.extract(sample),'head':lambda:head(sh),
                           'fusion':lambda:system.predict_scores(ss0,sg,sq,sv),
                           'final_temperature_operator':lambda:(logmix/1.).log_softmax(-1),
                           'save':lambda:torch.save(system.export_state(),io.BytesIO())},warmup=1,repeats=3,device=args.device,cache_hit=True)
    profile.update(batch_size=len(sample),calibration_scope='Timing operator at T=1 only; no V fit and no calibrated scientific result.',stage_devices={name:'cpu' for name in profile['stages']})
    save_json(out/'runtime_profile.json',profile)
    blocks=extract_evidence(anchor.extract_aux(sample),'blocks')
    save_json(out/'real_block_schema.json',dict(names=['t','f','pa'],sizes=list(blocks.block_sizes),joint_width=rows.h.shape[1],block_width=blocks.z.shape[1],scope='Actual identity auxiliary extraction; diagnostic feature masks, not missing IQ reconstruction.'))
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
