import argparse,json,os,sys,time
from pathlib import Path
import torch
from . import design as d
from .graph import fit,predict,relation_diagnostics
from .gradients import audit
from experiments.cvs_report_ssl.source import load_source
from experiments.cvs_report_ssl.model import build,Teacher
from experiments.cvs_report_ssl.auxiliary import PhysicalHeads
from experiments.cvs_stage2_decouple.source import seed_all
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags

def verify_source(c):
    from experiments.cvs_report_ssl import design as original
    config=c['source_config'];original.validate(config)
    folder=Path(config['output_root']);done=d.read(folder/'completion.json');frozen=d.read(folder.parent/'source_frozen.json')
    if any(done.get(k)!=v for k,v in dict(status='SOURCE_TRAINED',epoch=40,optimizer_steps=8880,logged_steps=8880,target_access=False,config=config).items()):raise ValueError('Frozen source provenance differs')
    if frozen['config']!=config or frozen['target_access'] or frozen['target_scores_consumed'] or frozen['selection']!='final_only':raise ValueError('Source selection provenance differs')
    aux=torch.load(c['teacher_and_heads'],map_location='cpu',weights_only=False)
    if aux['teacher_updates']!=7992 or aux['selected_for_test'] or aux['frontend_version']!=config['frontend']['version'] or aux['teacher'] is None:raise ValueError('Frozen auxiliary/teacher source differs')
    return aux

@torch.no_grad()
def extract(model,cache,device):
    if model.training:raise ValueError('Frozen feature extraction requires eval mode')
    features=[];probabilities=[]
    for start in range(0,len(cache['x']),256):
        x=cache['x'][start:start+256].to(device);out=model.forward_identity_only(x)
        features.append(torch.nn.functional.normalize(out['id_feat_joint'],dim=1));probabilities.append(out['tx_logits'].softmax(1))
    return torch.cat(features),torch.cat(probabilities)

def paired(q,base,y):
    correct=q.argmax(1)==y;old=base.argmax(1)==y;disagree=q.argmax(1)!=base.argmax(1)
    return dict(count=len(y),accuracy=float(correct.float().mean()),base_accuracy=float(old.float().mean()),
        fix_count=int((~old&correct).sum()),break_count=int((old&~correct).sum()),net_corrections=int((~old&correct).sum())-int((old&~correct).sum()),
        disagreement_count=int(disagree.sum()),accuracy_on_disagreement=float(correct[disagree].float().mean()) if disagree.any() else None,
        base_accuracy_on_disagreement=float(old[disagree].float().mean()) if disagree.any() else None)


def metrics(probabilities,y,rx,day):
    result={}
    for name,q in probabilities.items():
        confidence,pred=q.max(1);correct=pred==y;rows=[]
        baseline=probabilities['model'];baseline_order=torch.argsort(baseline.max(1).values,descending=True,stable=True)
        for coverage in (.5,.9,.95,1.):
            idx=torch.argsort(confidence,descending=True,stable=True)[:int(len(q)*coverage)]
            base_idx=baseline_order[:len(idx)];common=idx[torch.isin(idx,base_idx)]
            rows.append(dict(coverage=coverage,count=len(idx),error_rate=float((~correct[idx]).float().mean()),NLL=float(-q[idx,y[idx]].clamp_min(1e-8).log().mean()),
                on_baseline_selected=paired(q[base_idx],baseline[base_idx],y[base_idx]),
                on_common_selected=paired(q[common],baseline[common],y[common]) if len(common) else None))
        byrx={str(r):float(correct[rx==r].float().mean()) for r in rx.unique().tolist()}
        result[name]=dict(accuracy=float(correct.float().mean()),RX_accuracy=byrx,worst_RX_accuracy=min(byrx.values()),fixed_coverage=rows,
            paired=paired(q,baseline,y),by_stratum={key:{str(v):paired(q[values==v],baseline[values==v],y[values==v]) for v in values.unique().tolist()} for key,values in [('RX',rx),('day',day),('TX',y)]})
    return result


@torch.no_grad()
def geometry(sf,sq,ef,eq):
    distance=(sf-ef).norm(dim=1);cos=(sf*ef).sum(1).clamp(-1,1)
    return dict(count=len(sf),normalized_feature_L2_mean=float(distance.mean()),normalized_feature_L2_p95=float(distance.quantile(.95)),
        feature_cosine_mean=float(cos.mean()),feature_cosine_p05=float(cos.quantile(.05)),
        prediction_disagreement_count=int((sq.argmax(1)!=eq.argmax(1)).sum()),probability_L1_mean=float((sq-eq).abs().sum(1).mean()),
        student_confidence_mean=float(sq.max(1).values.mean()),EMA_confidence_mean=float(eq.max(1).values.mean()),
        accuracy=None,reason='Role-independent paired comparison; no labels consumed')

def run(row_id,device='cuda:0'):
    row=next(r for r in d.rows() if r['row_id']==row_id);c=d.config(row);d.validate(c)
    out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    d.write(out/'resolved_config.json',dict(config=c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,
        commit=(d.ROOT/'release_commit.txt').read_text().strip(),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),target_loaders_constructed=False))
    torch.set_num_threads(2);seed_all(c['model_seed'])
    with numerical_context(d.parent.FULL_FP32_POLICY):
        aux=verify_source(c);L,U,V,contract,provenance=load_source(c['source_config'])
        model=build(c['source_config'],device,checkpoint=c['source_checkpoint']).eval();teacher=Teacher(model,.99)
        teacher.model.load_state_dict(aux['teacher'],strict=True)
        heads=PhysicalHeads(c['source_config']['enabled']['linear'],False).to(device);heads.load_state_dict(aux['heads'],strict=True)
        d.write(out/'provenance.json',dict(original_E200=provenance,diagnostic_source_checkpoint=c['source_checkpoint'],teacher_source=c['teacher_and_heads'],source_config=c['source_config'],identity_updates=0,target_files_read=False))
        torch.cuda.reset_peak_memory_stats()
        student_features={};geometry_results={};all_results={};graph_results={};V_models={}
        for coordinate,current in [('student',model),('EMA',teacher.model)]:
            lf,lq=extract(current,L,device);uf,uq=extract(current,U,device)
            print('SOURCE_FEATURES '+json.dumps(dict(coordinate=coordinate,L=len(lf),U=len(uf),U_TX_truth=False)),flush=True)
            graph=fit(lf,L['y'].to(device),uf,uq,c['graph'])
            graph['diagnostics'].update(relation_diagnostics(graph,torch.cat((L['rx'],U['rx'])).to(device)))
            graph_results[coordinate]=graph['diagnostics'];d.write(out/'graph_fit.json',graph_results)
            print('GRAPH_FROZEN '+json.dumps(dict(coordinate=coordinate,**graph['diagnostics'])),flush=True)
            # V is introduced only after this graph is frozen, and never mutates either fit.
            vf,vq=extract(current,V,device);predictions=predict(graph,vf,vq)
            all_results[coordinate]=metrics(predictions,V['y'].to(device),V['rx'].to(device),V['day'].to(device));V_models[coordinate]=vq
            for role,f,q in [('L',lf,lq),('U',uf,uq),('V',vf,vq)]:
                if coordinate=='student':student_features[role]=(f,q)
                else:geometry_results[role]=geometry(*student_features[role],f,q)
            del graph,lf,lq,uf,uq,vf,vq,predictions
        d.write(out/'V_readouts.json',dict(source_role='V',coordinates=all_results,EMA_vs_student=paired(V_models['EMA'],V_models['student'],V['y'].to(device)),fit_roles=['L_s','U_s'],target_access=False))
        d.write(out/'student_EMA_geometry.json',dict(roles=geometry_results,temporal_scope='E40 endpoints only; not E200-to-E40 movement',U_truth_access=False,target_access=False))
        del student_features,V_models
        gradients=audit(c,model,teacher,heads,L,U,device);d.write(out/'gradient_diagnostics.json',gradients)
        d.write(out/'completion.json',dict(status='SOURCE_DIAGNOSTIC_COMPLETE',row_id=row_id,config=c,identity_updates=0,
            target_access=False,U_truth_access=False,V_in_fit=False,virtual_steps_restored_exactly=True,
            elapsed_seconds=time.perf_counter()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            hardware=torch.cuda.get_device_name(),backend=actual_flags(),gradient_batches=len(gradients['rows']),source_model_unchanged=True))
        print('SOURCE_DIAGNOSTIC_COMPLETE '+row_id,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--row',required=True);args=parser.parse_args();run(args.row)
