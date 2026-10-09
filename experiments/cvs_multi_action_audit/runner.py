"""Use only source-L packets, fixed reference and independently fitted audit models."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from experiments.cvs_multi_action_audit import design as d
from experiments.cvs_multi_disentangle.model import intermediate, identity_from_intermediate, classify_intermediate
from experiments.cvs_multi_disentangle.physics import factorial_batch
from experiments.cvs_multi_disentangle.runtime import installed
from experiments.cvs_multi_disentangle.evaluate import source_provenance, checkpoint_provenance
from experiments.cvs_equivariant_identity.precision import numerical_context


def role_split(records, expected_ids, fit_n=64, audit_n=32, seed=20261009):
    """Records contain source-L metadata only. Hash order has no signal/score input."""
    ids=[r['id'] for r in records]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected_ids):
        raise ValueError('Source-L physical IDs differ')
    groups=defaultdict(list)
    for r in records:
        if r['role']!='L_s' or r['y']<0: raise ValueError('Only visible source-L allowed')
        groups[(r['y'],r['rx'])].append(r)
    fit=[]; audit=[]
    for key,items in sorted(groups.items()):
        if len(items)<fit_n+audit_n: raise ValueError('Insufficient registered source stratum '+str(key))
        ordered=sorted(items,key=lambda r:hashlib.sha256((str(seed)+':'+r['id']).encode()).hexdigest())
        fit.extend(ordered[:fit_n]);audit.extend(ordered[fit_n:fit_n+audit_n])
    if set(r['id'] for r in fit)&set(r['id'] for r in audit): raise ValueError('Role overlap')
    return fit,audit


def provenance(c):
    d.validate(c)
    source=source_provenance(d.parent,c['parent_config'])
    ck=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
    checkpoint_provenance(d.parent,c['parent_config'],ck)
    return ck,dict(status='VERIFIED',checkpoint=c['checkpoint'],parent_run=d.parent.RUN,
        parent_arm='multi',selection='ALL_FOUR_FIXED_E200_MULTI_ROWS_NO_RANKING',
        parent_initialization='scratch; no ancestors; exact source physical roles',
        parent_training_target_access=False,prior_target_scores_consumed=False,
        teacher_state='final student model, fixed for diagnostic',
        audit_scope='unseen by newly fitted auxiliaries; seen by original identity training',
        identity_training=False,target_access=False)


def source_packets(native,c,device):
    from cvsrffi.xuc_fusion.native import role_ids_from_native
    from cvsrffi.game_tracking.data import opaque_id
    a=d.parent.make_args(c['parent_config'],str(device))
    ctx=native._build_ssdg_wisig_data(a,device)
    expected=d.read(d.SOURCE)
    if ctx['named_test_loaders'] or role_ids_from_native(ctx)!=expected['role_ids']:
        raise ValueError('Actual source roles differ or target loader constructed')
    ds=ctx['train_loader'].dataset
    records=[]
    for i,item in enumerate(ds.index):
        records.append(dict(id=opaque_id(item),index=i,role='L_s',y=int(item.tx_i),rx=int(item.rx_i)))
    if sorted({r['rx'] for r in records})!=d.RECIPE['source_receivers'] or len({r['y'] for r in records})!=6:
        raise ValueError('Source TX/RX coverage differs')
    fit,audit=role_split(records,expected['role_ids']['L_s'])
    def load(rows):
        values=[ds[r['index']] for r in rows]
        return dict(x=torch.stack([v[0] for v in values]).to(device),
            y=torch.tensor([int(v[1]) for v in values],device=device),
            rx=torch.tensor([int(v[3]['rx_i']) for v in values],device=device),
            ids=[r['id'] for r in rows])
    return load(fit),load(audit),dict(fit=fit,audit=audit,unselected_count=len(records)-len(fit)-len(audit),
        source_contract=d.SOURCE,source_roles='EXACT_MATCH',U_labels_read=False,V_read=False,target_read=False)


def channel_view(data):
    from leo_practical.channel import Config
    from leo_practical.batch import apply_leo_practical_channel_batch
    cfg=Config(fs_hz=25000000.,fc_hz=2462000000.,scenario='practical_mid',
               processing_route='residual',mode='post_sync')
    x,_,_=apply_leo_practical_channel_batch(data['x'],cfg,seed=d.RECIPE['augmentation_seed'],
        receiver_seed=2027,sample_ids=data['ids'],session_ids=['source-rx:'+str(r) for r in data['rx'].tolist()],
        realization_namespace=d.RUN+':source_audit',return_meta=False)
    return dict(data,x=x)


def gradient_audit(identity,x,y,generator):
    """Source gradient diagnostics only. No optimizer step, no model/buffer writes."""
    state={n:p.requires_grad for n,p in identity.named_parameters()}
    for p in identity.parameters():p.requires_grad_(True)
    identity.eval()
    params=[(n,p) for n,p in identity.named_parameters()]
    batch=factorial_batch(x,generator)
    with torch.no_grad():
        h0=intermediate(identity,x);h1=intermediate(identity,batch['x10']);h2=intermediate(identity,batch['x01'])
    h=intermediate(identity,x)
    losses={'clean_CE_reference':F.cross_entropy(identity(x),y),
        'real_L_CE':F.cross_entropy(identity(batch['x10']),y),
        'real_T_CE':F.cross_entropy(identity(batch['x01']),y),
        'exact_feature_L_CE':F.cross_entropy(classify_intermediate(identity,h+(h1-h0).detach()),y),
        'exact_feature_T_CE':F.cross_entropy(classify_intermediate(identity,h+(h2-h0).detach()),y)}
    gradients={k:torch.autograd.grad(v,[p for _,p in params],retain_graph=True,allow_unused=True) for k,v in losses.items()}
    result=[]
    for component in ('all','base','pa','G'):
        selected=[i for i,(n,_) in enumerate(params) if component=='all' or
            (component=='G' and any(t in n for t in ('cls_head','response_projection','response_gain','classifier'))) or
            (component=='pa' and 'pa' in n and 'cls_head' not in n) or
            (component=='base' and 'pa' not in n and not any(t in n for t in ('cls_head','response_projection','response_gain','classifier')))]
        def vec(key):
            return torch.cat([(gradients[key][i] if gradients[key][i] is not None else torch.zeros_like(params[i][1])).flatten() for i in selected]) if selected else x.new_zeros(1)
        base=vec('clean_CE_reference');bn=base.norm()
        for key in losses:
            v=vec(key);vn=v.norm()
            result.append(dict(component=component,loss=key,value=float(losses[key].detach()),norm=float(vn),
                norm_ratio=float(vn/bn) if bn>1e-12 else None,
                cosine=float(torch.dot(v,base)/(vn*bn)) if vn*bn>1e-12 else None))
    for n,p in params:p.requires_grad_(state[n])
    return dict(reference='clean labeled CE, NOT full native clean+LEO+pseudo objective',
        actual_identity_update=False,learned_action_gradient='deferred until action reliability established',rows=result)


def gradient_indices(data):
    """One packet per TX/RX stratum, never the first sorted group alone."""
    selected={}
    for i,(y,rx) in enumerate(zip(data['y'].tolist(),data['rx'].tolist())):
        selected.setdefault((int(y),int(rx)),i)
    if len(selected)>32:raise ValueError('Gradient diagnostic would exceed source-L budget')
    return torch.tensor(list(selected.values()),device=data['x'].device),[list(k) for k in selected]


def save_result(output,name,result):
    result=dict(result)
    for key in ('state_dicts','state_dict'):
        if key in result:torch.save(result.pop(key),output/(name+'_'+key+'.pth'))
    report=result.pop('report',result.copy())
    d.write(output/(name+'.json'),report)
    logs=result.get('step_logs',report.get('step_logs',[])) if isinstance(report,dict) else []
    if logs:
        with (output/(name+'_steps.jsonl')).open('w',encoding='utf-8') as f:
            for row in logs:f.write(json.dumps(row,allow_nan=False)+'\n')
        keys=sorted({k for r in logs for k,v in r.items() if not isinstance(v,(dict,list))})
        with (output/(name+'_steps.csv')).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(logs)


def run(c,smoke=False):
    d.validate(c);torch.set_num_threads(2);device=torch.device('cpu' if smoke else 'cuda:0')
    started=time.perf_counter();torch.manual_seed(c['model_seed'])
    with numerical_context(d.parent.FULL_FP32_POLICY),installed(c['parent_config'],training=False) as native:
        ck,proof=provenance(c)
        model=native.build_baseline_model(SimpleNamespace(**ck['baseline_args']),device)
        model.load_state_dict(ck['model'],strict=True);model.eval();del ck
        with torch.no_grad():z=model(torch.zeros(2,2,256,device=device))
        if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Checkpoint no-query smoke failed')
        if smoke:return dict(proof,checkpoint_smoke='PASS')
        output=Path(c['output_root']);output.mkdir(parents=True,exist_ok=False)
        d.write(output/'provenance.json',proof)
        d.write(output/'resolved_config.json',dict(c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,
            gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),hardware=torch.cuda.get_device_name(),
            commit=(d.ROOT/'release_commit.txt').read_text().strip()))
        fit,audit,split=source_packets(native,c,device);d.write(output/'physical_roles.json',split)
        identity=model.id_backbone;identity.eval()
        saved={n:p.detach().clone() for n,p in identity.named_parameters()}
        for p in identity.parameters():p.requires_grad_(False)
        from experiments.cvs_multi_action_audit.actions import fit_and_audit, composition_audit
        from experiments.cvs_multi_action_audit.receiver import receiver_audit
        for exposure in d.EXPOSURES:
            print('AUDIT_CONFIG '+json.dumps(dict(exposure=exposure,recipe=d.RECIPE,fit=len(fit['ids']),audit=len(audit['ids']))),flush=True)
            f,a=(fit,audit) if exposure=='clean' else (channel_view(fit),channel_view(audit))
            gen=torch.Generator().manual_seed(d.RECIPE['evaluation_seed'])
            def progress(row):
                if row['step']==1 or row['step']%25==0:
                    print('ACTION '+json.dumps(dict(exposure=exposure,**row)),flush=True)
            actions=fit_and_audit(identity,f['x'],f['y'],a['x'],a['y'],gen,
                fit_ids=f['ids'],audit_ids=a['ids'],steps=d.RECIPE['steps'],batch_size=32,
                learning_rate=d.RECIPE['learning_rate'],parameter_weight=1.,alpha_z=1.,
                fit_modes=tuple(d.RECIPE['fit_modes']),on_step=progress)
            d.write(output/(exposure+'_composition.json'),composition_audit(identity,a['x'],a['y'],
                torch.Generator().manual_seed(d.RECIPE['evaluation_seed']+2),actions['state_dicts']))
            save_result(output,exposure+'_actions',actions)
            receiver=receiver_audit(identity,f['x'],f['y'],f['rx'],a['x'],a['y'],a['rx'],
                torch.Generator().manual_seed(d.RECIPE['evaluation_seed']+1),
                fit_ids=f['ids'],audit_ids=a['ids'],steps=d.RECIPE['steps'])
            save_result(output,exposure+'_receiver',receiver)
            gi,coverage=gradient_indices(a)
            gradients=gradient_audit(identity,a['x'][gi],a['y'][gi],gen)
            gradients.update(tx_rx_coverage=coverage,packets=len(gi),selection='one fixed heldout physical packet per TX/RX')
            d.write(output/(exposure+'_gradients.json'),gradients)
            print('EXPOSURE_COMPLETE '+exposure,flush=True)
        if any(not torch.equal(saved[n],p.detach()) for n,p in identity.named_parameters()):
            raise ValueError('Frozen identity parameters changed')
        d.write(output/'completion.json',dict(status='SOURCE_AUDIT_COMPLETE',row_id=c['row_id'],
            exposures=list(d.EXPOSURES),identity_unchanged=True,target_access=False,
            elapsed_seconds=time.perf_counter()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            no_new_target_accuracy=True,physical_role_counts={k:len(split[k]) for k in ('fit','audit')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--row',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    row=next(r for r in d.rows() if r['row_id']==a.row)
    result=run(d.config(row),a.smoke)
    if result:print(json.dumps(result),flush=True)
