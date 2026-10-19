"""Frozen external heads on the exact cached IQ used by P79's AS30 student."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
from experiments.cvs_sameiq_experts import protocol as p

RUN = '20261019-diagnostic-FcrossRX-sameIQ-experts-s2026092701-m2-r01'
P43 = '20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
P79 = '20261019-diagnostic-FcrossRX-current-controls-e30-s2026092701-m2-r01'
HISTORY = '20260930-phase1-native-baselines-practical-manysig-m5-r01'
PANEL = '20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def analyze(teacher, student, y, rx, day, physics):
    import numpy as np
    result = {'clean': p.paired(teacher['clean'].argmax(-1), student['clean'].argmax(-1), y), 'observations': []}
    bydraw = {}
    for draw in (0, 1):
        for si, scene in enumerate(p.SCENES):
            t = teacher['satellite'][draw, si].argmax(-1)
            s = student['satellite'][draw, si, 0].argmax(-1)
            groups = {'ALL': np.ones(p.N, dtype=bool)}
            for name, values in [('RX',rx),('day',day),('TX',y),
                                 ('state',physics['state'][draw,si,0]),('lock',physics['lock'][draw,si,0])]:
                for value in np.unique(values):groups[f'{name}={int(value)}'] = values == value
            noise = physics['signal_to_added_noise_db'][draw,si,0]
            for name, mask in [('noise<3',noise<3),('noise3to8',(noise>=3)&(noise<8)),
                               ('noise8to15',(noise>=8)&(noise<15)),('noise>=15',noise>=15),
                               ('noise_missing',np.isnan(noise))]:groups[name]=mask
            for group, mask in groups.items():
                if not mask.any():continue
                row=dict(draw=draw,scene=scene,group=group,**p.paired(t[mask],s[mask],y[mask]))
                result['observations'].append(row)
                if group=='ALL':bydraw[(draw,scene)]=row
    result['qualification'] = p.qualification(bydraw)
    return result


def run(project, device):
    import numpy as np
    import torch
    project=Path(project).resolve()
    if not project.is_absolute() or os.geteuid()==0:raise ValueError('Ordinary N607 account and absolute project required')
    out=project/'runs'/RUN
    out.mkdir(exist_ok=False)
    start=time.perf_counter()
    try:
        torch.set_num_threads(2)
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.set_float32_matmul_precision('highest')
        current=read(project/'runs'/P43/'B01-s2026092701/source/source_contract.json')
        hist=project/'paper_reproduction/runs'/HISTORY
        # Import the same factory that trained these checkpoints, without source loaders.
        historical_release=project/'releases/native_comparisons_20260930_r01'
        sys.path.insert(0,str(historical_release))
        from comparison_suite.models import factory, classifier_logits
        models={};snapshots={};origins={};nn=0
        for method in p.METHODS:
            root=hist/(method+'-s2026092701')
            contract=read(root/'source_contract.json');init=read(root/'initialization.json')
            done=read(root/'completion.json');config=read(root/'resolved_config.json')
            p.provenance(method,current,contract,init,done,config)
            checkpoint=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
            if (checkpoint['source_contract']['role_ids']!=current['role_ids']
                    or checkpoint['classes']!=list(p.CLASSES)
                    or checkpoint['class_mapping']!={c:i for i,c in enumerate(p.CLASSES)}
                    or checkpoint['epoch']!=200 or checkpoint['selection']!='fixed_last_epoch'
                    or checkpoint['initialization']!=init or checkpoint['method']!=method
                    or checkpoint['config']!=config):raise ValueError('Checkpoint payload/provenance differs')
            model=factory(method,checkpoint['num_classes'],checkpoint['num_receivers'],**checkpoint['model_kwargs'])
            model.load_state_dict(checkpoint['model'],strict=True)
            model.to(device).eval().requires_grad_(False)
            if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite external checkpoint')
            snapshots[method]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            with torch.inference_mode():smoke=classifier_logits(model,method,torch.zeros(2,2,256,device=device))
            if smoke.shape!=(2,6) or not torch.isfinite(smoke).all():raise ValueError('Real checkpoint smoke failed')
            nn+=2;models[method]=model
            origins[method]=dict(checkpoint=str(root/'last.pt'),scratch=True,ancestors=[],fixed_epoch=200,
                                 updates=9800,source_role_ids_exact=True,target_contact_before_freeze=False,
                                 model_seed=2026092701,classes=list(p.CLASSES),factory_file=importlib.import_module('comparison_suite.models').__file__)
        write(out/'checkpoint_smoke.json',dict(status='PASS',NN=nn,source_IQ_before_smoke=0,query_access=False,provenance=origins))
        print('SMOKE PASS: both frozen native heads; 4 synthetic packets; 0 query',flush=True)
        # Source-only inputs are opened only after the two real-checkpoint smokes.
        panelroot=project/'runs'/PANEL/'diagnostic'
        roles=read(panelroot/'data_roles.json');obs=read(panelroot/'observations/completion.json')
        if obs['target_access'] or obs['draws']!=2 or obs['conditions'][0]!='full':raise ValueError('Original source observations required')
        original=read(project/'runs'/P79/'completion.json')
        if original.get('status')!='SOURCE_DIAGNOSTIC_COMPLETE' or original.get('target_access') is not False:raise ValueError('Completed source AS30 reference required')
        with np.load(project/'runs'/P79/'AS30/predictions.npz',allow_pickle=False) as f:student={k:f[k].copy() for k in f.files}
        ids=student['ids'].tolist();y=student['TX'];rx=student['RX'];day=student['day']
        p.panel_contract(ids,y,rx,day,roles,current)
        if student['clean'].shape!=(1800,6) or student['satellite'].shape!=(2,3,11,1800,6):raise ValueError('P79 reference shape differs')
        cachepath=project/'runs/20261011-phase1-stage2-geometry-source-manysig-m3-r01/s2026092701'
        cachemeta=read(cachepath/'preparation_complete.json')
        if cachemeta.get('status')!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or cachemeta.get('target_access') or cachemeta.get('identity_updates'):raise ValueError('Existing legal source cache required')
        cache=torch.load(cachepath/'source_cache.pt',map_location='cpu',weights_only=False)
        if cache['contract']['role_ids']!=current['role_ids']:raise ValueError('Source cache contract mismatch')
        v=cache['validation'];lookup={pid:i for i,pid in enumerate(v['ids'])}
        if v['role']!='V' or set(v['ids'])!=set(current['role_ids']['V']):raise ValueError('Validation cache role differs')
        index=torch.tensor([lookup[pid] for pid in ids]);x=v['x'][index].clone()
        for field,want in [('y',y),('rx',rx),('day',day)]:
            if not np.array_equal(v[field][index].numpy(),want):raise ValueError('Source row order/labels differ')
        del cache,v
        if x.shape!=(1800,2,256) or not torch.isfinite(x).all():raise ValueError('Finite original clean IQ required')
        iq=np.load(panelroot/'observations/iq.npy',mmap_mode='r',allow_pickle=False)
        if iq.shape!=(2,3,11,1800,2,256) or iq.dtype!=np.float32:raise ValueError('Original observation shape/type differs')
        with np.load(panelroot/'observations/physics.npz',allow_pickle=False) as f:physics={k:f[k].copy() for k in f.files}
        results={}
        with torch.inference_mode():
            for method,model in models.items():
                dest=out/method;dest.mkdir()
                clean=np.empty((1800,6),np.float32);sat=np.empty((2,3,1800,6),np.float32)
                for begin in range(0,1800,90):
                    clean[begin:begin+90]=classifier_logits(model,method,x[begin:begin+90].to(device)).cpu().numpy();nn+=90
                for draw in (0,1):
                    for si,scene in enumerate(p.SCENES):
                        for begin in range(0,1800,90):
                            batch=np.array(iq[draw,si,0,begin:begin+90],copy=True)
                            if not np.isfinite(batch).all():raise ValueError('Nonfinite cached observation')
                            t=torch.from_numpy(batch).to(device)
                            sat[draw,si,begin:begin+90]=classifier_logits(model,method,t).cpu().numpy();nn+=90
                if not np.isfinite(clean).all() or not np.isfinite(sat).all():raise ValueError('Nonfinite predictions')
                np.savez_compressed(dest/'predictions.npz',ids=np.asarray(ids),clean=clean,satellite=sat)
                result=analyze(dict(clean=clean,satellite=sat),student,y,rx,day,physics)
                write(dest/'comparison.json',result);results[method]=result
                if model.training or any(m.training for m in model.modules()) or any(q.requires_grad or q.grad is not None for q in model.parameters()):raise ValueError('Frozen model state changed')
                if any(not torch.equal(value,model.state_dict()[key].detach().cpu()) for key,value in snapshots[method].items()):raise ValueError('Model weights or BatchNorm buffers changed')
                print(json.dumps(dict(method=method,qualification=result['qualification'],NN=nn)),flush=True)
        if nn!=25204:raise ValueError('Fixed source forward budget differs')
        complete=dict(status='SOURCE_DIAGNOSED',run_id=RUN,methods=list(p.METHODS),NN=nn,
                      source_unique_V=1800,channel_draws=2,model_seed_count=1,optimizer_updates=0,
                      existing_PAMF_predictions_reused=True,query_access=False,target_access=False,
                      new_IQ_generated=0,U_truth_read=False,state_unchanged=True,
                      wall_seconds=time.perf_counter()-start,qualification={k:v['qualification'] for k,v in results.items()},
                      goal_complete=False)
        write(out/'completion.json',complete);print(json.dumps(complete),flush=True)
        return complete
    except BaseException as exc:
        write(out/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),outputs_preserved=True,automatic_retry=False))
        raise


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--project',required=True);a.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0')
    args=a.parse_args();run(args.project,args.device)
