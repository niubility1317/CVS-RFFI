"""One source V diagnostic panel, shared unchanged across all five rows."""
import os
import time
import numpy as np
import torch
from . import train_design as d
from . import probe_design as p
from .train_channel import practical,audit


def publish_panel(root,value):
    final=root/'source_panel_results.json'
    if final.exists():raise FileExistsError('Preserve an already published source panel')
    temporary=root/'source_panel_results.pending.json'
    d.write(temporary,value)
    os.replace(temporary,final)


def prepare():
    d.source_safety()
    from experiments.cvs_sigtime_rdrop_legacy.provenance import own_model
    ref=own_model(d.parent_recipe(2026092701),'cpu')
    with torch.no_grad():smoke=ref.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
    if smoke.shape!=(2,6) or not bool(torch.isfinite(smoke).all()):raise ValueError('Real checkpoint/default-head no-query smoke failed')
    d.write(d.BASE/'source_smoke.json',dict(status='PASS',forward_packets=2,query_read=False,
        checkpoint=p.PARENT+'/final_ssdg.pth',initializer=False))
    print('SMOKE PASS fixed E20 default head, no query; scratch matrix continues',flush=True)
    from experiments.cvs_phase1_stack.recover import verify_contract
    prep=d.read(__import__('pathlib').Path(p.PREPARATION)/'preparation_complete.json')
    if prep['target_access'] or prep['identity_updates'] or prep['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE':raise ValueError('Raw source preparation differs')
    pack=torch.load(__import__('pathlib').Path(p.PREPARATION)/'source_cache.pt',map_location='cpu',weights_only=False)
    verify_contract(pack['contract'],d.read(d.SOURCE));verify_contract(prep['source_contract'],pack['contract'])
    V={k:pack['validation'][k] for k in ('x','y','rx','day','ids')};del pack
    from .probe import validate_source
    validate_source(V,d.read(d.SOURCE))
    selected=[];rng=np.random.default_rng(20261015033)
    for tx in range(6):
        for rx in (1,3,4,6,8):
            for day in (1,2,3):
                candidates=torch.where((V['y']==tx)&(V['rx']==rx)&(V['day']==day))[0].tolist()
                selected.extend(rng.choice(candidates,8,replace=False).tolist())
    idx=torch.tensor(selected);panel={k:V[k][idx] for k in ('x','y','rx','day')};panel['ids']=[V['ids'][i] for i in selected]
    meta=dict(rx_i=panel['rx'].tolist(),day_i=panel['day'].tolist(),eq_i=[1]*720,ids=panel['ids'])
    views={};details={};started=time.perf_counter()
    for scene in d.SCENES:
        views[scene],details[scene]=practical(panel['x'],meta,0,0,source_scene=scene)
    with torch.no_grad():parent_predictions=ref.forward_identity_only(panel['x'])['tx_logits'].argmax(1)
    fixed=parent_predictions==panel['y']
    torch.save(dict(panel=panel,views=views,metadata=details,fixed_parent_clean_correct=fixed,
                    parent_clean_predictions=parent_predictions),d.BASE/'source_panel.pt')
    d.write(d.BASE/'source_panel.json',dict(source_role='V diagnostics, no new role',count=720,seed=20261015033,
        source_only=True,U_opened=False,query_read=False,V_updates=0,physical_ids=panel['ids'],
        fixed_parent_clean_correct=int(fixed.sum()),shared_preparation_forward_packets=722,
        actual_channel=audit(),generation_seconds=time.perf_counter()-started))


@torch.no_grad()
def evaluate(c):
    from .train_model import own_model
    from .probe import summary
    model=own_model(c,'cuda:0');loaded=torch.load(d.BASE/'source_panel.pt',map_location='cpu',weights_only=False)
    panel=loaded['panel'];y=panel['y'];fixed=loaded['fixed_parent_clean_correct'];results={};predictions={}
    before={k:v.clone() for k,v in model.state_dict().items()}
    for view in d.VIEWS:
        x=panel['x'] if view=='clean' else loaded['views'][view];logits=[]
        for start in range(0,720,128):logits.append(model.forward_identity_only(x[start:start+128].to('cuda:0'))['tx_logits'].cpu())
        logits=torch.cat(logits);pred=logits.argmax(1)
        if not bool(torch.isfinite(logits).all()):raise ValueError('Source panel head must be finite')
        wrong=logits.masked_fill(torch.nn.functional.one_hot(y,6).bool(),-torch.inf).max(1).values
        margins=(logits.gather(1,y[:,None]).squeeze(1)-wrong)/30
        ref=loaded['parent_clean_predictions']
        result=summary(pred,y,margins,ref)
        result['RX']={str(rx):summary(pred[panel['rx']==rx],y[panel['rx']==rx],margins[panel['rx']==rx],ref[panel['rx']==rx]) for rx in (1,3,4,6,8)}
        if view!='clean':
            def quality(db):
                if db is None:return 'unknown'
                return '<0' if db<0 else '0-3' if db<3 else '3-10' if db<10 else '10-20' if db<20 else '>=20'
            groups={}
            for i,meta in enumerate(loaded['metadata'][view]):
                key=(meta['state_end'],meta['receiver_lock_status'],quality(meta['signal_to_added_noise_db']))
                groups.setdefault(key,[]).append(i)
            result['state_quality']=[dict(state=k[0],lock=k[1],added_noise_SNR_bin=k[2],
                **summary(pred[torch.tensor(ix)],y[torch.tensor(ix)],margins[torch.tensor(ix)],ref[torch.tensor(ix)]))
                for k,ix in sorted(groups.items())]
        results[view]=result;predictions[view]=pred
    if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):raise ValueError('Source diagnostics changed frozen state')
    root=__import__('pathlib').Path(c['output_root'])
    torch.save(dict(ids=panel['ids'],predictions=predictions),root/'source_panel_decisions.pt')
    publish_panel(root,dict(source_only=True,target_metrics_read=False,config=c,
        views=results,forward_packets=5040,parameter_updates=0,EMA_updates=0,head_fit_updates=0))


def qualification():
    rows={r['arm']:d.config(r) for r in d.rows()};full={};panels={}
    for arm,c in rows.items():
        root=__import__('pathlib').Path(c['output_root']);done=d.read(root/'completion.json')
        if done['config']!=c or done['epoch']!=20 or done['optimizer_steps']!=4440 or done['target_access']:raise ValueError('Every source row must freeze final20')
        full[arm]=done['final_source_metrics'];panels[arm]=d.read(root/'source_panel_results.json')['views']
    clean=lambda a:full[a]['source_val_accuracy']
    urban=lambda a:sum(panels[a][s]['accuracy'] for s in d.SCENES[3:])/3
    low=lambda a:panels[a]['practical_low_urban']['accuracy']
    checks=dict(B10_clean_gain=clean('B10')>=clean('B00')+.005,
        B11_clean_gain=clean('B11')>=max(clean('B00'),clean('R_LEGACY')),
        B11_low_gain=low('B11')>=low('B01')+.01,B11_urban_gain=urban('B11')>=urban('B01'),
        clean_health=clean('B11')>=clean('R_LEGACY')-.005,
        worst_RX_health=full['B11']['source_val_worst_rx']>=full['R_LEGACY']['source_val_worst_rx']-.01)
    value=dict(status='SOURCE_QUALIFICATION_FROZEN',passed=all(checks.values()),checks=checks,
        target_scores_read=False,source_V=full,source_panel=panels,recorded_at=time.time(),
        future_seed_expansion_eligible=all(checks.values()),automatic_seed_expansion=False)
    d.write(d.BASE/'source_qualification.json',value);return value
