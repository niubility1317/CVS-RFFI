"""Once-only fixed epoch5 source norm calibration, no model/EMA updates."""
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from . import design as d,objective
from experiments.cvs_urban_branch50 import branches as b,gradient as g
from experiments.cvs_urban_branch50.model import build
from experiments.cvs_urban_branch50.mechanism import developer_panel
from experiments.cvs_urban_sigtime.channel import imports,configurations,numpy_iq,tensor_iq,usability


def fixed(x,panel,start,scene):
    ch,residual=imports();cfg=configurations()[scene];raw=numpy_iq(x);arrays=[];records=[]
    for j,pid in enumerate(panel['ids'][start:start+len(x)]):
        session=f"rx{int(panel['rx'][start+j])}:day{int(panel['day'][start+j])}:eq1"
        seed=ch.stable_seed(d.CALIBRATION_SEED,'semantic_strength/source_only',scene,pid)
        z,_=ch.normalize_input(raw[j,0]+1j*raw[j,1])
        wave,meta=residual.ResidualChannel(cfg,seed,ch.receiver_for_session(cfg,2026092701,session)).process(z)
        arrays.append(np.stack((wave.real,wave.imag)));records.append(meta)
    return tensor_iq(np.stack(arrays),x.device),records


def reuse_original(result,original_path):
    expected=(d.parent.BASE/'A0-s2026092701/source/warmup_epoch5.pth').as_posix()
    if (result.get('status')!='SOURCE_SEMANTIC_CALIBRATED' or not result.get('source_only') or result.get('query_read')
            or result.get('U_truth_read') or result.get('V_forward_packets') or result.get('optimizer_updates')
            or result.get('head_fit_updates') or result.get('source_count')!=720 or result.get('warmup_epoch')!=5
            or result.get('feature_parameters')!=195041 or result.get('calibration_seed')!=d.CALIBRATION_SEED
            or result.get('forward_packets')!=2160 or result.get('margin')!=.1 or not result.get('model_state_unchanged')
            or not result.get('training_gradients_unchanged') or result.get('source_checkpoint')!=expected):
        raise ValueError('Original fixed source calibration differs; do not rerun or retune')
    if set(result['coefficients'])!={'D3','D4'}:
        raise ValueError('Both predefined semantic strengths required')
    for v in result['coefficients'].values():
        if not 1e-4<=v['coefficient']<=10 or v['target_gradient_ratio']!=.1:
            raise ValueError('Original source coefficient is invalid')
    return dict(result,reused_from=str(original_path),new_forward_packets=0,new_optimizer_updates=0,
        reuse_reason='Only CPU/GPU metadata placement fixed; unchanged source relation math and scientific matrix')


def run():
    path=d.BASE/'semantic_calibration.json'
    if path.exists():raise FileExistsError('Reuse completed source estimate; do not recalculate/overwrite')
    if d.ORIGINAL_RUN:
        original=d.ORIGINAL_BASE/'semantic_calibration.json'
        result=reuse_original(d.read(original),original)
        d.write(path,result)
        print('SOURCE calibration reused unchanged; new forwards/updates=0',flush=True)
        return result
    started=time.perf_counter();torch.set_num_threads(2);reference=d.parent.config(d.parent.rows()[0]);source=Path(reference['output_root'])
    inner=d.read(source/'inner_source_contract.json');init=d.read(source/'initialization.json')
    if not init['scratch_only'] or init['checkpoint_sources'] or init['ancestors'] or init['target_contact']:
        raise ValueError('Warmup has forbidden initialization ancestry')
    from experiments.cvs_phase1_stack.recover import verify_contract
    verify_contract(d.read(source/'source_contract.json'),d.read(d.SOURCE))
    if (inner['counts']!=dict(L_train=5580,L_dev=720,U_s=56700,V=27000) or inner['fold_seed']!=2026101503 or inner['L_dev_updates']):
        raise ValueError('Fixed source warmup physical roles differ')
    cp=torch.load(source/'warmup_epoch5.pth',map_location='cpu',weights_only=False)
    if (cp['epoch']!=5 or cp['optimizer_steps']!=1110 or cp['config']!=reference or cp['inner_source_contract']!=inner
            or not cp['source_only'] or cp['target_access'] or cp['initialization']!='scratch'):
        raise ValueError('Need exact predefined source A0 epoch5, not a ranked checkpoint')
    model=build(reference,'cpu');model.load_state_dict(cp['model'],strict=True);model.eval()
    before={k:v.clone() for k,v in model.state_dict().items()};panel=developer_panel(inner)
    norms={k:[] for k in ('CE','D3','D4')};records=[];forwards=0;theta=None
    for start in range(0,720,120):
        x=panel['x'][start:start+120];y=panel['y'][start:start+120]
        _,clean=b.forward(model,x);forwards+=len(x)
        for scene in ('practical_mid_urban','practical_low_urban'):
            sat,metadata=fixed(x,panel,start,scene);out,urban=b.forward(model,sat);forwards+=len(x)
            if theta is None:
                theta=b.feature_parameters(model,out['id_feat_joint'])
                if sum(p.numel() for _,p in theta)!=195041:raise ValueError('Actual global feature-gradient scope changed')
            ce=.5*F.cross_entropy(out['tx_logits'],y,label_smoothing=.01);cg=g.vectors(ce,theta);norms['CE'].append(g.norm(cg))
            q=x.new_tensor([usability(m) for m in metadata])
            for arm in ('D3','D4'):
                loss,stats=objective.semantic(clean['J'],urban['J'],y,panel['rx'][start:start+120],panel['day'][start:start+120],q,bidirectional=arm=='D4')
                ag=g.vectors(loss,theta);norms[arm].append(g.norm(ag))
                records.append(dict(arm=arm,batch=start//120,scene=scene,**g.compare(cg,ag),
                    triplets=stats['triplets'],active_triplets=stats['active_triplets'],anchors_with_references=stats['anchors_with_references']))
    if forwards!=2160 or any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()) or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Zero-update source estimate changed state or budget')
    result=dict(status='SOURCE_SEMANTIC_CALIBRATED',source_only=True,query_read=False,U_truth_read=False,V_forward_packets=0,
        optimizer_updates=0,head_fit_updates=0,warmup_epoch=5,source_role='L_dev',source_count=720,
        source_checkpoint=str(source/'warmup_epoch5.pth'),feature_parameters=195041,feature_tensors=len(theta),
        default_W_excluded=True,model_state_unchanged=True,training_gradients_unchanged=True,
        coefficients={arm:g.coefficient(norms['CE'],norms[arm]) for arm in ('D3','D4')},global_norms=norms,records=records,
        calibration_seed=d.CALIBRATION_SEED,margin=.1,forward_packets=forwards,seconds=time.perf_counter()-started,
        commit=(d.ROOT/'release_commit.txt').read_text().strip(),
        scope='Same fixed A0 epoch5/internal source fold; static global norm ratio.1 for each arm; not constant in training; native Dropout unchanged')
    temporary=path.with_suffix('.json.writing')
    if temporary.exists():raise FileExistsError('Reconcile partial calibration artifact')
    d.write(temporary,result);temporary.rename(path)
    print('SOURCE CALIBRATION PASS semantic D3/D4 fixed global coefficients; query_read=false',flush=True)
    return result
