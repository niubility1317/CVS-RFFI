"""Frozen source-only teacher predictions and matched RX1 correction counts."""
import copy,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import design as d
from .model import own_model
from .gate import replay
from experiments.cvs_urban_branch50.model import own_model as original_model
from experiments.cvs_urban_branch50.mechanism import developer_panel
from experiments.cvs_urban_sigtime.channel import imports,configurations,numpy_iq,tensor_iq

VIEWS=('clean',*d.SCENES)


def load_panel(c):
    baseline=Path(d.operational_parent()['output_root']);inner=d.read(baseline/'inner_source_contract.json')
    actual=d.read(Path(c['output_root'])/'inner_source_contract.json')
    if (len(actual['role_ids']['L_dev'])!=720 or set(actual['role_ids']['L_dev'])!=set(inner['role_ids']['L_dev'])
            or not set(actual['role_ids']['L_train'])<=set(inner['role_ids']['L_train'])):
        raise ValueError('Teachers must use the original fixed L_train/L_dev physical fold')
    return developer_panel(inner)


def source_view(panel,scene):
    if scene=='clean':return panel['x']
    ch,residual=imports();cfg=configurations()[scene];raw=numpy_iq(panel['x']);values=[]
    for i,pid in enumerate(panel['ids']):
        x,_=ch.normalize_input(raw[i,0]+1j*raw[i,1]);seed=ch.stable_seed(d.EVALUATION_SEED,'crossfit/source',scene,pid)
        session=f"rx{int(panel['rx'][i])}:day{int(panel['day'][i])}:eq1"
        wave,_=residual.ResidualChannel(cfg,seed,ch.receiver_for_session(cfg,d.MODEL_SEED,session)).process(x)
        values.append(np.stack((wave.real,wave.imag)))
    return tensor_iq(np.stack(values),'cpu')


def predict(row_id):
    row=next(r for r in d.rows() if r['row_id']==row_id);c=d.config(row);source=Path(c['output_root'])
    out=source.parent/'source_evaluation'
    if out.exists():raise FileExistsError('Never overwrite source teacher predictions')
    model=own_model(c,'cpu');before={k:v.clone() for k,v in model.state_dict().items()};panel=load_panel(c)
    with torch.no_grad():smoke=model.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
    if smoke.shape!=(2,6) or not torch.isfinite(smoke).all():raise ValueError('Real teacher default-head smoke failed')
    out.mkdir();started=time.perf_counter();forwards=2;arrays={}
    for scene in VIEWS:
        x=source_view(panel,scene);values=[]
        with torch.no_grad():
            for start in range(0,720,90):values.append(model.forward_identity_only(x[start:start+90])['tx_logits'])
        logits=torch.cat(values)
        if logits.shape!=(720,6) or not torch.isfinite(logits).all():raise ValueError('Invalid actual source teacher logits')
        arrays[scene]=np.asarray(logits.tolist(),dtype=np.float32);forwards+=720
    if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):raise ValueError('Source evaluation changed teacher state')
    with (out/'predictions.npz').open('xb') as stream:
        np.savez_compressed(stream,ids=np.asarray(panel['ids'],dtype=str),y=np.asarray(panel['y'].tolist(),dtype=np.int64),
            RX=np.asarray(panel['rx'].tolist(),dtype=np.int64),day=np.asarray(panel['day'].tolist(),dtype=np.int64),**arrays)
    result=dict(status='SOURCE_PREDICTIONS_COMPLETE',source_only=True,query_read=False,U_truth_read=False,
        head_fit_updates=0,optimizer_updates=0,EMA_updates=0,forward_packets=forwards,model_state_unchanged=True,
        views=list(VIEWS),source_count=720,primary_RX1_count=int((panel['rx']==1).sum()),seconds=time.perf_counter()-started,
        scope='Legal source L_dev labels only; no target arrays or truth-last target scorer')
    if forwards!=5042 or result['primary_RX1_count']!=144:raise ValueError('Source teacher evaluation budget differs')
    d.write(out/'complete.json',result);return result


def summarize(sample,excluded,y,selected):
    if sample.shape!=excluded.shape or sample.shape!=(len(y),6) or selected.shape!=(len(y),):raise ValueError('Matched actual source predictions required')
    n=int(selected.sum())
    if not n:return dict(count=0,sample_accuracy=None,excluded_accuracy=None,repairs=0,new_errors=0,net=0)
    s,e=(logits.argmax(1)==y for logits in (sample,excluded));repairs=int((selected&~s&e).sum());damage=int((selected&s&~e).sum())
    wrong=[];truep=[];margin=[]
    for logits,correct in ((sample,s),(excluded,e)):
        p=logits.softmax(1);conf=p.max(1).values;other=logits.masked_fill(F.one_hot(y,6).bool(),-torch.inf).max(1).values
        wrong.append(float(((conf>=.95)&~correct&selected).sum())/n)
        truep.append(float(p.gather(1,y[:,None]).squeeze(1)[selected].mean()))
        margin.append(float(((logits.gather(1,y[:,None]).squeeze(1)-other)/30)[selected].mean()))
    return dict(count=n,sample_accuracy=float(s[selected].float().mean()),excluded_accuracy=float(e[selected].float().mean()),
        repairs=repairs,new_errors=damage,net=repairs-damage,sample_highconfidence_wrong=wrong[0],excluded_highconfidence_wrong=wrong[1],
        sample_true_class_probability=truep[0],excluded_true_class_probability=truep[1],sample_margin=margin[0],excluded_margin=margin[1],
        sample_confusion=torch.bincount((y[selected]*6+sample.argmax(1)[selected]),minlength=36).reshape(6,6).tolist(),
        excluded_confusion=torch.bincount((y[selected]*6+excluded.argmax(1)[selected]),minlength=36).reshape(6,6).tolist())


def source_decision(primary,rejected):
    if primary['count']!=144:raise ValueError('Actual fixed RX1 L_dev144 required')
    passed=(primary['net']>=3 and rejected['net']>=0
        and primary['excluded_highconfidence_wrong']-primary['sample_highconfidence_wrong']<=.02)
    return dict(passed=passed,rule=d.config(d.rows()[0])['evaluation']['expansion'],primary=primary,rejected=rejected,
        automatic_bank_or_student_expansion=False)


def compare():
    output=d.BASE/'comparison'
    if output.exists():raise FileExistsError('Never overwrite source teacher comparison')
    arrays={}
    for row in d.rows():
        folder=Path(d.config(row)['output_root']).parent/'source_evaluation';done=d.read(folder/'complete.json')
        if (done['status']!='SOURCE_PREDICTIONS_COMPLETE' or not done['source_only'] or done['query_read'] or done['U_truth_read']
                or done['head_fit_updates'] or done['forward_packets']!=5042 or not done['model_state_unchanged']):
            raise ValueError('All two legal frozen source prediction artifacts required')
        with np.load(folder/'predictions.npz',allow_pickle=False) as data:arrays[row['arm']]={k:data[k].copy() for k in data.files}
    a,b=(arrays[k] for k in d.ARMS)
    if any(not np.array_equal(a[k],b[k]) for k in ('ids','y','RX','day')) or len(set(a['ids']))!=720:
        raise ValueError('Teacher source physical records or TX/RX/day labels differ')
    y=torch.tensor(a['y'].tolist());rx=torch.tensor(a['RX'].tolist());day=torch.tensor(a['day'].tolist())
    baseline=original_model(d.operational_parent(),'cpu');cp=torch.load(Path(d.operational_parent()['output_root'])/'final_ssdg.pth',map_location='cpu',weights_only=False)
    teacher=copy.deepcopy(baseline);teacher.load_state_dict(cp['ema_model'],strict=True);teacher.eval().requires_grad_(False)
    panel=load_panel(d.config(d.rows()[0]));metadata=d.read(Path(d.config(d.rows()[0])['output_root'])/'L_dev_native_metadata.json')['records']
    if panel['ids']!=a['ids'].tolist():raise ValueError('Source gate physical order differs from frozen predictions')
    gates=replay(baseline,teacher,panel,metadata);accepted=torch.tensor([r['accepted'] for r in gates['records']])
    records=[]
    for scene in VIEWS:
        s,e=(torch.tensor(arrays[k][scene].tolist()) for k in d.ARMS)
        if not torch.isfinite(s).all() or not torch.isfinite(e).all():raise ValueError('Nonfinite frozen teacher logits')
        for region,mask in (('all_source',torch.ones(720,dtype=torch.bool)),('RX1',rx==1),
                ('RX1_old_gate_rejected',(rx==1)&~accepted),('RX1_old_gate_accepted',(rx==1)&accepted)):
            records.append(dict(view=scene,region=region,**summarize(s,e,y,mask)))
        for date in (1,2,3):records.append(dict(view=scene,region='RX1_day'+str(date),**summarize(s,e,y,(rx==1)&(day==date))))
    primary=next(r for r in records if r['view']=='clean' and r['region']=='RX1')
    rejected=next(r for r in records if r['view']=='clean' and r['region']=='RX1_old_gate_rejected')
    result=dict(status='ANALYZED',source_only=True,query_read=False,U_truth_read=False,teacher_models_trained_in_study=2,
        new_models_trained_by_comparison=0,
        student_models_trained=0,seed_count=1,records=records,gate=gates,
        source_decision=source_decision(primary,rejected),comparison_forward_packets=1440,
        teacher_evaluation_forward_packets=10084,head_fit_updates=0,EMA_updates=0,
        claim='Equal-budget source RX exposure comparison only; no teacher feature alignment, deployed student, target result or cross-seed claim')
    output.mkdir();d.write(output/'results.json',result);d.write(output/'complete.json',dict(status='ANALYZED',source_only=True,query_read=False))
    return result
