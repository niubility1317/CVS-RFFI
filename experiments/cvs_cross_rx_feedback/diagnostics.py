"""Read-only full telemetry and source-input census, with bounded gradient probes."""
import gzip,json,statistics as st
import torch
from torch.nn import functional as F
from . import design as d
from .method import parameter_selection,scores
from experiments.cvs_scratch30_baseline import design as b
from experiments.cvs_scratch30_baseline.predict import own_model
from experiments.cvs_scratch30_baseline.source import load
from experiments.cvs_nocfo_foundation.model import groups
from experiments.cvs_finite_margin.objective import deterministic
from experiments.cvs_finite_action.operators import quality

def run(device='cpu'):
 torch.set_num_threads(2);rows=[];L,U=None,None
 for seed in b.SEEDS:
  c=b.config(next(r for r in b.rows() if r['arm']=='SELF_SSL30' and r['model_seed']==seed));L,U,V,contract=load(c);model=own_model(c,device);pg,_=groups(model,c['source_loader_config']);names,params=parameter_selection(model,pg);before={k:v.clone() for k,v in model.state_dict().items()}
  with gzip.open(d.b.BASE/c['row_id']/'source'/'step_metrics.jsonl.gz','rt',encoding='utf-8') as f:steps=[json.loads(line) for line in f]
  if len(steps)!=6660 or [v['step'] for v in steps]!=list(range(1,6661)):raise ValueError('Historical SSL telemetry incomplete')
  active=[v['metrics'] for v in steps[1110:]];total=sum(v['U_packets'] for v in active);accepted=sum(v['accepted_U'] for v in active)
  if total!=710400:raise ValueError('Historical U occurrence budget differs')
  census=[];probes=[]
  for rx in d.RX:
   for day in (1,2,3):
    ix=((U['rx']==rx)&(U['day']==day)).nonzero().flatten();confidence=torch.empty(len(ix));pseudo=torch.empty(len(ix),dtype=torch.long)
    with deterministic(model),torch.no_grad():
     for begin in range(0,len(ix),256):
      q=model(U['x'][ix[begin:begin+256]].to(device)).softmax(1);conf,pred=q.max(1);confidence[begin:begin+len(q)]=conf.cpu();pseudo[begin:begin+len(q)]=pred.cpu()
    mask=confidence>=.95;census.append(dict(RX=rx,day=day,count=len(ix),confidence_proxy_accepted=int(mask.sum()),prediction_classes=torch.bincount(pseudo[mask],minlength=6).tolist(),mean_confidence=float(confidence.mean())))
    # Exactly64 deterministic source U samples per RX/day; no U truth or V.
    def labeled64(r):return torch.cat([((L['rx']==r)&(L['day']==day)&(L['y']==tx)).nonzero().flatten()[:11 if tx<4 else 10] for tx in range(6)])
    ux=U['x'][ix[:64]].to(device);li=labeled64(rx);other=d.RX[(d.RX.index(rx)+1)%5];mi=labeled64(other)
    model.train()
    with deterministic(model),torch.no_grad():q=model(ux).softmax(1).detach()
    gen=torch.Generator(device=device).manual_seed(seed+rx*1009+day*17);strong=ux+torch.randn(ux.shape,device=device,generator=gen)*.015
    with torch.random.fork_rng(devices=[torch.device(device).index or 0] if str(device).startswith('cuda') else []):
     torch.manual_seed(seed+rx*977+day);uloss=(F.kl_div(model(strong).log_softmax(1),q,reduction='none').sum(1)*(q.max(1).values>=.95)).mean()
    gu=torch.autograd.grad(uloss,params,allow_unused=True)
    with deterministic(model):
     lloss=F.cross_entropy(model(L['x'][li].to(device)),L['y'][li].to(device));mloss=F.cross_entropy(model(L['x'][mi].to(device)),L['y'][mi].to(device))
    gl=torch.autograd.grad(lloss,params,allow_unused=True);gm=torch.autograd.grad(mloss,params,allow_unused=True);cos,_,_=scores(gm,gu,[torch.ones_like(p) for p in params]);norm=lambda g:sum(float(v.square().sum()) for v in g if v is not None)**.5
    ln=norm(gl);un=norm(gu);probes.append(dict(RX=rx,day=day,feedback_RX=other,U_occurrences=64,gradient_cosine_SGD_P_identity=cos,weighted_U_L_gradient_ratio=.25*un/max(ln,1e-20),L_gradient_norm=ln,U_gradient_norm=un,mask='frozen confidence>=.95 proxy,not historical2pass gate'))
  if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):raise ValueError('Read-only gradient probes changed frozen checkpoint state')
  rows.append(dict(seed=seed,complete_historical_steps=6660,historical_U_occurrences=total,historical_accepted_U=accepted,historical_actual_acceptance=accepted/total,full_source_U_census=census,bounded_gradient_probes=probes,feedback_parameter_names=names,source_checkpoint_state_unchanged=True))
 coords=quality(L['x']);names=('logRMS','FFTflatness','abs_lag16coherence');physical=[]
 for i,name in enumerate(names):
  v=coords[:,i];physical.append(dict(name=name,count=6300,std=float(v.std(unbiased=False)),quantiles=torch.quantile(v,torch.tensor([0.,.01,.5,.99,1.])).tolist(),RX_means={str(r):float(v[L['rx']==r].mean()) for r in d.RX},TX_means={str(y):float(v[L['y']==y].mean()) for y in range(6)}))
 # Existing predictions only: all15 models strictlychecked by original sourceanalyzer.
 from experiments.cvs_stochastic_ensemble import design as s
 from experiments.cvs_stochastic_ensemble.summarize import analyze
 analyze(s.BASE);complements=[]
 for rx in d.RX:
  paths=[s.REFERENCE/s.baseline_id(rx)]+[s.BASE/r['row_id'] for r in s.rows() if r['held_rx']==rx];zs=[torch.load(p/'source_predictions.pt',map_location='cpu',weights_only=False)['logits'] for p in paths];y=V['y'][V['rx']==rx];pp=torch.stack([z.softmax(1) for z in zs]);avg=pp.mean(0);ok=torch.stack([z.argmax(1)==y for z in zs]);ens=avg.argmax(1)==y
  memberNLL=-pp.gather(2,y[None,:,None].expand(3,-1,1)).squeeze(2).clamp_min(1e-12).log().mean(1);nll=-avg[torch.arange(len(y)),y].clamp_min(1e-12).log().mean()
  complements.append(dict(RX=rx,count=5400,all_members_wrong=int((~ok).all(0).sum()),rescue_vs_member0=int((~ok[0]&ens).sum()),harm_vs_member0=int((ok[0]&~ens).sum()),net_vs_member0=int((ens.to(torch.int)-ok[0].to(torch.int)).sum()),mean_member_NLL=float(memberNLL.mean()),ensemble_NLL=float(nll),Jensen_noninferior_mean=bool(nll<=memberNLL.mean()+1e-6)))
 return dict(status='VERIFIED',target_access=False,U_truth_read=False,parameter_updates=0,historical_telemetry_steps=19980,full_U_probability_census=170100,gradient_probe_U_occurrences=2880,gradient_scope='66 optimizer backbone tensors excludingcos classweight; P=I SGD diagnostic,not old AdamW exactattribution; bounded45groups,not full U gradient census',rows=rows,physical_input_census=physical,ensemble_complements=complements,observation_boundary='RMSalready perpacketnormalized;lag16empirical notLSTFclaim;lag20 numericalexistingcompensation;eq0/eq1 sig_i no trustedpairmapping;out_len padding notnewobservation',new_run='source3RX+1feedbackL rotation,heldRXL/Uexcluded,normalcos;fixed3armrecipe notselectedfromtarget')
