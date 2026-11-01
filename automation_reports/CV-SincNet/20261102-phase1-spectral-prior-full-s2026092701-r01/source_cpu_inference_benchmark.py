import pathlib,sys,time,torch,numpy as np,json
root=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet');folder=root/'releases/20261102-phase1-spectral-prior-full-s2026092701-r01-inference-f8d9dc32-release/experiments/phase1_spectral_prior_gate'
for n in ['20261027-phase1-pamf-repl-shuffled-views-m16-r02-release','20261101-phase1-physical-rich-router-s2026092701-r01-release']:sys.path.insert(0,str(root/'releases'/n))
sys.path.insert(0,str(folder));import spectral_prior_source as s;from spectral_prior_gate import SpectralPriorGate;from spectral_prior_inference import apply
from experiments.pamf_phase1_flow import experiment as core
torch.set_num_threads(2);torch.manual_seed(2026112754);cache,clean,rx,train,pv,ph=s.s.source_contract(root);received,scene=s.s.epoch_data(cache,20);bank_ids=train[:8192];pool=train[8192:];ids=pool[scene[pool]!='clean'][:320];assert len(ids)==320;y=core.tensor(clean[bank_ids],'cpu');bank=torch.complex(y[:,0],y[:,1]);bc=bank[:,64:-64];power=bc.abs().square().sum(1).clamp_min(1e-12);unit=bc/power.sqrt()[:,None];phase,gate=s.s.initial_models(root,'cpu');phase.eval().requires_grad_(False);net=SpectralPriorGate().eval().requires_grad_(False);teacher,_=core.teacher(root,'cpu');xs=[core.tensor(received[ids[i:i+64]],'cpu') for i in range(0,320,64)]
with torch.no_grad():
 y,g=apply(net,phase,gate,bank,power,unit,1.,.8,xs[0]);teacher.classify_features(teacher.features(y));times=[];active=0
 for x in xs[1:]:
  st=time.perf_counter();y,g=apply(net,phase,gate,bank,power,unit,1.,.8,x);z=teacher.classify_features(teacher.features(y));times.append(time.perf_counter()-st);active+=int(g.sum());assert torch.isfinite(z).all()
print(json.dumps({'source_only':True,'model':'synthetic initialized gate; timing only','device':'N607 CPU, two PyTorch threads, shared host','warmup_records':64,'timed_records':256,'active':active,'batch64_seconds':times,'mean_ms_per_record':sum(times)*1000/256,'TX_labels_read':0,'U_s_truth_reads':0,'V_reads':0,'test_reads':0,'caveat':'shared-load measurement; no accuracy claim, may differ from actual selected model or full test'},allow_nan=False))
