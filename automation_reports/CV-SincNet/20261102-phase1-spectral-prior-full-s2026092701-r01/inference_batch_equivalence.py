import pathlib,sys,torch,numpy as np,json
root=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet');folder=root/'releases/20261102-phase1-spectral-prior-full-s2026092701-r01-inference-f8d9dc32-release/experiments/phase1_spectral_prior_gate'
for n in ['20261027-phase1-pamf-repl-shuffled-views-m16-r02-release','20261101-phase1-physical-rich-router-s2026092701-r01-release']:sys.path.insert(0,str(root/'releases'/n))
sys.path.insert(0,str(folder));import spectral_prior_source as s;from spectral_prior_gate import SpectralPriorGate;from spectral_prior_inference import apply,make_prior
from experiments.pamf_phase1_flow import experiment as core
torch.set_num_threads(2);torch.manual_seed(2026112654);cache,clean,rx,training,pv,ph=s.s.source_contract(root);received,scene=s.s.epoch_data(cache,20);bank_ids=training[:8192];pool=training[8192:];clean_ids=pool[scene[pool]=='clean'][:48];bad_ids=pool[scene[pool]!='clean'][:48];assert len(clean_ids)==48 and len(bad_ids)==48;ids=np.concatenate([np.r_[clean_ids[i:i+16],bad_ids[i:i+16]] for i in (0,16,32)]);y=core.tensor(clean[bank_ids],'cpu');bank=torch.complex(y[:,0],y[:,1]);bc=bank[:,64:-64];power=bc.abs().square().sum(1).clamp_min(1e-12);unit=bc/power.sqrt()[:,None];phase,gate=s.s.initial_models(root,'cpu');phase.eval().requires_grad_(False);net=SpectralPriorGate().eval();teacher,_=core.teacher(root,'cpu')
with torch.no_grad():net.output.weight.normal_(0,.01);net.output.bias.copy_(torch.tensor([.1,-.07]));net.requires_grad_(False)
out={}
with torch.no_grad():
 for alpha in (.5,1.):
  for threshold in (.8,.99):
   differences=[];mismatches=0;active=0
   for start in range(0,len(ids),32):
    x=core.tensor(received[ids[start:start+32]],'cpu');a=phase.anchor(x);ref=s.make_prior(a,bank,power,unit);full=net(a,ref);enabled=gate(x).sigmoid()>threshold;expected=torch.where(enabled[:,None,None],a+alpha*(full-a),x);got,enabled2=apply(net,phase,gate,bank,power,unit,alpha,threshold,x);assert torch.equal(enabled,enabled2);differences.append(float((got-expected).abs().max()));u=teacher.classify_features(teacher.features(expected)).argmax(1);v=teacher.classify_features(teacher.features(got)).argmax(1);mismatches+=int((u!=v).sum());active+=int(enabled.sum())
   out[f'{alpha}:{threshold}']={'records':len(ids),'active':active,'max_abs_difference':max(differences),'F_decision_mismatches':mismatches}
print(json.dumps({'source_only':True,'model':'synthetic nonzero gate weights, not selected candidate','TX_labels_read':0,'U_s_truth_reads':0,'V_reads':0,'test_reads':0,'results':out},allow_nan=False))
