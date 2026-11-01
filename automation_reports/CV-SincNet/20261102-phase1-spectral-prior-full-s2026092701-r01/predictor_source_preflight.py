import os,pathlib,sys,tempfile,json,torch,numpy as np
root=pathlib.Path('/home/szu2070436088/2510044040/CV-SincNet');r='20261102-phase1-spectral-prior-full-s2026092701-r01-inference-f8d9dc32-release';folder=root/'releases'/r/'experiments/phase1_spectral_prior_gate'
for n in ['20261027-phase1-pamf-repl-shuffled-views-m16-r02-release','20261101-phase1-physical-rich-router-s2026092701-r01-release']:sys.path.insert(0,str(root/'releases'/n))
sys.path.insert(0,str(folder));import spectral_prior_predict as p
from spectral_prior_gate import SpectralPriorGate
from experiments.pamf_phase1_flow import experiment as core
torch.set_num_threads(2);cache,clean,rx,training,pv,ph=p.s.source_contract(root);ids=training[:8192];x=core.tensor(clean[ids],'cpu');bank=torch.complex(x[:,0],x[:,1]);phase,gate=p.s.initial_models(root,'cpu');model=SpectralPriorGate();sel={'kind':'spectral_prior_gate','epoch':1,'alpha':.5,'threshold':.8,'validation_net':10};commit='1f8fba0af54d73a2e8418aa606e8396e3f575e0d'
with tempfile.TemporaryDirectory(prefix='phase1_synthetic_inference_preflight_') as tmp:
 out=pathlib.Path(tmp)/'source';out.mkdir();(out/'source_selection.json').write_text(json.dumps({'selected':sel}));(out.parent/'completion.json').write_text(json.dumps({'status':'SOURCE_SELECTION_COMPLETE','selected':sel,'source_only':True,'target_predictions':False}));fake={'commit':commit,'selected':sel,'model':model.state_dict(),'phase':phase.phase.state_dict(),'gate':gate.state_dict(),'bank':bank,'bank_ids':ids};torch.save(fake,out/'selected.pt');p.smoke(root,out,commit,'cpu');print('SYNTHETIC_SOURCE_PREFLIGHT', (out/'prediction_smoke.json').read_text(),flush=True)
 fake['bank']=bank.clone();fake['bank'][0,0]+=1;torch.save(fake,out/'selected.pt')
 try:p.load_fixed(root,out,commit,'cpu');raise AssertionError('Altered source bank accepted')
 except ValueError as e:
  assert 'differs' in str(e);print('BANK_CORRUPTION_REJECTED',str(e),flush=True)
print('NO_TEST_IQ_OR_TRUTH_READ; synthetic candidate only',flush=True)
