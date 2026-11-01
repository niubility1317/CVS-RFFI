import subprocess
from pathlib import Path

remote=r'''
from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');run='20261101-diagnostic-phase1-source-fir-oracle-s2026092701-r01'
release=root/'releases'/(run+'-release');native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
inner=r"""
from pathlib import Path
import json,numpy as np,torch
from experiments.pamf_phase1_flow import experiment as core,anchor_balanced as a
from experiments.pamf_phase1_flow.dura_source_trisig_diagnostic import PARENT
from experiments.pamf_phase1_flow.source_fir_oracle import fit_paired_fir
root=Path('/home/szu2070436088/2510044040/CV-SincNet');torch.set_num_threads(2)
teacher,contract=core.teacher(root,'cpu');cache,clean,roles=core.data(root,contract)
fixed=np.concatenate([np.flatnonzero(roles==role)[np.linspace(0,count-1,n,dtype=int)] for role,count,n in (('L_s',6300,128),('U_s',56700,384))])
ids=np.load(cache/'train/physical_ids.npy',allow_pickle=False);results=[]
with torch.no_grad():
 for view in (1,20):
  with np.load(root/'runs'/PARENT/f'DURA_default-{view}.npz',allow_pickle=False) as z:
   saved={k:z[k] for k in ('source_positions','physical_ids','roles','clean','native','state')}
  ix=saved['source_positions'];assert np.array_equal(ix,fixed) and np.array_equal(saved['physical_ids'],ids[ix]) and np.array_equal(saved['clean'],clean[ix])
  records={k:[] for k in ('base','reference','ratio','p05','p10','p20','p50','full')}
  for start in range(0,512,32):
   sl=slice(start,start+32);ref=core.tensor(saved['clean'][sl],'cpu');old=core.tensor(saved['native'][sl],'cpu')
   fitted=fit_paired_fir(old,ref,taps=9,widely_linear=True,ridge=1e-4);delta=fitted-old
   norm=old.flatten(1).norm(dim=1).clamp_min(1e-8);ratio=delta.flatten(1).norm(dim=1)/norm
   records['ratio'].append(ratio);records['base'].append(teacher.classify_features(teacher.features(old)).argmax(1))
   records['reference'].append(teacher.classify_features(teacher.features(ref)).argmax(1))
   for key,radius in [('p05',.05),('p10',.1),('p20',.2),('p50',.5),('full',float('inf'))]:
    y=old+delta*(radius/ratio.clamp_min(1e-12)).clamp(max=1)[:,None,None]
    records[key].append(teacher.classify_features(teacher.features(y)).argmax(1))
  z={k:np.from_dlpack(torch.cat(v).cpu()) for k,v in records.items()}
  view_result={'view':view,'groups':[]}
  for group,mask in [('all',np.ones(512,dtype=bool)),('blocked',saved['state']=='BLOCKED')]:
   vals=z['ratio'][mask];base=z['base']==z['reference'];row=dict(group=group,count=int(mask.sum()),
      ratio_p50=float(np.quantile(vals,.5)),ratio_p90=float(np.quantile(vals,.9)),ratio_p95=float(np.quantile(vals,.95)),ratio_max=float(vals.max()),methods={})
   for key in ('p05','p10','p20','p50','full'):
    new=z[key]==z['reference'];row['methods'][key]=dict(matches=int((mask&new).sum()),repairs=int((mask&~base&new).sum()),damages=int((mask&base&~new).sum()))
   view_result['groups'].append(row)
  results.append(view_result)
print(json.dumps(dict(status='SOURCE_ONLY_RADIUS_DIAGNOSTIC',taps=9,source_records=512,source_views=[1,20],
 clean_per_record_used=True,U_s_labels_read=0,V_reads=0,test_IQ_reads=0,results=results),ensure_ascii=False))
"""
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-c',inner],cwd=release,env=env,check=True,timeout=200)
'''
p=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],input=remote.encode(),capture_output=True,timeout=220)
stdout=p.stdout.decode('utf8',errors='replace');print(stdout);print(p.stderr.decode('utf8',errors='replace'))
if p.returncode==0:
 path=Path(__file__).parent/'phase1_source_fir_radius.json'
 with path.open('x',encoding='utf8') as f:f.write(stdout)
raise SystemExit(p.returncode)
