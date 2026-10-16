"""Focused profile/seed/entry regression; no real IQ,CP or NN."""
import json,subprocess,sys,tempfile
from copy import deepcopy
from pathlib import Path
import torch
from .confirmation import activate,FIRST
from . import design as d,model as api,runner

def checks():
 torch.set_num_threads(2);d=activate();old=d.read(d.ROOT/'automation_reports/CV-SincNet'/FIRST/'experiment.json');proof=d.read(d.ROOT/'automation_reports/CV-SincNet'/FIRST/'independent_readback.json')
 assert proof['source_only'] and proof['qualified_independent_Ag_source_screen'] and not proof['target_access'] and not proof['joint_goal_complete']
 before={v['config']['arm']:v['config'] for v in old['rows'] if v['purpose']!='reference_only_reuse'}
 snapshots=[];rows=[];neg=0
 for row in d.rows():
  c=d.validate(d.config(row));a=vars(d.make_args(c,'cpu'));ref=before[row['arm']]
  assert c['model_seed']==a['seed']==d.parent.base_model_config()['model_seed']==2026092701
  assert a['epochs']==20 and a['label_epochs']==13 and a['pseudo_epochs']==7 and a['source_fold_held_rx']==1 and not a['baseline_ckpt'] and not a['teacher_ckpt'] and c['baseline_reuse'] is False
  for key in ('coherent_mix','training_mechanism','additive','supcon','channel','fold','source_selection'):assert c[key]==ref[key],key
  torch.manual_seed(2026092701);m=api.build(c,'cpu');snapshots.append({k:v.detach().clone() for k,v in m.state_dict().items()})
  assert m._mix_runtime is None and m._mix_calls==m._mix_packets==m._joint_calls==m._additive_calls==m._supcon_applied==0
  assert snapshots[-1]['id_backbone.encoder.core.id_backbone.cls_head.weight'].shape==(6,160)
  rows.append(dict(row_id=row['row_id'],actual_model_seed=d.parent.base_model_config()['model_seed'],native_seed=a['seed'],scratch20=True))
  for path,value in ((('epochs',),50),(('model_seed',),2026092702),(('target_access',),True),(('baseline_reuse',),True),(('coherent_mix','max_relative_L2_change'),.2),(('coherent_mix','reference_window'),[0,80]),(('coherent_mix','U_or_V_donors'),True),(('additive','extra_weight'),.5),(('training_mechanism','As'),True),(('channel','equalization_enabled'),True),(('fold','held_RX'),6),(('classifier',),'newhead')):
   bad=deepcopy(c);part=bad
   for key in path[:-1]:part=part[key]
   part[path[-1]]=value
   try:d.validate(bad)
   except ValueError:neg+=1
   else:raise AssertionError('Unregistered profile/recipe override accepted')
  del m
 assert all(torch.equal(v,snapshots[1][k]) for k,v in snapshots[0].items())
 from experiments.cvs_additive_as20.model import build
 torch.manual_seed(2026092701);native=build(d.template(),'cpu');assert all(torch.equal(v,native.state_dict()[k]) for k,v in snapshots[0].items());del native
 with tempfile.TemporaryDirectory() as tmp:
  fake=Path(tmp)/'experiments/cvs_f_coherentmix20/runner.py';fake.parent.mkdir(parents=True);(fake.parent/'runtime.json').write_text(json.dumps(dict(profile='unregistered')),encoding='utf8');oldfile=runner.__file__;runner.__file__=str(fake)
  try:
   try:runner.bootstrap()
   except ValueError as e:assert 'Unregistered coherent source profile' in str(e);neg+=1
   else:raise AssertionError('Unknown runtime profile accepted')
  finally:runner.__file__=oldfile
 from .entrypoint_checks import check
 entry=check();assert activate().rows()==d.rows()
 return dict(status='PASS',rows=rows,negative_cases=neg,actual_constructor_native_seed1=True,initial_states_equal=True,initial_states_equal_same_seed_raw_backend=True,unchanged_core_scientific_configuration=True,state_key_count=len(snapshots[0]),direct_entry=entry,synthetic_NN_packets=0,synthetic_optimizer_updates=0,real_IQ_reads=0,real_checkpoint_reads=0,target_access=False,previous_checks='Original3090NN/3updates/20negative+16scope+directentryfix retained;unchangedIQ/model/fold not repeated')
if __name__=='__main__':print(json.dumps(checks()))
