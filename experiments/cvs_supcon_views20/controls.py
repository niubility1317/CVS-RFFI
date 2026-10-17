"""Reuse frozen rows as references only;never initialize a new model from them."""
from . import design as d
from experiments.cvs_scene_cycle20 import design as old

def load(bases=None):
 bases=d.controls() if bases is None else bases;results={};contracts=[]
 if set(bases)!={'Q0','Q_ord'}:raise ValueError('Exactly two fixed controls')
 for name,path in bases.items():
  arm={'Q0':'B00','Q_ord':'C_IID'}[name];config=old.config(next(r for r in old.rows() if r['arm']==arm));resolved=d.read(path/'resolved_config.json');done=d.read(path/'completion.json');init=d.read(path/'initialization.json');contract=d.read(path/'source_contract.json');fold=d.read(path/'source_fold.json');frozen=d.read(path/'source_frozen.json');r=d.read(path/'satmix_result.json');args=d.read(path/'actual_native_args.json');audit=d.read(path/'additive_audit.json')
  if resolved['commit']!=d.CONTROL_COMMIT or resolved['config']!=config or done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources'] or done['target_access'] or not init['scratch_only'] or init['ancestors'] or init['checkpoint_sources'] or init['seed']!=d.SEED or init['target_contact'] or frozen['selection']!='final_only':raise ValueError('Frozen control actual config/initialization/budget differs')
  if contract['held_RX']!=1 or fold['held_RX']!=1 or set(contract['source_receivers'])!=set(d.FIT) or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}:raise ValueError('Frozen control effective source roles differ')
  if args['epochs']!=20 or args['label_epochs']!=13 or args['pseudo_epochs']!=7 or args['baseline_ckpt'] or args['teacher_ckpt'] or not args['use_aug'] or args['source_fold_held_rx']!=1:raise ValueError('Frozen control actual native execution differs')
  if r['actual_steps']!=4440 or r['L_forward_packets']!=4546560 or r['extra_L_forward_packets']!=1136640 or r['supcon_applied']!=3330 or r['supcon_layer_forward_terms']!=13320 or not r['default_head_joint'] or r['head_fit_updates'] or r['target_access'] or r['U_truth_read'] or r['checkpoint_sources'] or r['As_training_observations']!=(0 if name=='Q0' else 568320) or r['total_pair_weight']!=1.25 or r['teacher_additive_calls'] or audit['extra_BN_updates'] or audit['base_raw_total_weight']!=.75 or audit['base_ordinary_weight']!=.25:raise ValueError('Frozen control real view/loss/head/permission execution differs')
  contracts.append(contract['role_ids']);results[name]=r
 if contracts[0]!=contracts[1]:raise ValueError('Control physical L/U/V IDs differ')
 return results,dict(status='VERIFIED',control_run=old.RUN,control_commit=d.CONTROL_COMMIT,controls={k:str(v) for k,v in bases.items()},all_effective_physical_roles_equal=True,role_counts={'L_s':5040,'U_s':45360,'V':21600},same_seed_E20_4440_eight_L_forwards_pairweight1_25=True,training_checkpoint_sources=[],control_weights_loaded=False,reused_model_epochs=40,reused_optimizer_steps=8880,target_access=False)
