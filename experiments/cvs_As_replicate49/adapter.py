"""Only seed/run bindings change; no learning rule or IQ operator change."""
from copy import deepcopy
SEEDS=(2026092702,2026092703)
ARMS=('B00','B01')
def row_ok(row):return row in [dict(arm=a,row_id=a+'-s'+str(s),model_seed=s) for s in SEEDS for a in ARMS]
def adapt(c,row,segment,run,base,owner):
 if not row_ok(row) or segment not in (20,30):raise ValueError('Only fixed02/03 and original two controls/E30')
 c=deepcopy(c);tail=segment==30
 if c['segment']!=segment or c['new_Ag'] or c['Ag_mode']!='NONE' or c['L_NN_per_step']!=768 or c['As']!=(row['arm']=='B01'):raise ValueError('Unchanged noAg/6L matching control required')
 source=base+'/'+row['row_id']+'/stage20';out=base+'/'+row['row_id']+('/source' if tail else '/stage20')
 c.update(schema='FcrossRX_As_seed_replication49_v1',run_id=run,**row,launch_owner=owner,output_root=out,parent_source=source if tail else None,checkpoint_sources=[source+'/final_ssdg.pth'] if tail else [],baseline_reuse=False)
 c['seed_binding']=dict(model_and_native_training=row['model_seed'],physical_split=392005,physical_roles='unchanged exact IDs',F_factory_model_seed=row['model_seed'],train_channel=20261016421,validation_channel=20261016422,receiver=20261016423,legacy_reference_config='architecture/recipe reference only;never checkpoint or actual model seed')
 return c

def apply_args(a,c):
 a.seed=c['model_seed'];a.run_id=c['run_id'];a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.baseline_ckpt='' if c['segment']==20 else c['parent_source']+'/final_ssdg.pth'
 if a.epochs!=(20 if c['segment']==20 else 10) or a.label_epochs!=(13 if c['segment']==20 else 0) or a.pseudo_epochs!=(7 if c['segment']==20 else 10) or a.teacher_ckpt:raise ValueError('Only original13L+17U/same ownEMA recipe')
 return a
