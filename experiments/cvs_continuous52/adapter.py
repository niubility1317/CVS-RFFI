from copy import deepcopy
SEED=2026092701
ARMS=('B00','B01')
def rows():return [dict(arm=a,row_id=a+'-s'+str(SEED),model_seed=SEED) for a in ARMS]
def adapt(c,row,run,base,owner):
 if row not in rows() or c['segment']!=20 or c['epochs']!=20 or c['label_epochs']!=13 or c['pseudo_epochs']!=7 or c['updates']!=4440 or c['new_Ag'] or c['Ag_mode']!='NONE' or c['L_NN_per_step']!=768 or c['As']!=(row['arm']=='B01'):raise ValueError('Only original6L/noAg/scratch20 matching reference')
 c=deepcopy(c);c.update(schema='FcrossRX_continuous52_v1',run_id=run,**row,launch_owner=owner,output_root=base+'/'+row['row_id']+'/source',epochs=30,label_epochs=13,pseudo_epochs=17,updates=6660,epoch_offset=0,step_offset=0,cumulative_epochs=30,cumulative_updates=6660,parent_source=None,checkpoint_sources=[],baseline_reuse=False,native_scratch_only=True,model_initialization='scratch',training_mode='single_continuous_E30',legacy_attachment_segment=20,optimizer_schedule='single_cosine30_no_restart',training_recipe='Unchanged6L/1.25/raw.25+ordinary.75+As_or_raw.25;singlecos30;13L17U;originalU/EMA/SupCon')
 c['training_mechanism']['expected_L_forward_packets']=6660*768;c['training_mechanism']['extra_L_identity_forwards']=6660*640
 c['baseline_reference']='P43 fixed seed01 B00/B01 own20+10 reused for reporting;new rows scratch single30,no parent'
 return c

def apply_args(a,c):
 if a.epochs!=20 or a.label_epochs!=13 or a.pseudo_epochs!=7 or a.teacher_ckpt or a.baseline_ckpt:raise ValueError('Expected original scratch20 args only')
 a.seed=c['model_seed'];a.run_id=c['run_id'];a.candidate_id=c['row_id'];a.output_dir=c['output_root'];a.epochs=30;a.label_epochs=13;a.pseudo_epochs=17;a.from_scratch=True;a.a1_scratch_only=True;a.baseline_ckpt='';a.teacher_ckpt=''
 return a
