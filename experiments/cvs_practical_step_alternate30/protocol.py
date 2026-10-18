"""Concrete fixed-window and source-lineage constraints; no target access."""
def cumulative(epoch,step):
 if epoch not in range(1,11) or step not in range(1,2221) or epoch!=(step-1)//222+1:raise ValueError('Only fixed localE1..10/step1..2220')
 return epoch+20,step+4440

def slot_plan(epoch,step):
 ep,cs=cumulative(epoch,step);on=bool(cs%2)
 return dict(cumulative_epoch=ep,cumulative_step=cs,on=on,As_packets=128 if on else 0,L_NN=768,L_loss_scale=1.25)

def check_checkpoint(cp,*,run,row,epoch,updates,baseline):
 a=cp['args']
 if cp['epoch']!=epoch or cp['a1_ema_successful_updates']!=updates or cp['run_id']!=run or cp['candidate_id']!=row or a['baseline_ckpt']!=baseline or a['teacher_ckpt'] or cp['checkpoint_selection']!='final_only' or cp.get('ema_model') is None or a['epochs']!=epoch or not a['use_ema_teacher'] or a['label_smoothing']!=.01 or a['batch_size']!=128 or a['seed']!=2026092701:raise ValueError('Fixed student/ownEMA/nativecounter/defaulthead args differ')
 return cp

def check_source_metadata(done,frozen,init,contract,expected,role_ids):
 if done['config']!=expected or done['epoch']!=10 or done['optimizer_steps']!=2220 or done['target_access'] or done.get('target_evaluated',False) or frozen['config']!=expected or frozen['selection']!='final_only' or frozen['target_access'] or init['target_contact'] or init['checkpoint_sources']!=expected['checkpoint_sources'] or init['scratch_only'] or init['ancestors']!=[expected['parent_source']] or contract['role_ids']!=role_ids or done.get('head_fit_updates',0):raise ValueError('Fixed frozen source lineage/data/defaulthead differs')
