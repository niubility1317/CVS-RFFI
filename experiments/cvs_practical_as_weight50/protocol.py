"""Concrete fixed-window and source-lineage constraints; no target access."""
def cumulative(epoch,step):
 if epoch not in range(1,21) or step not in range(1,4441) or epoch!=(step-1)//222+1:raise ValueError('Only fixed localE1..20/step1..4440')
 return epoch+30,step+6660

def check_checkpoint(cp,*,run,row,epoch,updates,baseline):
 a=cp['args']
 if cp['epoch']!=epoch or cp['a1_ema_successful_updates']!=updates or cp['run_id']!=run or cp['candidate_id']!=row or a['baseline_ckpt']!=baseline or a['teacher_ckpt'] or cp['checkpoint_selection']!='final_only' or cp.get('ema_model') is None or a['epochs']!=epoch or not a['use_ema_teacher'] or a['label_smoothing']!=.01 or a['batch_size']!=128 or a['seed']!=2026092701:raise ValueError('Fixed student/ownEMA/nativecounter/defaulthead args differ')
 return cp

def check_source_metadata(done,frozen,init,contract,expected,role_ids,epoch,updates):
 if done['config']!=expected or done['epoch']!=epoch or done['optimizer_steps']!=updates or done['target_access'] or done.get('target_evaluated',False) or frozen['config']!=expected or frozen['selection']!='final_only' or frozen['target_access'] or init['target_contact'] or init['checkpoint_sources']!=expected['checkpoint_sources'] or init['scratch_only'] or init['ancestors']!=[expected['parent_source']] or contract['role_ids']!=role_ids or done.get('head_fit_updates',0):raise ValueError('Fixed frozen source lineage/data/defaulthead differs')

def check_default_head(shape):
 if tuple(shape)!=(6,160):raise ValueError('Existing jointly trained6x160 cosine head required')

def check_parent_budget(config,result):
 if config['cumulative_epochs']!=30 or config['cumulative_updates']!=6660 or result['cumulative_epochs']!=30 or result['cumulative_optimizer_steps']!=6660 or result['parent_epochs']!=20 or result['parent_optimizer_steps']!=4440 or result['U_truth_read'] or result['target_access'] or result['head_fit_updates']:raise ValueError('Actual weighted30 cumulative metadata differs')
