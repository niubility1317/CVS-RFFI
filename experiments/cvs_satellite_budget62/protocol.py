"""Only concrete parent/data/freeze checks; no receipt or extra data validation."""
def check_source_metadata(done,frozen,init,contract,expected,role_ids,epoch,updates):
 if done['config']!=expected or done['epoch']!=epoch or done['optimizer_steps']!=updates or done['target_access'] or done.get('target_evaluated',False) or frozen['config']!=expected or frozen['selection']!='final_only' or frozen['target_access'] or init['target_contact'] or init['checkpoint_sources']!=expected['checkpoint_sources'] or init['scratch_only'] or init['ancestors']!=[expected['parent_source']] or contract['role_ids']!=role_ids:raise ValueError('Fixed source lineage/default-head freeze/data differs')
 if done.get('head_fit_updates',0):raise ValueError('Separate classifier fitting forbidden')

def check_checkpoint(cp,*,run,row,epoch,updates,baseline):
 a=cp['args']
 if cp['epoch']!=epoch or cp['a1_ema_successful_updates']!=updates or cp['run_id']!=run or cp['candidate_id']!=row or a['baseline_ckpt']!=baseline or a['teacher_ckpt'] or cp['checkpoint_selection']!='final_only' or cp.get('ema_model') is None or a['epochs']!=epoch or not a['use_ema_teacher'] or a['label_smoothing']!=.01 or a['batch_size']!=128:raise ValueError('Fixed checkpoint/localcounter/ownEMA/defaulthead args differ')
 return cp

def cumulative(epoch,step):
 if epoch not in range(1,21) or step not in range(1,4441) or epoch!=(step-1)//222+1:raise ValueError('Only local twenty-epoch resume window')
 return epoch+30,step+6660


def capacity_available(state,reserved=()):
 return len(set(state['pids'])|set(reserved))<2 and state['free_mb']>=12000
