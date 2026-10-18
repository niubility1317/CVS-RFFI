"""Actual seed/stage/factory and one source-only eligibility decision."""
def cumulative(epoch,step,budget):
 if budget not in (30,50):raise ValueError('Only weighted30 or50')
 epochs,offset,steps=(10,20,4440) if budget==30 else (20,30,6660)
 if epoch not in range(1,epochs+1) or step not in range(1,epochs*222+1) or epoch!=(step-1)//222+1:raise ValueError('Fixed native staged window differs')
 return epoch+offset,step+steps

def check_checkpoint(cp,*,seed,run,row,epoch,updates,baseline):
 a=cp['args']
 if cp['epoch']!=epoch or cp['a1_ema_successful_updates']!=updates or cp['run_id']!=run or cp['candidate_id']!=row or a['seed']!=seed or a['baseline_ckpt']!=baseline or a['teacher_ckpt'] or cp['checkpoint_selection']!='final_only' or cp.get('ema_model') is None or a['epochs']!=epoch or not a['use_ema_teacher'] or a['label_smoothing']!=.01 or a['batch_size']!=128:raise ValueError('Actual seed/ownEMA/local stage checkpoint differs')
 return cp

def check_metadata(done,frozen,init,contract,c,roles):
 if done['config']!=c or done['epoch']!=c['epochs'] or done['optimizer_steps']!=c['updates'] or done['target_access'] or done.get('target_evaluated',False) or frozen['config']!=c or frozen['selection']!='final_only' or frozen['target_access'] or init['target_contact'] or init['scratch_only'] or init['checkpoint_sources']!=c['checkpoint_sources'] or init['ancestors']!=[c['parent_source']] or contract['role_ids']!=roles or done.get('head_fit_updates',0):raise ValueError('Own staged frozen metadata differs')
 if done['cumulative_epochs']!=c['target_budget'] or done['cumulative_optimizer_steps']!=c['cumulative_updates']:raise ValueError('Own cumulative budget metadata differs')

def check_factory(actual,c):
 if actual!=c['model_seed'] or c['seed_binding']['F_factory_model_seed']!=actual:raise ValueError('True F factory seed differs')

def eligibility(summary,run):
 decision=summary['decision']
 if summary['run_id']!=run or summary['target_scores_read'] is not False or any(summary[k]!=0 for k in ('NN','IQ_reads','checkpoint_reads','updates')):raise ValueError('Only fixed seed01 independent source decision may authorize replication')
 required=('source_criteria_pass','fixed_02_03_source_eligible','clean_nonnegative','mid_low_gain_at_least_0p5pp')
 if any(decision[k] is not True for k in required) or decision['newAs50_minus_ownAs30_fullV_clean_pp']<0. or decision['newAs50_minus_ownAs30_mid_low_mean_pp']<.5:raise ValueError('Conditional fixedsource eligibility not satisfied;do not register or launch')
 return dict(status='SOURCE_ELIGIBILITY_PASS',fixed_seeds=[2026092702,2026092703],formal_scores_read=False)
