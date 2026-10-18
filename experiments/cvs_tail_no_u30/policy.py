"""Concrete tail behavior: keep data role handles, never fetch U or build a teacher."""
U_WEIGHTS=('lambda_u','lambda_ent','lambda_u_domain','lambda_u_adv','lambda_u_sat_cons','lambda_u_direct_metric_accept','lambda_u_quarantine_accept','lambda_u_zid_receiver_invariance','lambda_u_zid_day_invariance','lambda_u_zid_channel_invariance')
TEACHER_WEIGHTS=('lambda_teacher_clean_kl','lambda_teacher_sat_kl','lambda_teacher_zid_mse')
def validate_no_u_args(a):
 if (a.epochs,a.label_epochs,a.pseudo_epochs)!=(10,0,10) or a.use_unlabeled or a.use_ema_teacher or a.teacher_ckpt or a.from_scratch or a.a1_scratch_only or not a.baseline_ckpt:raise ValueError('Tail must be own20 student +10 L-only with no teacher')
 if any(bool(getattr(a,k,False)) for k in ('use_muse_ssdg','use_adv3b02_daot_stn','fasttrust_rc4','sat_anchor_ssl')) or any(float(getattr(a,k,0.))!=0. for k in U_WEIGHTS+TEACHER_WEIGHTS):raise ValueError('No alternate U/EMA/frozen teacher objective may remain active')
 if getattr(a,'source_unlabeled_epochs',None) is not None:raise ValueError('Tail has no U calendar')
 if a.batch_size!=128 or a.label_smoothing!=.01 or not a.use_aug or a.use_concat_sat_channel_aug or a.freeze_backbone or a.phase1_lr_schedule!='cosine':raise ValueError('Original L/default joint head/ordinary/cosine recipe required')
 return a
class NoUTraversal:
 def __init__(self,loader):self.loader=loader;self.iteration_attempts=0;self.packets=0
 def __len__(self):return len(self.loader)
 def __getattr__(self,key):
  if key=='loader':raise AttributeError(key)
  return getattr(self.loader,key)
 def __iter__(self):
  self.iteration_attempts+=1
  raise RuntimeError('This child retains the U role handle only;U traversal is disabled')
