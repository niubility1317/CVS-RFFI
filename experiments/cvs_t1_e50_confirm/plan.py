"""The user-authorized three-seed continuation; seed01 is never retrained."""
SEEDS=(2026092701,2026092702,2026092703)
NEW_SEEDS=SEEDS[1:]

def profile(seed):
 if seed not in NEW_SEEDS:raise ValueError('Only seed02/03 are new training rows; seed01 is completed reuse')
 return dict(arm='E30PLUS20',model_seed=seed,epochs=20,label_epochs=0,pseudo_epochs=20,
             updates=4440,epoch_offset=30,step_offset=6660,cumulative_epochs=50,
             cumulative_updates=11100,As_KL=0.,target_training=False)

def validate(p):
 if p!=profile(p.get('model_seed')):raise ValueError('Fixed continuation seed/budget/permissions changed')
 return p
