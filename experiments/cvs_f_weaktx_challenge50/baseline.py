"""User-selected 85.113889% original F+CROSS_RX baseline, never E20/satellite."""
REFERENCE_RUN='20261016-phase1-F-supcon-e50-clean-manysig-m9-r02'
REFERENCE_COMMIT='4df7b94d0fd8ce29056bd8236364e45afa92e07f'
SEEDS=(2026092701,2026092702,2026092703)
EPOCHS=50;LABEL_EPOCHS=33;PSEUDO_EPOCHS=17;UPDATES=11100
TEST_VIEWS=('clean',)

def original_config(seed):
 if seed not in SEEDS:raise ValueError('Use the registered three model seeds')
 from experiments.cvs_f_supcon50 import design as original
 config=original.config(next(r for r in original.training_rows() if r['arm']=='CROSS_RX' and r['model_seed']==seed))
 combo=config['combination'];mechanism=config['training_mechanism'];metric=config['supcon']
 if combo['SIG_TIME']!='SIG_TIME_X3' or combo['Sinc_optimizer']!='CENTER_UNITS' or mechanism['alpha']!=1. or mechanism['label_smoothing']!=.01 or not mechanism['default_W_joint'] or metric['mode']!='CROSS_RX' or metric['levels']!=['time_down','t2.pool'] or config['test_views']!=['clean']:raise ValueError('Original F+CROSS_RX baseline differs')
 return config

def assert_recipe(args):
 required=dict(epochs=50,label_epochs=33,pseudo_epochs=17,batch_size=128,label_smoothing=.01,use_aug=True,use_ema_teacher=True,max_grad_norm=5.,use_concat_sat_channel_aug=False)
 if any(getattr(args,key,None)!=value for key,value in required.items()):raise ValueError('User selected original scratch50=33L+17U F+CROSS_RX, no satellite augmentation')
 if getattr(args,'baseline_ckpt',None) or getattr(args,'teacher_ckpt',None):raise ValueError('Fresh candidate cannot inherit an unverified external checkpoint or teacher')
 return args

def make_args(seed,device='cuda:0'):
 from experiments.cvs_f_supcon50 import design as original
 return assert_recipe(original.make_args(original_config(seed),device))
