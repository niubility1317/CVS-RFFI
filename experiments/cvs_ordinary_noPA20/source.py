from . import design as d,model as api
from experiments.cvs_ordinary_fold20.source import train as original

def prohibit_PA(augmentor):
    def forbidden(*args,**kwargs):raise ValueError('N2 forbids any actual added PA call')
    augmentor.simulate_pa=forbidden
    return augmentor

def train(c):
    import os
    from pathlib import Path
    from experiments.cvs_receiver_residual_capacity4.control import lock
    from experiments.cvs_multi_state_action.dispatch import capacity
    with lock(d.BASE/(c['row_id']+'.owner.lock'),nonblocking=True):
        gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
        with lock(Path(d.PROJECT)/'runs/receiver_residual_capacity4.lock'):
            if len(capacity()[gpu]['pids']|{os.getpid()})>3:raise ValueError('All-owner GPU capacity exceeded')
        return guarded_train(c)

def guarded_train(c):
    from SSDG import train_ssdg as native
    saved=native.make_augmentor
    native.make_augmentor=lambda cfg:prohibit_PA(saved(cfg))
    try:result=original(c,design=d,model_api=api)
    finally:native.make_augmentor=saved
    args=d.read(c['output_root']+'/actual_native_args.json');args=args.get('args',args)
    if args['aug_enable_pa_normal'] or not args['use_aug'] or result['ordinary']['calls']!=4440:
        raise ValueError('Actual unchanged ordinary except PA switch required')
    d.write(c['output_root']+'/PA_removal_proof.json',dict(actual_PA_calls=0,guard_enabled=True,
        ordinary_calls=4440,actual_aug_enable_pa_normal=False,other_ordinary_enabled=True,U_recipe_changed=False,target_access=False))
    return result
