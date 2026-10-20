"""The two user-authorized E50 paths; no data or model imports."""
ARMS = ('E30PLUS20', 'SCRATCH50')


def profile(arm):
    if arm not in ARMS:
        raise ValueError('Only the two registered T1 E50 paths are authorized')
    tail = arm == 'E30PLUS20'
    epochs = 20 if tail else 50
    return dict(arm=arm, model_seed=2026092701, epochs=epochs,
                label_epochs=0 if tail else 33, pseudo_epochs=20 if tail else 17,
                updates=epochs * 222, epoch_offset=30 if tail else 0,
                step_offset=6660 if tail else 0, cumulative_epochs=50,
                cumulative_updates=11100, As_KL=0., target_training=False)


def validate(value):
    if value != profile(value.get('arm')):
        raise ValueError('Fixed budget, seed, As KL, or data permission changed')
    return value


def training_clock(arm, epoch, step):
    p = profile(arm)
    if not isinstance(step, int) or step not in range(1, p['updates'] + 1) or epoch != (step - 1) // 222 + 1:
        raise ValueError('Training epoch/step outside registered budget')
    return epoch + p['epoch_offset'], step + p['step_offset']
