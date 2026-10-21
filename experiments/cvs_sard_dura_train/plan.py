ARMS = ('SARD', 'SARD_DURA_AUG')
VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
         'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')
SEED = 2026092701

def profile(arm):
    if arm not in ARMS:
        raise ValueError('Only matched SARD and DURA-augmentation rows')
    return dict(arm=arm, model_seed=SEED, epochs=50, label_epochs=13, pseudo_epochs=37,
                updates=11100, epoch_offset=0, step_offset=0, cumulative_epochs=50,
                cumulative_updates=11100, As_KL=0., target_training=False)

def require_complete_predictions(keys):
    expected = {m+'__'+v for m in ('raw', 'dura') for v in VIEWS}
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError('Truth closed until all14 registered prediction cells are fixed')
