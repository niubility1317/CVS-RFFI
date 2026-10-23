"""Preserve native T1; replace only the approved supervised channel inputs."""
def train(c):
    from experiments.cric_dura_comparison import t1_train,t1_model
    if c['regime']=='native_cric':
        if c['satellite_concat'] or c.get('channel_augmentation') is not None:raise ValueError('Native T1 recipe altered')
        return t1_train.train(c)
    if c['regime']!='matched_sat_supervised' or c['own_EMA'] or c['U_use']:raise ValueError('Strict supervised scope required')
    from .pamf_model import install
    saved=t1_model.install;t1_model.install=install
    try:return t1_train.train(c)
    finally:t1_model.install=saved
