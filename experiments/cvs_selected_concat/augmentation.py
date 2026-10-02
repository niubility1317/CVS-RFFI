"""User-fixed two-scene restriction of the existing practical augmentation."""
from baselines.common.practical_source import PracticalResidualAugment

SCENES=('practical_mid','practical_low_urban')
RECIPE=dict(scenes=list(SCENES),satellite_ce_start=80,probability_E80_E90=.60,
    probability_E91_E200=.80,processing_route='residual',mode='post_sync',
    equalization_enabled=False,fs_hz=25000000,augmentation_seed=2027,receiver_seed=2027,
    clean_weight=1.,satellite_weight=.68,concat_sat_ce_only=True,consistency_weight=0.)

class MidUrbanAugment(PracticalResidualAugment):
    def __init__(self):
        super().__init__(fs_hz=25e6,seed=2027,receiver_seed=2027)
        self.configs={s:self.configs[s] for s in SCENES}

    def __call__(self,iq,*,metadata=None):
        if self.epoch<80: raise ValueError('No satellite forward before registered E80')
        # Parent E80..90 chooses mid/urban; later tuple(configs) is now mid/urban.
        return super().__call__(iq,metadata=metadata)
