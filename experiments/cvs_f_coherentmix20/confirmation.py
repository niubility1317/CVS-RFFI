"""Fixed seed01 confirmation of the already passed seed02 coherent IQ screen."""
from pathlib import Path
PROFILE='coherent_confirmation_seed01'
SEED=2026092701
RUN='20261017-phase1-FcrossRX-coherentmix-confirm-source-RX1-e20-s2026092701-m3-r01'
RELEASE='cvs_FcrossRX_coherentmix_confirm_source_RX1_E20_20261017_r01'
OWNER='codex/root/FcrossRX-coherentmix-confirm-20261017'
REFERENCE_RUN='20261016-phase1-F-crossRX-additiveAs-sourceRX1-e20-s2026092701-m3-r01'
BACKEND='/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_F_crossRX_additiveAs_sourceRX1_e20_20261016_r01'
BACKEND_COMMIT='7f398321cd2ffd2c2d2cb11f52f67bfd8c44987b'
FIRST='20261016-phase1-FcrossRX-coherentmix-source-RX1-e20-s2026092702-m3-r02'
def activate():
 from . import design as d
 # Pin the configuration that actually constructs F,not only native args.seed.
 d.parent.old.prior.SEED=SEED;d.parent.old.SEED=SEED;d.parent.SEED=SEED
 d.SEED=SEED;d.RUN=RUN;d.RELEASE=RELEASE;d.OWNER=OWNER;d.REFERENCE_RUN=REFERENCE_RUN;d.BACKEND=BACKEND;d.BACKEND_COMMIT=BACKEND_COMMIT
 d.BASE=Path(d.PROJECT)/'runs'/RUN;d.SMOKE_SOURCE=Path(d.PROJECT)/'runs'/REFERENCE_RUN/'D_raw-RX1-s2026092701/source'
 if d.parent.base_model_config()['model_seed']!=SEED:raise ValueError('Actual F constructor seed differs from confirmation')
 return d
