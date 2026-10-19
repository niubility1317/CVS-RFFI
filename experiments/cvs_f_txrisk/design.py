"""Source-only weak-TX calibration on all three fixed F CROSS_RX models."""
from pathlib import Path
RUN='20261019-diagnostic-FcrossRX-TXrisk-sourceV-m3-r01'
RELEASE='cvs_F_TXrisk_sourceV_20261019_r01'
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
ORIGINAL=PROJECT+'/releases/cvs_F_supcon50_clean_20261016_r02'
ORIGINAL_COMMIT='4df7b94d0fd8ce29056bd8236364e45afa92e07f'
ROOT=Path(__file__).resolve().parents[2];BASE=Path(PROJECT)/'runs'/RUN
SEEDS=(2026092701,2026092702,2026092703)
OWNER='codex/root/F-source-TXrisk-20261019'
GPUS=('2','3');BETA=.5
