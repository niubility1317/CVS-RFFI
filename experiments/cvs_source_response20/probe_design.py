"""One source-only finite-IQ falsification, no new model training."""
import json
from pathlib import Path
from experiments.cvs_mixed_iq20 import probe_design as old

ROOT=Path(__file__).resolve().parents[2]
PROJECT=old.PROJECT
RUN='20261016-phase1-source-response-finiteIQ-s2026092701-r01'
RELEASE='cvs_source_response_finiteIQ_20261016_r01'
OWNER='codex/root/source-response-finiteIQ-20261016'
BASE=Path(PROJECT)/'runs'/RUN
PARENT=old.PARENT
SOURCE=old.SOURCE
PREPARATION=old.PREPARATION
CONDITIONS=('RAW','EMP_E1_REFLECT','ROT_E1_REFLECT','EMP_E20_REFLECT','ROT_E20_REFLECT','EMP_E20_EDGE','ROT_E20_EDGE')
FORWARDS=27000*len(CONDITIONS)+2
read=old.read
write=old.write


def config():
    return dict(run_id=RUN,launch_owner=OWNER,source_only=True,query_read=False,source_role='V',source_count=27000,
        checkpoint=PARENT+'/final_ssdg.pth',parent_model_seed=2026092701,parent_epochs=20,parent_updates=4440,
        source_RX=[1,3,4,6,8],source_days=[1,2,3],equalized=1,IQ_length=256,sampling_rate=25000000,
        source_bank_run='20261011-phase1-rx-spectral-crossfit-manysig-m1-r01',source_bank_primary='rectangular/DCT1to8',
        source_bank_role='L6300-derived predictions only; average six excluded TX for each day/RX',bank_entries=15,
        rotation_seed=20261016011,augmentation_seed=20261016023,bank_taps=33,delta_L1_bound=.06,
        paired_control='same kernel L2 per bank entry, common L1 bound, same sampled strength',
        conditions=list(CONDITIONS),candidate_conditions=['EMP_E1_REFLECT','EMP_E20_REFLECT'],
        max_accuracy_drop_pp=1.,max_worst_RX_drop_pp=2.,
        forward_packets=FORWARDS,threads=2,device='CPU',optimizer_updates=0,EMA_updates=0,head_fit_updates=0,
        new_training_epochs=0,new_models=0,target_predictions=0,automatic_training=False,
        interpretation='Frozen source safety and finite-IQ realization only; not training benefit or target generalization')


def validate(value):
    if value!=config():raise ValueError('Source-only finite-IQ config differs')


def gate(results):
    raw=results['RAW'];drops={};rx_drops={}
    for name in config()['candidate_conditions']:
        drops[name]=100*(raw['overall']['accuracy']-results[name]['overall']['accuracy'])
        base={x['value']:x['accuracy'] for x in raw['strata'] if x['dimension']=='RX'}
        rx_drops[name]=max(100*(base[x['value']]-x['accuracy']) for x in results[name]['strata'] if x['dimension']=='RX')
    passed=all(v<=1.+1e-9 for v in drops.values()) and all(v<=2.+1e-9 for v in rx_drops.values())
    return dict(status='PASS' if passed else 'FAIL',accuracy_drop_pp=drops,worst_RX_drop_pp=rx_drops,
        scope='Necessary source operator safety only; no automatic promotion or training improvement',target_metrics_read=False)
