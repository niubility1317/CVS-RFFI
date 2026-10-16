from copy import deepcopy
from pathlib import Path
from experiments.cvs_f_crossrx_practical20 import design as prior
from experiments.cvs_f_phasecycle20 import design as control
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,FULL_FP32_POLICY,SEED,UPDATES,SCENES,read,write=(getattr(prior,k) for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','SEED','UPDATES','SCENES','read','write'))
RUN='20261016-phase1-FcrossRX-rxmix-source-RX1-e20-s2026092701-m1-r01'
RELEASE='cvs_FcrossRX_rxmix_source_RX1_E20_20261016_r01'
OWNER='codex/root/FcrossRX-rxmix-source-20261016'
BASE=Path(PROJECT)/'runs'/RUN
MIX_SEED=20261016921
REFERENCE=control.REFERENCE
STATISTIC=Path(PROJECT)/'runs/20261016-diagnostic-source-ellipticity-crossfit-manysig-m1-r01/ELLIP-fitRX3_4_6_8'
def rows():return [dict(arm='RXMIX',held_RX=1,row_id=f'RXMIX-RX1-s{SEED}')]
def parent_config():return prior.parent_config()
def config(row):
    if row not in rows():raise ValueError('One fixed source RXMIX row only')
    c=deepcopy(control.config(control.rows()[0]));del c['phase_cycle']
    c.update(schema='FcrossRX_RXMIX_E20_v1',run_id=RUN,**row,launch_owner=OWNER,
        output_root=(BASE/row['row_id']/'source').as_posix(),method='FcrossRX_RXMIX',As=False,new_Ag='UNPROVEN bounded widely-linear received-IQ change',
        rxmix=dict(operation='y=(x+b*conj(x))/sqrt(1+abs(b)**2)',seed=MIX_SEED,statistics_root=STATISTIC.as_posix(),
            amplitude='six absolute pair differences of full fitRX ellipticity mean contrasts, divided by2 and capped at .02',b_abs_max=.02,
            amplitude_schedule='private perRX/day offset then cycle six bank entries by epoch; no TX selection',
            direction='private uniform angle keyed by seed/epoch/physicalRX/day; one b per RX/day/session/epoch for all TX',
            scope='Forward perturbation of already received IQ, not per-packet calibration, unknown TX inversion or replacement by a real RX',
            excluded=['CFO','phase noise','PA','DAC','noise','multipath','time shift','boundary extension','per-packet normalization']),
        training_recipe='F+CROSS_RX unchanged; .5RD(original ordinary IQ)+.5RD(RXMIX(same actual ordinary IQ)); scratch20=13L+7U, original ownEMA/oldU/cos',
        training_mechanism=dict(c['training_mechanism'],second_pair='bounded receivedIQ widely-linear transform of same generated ordinary IQ',ordinary_policy='Common old ordinary unchanged, generated once; RXMIX is the only added IQ operator'),
        source_falsification=dict(c['source_falsification'],on_fail='Close fixed source-ellipticity RXMIX candidate; no bound/coordinate/probability/RX/seed/ratio rescue',on_pass='Only separately preregister remaining source folds; no automatic target or Ag+As promotion'),
        cost_claim='1newE20/4440updates/3579018network packets including own smoke and seven source views; same4Lforwards/oneordinary draw/primaryL-U as C0.568320 boundedIQ transforms; reference C0 separate,no inherited weights.')
    return c
def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed source RXMIX design differs')
def make_args(c,device='cuda:0'):
    validate(c);a=prior.make_args(prior.config(prior.rows()[0]),device)
    a.run_id=RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root'];return a
reference_result=control.reference_result
qualify=control.qualify
