"""Fixed, source-only comparison; no checkpoint selection by target performance."""
from pathlib import Path
from experiments.cvs_urban_bridge50 import design as training

ROOT,PROJECT,SOURCE=training.ROOT,training.PROJECT,training.SOURCE
RUN='20261015-phase1-sigtime-bridge-source-manysig-s2026092701-m3-r01'
RELEASE='cvs_sigtime_bridge_teaching_source_20261015_r01'
OWNER='codex/root/bridge-teaching-source-20261015'
BASE=Path(PROJECT)/'runs'/RUN
ARMS=('NOMINAL','BRIDGE_CE','BRIDGE_SIG')
VIEWS=('full','HQ_independent')
SOURCE_STREAM=2026101509
RECEIVER_SEED=2026092701
FORWARD_PACKETS=56882
read,write=training.read,training.write


def configs():
    return dict(NOMINAL=training.references()['NOMINAL'],**{r['arm']:training.config(r) for r in training.rows()})


def eligible_rows():
    result=[];fold=None
    for arm,c in configs().items():
        source=Path(c['output_root']);done=read(source/'completion.json');frozen=read(source/'source_frozen.json')
        inner=read(source/'inner_source_contract.json')
        if (done['config']!=c or frozen['config']!=c or done['epoch']!=50 or frozen['epoch']!=50
                or done['optimizer_steps']!=11100 or done['target_access'] or done['head_fit_updates']
                or frozen['selection']!='final_only' or frozen['target_access']):
            raise ValueError('All three own source-only final50/default-head freezes required')
        if fold is None:fold=inner
        if inner!=fold:raise ValueError('All actual source physical folds must match')
        if arm!='NOMINAL':
            mechanism=read(source/'training_mechanism_complete.json');cal=read(source/'bridge_calibration.json')
            if (mechanism['status']!='VERIFIED' or mechanism['updates']!=11100 or mechanism['main_forward_packets']!=9031800
                    or mechanism['head_fit_updates'] or mechanism['U_truth_read'] or cal['status']!='SOURCE_BRIDGE_CALIBRATED' or cal['warmup_epoch']!=5
                    or not cal['source_only'] or cal['query_read'] or cal['forward_packets']!=5760
                    or cal['optimizer_updates'] or cal['EMA_updates'] or cal['head_fit_updates']):
                raise ValueError('Real own five-view training and source epoch5 calibration required')
        result.append((arm,c))
    return result,fold


def configuration():
    return dict(run_id=RUN,owner=OWNER,source_only=True,query_read=False,source_role='L_dev720',
        configs=configs(),views=list(VIEWS),scenes=list(training.SCENES),source_stream=SOURCE_STREAM,
        receiver_seed=RECEIVER_SEED,noise_improvement_db=6.,receiver_actions='full frozen',
        forward_packets=FORWARD_PACKETS,optimizer_updates=0,EMA_updates=0,head_fit_updates=0,
        V_forward_packets=0,U_opened=False,formal_test_IQ_changed=False,
        common_descriptor_scales='BRIDGE_CE own source epoch5 HQ RMS; frozen diagnostic comparator only',
        method_scales='Each new model own source epoch5 HQ RMS; no recalibration or training',
        fixed_correct_set='NOMINAL student clean default head; same physical L_dev720',
        claim='Source mechanism association and repair evidence; not formal target performance or cross-seed proof')
