from pathlib import Path
from experiments.cvs_sigtime_rdrop_completion20 import design as parent
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,read,write,FULL_FP32_POLICY=(getattr(parent,k) for k in ('PROJECT','SOURCE','read','write','FULL_FP32_POLICY'))
RUN='20261016-phase1-SIGTIME-rdrop-feature-geometry-source-s2026092702-m4-r01'
RELEASE='cvs_SIGTIME_rdrop_feature_geometry_source_20261016_r01'
OWNER='codex/root/SIGTIME-RDrop-feature-geometry-20261016'
BASE=Path(PROJECT)/'runs'/RUN
PARENT_RELEASE=PROJECT+'/releases/'+parent.RELEASE
PARENT_COMMIT='78ad998e9dcb5653573fa801538e914189833d20'
ARMS=parent.ARMS

def row_config(row):
    if row not in parent.rows():raise ValueError('Fixed parent condition required')
    return dict(row_id=row['arm'],parent_row=row,checkpoint=(Path(parent.config(row)['output_root'])/'final_ssdg.pth').as_posix(),
        output_root=(BASE/row['arm']).as_posix(),model_seed=parent.SEED,optimizer_steps=0,head_fit_updates=0,target_access=False)

def request():
    return dict(run_id=RUN,launch_owner=OWNER,source_only=True,target_access=False,
        parent_release=PARENT_RELEASE,parent_commit=PARENT_COMMIT,parent_rows=parent.rows(),rows=[row_config(r) for r in parent.rows()],
        model_release=parent.MODEL_RELEASE,dependency_release=parent.DEPENDENCY_RELEASE,fold_release=parent.FOLD_RELEASE,
        checkpoints={r['arm']:(Path(parent.config(r)['output_root'])/'final_ssdg.pth').as_posix() for r in parent.rows()},
        output_root=BASE.as_posix(),selection='First2 native L per TX/fitRX/day cell=144;all70 L per TX/RX1/day cell=1260;single shared materialization',
        feature='Actual eval id_feat_joint160,FP64 cosine distances after normalization eps1e-4;all raw norms must exceed1e-4',
        relationship='For each fit anchor,compare70sameTX/sameDay heldRX1 distances against350otherTX/sameDay heldRX1 distances;difference<-1e-12 wins,abs difference<=1e-12 ties half',
        per_model_counts=dict(fit_L=144,held_L=1260,same_identity_pairs=10080,different_identity_pairs=50400,triplets=3528000,anchor_records=144,aggregate_groups=86),
        aggregate_groups=['overall','fit_RX','day','TX','TX_fitRX_day_cell'],
        actual_L_materializations=1404,actual_identity_forward_packets=5616,synthetic_CP_smoke_packets=8,
        model_seeds=1,model_conditions=4,synthetic_seed=2026101625,optimizer_steps=0,head_fit_updates=0,
        U_IQ_reads=0,V_IQ_reads=0,target_IQ_reads=0,target_truth_reads=0,
        interpretation='Descriptive known-identity source feature geometry,not a trained readout,classification accuracy,causal attribution,newTX or target generalization. No promotion threshold;failed NULL recipe remains closed.')

def validate(c):
    if c!=request():raise ValueError('Fixed four-model source geometry diagnostic required;no training or target scope')
    return c
