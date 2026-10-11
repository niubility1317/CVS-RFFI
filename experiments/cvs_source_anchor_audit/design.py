from pathlib import Path
from experiments.cvs_report_ssl import design as parent

ROOT=Path(__file__).resolve().parents[2]
PROJECT=parent.PROJECT
RUN='20261011-phase1-source-anchor-gradient-manysig-m6-r01'
RELEASE='cvs_source_anchor_gradient_20261011_r01'
BASE=Path(PROJECT)/'runs'/RUN
OWNER='codex/root/source-anchor-gradient-20261011'
SEEDS=parent.SEEDS
SOURCE_ARMS=('C1','C2')
read,write=parent.read,parent.write

def rows():
    return [dict(row_id=f'{a}-s{s}',source_arm=a,model_seed=s) for s in SEEDS for a in SOURCE_ARMS]

def config(row):
    if row not in rows():raise ValueError('Unregistered source diagnostic row')
    c=parent.config(dict(arm=row['source_arm'],model_seed=row['model_seed'],row_id=row['row_id']))
    return dict(run_id=RUN,**row,source_run=parent.RUN,source_config=c,
        output=(BASE/row['row_id']).as_posix(),source_checkpoint=c['output_root']+'/final_ssdg.pth',
        teacher_and_heads=c['output_root']+'/auxiliary_state.pt',
        source_only=True,target_access=False,identity_updates=0,source_roles=dict(L=6300,U=56700,V=27000),
        graph=dict(k=32,temperature=.07,restart=1.,max_iterations=100,tolerance=1e-6,
            symmetric=True,scale='mean degree one',anchors='all legal L true TX',
            coordinates=['student','EMA'],feature='separate frozen E40 normalized identity160 coordinates',U_restart='corresponding model soft probabilities',
            U_restart_weight='max corresponding model class probability',V_in_fit=False,forced_cross_RX_edges=False),
        gradient=dict(batches=32,L_batch=128,U_batch=128,L_probe_batch=128,
            sample_seed=row['model_seed']+711337,mode='deterministic eval with gradients enabled',
            parameter_scope='dense identity only; native Hz excluded',virtual_relative_step=1e-5,
            teacher_targets=['EMA','current_student_snapshot_without_averaging'],paired_batches=True,
            virtual_update='normalized gradient diagnostic; restored exactly; not AdamW or persisted training'),
        source_decision='Report all six rows and all fixed readouts; no target files or automatic training choice',
        launch_owner=OWNER)

def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Source diagnostic config differs')
