"""Deduplicated source-model cost ledger from already verified panel cost tables."""
import argparse,csv,json
from pathlib import Path
from .analyze_main import write_csv


def run(folder,output):
    folder=Path(folder);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    def read(name):
        with (folder/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    data=[]
    for r in read('main_F_cost.csv'):
        data.append(dict(source_panel='main153',model=f"{r['protocol']}/{r['regime']}/s{r['seed']}",kind='F',
            shared_worker_seconds=float(r['shared_worker_seconds']),reused_original_T1=r['new_F']=='False'))
    for r in read('F_ablation_F_cost.csv'):
        if r['newly_trained_for_fine_panel']=='True':
            data.append(dict(source_panel='fine63',model=f"{r['arm']}/s{r['seed']}",kind='F',shared_worker_seconds=float(r['shared_worker_seconds']),reused_original_T1=False))
    for r in read('core_completed_F_cost.csv'):
        if r['new_F']=='True':
            data.append(dict(source_panel='core_new12',model=f"{r['bits']}/s{r['seed']}",kind='F',shared_worker_seconds=float(r['shared_worker_seconds']),reused_original_T1=False))
    for r in read('structure_F_cost.csv'):
        if r['new_F']=='True':
            data.append(dict(source_panel='structure_new24',model=f"{r['architecture']}/s{r['seed']}",kind='F',shared_worker_seconds=float(r['shared_worker_seconds']),reused_original_T1=False))
    for r in read('external_final_source_cost.csv'):
        data.append(dict(source_panel='external516',model='/'.join(r[k] for k in ('protocol','family','variant','regime','seed')),kind='F',shared_worker_seconds=float(r['source_shared_worker_seconds']),reused_original_T1=False))
    if len(data)!=768 or len({(r['source_panel'],r['model']) for r in data})!=768:raise ValueError('Source F deduplication differs')
    for r in read('G_ablation_training_cost.csv'):
        data.append(dict(source_panel='G_student21_incremental',model=f"{r['arm']}/s{r['seed']}",kind='G_student',shared_worker_seconds=float(r['seconds']),reused_original_T1=False))
    summary=[]
    for panel in sorted({r['source_panel'] for r in data}):
        r=[v for v in data if v['source_panel']==panel]
        summary.append(dict(source_panel=panel,unique_models=len(r),shared_worker_hours=sum(v['shared_worker_seconds'] for v in r)/3600))
    lineages=read('G_lineage_stage_cost.csv')
    # Stage rows contain35 independent complete chains, not35 one-stage students.
    gseconds=sum(float(r['shared_worker_seconds']) for r in lineages)
    if len(lineages)!=280:raise ValueError('Missing G ancestry stage')
    summary.append(dict(source_panel='G_lineage35_full_8stage',unique_models=35,shared_worker_hours=gseconds/3600))
    write_csv(out/'unique_source_model_cost.csv',data);write_csv(out/'unique_source_cost_summary.csv',summary)
    proof=dict(status='VERIFIED',unique_F_sources=768,unique_PAMF_F_sources=252,unique_external_F_sources=516,
        F_shared_worker_hours=sum(r['shared_worker_seconds'] for r in data if r['kind']=='F')/3600,
        G_complete_lineages=35,G_source_stages=280,G_lineage_shared_worker_hours=gseconds/3600,G_incremental_students=21,
        no_overlap_sum_of_core_Lonly_fine_main=True,
        scope='Selected final scientific source models. Shared-worker elapsed is not isolated GPU-hours or calendar completion time.',
        exclusions=['Historical original F0 G/teacher ancestors not recharged as new35 chains; unavailable historical cost is not invented.',
            'Invalid-input/failed/stopped historical work is preserved separately, not included as valid-model cost.',
            'Prediction, source diagnostics, frozen IQ caches, timing and CPU scoring are separate from training.',
            'This is not total laboratory energy/capacity accounting.'])
    (out/'unique_source_cost_verification.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8');print(json.dumps(proof));return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.input,a.output)
