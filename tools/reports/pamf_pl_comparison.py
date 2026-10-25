"""Read frozen aggregates; publish the current PL comparison without retraining."""
import argparse,csv,gzip,json,shutil,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path('E:/type10-7')
sys.path.insert(0,str(ROOT))
FROZEN=ROOT/'exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/frozen_statistics'
EXT=ROOT/'local_artifacts/external_practical_residual_2x2_design_20261023/full_analysis_20261024'
OUT=ROOT/'exports/reports/20261025-PAMF-PL-comparison-r20'
PREVIOUS=ROOT/'exports/reports/20261025-PAMF-PL-comparison-r18'
SEEDS=(2026092701,2026092702,2026092703)
PIDS=('F0','F1_rx_repartition_r01','F2_rx_repartition_r01','ManyRx','label01','label05','m1-natural','m2-natural','m3-natural','m4-natural','m6-natural')
METHODS=('riei','drift','ibrff','sdg','supervised','ordinary_PL')
NAMES=dict(zip(METHODS,('RIEI','DRIFT','IB-RFF','SDG','PAMF（监督训练）','PAMF（普通伪标签）')))
SHORT=dict(zip(PIDS,('标准接收机划分','接收机划分一','接收机划分二','大规模十类数据','1%标注','5%标注','1个源接收机','2个源接收机','3个源接收机','4个源接收机','6个源接收机')))
TASKS=('cross_date','cross_rx','cross_both');TN=dict(zip(TASKS,('跨日','跨接收机','跨日＋跨接收机')))
SCENES=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
SN=dict(zip(SCENES,('clean','高郊区','中郊区','低郊区','高城市','中城市','低城市')))

def allowed(r):
    return r['cell'] not in ('A','B') or (r['view']=='clean' and r['test'] in ('cross_rx','cross_both'))

def project():
    """Reuse verified aggregates; exclude no-channel practical rows from this report."""
    if OUT.exists():raise FileExistsError(OUT)
    (OUT/'tables').mkdir(parents=True);(OUT/'figures').mkdir()
    for name in ('complete_summary.csv','per_seed_overall.csv.gz'):
        write(OUT/'tables'/name,(r for r in rows(PREVIOUS/'tables'/name) if allowed(r)))
    for name in ('PAMF_actual_exposures.csv','external_training_and_exposures.csv','PAMF_main_unique_stage_cost.csv','PAMF_current_timing.csv','data_contracts.csv'):
        shutil.copy2(PREVIOUS/'tables'/name,OUT/'tables'/name)
    summary=list(rows(OUT/'tables/complete_summary.csv'))
    seed=list(rows(OUT/'tables/per_seed_overall.csv.gz'))
    groups=[]
    for p in PIDS:
        for m in METHODS:
            cells=('native',) if m in ('supervised','ordinary_PL') else ('A','B','C','D')
            for cell in cells:
                on=cell not in ('A','B')
                section='PAMF_main' if m in ('supervised','ordinary_PL') else 'external_channel_on' if on else 'external_no_channel_CLEAN_only'
                for s in SEEDS:
                    groups.append(dict(protocol=p,display_protocol=SHORT[p],method=m,display_method=NAMES[m],cell=cell,seed=s,channel_augmentation='on' if on else 'off',PL='on' if m=='ordinary_PL' or cell in ('B','D') else 'off',report_views='clean;'+(';'.join(SCENES[1:])) if on else 'clean',report_populations='cross_date;cross_rx;cross_both' if on else 'cross_rx;cross_both',section=section,main_comparison=bool(m in ('supervised','ordinary_PL') or cell in ('C','D')),DURA_readout_section='generality_only' if m in METHODS[:4] else 'PAMF_system_and_restoration_control'))
    write(OUT/'tables/experiment_groups.csv',groups)
    v=json.loads((PREVIOUS/'verification.json').read_text(encoding='utf-8'))
    v.update(summary_rows=len(summary),per_seed_overall_rows=len(seed),main_display_names=['RIEI','DRIFT','IB-RFF','SDG','PAMF','DURA+PAMF'],excluded_historical_no_channel_practical_summary_rows=7392,excluded_historical_no_channel_source_date_CLEAN_rows=352,report_group_classifier_rows=len(groups))
    (OUT/'verification.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # The reporting model is the single owner of regime-specific projections.
    # Retained frozen inputs above are historical evidence, not a mixed main panel.
    from tools.reports.pamf_reporting_model import refresh
    refresh()
def rows(path):
    with (gzip.open if str(path).endswith('.gz') else open)(path,'rt',encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)
def records(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:yield json.loads(line)
def write(path,data):
    data=list(data);path.parent.mkdir(parents=True,exist_ok=True)
    with (gzip.open if str(path).endswith('.gz') else open)(path,'wt',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def figures():
    from tools.reports.pamf_weekly_data import main as workbook_data
    from tools.reports.pamf_experiment_figures import main as vertical_figures
    workbook_data()
    vertical_figures()

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['project','figures']);x=a.parse_args();project() if x.phase=='project' else figures()
