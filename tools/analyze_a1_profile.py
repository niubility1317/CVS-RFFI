"""Read complete archived A1-A7 logs; reproduce descriptive A1 comparisons."""
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean

ROOT = Path('E:/type10-7')
SRC = ROOT / 'local_artifacts/adv3b02_two_batch_report_20260905'
OUT = ROOT / 'automation_reports/CV-SincNet/adv3b02_a1_comprehensive_20260907'
OUT.mkdir(parents=True, exist_ok=True)
scenes = ['clean', 'leo_clear_weak', 'leo_low_elev_weak', 'leo_rain_weak']
data = {}
curves = []
all_rx = []
fields = ['epoch', 'val_tx_acc', 'stage_source_val_sat_mean_tx', 'val_dom_acc',
          'train_loss', 'train_loss_labeled', 'train_loss_unlabeled',
          'train_w_loss_domain_labeled', 'train_w_loss_adv_labeled',
          'train_w_loss_tx_labeled', 'train_loss_daot_total',
          'train_loss_daot_unlabeled', 'train_loss_daot_orbit_logit',
          'train_loss_daot_orbit_z', 'train_daot_consensus_rate',
          'train_skipped_nonfinite_grad', 'train_skipped_nonfinite_loss',
          'epoch_time_s', 'lr']
for folder in sorted(SRC.glob('*/*')):
    if not folder.is_dir():
        continue
    match = re.search(r'_(A[1-7]|P3)_', folder.name)
    if not match:
        continue
    row = match[1]
    records = [json.loads(s) for s in (folder/'metrics_epoch.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    assert [r['epoch'] for r in records] == list(range(1, 201))
    csv_records = list(csv.DictReader((folder/'metrics_epoch.csv').open(encoding='utf-8-sig')))
    assert len(csv_records) == 200
    logs = {}
    for p in folder.glob('*.log'):
        lines = p.read_text(encoding='utf-8-sig').splitlines()
        logs[p.name] = {'lines': len(lines), 'markers': [
            {'line': i+1, 'text': s[:600]} for i, s in enumerate(lines)
            if re.search(r'Traceback|RuntimeError|out of memory|Killed|non.?finite|nan|inf|warning|resume', s, re.I)]}
    metrics = {}
    for scene in scenes:
        m = json.loads((folder/f'metrics_{scene}.json').read_text(encoding='utf-8-sig'))
        ag = m['aggregate']
        assert m['checkpoint_epoch'] == 200 and m['reconstruction_audit']['checkpoint_load_strict']
        assert ag['tx_total'] == 168000
        assert sum(r['tx_correct'] for r in m['rows']) == ag['tx_correct']
        assert abs(100*ag['tx_correct']/ag['tx_total']-ag['tx_acc']) < 1e-8
        metrics[scene] = ag
        for r in m['rows']:
            all_rx.append({'method': row, 'scene': scene, **{k:r[k] for k in ['rx_idx','rx_label','tx_acc','tx_correct','tx_total']}})
    numeric = {}
    for k in records[-1]:
        vals = [(r['epoch'], r.get(k)) for r in records]
        vals = [(e, v) for e, v in vals if isinstance(v, (float, int)) and math.isfinite(v)]
        if vals:
            numeric[k] = {'first': vals[0], 'last': vals[-1], 'min': min(vals,key=lambda x:x[1]), 'max': max(vals,key=lambda x:x[1]), 'nonzero_epochs': sum(v != 0 for _,v in vals)}
    data[row] = {'source': str(folder), 'epochs':200, 'csv_records':len(csv_records),
        'log_scan':logs, 'metrics':metrics, 'numeric_summary':numeric,
        'resource':json.loads((folder/'phase1_resource_summary.json').read_text(encoding='utf-8-sig')),
        'config':json.loads((folder/'config.json').read_text(encoding='utf-8-sig')),
        'breakpoints':[{k:r.get(k) for k in fields} for r in records if r['epoch'] in [1,20,21,30,40,60,80,90,94,100,130,140,180,200]],
        'first_domain_gt10':next((r['epoch'] for r in records if r['train_w_loss_domain_labeled']>10),None),
        'first_domain_gt100':next((r['epoch'] for r in records if r['train_w_loss_domain_labeled']>100),None),
        'last20_source_clean_range':[min(r['val_tx_acc'] for r in records[-20:]),max(r['val_tx_acc'] for r in records[-20:])],
        'last20_source_leo_range':[min(r['stage_source_val_sat_mean_tx'] for r in records[-20:]),max(r['stage_source_val_sat_mean_tx'] for r in records[-20:])]}
    curves.extend({'method':row, **{k:r.get(k) for k in fields}} for r in records)
comparisons=[]
for other in ['A2','A3','A4','A5','A6','A7','P3']:
    a,b=data['A1']['metrics'],data[other]['metrics']
    dc=a['clean']['tx_acc']-b['clean']['tx_acc']
    dl=mean(a[s]['tx_acc'] for s in scenes[1:])-mean(b[s]['tx_acc'] for s in scenes[1:])
    comparisons.append({'other':other,'clean_delta_pp':dc,'leo_delta_pp':dl,
        'extra_clean_correct':a['clean']['tx_correct']-b['clean']['tx_correct'],
        'clean_weight_break_even':-dl/(dc-dl),
        'equal_clean_leo_A1':(a['clean']['tx_acc']+mean(a[s]['tx_acc'] for s in scenes[1:]))/2,
        'equal_clean_leo_other':(b['clean']['tx_acc']+mean(b[s]['tx_acc'] for s in scenes[1:]))/2})
def writecsv(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
writecsv('full_curves.csv',curves)
writecsv('receiver_details.csv',all_rx)
writecsv('tradeoffs.csv',comparisons)
(OUT/'analysis.json').write_text(json.dumps({'rows':data,'comparisons':comparisons},ensure_ascii=False,indent=2),encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
for row in ['A1','A2','A4','P3']:
    rs=[r for r in curves if r['method']==row]
    for ax,k,title in zip(axs.flat,['val_tx_acc','stage_source_val_sat_mean_tx','train_w_loss_domain_labeled','val_dom_acc'],['Source clean accuracy (%)','Source LEO accuracy (%)','Weighted domain loss (log scale)','Source domain accuracy (%)']):
        ax.plot([r['epoch'] for r in rs],[r[k] for r in rs],label=row,linewidth=2 if row=='A1' else 1.2)
        ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
        if 'loss' in k:ax.set_yscale('log')
        ax.legend()
fig.suptitle('A1 training profile: full E1-E200 source-only curves')
fig.savefig(OUT/'training_curves.png',dpi=170)
plt.close(fig)
print(json.dumps({'rows':list(data),'epochs':len(curves),'logs':sum(len(x['log_scan']) for x in data.values()),'log_lines':sum(y['lines'] for x in data.values() for y in x['log_scan'].values()),'comparisons':comparisons,'A1_breakpoints':data['A1']['breakpoints'],'A1_extrema':{k:v for k,v in data['A1']['numeric_summary'].items() if k in fields},'A1_domain_gt10':data['A1']['first_domain_gt10'],'A1_domain_gt100':data['A1']['first_domain_gt100']},ensure_ascii=False))
