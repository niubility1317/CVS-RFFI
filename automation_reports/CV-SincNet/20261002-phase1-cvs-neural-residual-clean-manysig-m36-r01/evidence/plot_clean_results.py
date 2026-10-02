"""Plot only frozen, independently validated clean results; no model selection."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / 'final_readback.json').read_text(encoding='utf-8'))
audit = json.loads((HERE / 'analysis_validation.json').read_text(encoding='utf-8'))
assert audit['status'] == 'VERIFIED' and audit['target_feedback'] is False
candidate = audit['candidate']
methods = ['native', 'cvcnn', 'real_cnn', 'resnet1d', 'residual_fusion',
           'energy_equivariant', 'coupled_lag4', 'adaptive_volterra_lag4', candidate]
labels = ['Original CVS', 'CVCNN', 'Real CNN', 'ResNet1D', 'Residual fusion',
          'Energy equivariant', 'Coupled lag4', 'Adaptive lag4', 'New neural residual']
summary = {(r['method'], r['receiver']): r for r in data['summary']['summary']}
pairs = [r for r in data['summary']['paired']
         if r['baseline'] == 'adaptive_volterra_lag4' and r['receiver'] != 'ALL']
pairs.sort(key=lambda r: r['receiver'])
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 6), gridspec_kw={'width_ratios': [1.3, 1]})
colors = ['#94a3b8'] * 7 + ['#c08a28', '#007f78']
for i, (method, color) in enumerate(zip(methods, colors)):
    row = summary[method, 'ALL']
    mean, sd = 100 * row['accuracy_mean'], 100 * row['accuracy_seed_sd']
    left.errorbar(mean, i, xerr=sd, fmt='o', color=color, capsize=4, markersize=8)
    left.text(82.0, i, f'{mean:.2f}', va='center', ha='right', fontsize=10, color=color)
left.set_yticks(range(len(methods)), labels)
left.invert_yaxis()
left.set_xlim(65, 82.5)
left.set_xlabel('Clean accuracy (%) | mean and model-seed SD, n = 4')
left.set_title('Same 168,000 physical queries per model', loc='left', fontweight='bold', pad=15)
left.grid(axis='x', alpha=0.18)
left.set_axisbelow(True)

y = np.arange(len(pairs))
gains = [r['accuracy_delta_pp_mean'] for r in pairs]
right.barh(y, gains, color=['#007f78' if v >= 0 else '#c55645' for v in gains], height=0.58)
right.axvline(0, color='#475569', linewidth=1)
right.set_yticks(y, [r['receiver'] for r in pairs])
right.invert_yaxis()
for i, value in enumerate(gains):
    right.text(value + (0.07 if value >= 0 else -0.07), i, f'{value:+.3f}',
               va='center', ha='left' if value >= 0 else 'right')
right.set_xlim(-1.1, 2.7)
right.set_xlabel('Accuracy change vs Adaptive lag4 (percentage points)')
right.set_title('Receiver means: 6 improve, 1 declines', loc='left', fontweight='bold', pad=15)
right.grid(axis='x', alpha=0.18)
right.set_axisbelow(True)
fig.suptitle('Pure architecture: a local gain, with a historical performance gap',
             x=0.035, y=0.97, ha='left', fontsize=15, fontweight='bold')
fig.text(0.035, 0.025,
         'Source selection frozen before query. Single CE and original training unchanged. '
         '32 frozen controls + 4 new predictions.\n'
         'Historical clean benchmark; model-seed variation is not independent-dataset uncertainty. '
         'No target-driven reselection or retraining.', fontsize=9, color='#475569')
fig.tight_layout(rect=(0.02, 0.105, 1, 0.91), w_pad=3)
fig.savefig(HERE / 'clean_comparison.png', dpi=180)
fig.savefig(HERE / 'clean_comparison.pdf')
plt.close(fig)
print('VERIFIED: plotted 9 fixed methods and 7 receiver deltas')
