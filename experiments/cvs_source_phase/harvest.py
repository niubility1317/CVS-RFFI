"""Read-only source artifacts and fixed offline analysis; no IQ or model reads."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from . import design as d
from .publish import remote

INSPECT = r'''
import json,time
from pathlib import Path
c=CONFIG
release=Path(c['project'])/'releases'/c['release']; root=Path(c['root'])
def read(p):return json.loads(p.read_text()) if p.is_file() else None
receipt=read(release/'submit.json'); proc=Path('/proc')/str(receipt['pid']) if receipt else None
live=None
if proc and proc.exists():
 try:live=dict(pid=receipt['pid'],cwd=str((proc/'cwd').resolve()),argv=(proc/'cmdline').read_bytes().decode().split('\0'))
 except OSError:live=dict(status='UNKNOWN')
log=release/'phase_source.log'
print(json.dumps(dict(read_at=time.time(),submit=receipt,live=live,active=read(root/'active.json'),
 preflight=read(release/'preflight.json'),resolved=read(root/'resolved_config.json'),
 results=read(root/'phase_crossfit.json'),completion=read(root/'completion.json'),failure=read(root/'failure.json'),
 log_bytes=log.stat().st_size if log.is_file() else 0,log=log.read_text(errors='replace') if log.is_file() else '')))
'''


def get(output, name):
    value = remote(INSPECT.replace('CONFIG', repr(dict(project=d.PROJECT, release=d.RELEASE, root=d.BASE))))
    d.write(Path(output) / name, value)
    return value


def finish(output):
    output = Path(output)
    e = get(output, 'terminal_readback.json')
    if e['failure'] or not e['completion'] or e['completion']['status'] != 'SOURCE_DIAGNOSTIC_COMPLETE':
        raise ValueError('Source run has not completed; preserve snapshot without analysis claims')
    d.validate(e['resolved']['config'])
    c = e['completion']
    if (c['L_packets'], c['network_forwards'], c['identity_updates'], c['checkpoint_loads'], c['target_access']) != (6300, 0, 0, 0, False):
        raise ValueError('Observed source-only cost/access differs')
    windows = e['results']['windows']; p = windows[d.config()['primary']]
    rows = []
    for name, r in windows.items():
        if len(r['cells']) != 90 or sum(v['packets'] for v in r['cells']) != 6300 or len(r['folds']) != 18:
            raise ValueError('Incomplete source cells or folds')
        recomputed = float(np.mean([f['gain'] for f in r['folds']]))
        if abs(recomputed - r['mean_risk_gain']) > 1e-12:
            raise ValueError('Fold mean independently differs')
        for cell in r['cells']:
            rows.append(dict(window=name, TX=cell['TX'], RX=cell['RX'], day=cell['day'], packets=70,
                min_eligible=min(cell['eligible']), mean_eligible=np.mean(cell['eligible']),
                mean_split_cosine=np.mean(cell['split_cosine'])))
    with (output / 'source_cells.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for name, r in windows.items():
        ax[0, 0].plot(np.arange(18) + 1, [v['gain'] for v in r['folds']], marker='.', label=name)
        ax[0, 1].plot(list(d.RX), list(r['per_RX_risk_gain'].values()), marker='o', label=name)
        ax[1, 0].plot(list(d.RX), list(r['per_RX_split_cosine'].values()), marker='o', label=name)
    ax[0, 0].axhline(.02, color='black', ls='--', lw=1)
    ax[0, 0].axhline(0, color='grey', lw=.7)
    ax[0, 0].set(xlabel='Held TX/day fold', ylabel='Circular risk gain', title='Source association, not accuracy')
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].axhline(0, color='grey', lw=.7)
    ax[0, 1].set(xlabel='Source RX', ylabel='Mean circular risk gain', title='All five RX are retained')
    ax[1, 0].axhline(.8, color='black', ls='--', lw=1)
    ax[1, 0].set(xlabel='Source RX', ylabel='Odd/even cycle cosine', title='Cycle consistency on eligible tones')
    availability = np.array([v['eligible'] for v in p['cells']]).reshape(6, 3, 5, 10).mean(3).reshape(18, 5)
    im = ax[1, 1].imshow(availability, vmin=0, vmax=70, aspect='auto', cmap='viridis')
    ax[1, 1].set(xticks=np.arange(5), xticklabels=d.RX, xlabel='Source RX', ylabel='TX/day cell',
        title='Primary mean eligible packets / 70')
    fig.colorbar(im, ax=ax[1, 1])
    figure = output / 'phase_source_repeatability.png'; fig.savefig(figure, dpi=160); plt.close(fig)
    summary = dict(status='ANALYZED', source_decision=p['status'], checks=p['checks'], primary=d.config()['primary'],
        mean_risk_gain=p['mean_risk_gain'], positive_folds=p['positive_folds'],
        per_RX_risk_gain=p['per_RX_risk_gain'], minimum_fit_resultant=p['minimum_fit_resultant'],
        minimum_eligible=p['minimum_cell_coordinate_eligible'], eligible_fraction=p['eligible_fraction'],
        per_RX_split_cosine=p['per_RX_split_cosine'], cost=c, commit=e['resolved']['commit'],
        association_not_accuracy=True, automatic_training=False,
        next_action='Abandon this phase augmentation hypothesis' if p['status']=='FAIL' else 'Separate finite-IQ/default-head source safety design only')
    d.write(output / 'analysis_summary.json', summary)
    lines = ['# 源相位重复性结果', '', f"源规则：**{p['status']}**；真实运行与产物：VERIFIED。实际代码 `{e['resolved']['commit']}`。", '',
        '|窗口|risk gain|正收益折/18|最低cell坐标可用数/70|最小拟合resultant|结论|', '|---|---:|---:|---:|---:|---|']
    for name, r in windows.items():
        lines.append(f"|{name}|{r['mean_risk_gain']:.8f}|{r['positive_folds']}|{r['minimum_cell_coordinate_eligible']}|{r['minimum_fit_resultant']:.6f}|{r['status']}|")
    lines += ['', '主窗口规则逐项：', ''] + [f'- {key}：{value}' for key, value in p['checks'].items()]
    lines += ['', f"全量6300源L、18900窗口、75600周期；0模型/前向/更新/checkpoint/目标预测。墙钟 {c['seconds']:.3f} 秒。", '',
        '这里的risk gain是源RX相位统计的留出误差差，不是准确率或百分点。周期重复比不是总SNR；RX关联不能区分硬件、传播、均衡或采集条件。相位共轭、CFO混叠和模板时移规范仍未唯一识别。', '',
        '失败后不换窗口、降阈值或调幅启动训练。通过也仅支持另做有限IQ/defaulthead源安全验证，不能跳过身份保持和同预算训练对照。当前没有任何新clean/urban性能或跨seed结果。', '',
        f"图：{figure.as_posix()}；逐cell表：{(output/'source_cells.csv').as_posix()}。"]
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True); parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    if args.inspect:
        e = get(args.output, 'post_launch_readback.json')
        print(json.dumps(dict(live=e['live'], completion=e['completion'], log_bytes=e['log_bytes'], failure=e['failure'])))
    else:
        finish(args.output)
