"""Chinese accuracy/CE figures from an existing frozen source-fit JSON only.

No model, IQ, checkpoint, training or formal-test reader is imported. The raw L
reference uses all180 IDs; each scenario's old/new L pair uses its own subset.
"""
import argparse
import json
import math
import os
from pathlib import Path

RUN = '20261019-diagnostic-F-rawAs-sourceFit-e20-s2026092701-m2-r01'
ARMS = ('B0', 'B1', 'RAW_NOAS', 'RAW_AS')
SCENES = ('practical_low_suburban', 'practical_mid_urban', 'practical_low_urban')
TITLES = ('低仰角郊区', '中仰角城市', '低仰角城市（重点）')
LABELS = {'B0': 'B0：普通 NoAs', 'B1': 'B1：普通 As',
          'RAW_NOAS': 'RAW_NOAS：raw＋raw', 'RAW_AS': 'RAW_AS：raw＋As'}
COLORS = {'B0': '#7B858C', 'B1': '#286F9C', 'RAW_NOAS': '#A66D25', 'RAW_AS': '#C13F4A'}
MARKERS = {'B0': 'o', 'B1': 's', 'RAW_NOAS': '^', 'RAW_AS': 'D'}


def checked_metric(value):
    n, acc, ce = value['n'], value['accuracy'], value['hard_label_NLL']
    if (isinstance(n, bool) or not isinstance(n, int) or n <= 0
            or not isinstance(acc, (int, float)) or not math.isfinite(acc) or not 0 <= acc <= 1
            or not isinstance(ce, (int, float)) or not math.isfinite(ce) or ce < 0
            or abs(n*acc-round(n*acc)) > 1e-6):
        raise ValueError('Finite source counts/accuracy/default-logit CE required')
    return dict(observations=n, correct=round(n*acc), accuracy_percent=100*acc,
                standard_CE=ce)


def prepare(data):
    """Validate the known diagnostic schema and expose exactly the plotted data."""
    if (data['status'] != 'SOURCE_ONLY_FROZEN_READOUT_COMPLETE' or data['run_id'] != RUN
            or data['model_seed_count'] != 1 or data['target_access']
            or data['source_promotion_changed']):
        raise ValueError('Expected completed single-seed source-only diagnostic')
    raw = {a: checked_metric(data['L'][a]['raw']['overall']) for a in ARMS}
    if any(r['observations'] != 180 for r in raw.values()):
        raise ValueError('Raw reference must retain the original180 L IDs')
    scenes = {}
    for scene in SCENES:
        values = {}
        for arm in ARMS:
            values[arm] = [checked_metric(data['L'][arm]['observed_reconstructed']['by_scene'][scene]),
                           checked_metric(data['L'][arm]['new_paired']['by_scene'][scene]),
                           checked_metric(data['V'][arm]['new_by_scene'][scene]['overall'])]
        n = values['B0'][0]['observations']
        if any(v[0]['observations'] != n or v[1]['observations'] != 2*n
               or v[2]['observations'] != 3600 for v in values.values()):
            raise ValueError('Paired L subset/two-draw/V physical count mismatch')
        scenes[scene] = dict(L_physical_IDs=n, L_new_draws=2, V_physical_IDs=1800,
                             V_draws=2, arms=values)
    if sum(v['L_physical_IDs'] for v in scenes.values()) != 180:
        raise ValueError('Observed scenario subsets must partition the existing180 L panel')
    return dict(raw_all180=raw, scenes=scenes, stage_order=[
        '训练命名流重建 As', '同 L 新 As', '源 V 新 As'], model_seed_count=1,
        raw_is_common_reference_not_scenario_subset=True,
        standard_CE='Mean hard-label CE on default logits; no temperature or label smoothing')


def chinese_font(matplotlib):
    from matplotlib import font_manager
    configured = os.environ.get('FONT_PATH')
    candidates = ([Path(configured)] if configured else []) + [
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Light.ttc'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'),
        Path('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')]
    for path in candidates:
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=str(path)).get_name()
            matplotlib.rcParams['font.family'] = [name]
            return dict(family=name, path=str(path))
    for family in ('Noto Sans CJK SC', 'Noto Sans CJK JP', 'Source Han Sans SC',
                   'WenQuanYi Micro Hei', 'Microsoft YaHei', 'SimHei'):
        try:
            path = font_manager.findfont(font_manager.FontProperties(family=family),
                                         fallback_to_default=False)
        except ValueError:
            continue
        matplotlib.rcParams['font.family'] = [family]
        return dict(family=family, path=path)
    raise RuntimeError('No existing CJK font found; FONT_PATH may name an existing font file')


def figure(payload, metric, prefix, plt):
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    accuracy = metric == 'accuracy_percent'
    kind = 'accuracy' if accuracy else 'CE'
    fig = plt.figure(figsize=(16.8, 9.0), facecolor='white')
    grid = fig.add_gridspec(2, 3, height_ratios=[.6, 3.], left=.055, right=.985,
                           top=.80, bottom=.27, wspace=.20, hspace=.63)
    title = '源域识别准确率' if accuracy else '源域标准交叉熵'
    fig.text(.055, .955, title+'：固定 E20 默认头的冻结读出', fontsize=20, weight='bold')
    fig.text(.055, .913, '比较 raw＋As 的输入组织；全部为源域诊断，不是正式七视图测试结果。',
             fontsize=12, color='#3E4650')
    rawax = fig.add_subplot(grid[0, :]); rawax.set_axis_off()
    rawax.text(0., 1.28, '原始 L 参考：全部 180 个物理 ID（不是各场景子集的逐包配对值）',
               transform=rawax.transAxes, fontsize=11, weight='bold', color='#3E4650')
    for i, arm in enumerate(ARMS):
        left = .003+i*.25
        rawax.add_patch(FancyBboxPatch((left, -.06), .236, 1.02,
            transform=rawax.transAxes, boxstyle='round,pad=.006,rounding_size=.025',
            facecolor='#FAFBFC', edgecolor=COLORS[arm], linewidth=1.1, clip_on=False))
        item = payload['raw_all180'][arm]
        label = (f"{item[metric]:.2f}%（{item['correct']}/180）" if accuracy
                 else f"CE = {item[metric]:.4f}")
        rawax.text(left+.118, .64, arm, transform=rawax.transAxes, ha='center',
                   va='center', fontsize=11, color=COLORS[arm], weight='bold')
        rawax.text(left+.118, .20, label, transform=rawax.transAxes, ha='center',
                   va='center', fontsize=12, color='#1D2732')
    ce_max = max(v[metric] for s in payload['scenes'].values() for points in s['arms'].values() for v in points)
    ymax = 100 if accuracy else max(.1, ce_max*1.17)
    for j, scene in enumerate(SCENES):
        ax = fig.add_subplot(grid[1, j])
        if j == 2:
            ax.set_facecolor('#FFF5F4')
        info = payload['scenes'][scene]
        n = info['L_physical_IDs']
        for i, arm in enumerate(ARMS):
            points = [r[metric] for r in info['arms'][arm]]
            width = 2.4 if arm == 'RAW_AS' else 1.8
            # L old/new observations share physical IDs. V uses different IDs,
            # so its connector is explicitly descriptive and dashed.
            ax.plot([0, 1], points[:2], color=COLORS[arm], linewidth=width,
                    marker=MARKERS[arm], markersize=6, zorder=i+2)
            ax.plot([1, 2], points[1:], color=COLORS[arm], linewidth=width,
                    linestyle=(0, (3, 3)), marker=MARKERS[arm], markersize=6, zorder=i+2)
        ax.set_title(TITLES[j]+'\n'+scene, fontsize=12.5,
                     weight='bold' if j == 2 else 'normal', pad=14)
        ax.set_xticks([0, 1, 2], [f'训练流重建 As\n{n} 个 L ID',
            f'同 L 新 As\n{n} ID × 2 draw', '源 V 新 As\n1800 ID × 2 draw'])
        ax.tick_params(axis='x', labelsize=10, pad=8)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_xlim(-.18, 2.18); ax.set_ylim(0, ymax)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis='y', color='#D8DDE2', alpha=.75, linewidth=.7)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines['left'].set_color('#C4CBD2'); ax.spines['bottom'].set_color('#C4CBD2')
        if j == 0:
            ax.set_ylabel('准确率（%）' if accuracy else '标准交叉熵 CE（越低越好）', fontsize=12)
    handles = [Line2D([0], [0], color=COLORS[a], marker=MARKERS[a], linewidth=2,
                       markersize=6, label=LABELS[a]) for a in ARMS]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.52, .155),
               ncol=4, frameon=False, fontsize=11, columnspacing=1.8)
    notes = [
        'n = 1 个模型种子。两次信道 draw 不是独立模型种子；本图没有跨种子误差条。',
        '实线连接相同 L ID 的旧／新观察；虚线更换为 V ID，仅作描述性比较。旧／新实现的损伤难度组成可能不同。',
        '重建沿用既定训练命名随机流；历史 As IQ 未保存，无历史字节对照。B0 与 RAW_NOAS 从未接受 As 训练。',
        'CE 单独作图：默认 logits 的标准真标签交叉熵，不加温、不作标签平滑；与训练 smoothing = 0.01 的损失区分。']
    for i, line in enumerate(notes):
        fig.text(.055, .124-i*.027, line, fontsize=9.8, color='#4A525C')
    png, pdf = Path(str(prefix)+'_'+kind+'.png'), Path(str(prefix)+'_'+kind+'.pdf')
    fig.savefig(png, dpi=180, facecolor='white')
    fig.savefig(pdf, facecolor='white')
    plt.close(fig)
    return [str(png), str(pdf)]


def main(input_path, output_prefix):
    source, prefix = Path(input_path), Path(output_prefix)
    payload = prepare(json.loads(source.read_text(encoding='utf-8-sig')))
    names = [Path(str(prefix)+suffix) for suffix in (
        '_accuracy.png', '_accuracy.pdf', '_CE.png', '_CE.pdf', '_metadata.json')]
    if any(p.exists() for p in names):
        raise FileExistsError('Existing figure prefix; choose a new prefix without overwriting artifacts')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    font = chinese_font(matplotlib)
    matplotlib.rcParams.update({'font.size': 11, 'axes.unicode_minus': False,
                                'pdf.fonttype': 42, 'savefig.bbox': 'tight'})
    import matplotlib.pyplot as plt
    outputs = figure(payload, 'accuracy_percent', prefix, plt)+figure(payload, 'standard_CE', prefix, plt)
    metadata = dict(schema='pure_F_rawAs20_source_fit_plot_v1', source_run=RUN,
        input=str(source.resolve()), figures=outputs, font=font, model_access=False,
        model_forward_packets=0, training_updates=0, IQ_access=False,
        checkpoint_access=False, formal_data_access=False, computations='JSON metric validation; '
        'accuracy ×100; preserve CE and source counts; no fitted quantities',
        historical_IQ_byte_comparison=False, plotted=payload)
    names[-1].write_text(json.dumps(metadata, ensure_ascii=False, indent=2,
                                   allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps(dict(status='FIGURES_WRITTEN', files=outputs,
                          metadata=str(names[-1]), model_forward_packets=0), ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True); p.add_argument('--output-prefix', required=True)
    a = p.parse_args()
    main(a.input, a.output_prefix)
