"""Render source-only selected-direction and bounded-transform diagnostics."""


def lines(rows):
    result=['', '## 选中方向与实际统计界', '',
        '|方案／种子|clean 次数|正向次数|反向次数|总搜索次数|',
        '|---|---:|---:|---:|---:|']
    for row in rows:
        stats=row['style_search_summary']
        counts=[stats[k]['positive'] for k in ('identity_selected','positive_selected','negative_selected')]
        total=stats['identity_selected']['observations']
        if sum(counts)!=total:raise ValueError('Selected directions do not partition source searches')
        result.append('|'+row['row_id']+'|'+'|'.join(str(n) for n in (*counts,total))+'|')
    result+=['', '|方案／种子|层|均值位移最大值／自身 σ|最小／最大缩放|均值差截断率均值|log-scale 截断率均值|',
        '|---|---|---:|---:|---:|---:|']
    for row in rows:
        for level,entry in row['L_style'].items():
            m=entry['metrics']
            result.append(f"|{row['row_id']}|{level}|{m['mean_shift_in_sigma_max']['max']:.6g}|{m['scale_ratio_min']['min']:.6g}／{m['scale_ratio_max']['max']:.6g}|{m['mean_clipped_fraction']['mean']:.4%}|{m['log_scale_clipped_fraction']['mean']:.4%}|")
    result+=['', '统计界来自选中候选的训练重放，均值位移按该层自身 σ 计量。截断率为每次变换记录的通道比例再按调用平均，不是不同物理样本比例。clean 选择不安装 hook，不进入层统计分母。所有候选的方向、CE、选择及所选变换原始统计均保留；这些界不证明原始 IQ 的物理不变性或测试收益。']
    return result
