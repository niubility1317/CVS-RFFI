"""Create a portable statistical replay and preserve historical cost scope."""
import csv,gzip,json,re,shutil,sys
from pathlib import Path
ROOT=Path('E:/type10-7');P=ROOT/'local_artifacts/pamf_repl_completion_20261024';OUT=ROOT/'exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01';RUN='20261024-phase1-PAMF-REPL-completion-m201-r01'
EXT=ROOT/'local_artifacts/external_practical_residual_2x2_design_20261023/full_analysis_20261024'
def clean(x):
    if isinstance(x,str):
        x=re.sub(r'/home/[^/]+/\d+/CV-SincNet','PROJECT',x)
        x=re.sub(r'/home/[^/]+','N607_HOME',x)
        return x.replace('E:/type10-7','PROJECT').replace('E:\\type10-7','PROJECT')
    if isinstance(x,list):return [clean(v) for v in x]
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    return x
def rows(path):
    with (gzip.open if path.suffix=='.gz' else open)(path,'rt',encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)
def csvwrite(path,data):
    it=iter(data);first=next(it);path.parent.mkdir(parents=True,exist_ok=True)
    with (gzip.open if path.suffix=='.gz' else open)(path,'wt',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(first));w.writeheader();w.writerow(clean(first));w.writerows(clean(r) for r in it)
def portable(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if src.name.endswith('.jsonl.gz'):
        with gzip.open(src,'rt',encoding='utf-8') as f,gzip.open(dst,'wt',encoding='utf-8') as g:
            for line in f:g.write(json.dumps(clean(json.loads(line)),ensure_ascii=False,allow_nan=False)+'\n')
    elif src.suffix=='.json':dst.write_text(json.dumps(clean(json.loads(src.read_text(encoding='utf-8-sig'))),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    elif '.csv' in src.name:csvwrite(dst,rows(src))
    else:shutil.copy2(src,dst)
def prepare():
    frozen=OUT/'frozen_statistics';(frozen/'analysis').mkdir(parents=True,exist_ok=True)
    for src in (P/'final_analysis_20261025').iterdir():
        if src.is_file():portable(src,frozen/'analysis'/src.name)
    for src in (P/'complete_20261025/inputs').iterdir():
        if src.is_file() and src.name!='retrieval.json':portable(src,frozen/'inputs'/src.name)
    external=frozen/'external';external.mkdir(exist_ok=True)
    csvwrite(external/'per_seed_overall.csv.gz',(r for r in rows(EXT/'per_seed.csv.gz') if r['family']!='pamf' and r['dimension']=='overall'))
    csvwrite(external/'summary_overall.csv',(r for r in rows(EXT/'summary_overall.csv') if r['family']!='pamf'))
    cost=OUT/'source_data/costs';cost.mkdir(parents=True,exist_ok=True)
    for name in ['F_cost.csv','F_stage_cost.csv','restored_cache_cost.csv','augmentation_producer_cost.csv']:
        src=EXT/name
        if src.exists():csvwrite(cost/('historical_external_'+name),(r for r in rows(src) if r.get('family')!='pamf'))
    csvwrite(cost/'historical_matched_G_cost.csv',(dict(r,new_in_previous_external_run=r['new_G'],new_this_REPL_run=False) for r in rows(EXT/'G_cost.csv')))
    lineage=ROOT/'automation_reports/CV-SincNet/20261021-phase1-T1-DURA-G-lineage-m35-r01/G_lineage_stage_cost.csv'
    extra=[r for r in rows(lineage) if r['row_id'] in ('m3-natural-G-LU-s2026092702','m3-natural-G-LU-s2026092703')]
    if len(extra)!=16:raise ValueError(('Two extra m3 G ancestors/eight stages each',len(extra)))
    csvwrite(cost/'historical_extra_m3_G_stages.csv',(dict(r,new_this_REPL_run=False) for r in extra))
    gab=ROOT/'automation_reports/CV-SincNet/20261021-phase1-T1-DURA-G-ablation-m21-r01/G_ablation_training_cost.csv'
    csvwrite(cost/'historical_G_component_students.csv',(dict(r,new_this_REPL_run=False,shared_R65_ancestry_cost='shared historical teacher; do not multiply per component') for r in rows(gab)))
    for src in (P/'final_analysis_20261025').glob('*.csv'):
        portable(src,OUT/'source_data'/src.name)
    for name in ['replay313_20261025/comparison_verification.json','replay310_20261025/reproduction_verification.json','supplement_crossruntime_comparison.json']:
        portable(P/name,OUT/'source_data'/('Python310_replay.json' if name.startswith('replay310') else 'Python313_comparison.json' if name.startswith('replay313') else 'supplement_crossruntime_comparison.json'))
def assets():
    modules=['experiments/pamf_repl_completion/__init__.py','experiments/pamf_repl_completion/plan.py','experiments/pamf_repl_completion/analyze.py','experiments/pamf_repl_completion/results.py','experiments/pamf_repl_completion/reproduce.py','experiments/cric_dura_comparison/__init__.py','experiments/cric_dura_comparison/io.py','experiments/cric_dura_comparison/analyze_main.py']
    (OUT/'experiments').mkdir(exist_ok=True);(OUT/'experiments/__init__.py').write_text('',encoding='utf-8')
    for name in modules:
        dst=OUT/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst)
    # Keep all actual plotting tools and selected method definitions available.
    for name in ['tools/figures/build_repl_paper_kit.py','tools/figures/assemble_repl_paper_kit.py','tools/figures/prepare_repl_delivery.py','docs/PAMF_REPL.md','docs/PAMF_REPL_COMPLETE_EXPERIMENTS_20261024.md','configs/pamf_ssl_repl.json']:
        dst=OUT/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst)
    readonly='''# PAMF-REPL 完整论文交付（2026-10-25）

先看 `IEEE_experiments/main.pdf`（仅 IEEE 实验部分）、`论文结果与逐图说明.pdf` 和 `论文结果与边界.md`。27 张独立图的 PDF/SVG 保留可编辑文字，PNG/TIFF 为 4800 dpi；图中无标题、脚注。`tables/` 是每张图的来源表，`source_data/` 是完整汇总和去重成本。

201 逻辑 F = 189 新 F + 12 精确复用；84 唯一 E20 前段与 189 完成阶段不是 273 个模型。228 全冻结预测 / 228 独立 CPU 评分，4 个独占计时面板，733 有效阶段全部完成。DURA 新训练 0、新恢复 IQ 缓存 0。旧 T1 数据没有改名。

## 纯 Python 标准库复算

在解包根目录使用 Python 3.10 或更新版本：

```text
python -m experiments.pamf_repl_completion.reproduce --root frozen_statistics --output replay --external-seed frozen_statistics/external/per_seed_overall.csv.gz --external-summary frozen_statistics/external/summary_overall.csv --compare
```

`replay` 必须是尚不存在的新目录。复算 228 行、379239 份 CM、247863 条净纠错汇总、seed 内六景平均 / 三 seed SD、8 格交互、m3 交叉、去重 F 成本、实际 PL 计数、源诊断和计时样本。`--compare` 按完整键比较 9 张统计表；数值容差 1e-10，CSV 字节序不作一致承诺。复算器不导入 Torch，不读取 IQ、模型权重、逐样本目标真值或网络服务。

`frozen_statistics/external/` 只含此前已经核验的四外部 Full 全 A/B/C/D 汇总及 seed 内总体指标。新的 REPL-minus-外部配对由这些固定输入复算；本包不声称再次重算旧外部全部 CM。外部完整 CM 原包继续保留于 `20261024-practical-residual-full2x2-complete-r14.zip`。

## 统计与使用边界

S6 是同一 seed 内六种 practical 指标的算术均值；然后对三个实际 F seed 计算均值与样本 SD。通常每契约 G 固定，SD 不是三次完整系统重复、置信区间或显著性。m3 九格中只有三个对角 F/G 对是三组完整训练种子。Most source curves use six fixed targets; F0 不作为 m5 插入。

所有新 PAMF 读取 raw/G-L/G-LU。完全 L-only 仅 Sup F + G-L；PL/REPL F 经 F 使用 U。四外部主要列固定 Full D（channel on + ordinary PL on），A/C 配 G-L，B/D 配 G-LU；不是挑选测试最优格。

F0 REPL+G-LU S6 跨 RX 为 78.2863±0.9701%；固定同一 G-LU 后相对普通 PL 为 -0.2298±0.9544 个百分点，保留负结果。REPL 曾按公开 F0 clean 选择，披露选择后比较背景，不宣称盲选。D0 是更早未见日。ManyRx 为十类，其余为六类。

新增 F 成本为 8670 唯一模型轮、1924740 更新、265.956426 共享 worker 小时；不是独占 GPU 小时或墙钟耗时。G/缓存祖先成本在 `source_data/costs/` 单列；原 F0 G-LU 部分祖先计时缺失，保留缺失，不补零。历史 external 的 new_G 字段指此前外部 run，所有 G 在本 REPL run 均为复用。

F eager 与 G+F 优化图计时分列；五 round 是计时重复，不是五训练 seed。共驻留显存不当作单模型 VRAM，缓存分类耗时不当作在线 G+F。

## 科学执行与保留

模型执行 commit dacafc8bfd7d3ef16e6f648cdc4491b65133ae8f；控制/CPU 评分恢复 commit a79bdf385489cb3822b37434a07e3c43e5ce839a。两次旧 CPU 评分失败保留排除，由两新输出复用原预测完成评分，历史 735 / 有效 733；不新增模型。

本新包不覆盖 r14/r15。旧错误比较本地 2046 项已经归档验证后移除；隔离包 `F:/AI/exports/STRA-RFFI/quarantine/20261024-obsolete-comparisons-r01/obsolete_local_comparisons.zip` 保留，本次没有重复清理。

论文图表可直接复用。重新绘图需 matplotlib/Pillow/Times New Roman；`build_repl_paper_kit.py --source frozen_statistics/analysis --output <new-dir> --raster`。重新编译 IEEE 实验源用 pdflatex、bibtex、pdflatex 两遍。统计复算无需这些依赖。
'''
    (OUT/'README.md').write_text(readonly,encoding='utf-8')
    print(json.dumps(dict(status='PORTABLE_INPUTS_READY',modules=len(modules),extra_G_stages=16)),flush=True)
if __name__=='__main__':
    if '--assets-only' not in sys.argv:prepare()
    assets()
