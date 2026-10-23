"""Package the verified PAMF report, source summaries and publication figures."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import zipfile
from pypdf import PdfReader
from verify_completed_pamf import verify


def package(root,project):
    if (root/'delivery_manifest.json').exists():raise FileExistsError('Preserve completed delivery')
    source=project/'local_artifacts/final_delivery_20261023'
    shutil.copyfile(source/'complete_inputs/manifest.json',root/'source_data/collection_manifest.json')
    shutil.copyfile(source/'fetch_verification.json',root/'source_data/collection_verification.json')
    code=root/'reproduce';code.mkdir(exist_ok=True)
    for name in ('collect_completed_pamf.py','analyze_completed_pamf.py','plot_completed_pamf.py',
                 'report_completed_pamf.py','verify_completed_pamf.py','package_completed_pamf.py',
                 'paper_publication_style.py','plot_pamf_paper_figures.py','pamf_experiment_names.py',
                 'explain_completed_pamf.py'):
        shutil.copyfile(Path(__file__).with_name(name),code/name)
    (root/'README_zh.md').write_text('''# PAMF Phase1 已完成实验汇总

数据截至 2026 年 10 月 23 日。当前方法为 T1-E50（13L＋37U）＋DURA v1.0，不使用旧输入绑定错误结果，也不把历史 PAMF 配方改标为当前版本。

## 交付内容

- `PAMF_结果总结_20261023.pdf`：中文结果总结，主文突出三个综合模块的四行消融，21 项细诊断移到补充部分；科研图保留原始矢量。
- `PAMF_结果总结_20261023.md`：可编辑的同内容文本。
- `PAMF_逐图说明_20261023.pdf` 及 `.md`：每张图的研究问题、坐标／图例、结果和边界，并解释源数量图为何没有五源点。
- `main_figures.pdf`、`supplementary_figures.pdf`：主图和补充图分开，保留原 Fig. 编号用于追溯。
- `all_figures.pdf`：40 页矢量图合集。
- `实验命名说明.md`、`experiment_names.csv`：语义命名与原始 ID 对照；原 run、权重和评分目录不改名。
- `pdf/`、`svg/`：40 张独立矢量图，Times New Roman，SVG 文字可编辑。
- `png/`、`tiff/`：40 张 4800 dpi 位图，183 mm 宽对应 34,582 像素；TIFF 使用 LZW。
- `captions_en.md`：英文图注、统计定义与数据口径，独立于图片；图内无说明性副标题或图下说明。
- `source_data/`、`analysis/`：冻结汇总、逐种子聚合、配对效应、结论数据与来源记录。
- `reproduce/`：仅做读取、统计、制图和导出的 Python 源码。

40 张图均为单一数据图。恢复与训练对照同时展示两端实际准确率，增幅仅作辅助；细诊断改为实际准确率及完整识别器参照。资源图从零开始，最优在上、柱长为实际值／最优值×100%，柱右标实际数值。

主模块消融为图 13：完整识别器、去跨接收机身份对比、去成对预测一致性、去整组选择性自训练。图 12 的八组合及图 14/15/16/40 的细项为补充。源数量图 7 至 10 明示五源未测试；不能用目标集合不同的五源主实验拼入，也不插值造点。

## 主要结论

F0 当前 PAMF clean 为 84.4839 ± 0.9660%，六景均值为 77.6468 ± 0.8461%，相对 raw 的配对增益为 7.0455 ± 0.5949 个百分点。DURA-LU 在 17/17 契约提高原生 T1 六景均值，但 5/17 契约的 clean 均值变化非正。

同样接入 DURA 后，F0 外部最佳 Full/Full＋PL 为 SDG Full＋PL 的 66.1804 ± 0.2085%，与 PAMF 的均值差为 11.4664 个百分点。这是完整配方比较；不把差值归到单一组件。核心 B 在完整组合下有较一致收益，A/CRIC、C/原生选择性半监督和正协同没有稳定的普遍证据。严格 CE 结构支持 PA/signature/参考分支的作用，但不证明每个时间坐标、三视图或精细参考设计均不可替代。

low urban 仍低于 70% 目标，mid urban 距 raw clean 仍约 10 个百分点。参数总数 499,944；统一 r02 的 B1 GPU-wall 中位数为 4.541187 ms。外部统一计时为无 G 的 Full 单识别器，PAMF 为完整 G＋F，不能据此宣称等恢复条件速度领先。

## 统计与数据单位

六景先在每个 F seed 内平均，再计算三 seed 样本标准差；不平均场景 SD 来伪造六景 SD。除明确注明的配对 G 学生和 m3 三组对角完整系统外，三 F 的变化均条件于固定 G，不是三个独立完整系统。标准差不是置信区间或显著性检验。

ManyRx 为其原登记 10 类，其余契约 6 类；F0 每景 168000 包/七目标 RX，源数量曲线为每景 144000 包/六目标 RX，不把 F0 插入 m1/2/3/4/6 曲线。不同契约不计算混合总体平均。

`main/core/fine/physics/gfactor/external_full` 的准确率、Macro-F1、均值和 SD 均为百分数（0 至 100）。保留的上游 `external.csv` 中 `accuracy_*`/`macro_f1_*` 为 0 至 1 的比例，`external_effects.csv` 的净纠错也为比例；本包归一化的 `external_full.csv` 及配对效果则使用百分数/百分点。源干预的 flip fraction 为 0 至 1，图 21 乘 100 展示。几何距离按图轴说明展示，不冒充准确率。

未登记的外部 PL 组合用 N/A，不补零。去参考结构没有记录的 reference descriptor 为缺失项，见 `physics_source_missing.csv`，不伪造为 0。旧失败/替代/停止和 deferred 训练不补跑，不与有效结果混合。

三组时延按独立 cohort 分列；细 G 的 native/eager 版本和 graph＋reuse 默认路径不混成同一路径速度收益。参数量不等于 FLOPs、序列化文件体积或隔离显存。共享 worker elapsed 不是独占 GPU 小时；复用成本不能跨面板重复相加。

## 复现

依赖 Python 3.10 或以上、matplotlib、numpy、Pillow、pypdf；中文 PDF 另需 reportlab。Windows 需已安装 Times New Roman 和宋体。使用所附冻结 snapshot 即可重绘，不读取 IQ、权重、逐样本目标真值，也不执行训练或预测。

1. 将 `source_data/` 复制到一个新目录 `NEW/source_data/`。
2. `python reproduce/analyze_completed_pamf.py --source NEW/source_data/snapshot.json --output NEW/analysis`
3. `python reproduce/plot_completed_pamf.py --source NEW/source_data/snapshot.json --output NEW --dpi 4800`
4. `python reproduce/report_completed_pamf.py --root NEW`
5. `python reproduce/explain_completed_pamf.py --root NEW`
6. `python reproduce/verify_completed_pamf.py --root NEW`

`collect_completed_pamf.py` 仅用于在原项目中从既有已完成评分重新汇集；普通重绘不需要原项目或服务器。完整 RX/TX/day 分层与大混淆矩阵输入保留在原实验产物，本包提供图表使用的 overall 与逐 seed 汇总，不复制 IQ、权重、运行日志或私有控制器状态。
''',encoding='utf-8')
    result=verify(root)
    (root/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    outputs={}
    for kind in ('vector','png','tiff'):
        archive=root.with_suffix('.zip') if kind=='vector' else root.with_name(root.name+'-'+kind.upper()+'-4800dpi.zip')
        if archive.exists():raise FileExistsError('Preserve previous archive: '+str(archive))
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED if kind=='vector' else zipfile.ZIP_STORED,
                             compresslevel=6 if kind=='vector' else None) as z:
            for path in sorted(root.rglob('*')):
                if not path.is_file():continue
                relative=path.relative_to(root);top=relative.parts[0]
                if (kind=='vector' and top not in ('png','tiff','draft_figures.json')) or top==kind:
                    z.write(path,relative)
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:raise ValueError('Archive readback failed')
        outputs[kind]=dict(file=archive.name,bytes=archive.stat().st_size)
    verify(root,archives=True)
    delivery=dict(status='VERIFIED',as_of_hkt='2026-10-23',
        report_pages=len(PdfReader(root/'PAMF_结果总结_20261023.pdf').pages),
        guide_pages=len(PdfReader(root/'PAMF_逐图说明_20261023.pdf').pages),figures=40,dpi=4800,
        archives=outputs,negative_results_retained=True,new_training_or_inference=False,
        vector_report=True,archive_readback_verified=True)
    (root/'delivery_manifest.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(delivery,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    p.add_argument('--project',required=True,type=Path);a=p.parse_args();package(a.root,a.project)
