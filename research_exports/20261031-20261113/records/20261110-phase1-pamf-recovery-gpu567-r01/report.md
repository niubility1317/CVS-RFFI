# PAMF 两组实验故障恢复与 GPU 5–7 调度

## 原因与恢复边界

- 新结构 226 逻辑模型：前一队列记录 401 个阶段完成、214 个排队、12 个依赖阻塞、6 个技术失败。六个失败均为 E200 训练 200 轮、10,000 步和 `last.pt` 已完成后，资源统计深复制模型时钩子错误。逐行核对源合同、初始化、checkpoint 身份及完整训练记录，在新输出目录恢复完成标记和资源统计，不重训、不改 seed 或选择规则。
- 旧结构 190 逻辑模型：552 个阶段完成，汇总读取 E200 clean 评分的 `count` 时遇到真实字段 `query_count`。保留原评分，校验字段一致性后在独立输出根重新汇总。
- 原 run、模型权重、日志和结果不覆盖；新组已完成合法阶段引用原产物，其余阶段的输出、配置和依赖写入独立新 run。新训练记录恢复提交；已训练权重保留原执行提交并另记恢复提交。

## 固定矩阵与资源

沿用原模型、数据合同、源端固定选模、种子和预算；原 226/190 两组逻辑矩阵及 DURA-L、DURA-LU 的既定读出不变。正式评分按已登记的 clean 和六个 practical 场景，历史 E200 网络只做既定 clean 对照；预测冻结后独立 truth-last 评分，测试结果不回流选模。

只向物理 GPU 5、6、7 派发，GPU 5、6 每卡本队列最多 3 个，GPU 7 最多 1 个；结合显存和主机资源守卫，保留其他 owner。启动先在 5–7 中选择容量足够的 GPU 执行真实源 checkpoint、无 query smoke。源和预测输入依赖仍按原合同，评分不读取训练数据或模型。

## 发布与停止规则

代码和配置采用本报告所在 Git 提交；本地运行定向测试并推送，[计算环境] 单次 preflight，单个 release archive 校验一次、远端编译。命令：`python -m experiments.pamf_split_spectrum_paper.publish_recover_567 --repo <workspace> --output [已省略路径]`。

控制器 `experiments.pamf_split_spectrum_paper.recover_567` 在新 release CWD 执行；新 run 为 `20261110-phase1-pamf-recovery-gpu567-r01`，旧组汇总输出为 `20261110-phase1-pamf-recovery-gpu567-r01-old-analysis`。已完成阶段引用旧绝对路径，待完成阶段在新 run 的 `F/`、`readouts/`、`logs/` 和 `configs/` 中产生，`sources.json` 记录逐行来源。预期产物包括 smoke、各行预测和评分、独立汇总 CSV 与报告、调度状态。
