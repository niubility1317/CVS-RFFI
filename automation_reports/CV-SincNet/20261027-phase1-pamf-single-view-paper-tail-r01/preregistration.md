# 单视图论文复现余量接续

用户当前要求“PAMF 单视图论文复现，剩余任务补到最后完成”。科学矩阵仍为原 run `20261027-phase1-pamf-single-view-paper-m120-r01` 的120模型单元；不是新增120次训练。唯一结构化矩阵见 experiment.json，方法、数据权限、普通PL、默认联合头、种子及E50/11100预算均保持。

新执行 run 为 `20261027-phase1-pamf-single-view-paper-tail-r01`。等待 `20261027-phase1-pamf-repl-shuffled-views-m16-r02/analysis/complete.json` 确认 `ANALYZED/models=16` 后，才执行真实checkpoint无query smoke并立即进入剩余队列。等待期间0 GPU、0数据／预测／truth访问。前序健康控制器和worker不修改。

剩余43模型、84预测行、84独立评分行。65个已完成新模型、12个原始匹配复用模型及36行已完成预测／评分保持原路径。6个暂停模型复用11段完整保存的本行E20/E30；另外37行从零开始（35行原未启动、2行暂停时尚无完整阶段权重）。未保存的阶段内内存不冒称逐步恢复；旧日志及所有partial产物保留，新增阶段在独立目录执行。恢复核对本行seed、完整源配方、来源契约、连续父权重和原生阶段预算，不读取目标指标作选择。

不可覆盖根目录为 `PROJECT/runs/20261027-phase1-pamf-single-view-paper-tail-r01`：等待状态 `queue_state.json`，实际队列位于 `pipeline/`，新模型／预测／评分／分析也位于该子树。科学run_id保持原矩阵身份，continuation_run_id记录新执行身份。一个launch owner：`codex-pamf-single-view-tail-20261027`。

N607普通账户、既有Python/Torch、FP32/TF32off、每worker CPU2线程；CWD `PROJECT/releases/20261027-phase1-pamf-single-view-paper-tail-r01-release`。命令 `PYTHON -u -m experiments.pamf_single_view_paper.tail dispatch --config RELEASE/runtime.json`。代码身份记录在runtime及launch_evidence；归档一次传输比较、远端编译一次。GPU0–7每卡本run至多1条、全owner至多2条、free≥12000MiB，CPU scorer至多1条；不抢占其他健康任务。

每row冻结final后按原增强／无增强范围完成三群体、raw/DURA-L/DURA-LU及全部声明视图，prediction全部固定后独立CPU truth-last评分，最后汇总完整三种子、逐景/TX/RX/day、Macro-F1及成本。新G及新IQ缓存均0。首次技术／输入权限／非有限／输出冲突等原系统失败条件停止新派发，保留健康worker和产物；不因低分取消、重排或选择性重跑。

原6项协议检查和新增4项接续检查全部PASS；一次独立P0/P1审查PASS。检查覆盖前序两组屏障、禁止外来／改变配方的阶段继承、已完成任务不重复，以及预测和评分依赖迁移。新真实checkpoint smoke在前序完成后作为launcher第一步执行；当前不将本地检查冒称GPU smoke或训练启动。
