# SDG 作者 WiSig 跨天实验预登记

run_id：`20261019-phase1-sdg-author-wisig-s2023-r01`。完成第二篇真正的 RFFI 域泛化作者代码复现，不把 ASKNet/WaveMLP 结构基线当作新增 DG 方法。当前未获得论文全文，因此工程参照固定为作者公开日志；不据此宣称论文表已核准。

作者代码固定 `0090f4ba1d5cf76a4200a5be4830a9b08eb39699` 的 `2TWC_Version`；本地实现/科学配置为 `paper_reproduction/sdg_20261019/`。唯一行 `sdg-crossday-s2023`，不加载任何作者或历史 checkpoint。S MACNN 129894 参数，原6类头与骨干联合训练；ROA、监督类别对齐及监督域对抗对比损失调用未经修改作者函数。

数据为作者公开 Google Drive `Dataset_WiSig` 的4个pickle，每TX/day1000包、RX1-1、256点。TX顺序14-10、14-7、20-15、20-19、6-15、8-20。作者处理 IQ 的均衡状态未说明；直接使用作者文件，不改为本项目其他表示。源day1，先10%源holdout，再从剩余90%取30%验证：train3780/val1620/holdout600。目标day2/3/4各6000。model/split seed2023，data/augmentation无额外独立seed（主全局seed控制），support/K不适用，evaluation seed20261019只置换导出顺序。

固定200轮，Adam lr0.001、weight_decay0、batch32，验证/测试batch16，无学习率调度、无早停；主增强深度2、辅助深度1，类别/域对比权重1/100，源验证NLL最低选模。保留原训练前torchsummary forward对RNG及BN的影响。真实source小批checkpoint烟测为launcher首步，PASS后丢弃烟测权重，立即从零正式训练。

训练阶段只打开day1，禁止训练/选模使用其他日期。完成200轮并冻结源NLL最佳模型后，预处理器再准备匿名query IQ/ID和独立truth-side文件；预测器不接触truth，不拟合参数，不使用配额/批次真标签。全部18600条prediction固定后，独立scorer输出day1源holdout和day2/3/4各域准确率、Macro-F1、混淆矩阵及18000条目标汇总。

目标day2/3/4作者日志为99.2167%、83.2667%、97.2500%，均值93.2444%；源holdout99.5%。目标均值达到92.2444%是提前固定的工程参照下限（允许低1个百分点）；完整披露各域差值，不因分数修改超参数、seed或选择性重跑。本任务是全监督原数据任务，不是CVS弱标签、跨接收机或星地部署实验。

唯一owner=root-sdg-20261019；普通N607账户，首选GPU1单进程，每GPU总实验进程≤3，不修改其他任务。环境为已存在SDG-SEI Py3.8/torch1.11，独占release/deps包含NumPy1.23.5和torchsummary1.5.1（保留旧ragged permutation，不改共享环境）。CWD为独占release；精确路径、GPU、Gitcommit、环境和argv由本run的experiment.json引用私有runtime.json。固定命令模板：`SDG-SEI/python -u RELEASE/pipeline.py launch --runtime RELEASE/runtime.json`。

只允许本run技术错误停止：错误RX/day/标签映射/源契约，输出碰撞，无法执行，非有限源loss，无法形成合法prediction或错误truth连接。低目标分数不是技术失败。保留所有partial输出，恢复先核对原run再决定新run，不停止健康其他任务。

发布前完成聚焦源划分/原函数/契约和truth隔离检查、一次独立P0/P1审查、代码commit、一次N607 preflight、一次archive本地/远端SHA比较和远端编译；发布后核对PID/CWD/命令/GPU和日志增长。预期产物为smoke/result.json、train/best.pt/last.pt/epochs.jsonl/resolved_config.json/training_complete.json、predictions.npz、predictions_complete.json、scores.json、completion.json。真实文件存在并核查后才报告对应状态，不以exit0代替结果证据。
