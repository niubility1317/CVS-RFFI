# MoPC 仓库版 clean 原生注册 pilot

用户要求在其他显卡运行 MoPC。先进行三池、各一套 MIX 条件的单模型种子 pilot：P01/MIX01/1-1、P02/MIX05/20-1、P03/MIX01/7-7，seed 2026092702，K5/N20，使用原生 MoPC A→C、作者 attention ResNet 已冻结源 checkpoint 和 public_code 注册语义；不运行 B，不启用梯度裁剪。每行 raw clean support/raw clean query，同一物理记录和既定 split。原 satellite 预测不复用为 clean；旧非有限隔离保留，不重发。

启动器在 `20261114-phase2-mopc-public-clean-support-m3-r02` 三个 source/support 诊断全部完成且校验 capsule、权重、seed、K/N、语义和预算后启动；固定 pilot 身份使用原子目录保留唯一 owner。GPU 5–7，每卡最多 2 个本 run 预测进程、每次启动前空余至少 6000 MiB，避让其他 owner。每个 worker 真实源 checkpoint no-query smoke，异常停止新增派发、排空并保留结果。既定完整比较预测屏障尚未满足；不启动 truth-last 评分，不把 pilot 完成称为全部 MoPC 数值修复或完整比较。

代码版本为本报告所属提交；入口 `python -m experiments.pamf_phase2_practical.clean_mopc_pilot`，runtime 来自 `preparations/20261114-phase2-clean-non-issl-full-m4-r02/runtime/native.json`，source 来自 `runs/20261101-phase2-mopc-author-group1-m4-r04`；其余精确路径由 pilot 准备目录保存。输出 `runs/20261114-phase2-mopc-public-clean-pilot-m1-r01` 不覆盖，正式完整矩阵待合法全局评分与数值处理。
