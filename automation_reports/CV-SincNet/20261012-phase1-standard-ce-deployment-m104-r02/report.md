# 十三模型八种子：端到端推理时延、内存显存与FP32模型存储实测

- run_id：`20261012-phase1-standard-ce-deployment-m104-r02`
- group_id：`phase1-standard-ce-deployment-cost`；类别：`diagnostic`；阶段：`Phase1 deployment diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Frozen E200 real checkpoints; synthetic CPU IQ to CPU labels, no dataset/truth/prediction scoring. GPU0 sequential fixed matrix.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 修复与固定范围

r01 在远端编译依赖导入时失败，尚未启动模型测量，release/archive 保留。本版以标准库读取 Linux /proc/self/statm 采样 RSS，避免安装软件；矩阵、权重、种子、时延与显存口径不变。原候选唯一 P0/P1 审查及换行转义定点复核通过，修复依赖后复用该检查范围；launcher 首步仍执行真实 checkpoint no-query smoke。

## 完整实测与 Excel 交付

VERIFIED。104/104 既有 E200 权重完成测量，13 种模型各 8 个种子，B1/B128 共 208 个测量组。每个权重独立进程，GPU0 串行；启动前和测量中未发现其他 GPU0 进程。实际执行 Git 提交为 `45c734e147fe0828bbdbed2975e4bb0deef2e31d`；真实 checkpoint no-query smoke、来源契约核对、单份 release archive SHA 比对和远端编译通过。r01 失败产物保留，其原因是缺少可选依赖，测量未启动；r02 改用 Linux 原生 RSS 采样，不安装软件。

硬件为 RTX3090，CPU Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz，驱动 580.173.02，Linux 6.8.0-142-generic，PyTorch 2.1.0+cu121 / CUDA 12.1。固定 FP32，TF32 关闭，CPU 2 线程。

每个权重预热 30 次；B1 测 300 个请求，B128 测 100 个请求；另各测 50 次推理内存。共 41600 次计时请求。网络 P50/P95/P99 使用 8 个权重的等量请求合并（B1 2400、B128 800），线性插值计算；跨种子 SD 则为 8 个权重平均时延的样本 SD，二者不混用。

端到端范围为 CPU 中已同步/归一化合成 IQ 的缓冲构造、H2D、整个模型前向（含 FFT/reshape/tokenization）、有限值检查、argmax、回读 CPU 和标签列表构造。排除射频采集、同步前端、磁盘、网络、凑批与排队；因此是软件推理链路端到端，不是接收机或卫星系统全链路。没有读取真实数据、测试 IQ 或 truth，没有重测准确率。

GPU 列分别为 PyTorch allocated/reserved 推理峰值，在预热后清理缓存、重置统计后测量；不含全部驱动上下文，不能作为 NVIDIA 进程总显存。CPU 列为单独推理窗口中每 1 ms 采样的总进程 RSS 峰值，包含解释器、框架和初始化后运行时，可能漏掉更窄峰值。汇总采用 8 个权重的最大内存值。模型存储为实际 torch.save(FP32 state_dict) 序列化字节（含 buffer/格式开销），不含优化器、训练记录或模型代码；未声称完整部署软件包。

| 模型 | B1 均值（ms） | B1 P95（ms） | B1 P99（ms） | B1 GPU alloc峰值（MiB） | B128 GPU alloc峰值（MiB） | FP32存储（MiB） |
|---|---:|---:|---:|---:|---:|---:|
| CVS | 4.806 | 4.865 | 4.887 | 9.909 | 46.166 | 0.649 |
| ResNet-50-1D | 5.059 | 5.172 | 5.250 | 71.343 | 102.520 | 61.211 |
| DenseNet-121-1D | 21.369 | 21.474 | 21.544 | 566.020 | 208.924 | 21.626 |
| Complex ResNet-18-1D | 7.211 | 7.294 | 7.508 | 38.597 | 58.684 | 29.468 |
| ResNet-18-1D | 2.442 | 2.575 | 2.611 | 23.874 | 33.104 | 14.751 |
| ResNet-34-1D | 4.159 | 4.277 | 4.379 | 36.784 | 46.014 | 27.681 |
| VGG-16-BN-1D | 2.381 | 2.483 | 2.503 | 148.091 | 167.294 | 138.939 |
| BiLSTM (3 layers) | 3.217 | 3.289 | 3.319 | 170.668 | 411.524 | 56.249 |
| ConvNeXt-Tiny-1D | 3.777 | 3.797 | 3.818 | 111.438 | 143.498 | 102.170 |
| ResNet-50-2D | 5.052 | 5.852 | 5.950 | 124.085 | 107.322 | 90.016 |
| ResNet-18-2D | 2.122 | 2.192 | 2.209 | 76.819 | 55.937 | 42.710 |
| Complex ResNet-18-2D | 6.379 | 6.404 | 6.796 | 119.470 | 102.830 | 85.366 |
| Transformer (6 layers) | 1.668 | 1.690 | 1.698 | 84.309 | 211.557 | 72.332 |

Excel：`E:\type10-7\outputs\01a0fd26-2621-7662-a9bd-9ff19509e812\standard_ce_clean_results_deployment.xlsx`。14 个工作表；原 12 页、8 张论文图、4 张可编辑图表以及既有数据/公式/格式/列宽/表引用保留。新增效率汇总第 86 行起的端到端测量表、208 行部署测量、41600 行请求时延。均值、P50/P95/P99、内存/存储单位和吞吐率使用可追溯 Excel 公式；全部与独立 Python 汇总核对一致，无公式错误。

数据与口径：`E:\type10-7\exports\paper_figures\20261012-standard-ce-deployment-r02-source`；Excel QA：`local_artifacts/standard_ce_deployment_20261012_r02/excel_verification.json`。存储差异与更快基线均完整保留，不基于本次性能结果重新选模或改网络。
