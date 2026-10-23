# PAMF 半监督模型接入 DURA 的补充评估

状态：VERIFIED_CONTROLLER_QUEUED。用户在侧聊要求补充接入 DURA 的星地性能，本次只评估已有冻结模型，不改动主任务。

固定 PSM、RXPSM 各三个 E50/11100 学生和默认联合分类头，接入同一 DURA v1.0：R69 stage2/native_map_prior/final.pt，266951 参数，InferenceEngine(reuse=True, graph=True, native=False)，FP32、TF32off。监督 F20 和普通 PL F21 的相同 G 七景结果已完成，直接复用；新六行的 raw 决策也复用原物理 ID 对齐的冻结预测。监督 F20 指识别器监督训练，公共 G 使用过合法 L/U IQ，不标为整个系统仅 L 训练。

矩阵：PSM／RXPSM × model seed 2026092701／02／03；同一 F0 目标集，每景 168000 个物理样本，clean 和六种 practical。逐行 raw／DURA 七景全部预测固定后，独立 CPU scorer 才连接 truth。报告每景准确率、Macro-F1、六景均值、三种子样本 SD、配对净纠错及 RX／TX／day 分层。结果不回流调参、选模或影响其他健康实验。

源 F 的全部 own 20＋10＋20 祖先仍由原 frozen loader 检查；G 的 role_ids、counts、RX、day 与 F0 相同。数据句柄 VALIDATED_ONCE 不变，不重建数据。完整路径、种子口径、矩阵、环境和命令见 experiment.json。执行 commit 由不可变 release/runtime.json 与后续 launch_verification.json 记录。

GPU5／6／7 分别对应三个 seed，每卡最多一个本补测 worker，总 compute 进程计入其他 owner、上限两个，启动前空闲显存至少 12000 MiB；仅一个独立 CPU scorer。主任务和其他 owner 的健康进程保持运行。单独真实 F＋G source-only smoke 为控制器第一步，PASS 后直接继续；smoke 失败或两次相同技术失败停止新派发，保留所有产物，不自动重跑。

四项聚焦协议负测 PASS：predictor 拒绝 truth，拒绝替换 F／G／raw 来源，拒绝 scorer 指向另一预测，缺失预测时 truth 保持关闭。冻结 evaluation／loader／engine 复用既有独立审查；本次没有新增训练机制或候选筛选。

命令：
```text
E:/type10-7/local_envs/phase1-cu128/Scripts/python.exe -X utf8 -m experiments.pamf_ssl_research.dura_supplement publish --repo E:/type10-7/code/worktrees/pamf_ssl_dura_eval_20261023 --context E:/type10-7/local_artifacts/pamf_ssl_research_20261023/baseline_context.json --output E:/type10-7/local_artifacts/pamf_ssl_dura_eval_20261023/release-r01 --ssh-config E:/type10-7/tools/n607_ssh_config
```

工作目录为 code/worktrees/pamf_ssl_dura_eval_20261023；远端 release/output 使用本 run ID 且不可覆盖。此时没有新 DURA 目标分数，不宣称完成。

## 已核实启动状态

观测时间 2026-10-23T11:36:52.536957+00:00，执行 commit `e01ac1f474ef8083e4b9ab62ba78549d286eb0e8`。一次归档传输比较、远端编译完成；独立读回核实控制器 PID／CWD／命令匹配，状态心跳继续。六行均 QUEUED，当前每卡已有两个总计算进程，尚未开始本 run 的 GPU smoke 或预测。队列等待容量后自动继续。尚未收集新 DURA 目标评分，不宣称评估完成；其他 owner 保持原运行。

## 已核实启动状态

观测时间 2026-10-23T11:41:42.784726+00:00，执行 commit `e01ac1f474ef8083e4b9ab62ba78549d286eb0e8`。一次归档传输比较、远端编译完成；独立读回核实控制器 PID／CWD／命令匹配，状态心跳继续。各行的实际状态见 launch_verification.json。尚未收集新 DURA 目标评分，不宣称评估完成；其他 owner 保持原运行。


## 用户取消重复派发

VERIFIED：本侧聊六模型队列与已先启动的九模型补测指向同一 PSM/RXPSM 最终权重及测试样本。用户明确选择取消重复六模型队列、保留九模型队列与全部训练。确认期间侧队列已完成 seed2 的 PSM 评分并开始 RXPSM 预测，因此仅停止经 PID/CWD/完整命令核实的 CPU 调度器，取消另外四条尚未开始的预测；已完成产物保留，正在预测的子进程不接收信号并继续完成。独立后态确认控制器已退出，三个 VIEWANCHOR 训练进程及两条主控制器均保持原身份。

后续统一结果由 `20261023-phase1-PAMF-SSL-DURA-reference-m9-r01` 的原定九行产生。本侧聊部分评分不读取、不用于选模、调参或挑选结果；重叠工作及其额外推理成本如实保留。本队列不能再按旧 RUNNING 状态恢复，远端 cancellation.json/handoff.json 已标明用户取消和替代 owner。公开证据见 cancellation_verification.json。
