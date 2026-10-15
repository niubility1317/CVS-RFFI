# E50 冻结后源域机制分析

状态：PLANNED，训练期间不启动、不改动任何健康进程或配置。本分析为已授权同 run 的只读结果归因，训练更新数和新增 epoch 均为 0，不产生分类头、伪标签参数或选模反馈。

四个 E50 final50 checkpoint 全部 source-frozen 且已完成七视图评分后，使用与 E30 相同的 720 条 TX×源 RX×day 平衡 L_s IQ、固定物理 ID、命名随机流和冻结 E50 源参考模型的 clean 正确集合。U 和 query 数据均不打开，V 不参与本分析。

B0、C3、SSL_CONF、SSL_WEAK 各运行一次同场景 full residual 分层和一次原控制条件，另含参考模型 full 与控制两次，共 10 次 checkpoint pass。预期 267120 个前向样本（含 720 条固定参考正确集合计算），实际成本独立记录，不能计入训练提升或掩盖成本。控制包括移除噪声、展平传播核、移除残余载波和 RX IQ 残差的 total/fixed 两类效应、边界影响。所有波形、接收机状态及补偿依赖沿用已经验证过的源诊断实现。

沿真实 CVS 前向记录现有 time/frequency/PA/reference/signature 分支，默认头间隔统一除以 scale30；记录源状态、接收机锁定、signal_to_added_noise_db 区间、固定参考集合保持率、实际内部门控及各既有分支/默认头的梯度范数与夹角。autograd.grad 只用于读出，不执行 backward/optimizer step，不累积参数 grad；开始与结束逐项确认模型状态不变。source_role=L_s、query_read=false、U_truth_read=false、head_fit_updates=0。

普通 N607 账户，CPU-only，torch/OMP/MKL/OpenBLAS 各 2 线程；CUDA_VISIBLE_DEVICES 为空，不占用或停止其他 GPU 任务。唯一 release 为 cvs_sigtime_urban_ssl_e50_source_20261015_r01，唯一输出为原 E50 run/source_mechanism，代码和配置以实际 Git commit 标识。入口在原训练 release 之外，绝不热改运行文件。发布复用一次归档传输比较、远端编译及独立进程回读。技术失败保留输出、先核实原 PID/日志，不自动重试；未冻结、源角色错误、非有限数或状态变化均停止本分析。

本地发布命令：`E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 -m experiments.cvs_urban_sigtime50.source_publish --output E:/type10-7/local_artifacts/cvs_urban_sigtime_e50_20261015/source_delivery_r01`。远端命令：`python -u -m experiments.cvs_urban_sigtime50.mechanism`。分析结果只说明当前固定源面板和单 seed；小分层须报告样本数，不将损伤移除效应直接当成训练收益的因果证明。任何 E50 目标成绩均不得调整本方案。

预期 artifact：source_mechanism/results.json、sample_branches.jsonl.gz、progress.json、逐状态/质量/条件表与分层图。源码只复用已完成的 E30 分析实现，并增加四行 final50 来源检查。两项定点检查 PASS：缺行/非 E50 拒绝；真实分支观测保持 logits、参数与 grad 状态不变。尚未执行时不宣称有机制结果。
