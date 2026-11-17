# 改进实验队列扩卡接管

用户要求已登记实验不再等待单卡串行。原 `20261117-phase1-pamf-repl-priority-clean-m20-r01` 已 smoke PASS，十类八行训练完成，六源四行训练运行，其余八条六源训练排队。补跑已 ANALYZED，GPU 0、2 空出。保留原 20 行全部方法、代码、配置、数据、seed、输出、冻结与测试矩阵，不增加训练行或重训已完成模型。

新 owner 先在停止瞬间核对原 dispatcher PID、CWD、命令、所有子进程与状态，确认没有已启动但仍标 queued 的任务；仅停止 dispatcher，不对训练、预测或评分 worker 发信号。新 dispatcher 检查原 owner 已退出，以原 PID/CWD/argv 和 completion artifact 接管原 active worker；随后用原已提交的不可变 worker release、原 config、原输出根和已有真实 checkpoint smoke 调度尚未启动的原工作。原 worker 退出码在重挂父进程后不可得，以预登记 completion artifact 判断完成并明确标注。任何 worker 失踪、未知日志、身份不一致或系统技术失败停止新增派发，保留原产物。

优先实验使用 GPU 0、1、2、3、5、7，与低权重 KL 独立 owner 的 GPU 4、6 分开，避免两个本轮控制器同时对同卡做非原子的容量判定。每卡本优先实验最多两个 worker，全部 owner 与待初始化进程合计不超过 5，启动前至少 12 GiB 空闲；逐次派发前重新读取显存和其他进程。其他 owner 任务不变。全部预测先于独立 truth-last clean 评分，U 真值只在冻结后离线诊断。输出为原 run 的任务和评分，加本次 `takeover.json`、`complete.json`；无 target 数据反馈到训练或选择。发布、接管和启动须由独立运行证据核实。
