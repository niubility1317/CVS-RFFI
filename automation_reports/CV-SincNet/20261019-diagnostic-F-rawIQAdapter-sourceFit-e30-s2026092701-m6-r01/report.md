# Raw IQ 适配器：冻结源拟合诊断预登记

当前 raw IQ 双教师 E30 矩阵已完成，源条件先固定 FAIL，固定配方关闭。本诊断只回答源端拟合与函数变化问题，不延长、调参、扩种子或重新测试目标。

固定 run：`20261019-diagnostic-F-rawIQAdapter-sourceFit-e30-s2026092701-m6-r01`。代码工作区：`E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014`。实际代码/config commit 将在发布前填入 experiment.json；沿用已核实的 raw 模型 release `cvs_F_rawIQ_dualTeacher_E30_20261019_r01`（Git 6ea2f71f6654f4ceeadd460905a9d719b5dc1b89）。新诊断 release 为 `cvs_F_rawIQAdapter_sourceFit_E30_20261019_r01`，唯一输出 `PROJECT/runs/20261019-diagnostic-F-rawIQAdapter-sourceFit-e30-s2026092701-m6-r01`。

## 固定观测与矩阵

复用旧诊断 L180：6 TX×5 源 RX×3 day×2 条，均属原合法 L6300；原 U56700 的 TX 真值隐藏，V27000 不作新前向，目标 IQ、truth、预测和评分均不读取。复用 VALIDATED_ONCE 数据，不重验数据或建立 hash/receipt 链。source-only 原规范化处理为 normalize=True、center=False、transform=None，2×256 received IQ。

四个新 raw FINAL30 模型为 NOAS_ADAPTER、S1、S2、S3；原 F 与默认 6×160 cosine30 分类头不变。六视图为 raw、各物理样本在 raw S3 训练中末次已见 As、同 L 新 mid urban 两 draw、新 low urban 两 draw。已见与新信道比较须限定相同 scene 组成，不把混合场景差解释为泛化损失。

TC、TS 使用旧 `20261019-diagnostic-F-dualAdapter-sourceFit-e30-s2026092701-m8-r01` 完整源域预测，新增 NN 为 0。全量 L/U 4440 条日志已经证明新 raw S3 的180个末次观测坐标与旧面板相同；运行时还核对 physical IDs、原始处理、完整信道配置、seed/session namespace 及实际生成记录。只核对既有消费接口，不新增数据 revalidation/hash。旧控制不重新前向；不再重复 INIT_ZERO 或 Dropout-only 研究。

四模型各180×6=1080 个真实包前向，合计4320；启动器首步真实新raw checkpoint/noquery smoke1，PASS 后立即继续，总计 **4321**。信道生成900包、原始源 L 读取180条；optimizer、EMA、分类头拟合、新 V 及 target NN 均为0。四组继承成本各 E20+10=E30/6660更新，本诊断训练增量为0；教师旧 E20/E200成本保留。

## 指标、边界与后续判断

记录各模型/视图 correct/n、accuracy、macro-F1、NLL、LS=.01 CE、TC clean 正确保持与新错、实际 normalized joint160 距离分位数、默认头余弦间隔和充分保持界覆盖、源 RX/day/TX 描述、同头 forward 一致及准确 NN/信道成本。冻结读出不能证明梯度冲突，也不把训练 Dropout 统计称为 frozen fit。

本 run 是描述性诊断，没有新的晋级阈值，不选择训练系数、温度、seed或 checkpoint，不更改已固定的 source FAIL，也不宣称目标完成。是否需要新的特征保持或表示机制，只能依据这些合法源证据另行固定，不能依据本轮正式测试成绩救援。

## 执行

普通 N607 账户；当前 Python3.10/Torch2.1，CPU2、FP32、TF32关闭。新 release 为 CWD。命令：`CVS-RFFI/python -u PROJECT/releases/cvs_F_rawIQAdapter_sourceFit_E30_20261019_r01/analysis/cvs_raw_iq_adapter_source_fit_20261019.py --runtime PROJECT/releases/cvs_F_rawIQAdapter_sourceFit_E30_20261019_r01/runtime.json --output PROJECT/runs/20261019-diagnostic-F-rawIQAdapter-sourceFit-e30-s2026092701-m6-r01`。

GPU4/6/7 中依共享锁选择一张，所有 owner 总进程上限2、free≥12000 MB，当前 run 仅1进程；不停止、重启或热改其他健康任务。只有输入/继承权限、状态或形状不匹配、非有限值、输出冲突、固定预算或持久参数意外改变等本 run 技术失败才停止自身；保留产物，不自动重试。

预期产物为 execution.json、source_smoke.json、source_panel.json、四模型及复用控制 readout.json、source predictions.npz、analysis.json、complete.json、最终解释和登记。保存 post-launch PID/CWD/argv/GPU/log-growth；唯一发布包作一次本地/远端 SHA 比较、远端 compile 和 preflight。一个独立 P0/P1 审核只针对直接影响本次合法读出的缺陷，不增审批或重复 unchanged 审核。
