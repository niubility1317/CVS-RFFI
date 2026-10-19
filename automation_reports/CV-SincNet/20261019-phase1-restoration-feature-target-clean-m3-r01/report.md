# 冻结特征恢复：完整目标 clean 三套原 F50 对照

状态：RUNNING / VERIFIED。性能以目标真实 TX 测试为准。源机制已通过预登记规则，但源 V 的 low-urban +3 个百分点不作为目标性能证据。尚未收到扩大目标场景的新选择，因此沿用已有 clean-only 授权，不读取六种 practical 目标 IQ。

## 固定矩阵

两组为同预算 waveform_control、frozenF_feature，各自复用已固定的源域末轮 checkpoint，不新增训练、校准、目标适配或另训分类头。原 F50 三套 `CROSS_RX-s2026092701/02/03` checkpoint 与各自默认头均冻结，分别对 168000 条同一目标 clean 样本比较原 IQ、父 R03、refined。三个 F 读出分别报告并计算均值与样本标准差，不作集成票决。恢复器只训练过 seed2026092701，不能把三套 F 读出称为三个恢复器训练种子；504000 次 F 决策仍只有 168000 条物理样本。

源域资格在目标访问前已固定：实验组 low-urban 83.2778% 对控制组 80.2778%，clean 两组均 98.5%，平均 practical NMSE 比值 1.01748，满足原 ≥0.5 个百分点／≤0.1 个百分点 clean 降幅／≤1.05 NMSE 比值的规则。模型及权重、损失、预算、种子和 checkpoint 选择不再据目标改变。

## 数据与评分

复用原 VALIDATED_ONCE 完整 formal split，以及已完成 R05 中同一物理 ID 的 clean 和父 R03 IQ。新预测只读取当前 clean IQ，经冻结恢复器和三套原默认头逐样本 argmax，不输入真实 TX、role、batch 真计数、配额或全局重分配。每组全部 clean IQ 和三个 F 的三路预测完成并验证后，单独无 CUDA 的 CPU scorer 才读取目标 TX/RX/day truth。同一 run 最多一条 CPU scorer。

主要报告各 F seed 与三 seed 均值的目标 accuracy、相对原 IQ 和控制组的配对差、修复/损坏/净修复，另报 Macro-F1、逐 TX/RX/day 混淆矩阵、clean NMSE 和完整预测耗时。原 F 三 seed 历史 clean 均值为 85.113889%，本轮会从实际预测独立重算，不能直接借用历史数字作为新结果。该基准已暴露，不宣称新的盲测或真实在轨证据；clean 测试不能替代星地目标性能证据。

## 执行边界

N607 普通账号，GPU0 为控制组、GPU1 为特征组，每卡一条本 run 预测 worker。共享容量锁、总 owner 进程少于 2、至少 12 GB 空闲显存、CPU 2 线程；控制器无 CUDA。首步真实两组末轮 checkpoint、三套原 F50 和 2 条 L_s 无 query smoke，PASS 立即继续。一个独立不可覆盖 root，非法源资格/父模型/代码、输出冲突、非有限或预测缺失则保留产物并停止，不自动重试；不终止其他健康 owner。

7 项聚焦检查通过，包括三 F 覆盖、类决策合法性、提前 truth、IQ 覆盖，以及未通过源规则不得扩大读出。沿用源候选已完成的一次独立 P0/P1 审查，不增加重复 runner review。预检确认三套原 F50 权重与实际训练/来源 sidecar 齐全、源比较完整通过、新 root 不存在及 GPU0/1 空闲。命令、环境、路径、矩阵与预期产物见 [experiment.json](experiment.json)。


## 启动核验

方法 commit：`ffef2a7fffe1a91677aadcfcc94807e841a3b36d`。单次发布归档 SHA 比较、远端 compile/import、真实两组末轮权重＋三套原 F50 默认头的 2 条 L_s smoke 均通过。独立后态核实控制器与两 worker 的 PID/CWD/命令、物理 GPU0/1 归属及两次日志增长。两组当前均已完成至少 49152/168000 条目标 clean 的三套 F 预测，暂无本候选目标评分结果。

登记中 model seed 指已训练恢复器 seed2026092701；data/augmentation/evaluation 本轮没有随机抽样，记为 null，原 split 为继承句柄。此处是纠正从源训练记录复制来的说明字段，不改变实际目标输入、顺序、F seed、模型、预算或科学配方。F 三套读出 seed 在独立字段明确列出。目标真实标签仍由各组完整预测完成后的独立 CPU scorer 接入。
