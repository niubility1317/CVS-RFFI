# ECRS V2第二批桥接实验

run_id：`phase1_ecrs_v2_bridge_s392005_e200_20260908_r1`。
状态：`LOCAL_VERIFIED`，部署后追加进程与队列证据。

## 授权与矩阵

用户在首批已运行后追加“再启动多点实验”。本次据此提前展开[已发布矩阵](ECRS_V1R_V2_MATRIX_RELEASE_20260907.md)的B3a/B3b/B3c/B4四个source-only候选，不等待首批性能结果再派发；这是执行顺序的明确化，不用尚未返回的结果声称source门槛已通过。首批B0/B2-V1/B2继续原run，无停止、修改或迁移。

|行|估计器|锚点|训练目标|
|---|---|---|---|
|B3a|legacy28_old_reference|real8|原基线＋response CE=0.15|
|B3b|legacy28_estimated_reference|real8|同上|
|B3c|compact8|real8|同上|
|B4|compact8|complex24|同上|

四行均为V2、seed392005、200轮、从头训练，物理估计冻结，旧物理loss=0，cross-RX/U/fusion关闭。B3a不是相对B2的solver单因素对照；B3b为参考路径包变化，B3c为字典/尺度/去冗余包变化，B4仅改变锚点。B3与B3c重复，不再启动B3。B5、B6、策略行和B7不在本次队列。

数据、骨干、增强、LR、训练预算和48次source验证保持已发布配置。ManySig equalized=1，source RX=1,3,4,6,8、day=1,2,3；L/U/V=0.07/0.63/0.30，V只读，U真值不进入训练，target不用于选择。四行固定raw路径选择，并保留响应路径证据；独立预测增益、TX/RX/view probe和真实数据效果仍未证明。

## 环境与有界派发

项目根：`/home/szu2070436088/2510044040/CV-SincNet`。Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，torch2.1.0+cu121。
数据：项目根下`Dataset_WigSig/ManySig.pkl`。release/CWD：`releases/<run-id>_<commit8>`。checkpoint/output：`runs/<run-id>/<row>`。stdout/queue/常规metrics：`logs/<run-id>`；现有ECRS逐批遥测随checkpoint写入对应runs行目录，不迁移历史产物。

2026-09-08约00:40盘点：GPU0有B0一个训练进程和首批非训练dispatcher；GPU1/2各有本批旧实验＋其他adapt-unit任务，GPU3—7各有两个其他adapt-unit任务。本次不干预这些进程。排除精确识别的首批dispatcher CUDA上下文后，当前仅GPU0可新增一行。

队列固定顺序B3a→B3b→B3c→B4，每次读取全卡计算PID，未知任务也计入容量；每GPU最多两个计算任务。已创建但尚未出现在nvidia-smi的本队列子进程预留名额，防止初始化窗口超发。优先选择占用少的卡，再按GPU编号，不保证所有行同时启动。真实GPU分配以children.jsonl为准。

仅在队列启动后的24小时内派发这四行；到期未启动者写入unstarted.json，不自动重新派发。logs下创建`STOP_LAUNCHING`可取消剩余未启动行，已运行任务继续。失败只记录，不自动重启、不停止其他行。唯一launch owner为当前主Agent。

```text
<python> -u <release>/code/scripts/run_ecrs_v2_bridge_batch_20260908.py --project-root <project-root> --run-id phase1_ecrs_v2_bridge_s392005_e200_20260908_r1 --smoke-checkpoint <project-root>/runs/phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_direct8_20260902_r2/ADV3B02_ECRS_R7/best.pth --detach
```

首次派发前，以单独子进程严格重建真实R7 checkpoint作无query合成IQ前向，同时检查四种V2响应路径的有限loss、encoder梯度及物理侧无梯度。检查子进程退出释放CUDA上下文，再核实容量并启动正式行。该旧checkpoint不用于正式行初始化，不创建smoke许可artifact。

## 验证与判定

复用已完成的36项相关回归及真实parser展开检查。此次四候选共享真实V2训练路径一次P0/P1审查、新队列helper一次定点审查均无阻断。新增4项测试通过：dispatcher排除、CUDA可见前预留、PID去重及精确四行范围；CLI检查通过。发布时本地Git固定代码、push读回、一次release SHA比较、一次远端编译与实际smoke。

低性能不停止健康训练；既定协议/路径错误、输出碰撞、无法执行/产生合法prediction等技术故障才处理所属run，保留产物。最终source输出需包含选定checkpoint、clean及三LEO逐样本prediction、训练/响应诊断和资源；source完成记SOURCE_SCREEN_COMPLETE，不冒充target确认闭合或默认方案晋级。当前仅启动授权，不含周期监控、无界候选或自动技术修复。
