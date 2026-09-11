# CORE90锚定几何单seed source实验

日期：2026-09-11。Run ID：`core90_anchored_s392005_20260911_r1`。唯一launch owner：当前任务主Agent。

用户授权：启动刚修复验收的实验，数据集配置和此前一样，seed392005。执行已接受的锚定几何设计首轮source矩阵；不重训H0、不调用旧target预测/评分链，不增加另外两个head seed。

## 方法、矩阵和预算

|候选|本次实际计划|预算/条件|GPU|
|---|---|---|---|
|A2/A3/A4/C_angle/C_angle_keep|共享L_s paired缓存，每种5个4RX头部OOF＋1个5RX最终专家；最终专家直接校准/导出|每个80轮，35步/轮，共30次拟合；A2只消费clean，其他clean/LEO各0.5|GPU0/1/2/3/5各一种并行|
|A0/A1|原H0与source类均值角度读出，统一V最终校准|无新增SGD拟合|GPU0，前述阶段完成后|
|P1|独立t/f/pa块证据拟合与pattern校准，full/missing_t/missing_f/missing_pa/all_missing|独立闭式E成本，不混入30/40个SGD头数量|GPU0，顺序阶段|
|A5-0/25/50/75/100|复用A4 OOF及最终专家，固定概率混合、实际保护动作审计、V校准/导出|无额外专家拟合|GPU0，顺序阶段|
|A6|A4固定OOF中至少一个非零动作正净收益后才执行嵌套门控评价；否则记录NOT_ACTIVATED_NO_SOURCE_UTILITY|触发时最多再拟合10个去重3RX专家，本seed共40次上限|GPU0，顺序阶段|

几何与科学参数沿用`code/configs/core90_anchored_geometry_v1.json`：r=4、ρ=log(2)/2、条件数≤2、冻结W/τ、Adam学习率0.001/weight_decay=0、λmetric=0.01；keep仅A4/C_angle_keep开启，λkeep=1/τ²、γmax=0.1τ。A6的λH=2、ridge=0.001、utility_margin=0、保护分位数0.9；V最终coverage=0.9、温度范围0.05–20。延期分支继续关闭。epoch日志新增实时输出，参数及优化步数不变。

配置允许三个head seed，但调度命令只运行392005；split/view也为392005。单seed不能形成三seed支持结论，完成时保留`PENDING_ALL_HEAD_SEEDS`。source门槛仍为嵌套净收益、RX/TX/clean保底与多seed一致性；本次不据单seed自动晋级或选择性重跑。

## 数据与继承来源核对：VERIFIED

复用原run`core90_evidence_frozen_392005_20260911`的完整数据契约和安全H0导出。实物检查：

- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`，equalized=1。
- source RX=`[1,3,4,6,8]`，day=`[1,2,3]`；L_s/U_s/V=`6300/56700/27000`，比例0.07/0.63/0.30，监督比例0.1。TX映射为`14-10,14-7,20-15,20-19,6-15,8-20`。
- 契约保留原target集合168000条、RX=`[0,2,5,7,9,10,11]`、day0–3；本次不构建或迭代target输入/预测/评分。G不使用U_s标签或样本。
- H0：原run`H0/ground_checkpoint.pt`，SHA256=`e2905467945db6c1a98c62f67d13a527c0cca3e55a5a7ec6697b8aa3d099478d`；契约`data_contract.json`的SHA256=`37d80ee312cf67744081298840fbff1c2895876a0b3ecde2243dad6a8f9ff0fb`，均与已验收实物一致。
- 远端载入独立核对payload契约与当前契约完全相等；lineage为from_scratch=true、upstream=[]、selection=final_only、target_feedback=false。沿用原H0从零E200训练，不加载其他seed、teacher或历史joint_safe权重。
- source输入严格按既有物理ID列表物化，不重新划分；out_len=256、中心裁剪、RMS归一化、center=false、equalized=1保持原处理。新物化的全部L_s输入必须与此前`source_training.pt`逐tensor/顺序一致，V使用同一处理且只作校准。

已确认来源和契约未变，不重复进行data capsule验证。此设计受先前target分析启发，不能把再次运行旧target称作独立确认；当前source-only终点遵守已接受计划。

## 本地验证与发布

实现基线`147baf94e82df7810c93d1ff372616b59aceebbf`已有77项回归通过及独立修复审查。本次仅补充source矩阵调度、普通用户一次性dispatcher与每epoch进度输出；本地`test_anchored_experiment_launch.py`和`test_anchored_fit.py`共11项通过，0失败/错误/跳过，CLI help及diff检查通过。JUnit位于`analysis/core90_anchored_launch_evidence_392005_20260911/local_launch_tests.xml`。

独立P0/P1审查已完成，未发现阻断项；范围限定新增启动路径，既有机制不重复审查。审查者只读模拟确认14候选/4缓存/5个OOF正确调度，检查dispatcher排他创建、seed及GPU绑定、容量等待与无自动重试。发布代码将由本报告同次Git提交固定，push后独立比较远端OID；归档只取该commit的code目录，远端一次SHA对比、一次compileall，之后由launcher执行真实H0/L_s六类smoke并立即进入正式source阶段，不建立额外smoke审批。

## 环境、路径与执行命令

普通身份`szu2070436088@dell-DSS8440`；使用`E:/type10-7/tools/n607_ssh_config`的direct短连接，未使用管理员。远端Python=`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，PyTorch2.1.0+cu121、NumPy2.2.5；本次实测tensor.numpy互操作PASS，无环境安装/修改。

2026-09-11 18:23 preflight：GPU0/1/2/3/5为空，GPU4/6/7分别有其他任务；/home可用7.1T。本次各GPU最多新增一个worker，不影响其他任务；子进程启动前读GPU计算进程数，已≥2则等待30秒再次检查，不杀任何现有进程。GPU资源会变化，最终以启动读回为准。

- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/core90_anchored_s392005_20260911_r1`。
- 日志root：`/home/szu2070436088/2510044040/CV-SincNet/logs/core90_anchored_s392005_20260911_r1`。
- coordinator日志：上述logs父目录`core90_anchored_s392005_20260911_r1.coordinator.log`。
- PID记录：上述runs父目录`core90_anchored_s392005_20260911_r1.coordinator.pid`。
- CWD：本次新建的不可变release，路径与commit在落地记录补充。

冻结启动命令（RELEASE替换为落地目录）：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u RELEASE/code/scripts/run_core90_anchored_experiment.py --root /home/szu2070436088/2510044040/CV-SincNet/runs/core90_anchored_s392005_20260911_r1 --logs /home/szu2070436088/2510044040/CV-SincNet/logs/core90_anchored_s392005_20260911_r1 --ground /home/szu2070436088/2510044040/CV-SincNet/runs/core90_evidence_frozen_392005_20260911/H0/ground_checkpoint.pt --contract /home/szu2070436088/2510044040/CV-SincNet/runs/core90_evidence_frozen_392005_20260911/data_contract.json --previous-source /home/szu2070436088/2510044040/CV-SincNet/runs/core90_evidence_frozen_392005_20260911/source_training.pt --dataset /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --seed 392005 --gpus 0,1,2,3,5 --threads 4
```

预期产物：plan/status/commands、source_inputs.json、joint/blocks两布局的L_s/V缓存、5专家各6个OOF/最终checkpoint与完整history/geometry、各候选V校准/风险曲线/导出bundle/profile、固定融合与条件嵌套预测/分组指标、最终matrix_summary.json。source阶段终点为SOURCE_SYSTEM_FROZEN；不生成target artifact或确认性结果。

## 停止与保护

root/log/PID存在即拒绝重复dispatch；输出不覆盖，不自动重试，不回退旧权重。协议越界、来源/数据身份不一致、源IQ处理不一致、无法执行、数值非有限、输出冲突或无法闭合合法source产物记TECHNICAL_FAILURE，保留日志和部分产物；低性能、负收益或未达到收敛阈值不停止固定80轮。其他正在运行的同批健康worker正常结束，不对无关PID操作。SSH超时先读回既有run/PID，不重发。

当前预登记状态：LOCAL_VERIFIED，等待固定Git版本、落地和实际启动读回；未把预登记写作已运行。

## 落地与正式启动读回：VERIFIED

上述预登记后的实际状态为**RUNNING / EXPERT_OOF_FITS**，不是实验完成。

- 实际代码commit：`1afa43a75c76b728d4e6352ba8e44acae44f277b`；GitHub远端分支OID与本地HEAD独立读回一致。
- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/core90_anchored_392005_1afa43a7`。发布归档5482816字节，本地/远端SHA256均为`98430a43545b967b0929a47747b627a50b2b25db8663410fa63bd7ad7e9f43bf`，一次远端compileall通过。
- coordinator PID=`966219`、PPID=1，`/proc`实际CWD/argv与release和本run完全一致。dispatcher只执行一次，没有超时重发。
- source_inputs.json确认L_s=6300、V=27000，全部L_s的IQ/labels/physical ID及顺序与上轮逐tensor一致。真实H0/L_s六类smoke打印PASS，未使用该smoke梯度作为正式专家初始化。
- joint与blocks的L_s/V四份缓存全部完成，每布局L_s=12600行、V=54000行；源物理集合和角色匹配。

启动快照的正式worker状态如下（各自在首个heldout RX=1折，随后继续其余折和全source拟合）：

|候选|GPU|PID/PPID|快照epoch/80|optimizer_steps|
|---|---|---|---|---|
|A2|0|967800/966219|27|945|
|A3|1|967805/966219|19|665|
|A4|2|967810/966219|19|665|
|C_angle|3|967815/966219|26|910|
|C_angle_keep|5|967823/966219|23|805|

所有worker的`/proc`CWD为本release，实际argv包含`--stage oof --seed 392005`及本run输出；`CUDA_VISIBLE_DEVICES`分别为0/1/2/3/5，nvidia-smi实际PID显存记录对应362–396MiB。其他原有GPU4/6/7任务保持运行，本次没有终止或修改它们。

日志从启动警告推进到多条连续epoch记录。A4快照E19的CE/keep/metric加权梯度范数分别约0.190869、0.000109683、0.00184518；`a_max_change≈0.0064003`，机制确实参与当前优化。A2/A3的keep为0、普通角度对照的metric为0均是设计分支，不属于漏开。此处只证明启动和实际接线，不能据此推断收益、收敛或晋级。

读回的全部当前日志无Traceback，status仍为EXPERT_OOF_FITS、target_access=false。证据为`analysis/core90_anchored_launch_evidence_392005_20260911/startup_snapshot_1.json`与`startup_snapshot_2.json`，含实际OS进程信息、GPU记录、缓存计数、日志末两条及source输入核对。完整epoch日志继续保留远端logs目录；以上数值是启动快照，不是最终结果。

后续由同一个coordinator自动执行剩余OOF、全部source校准/导出、P1、A5及条件性A6；本任务没有另外创建定时监控，也未启动target或另两个seed。
