# A1-Fast-V2从零训练发布记录

状态：LOCAL_VERIFIED，等待Git固定与N607落地。用户授权继续优化A1加速版、发布实验，每张显卡一个实验，并明确要求从头训练、不使用其他checkpoint。

## 设计、数据与输入权限

遵循当前`E:/type10-7/项目.md`，原A1主线，不加入R3。设计见[冻结矩阵](../../../docs/A1_FAST_V2_EXPERIMENT_20260909.md)。四行均为M/lite_d/dual、160维、seed392005、E200、L128/U256、final-only。ManySig equalized=1、source RX=1,3,4,6,8/day=1,2,3，L/U/V=6300/56700/27000；不改变已验证数据契约。

所有行`from_scratch=true/a1_scratch_only=true`，baseline/teacher路径为空；禁止历史初始化、resume、教师、EMA或派生状态。EMA仅由本轮随机学生深拷贝后随本轮成功更新推进。四行初始模型、heads和CPU RNG完全一致。未加载历史checkpoint；任何继承尝试由guard拒绝。首个权重读取是本行E200完成后的本行结果，用于最终推理。

|行|GPU|区别|
|---|---:|---|
|F0_FIXED_BASE|4|原A1快速顺序教师，共同EMA缓存修复|
|F1_RUNTIME|5|F0+梯度/预算批量回读+RC4身份教师|
|F2_WARM_EMA|6|F1+成功更新计数驱动EMA启动平均|
|F3_COVERAGE_KL|7|F1+归一化后按有效共识覆盖缩放logit蒸馏|

GPU0已有LoRa、GPU1—3已有R3实验保留。本轮补满GPU4—7，每卡一个训练，不停止已有任务，不在已占用卡上并放。所有变化在读取新目标分数之前冻结；已有目标分数不作为本轮参数选择依据。

## 版本、命令与运行位置

分支`codex/a1-fast-v2-20260909`，基线`9a5c9665`；实际发布commit在落地后追加，不更改运行release。

- 远端项目：`/home/szu2070436088/2510044040/CV-SincNet`。
- 解释器：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，Torch2.1+cu121。
- CWD：本轮唯一不可变release，名称随commit固定。
- 启动：`python -u <release>/code/scripts/run_a1_fast_v2.py --project-root /home/szu2070436088/2510044040/CV-SincNet --run-id a1_fast_v2_s392005_20260909_r1`。
- 输出：`<project>/runs/a1_fast_v2_s392005_20260909_r1/<row>`。
- 日志：`<project>/logs/a1_fast_v2_s392005_20260909_r1`。
- 实际完整参数：`effective_matrix.json`和每行`launch_config.json/process.json`。
- launch owner：本任务主Agent；各row只由本轮dispatcher启动一次。

## 已完成验证

本机`ssr-gpu`：运行helper40项、学习和旧路径29项、真实矩阵1项，共70项通过。CUDA源域形状检查无checkpoint/query读取，B512 FP32/AMP身份输出、route、BN/RNG精确相等；E1/E21/E61/E161 RC4步骤梯度有限且anchor loss为0。另有L后U真实DAOT目标E1/E21/E161 loss、梯度、AdamW、EMA、BN和RNG对照通过。

独立P0/P1审查PASS：真实parser与数据权限、成功计数、RC4下游、coverage归一化顺序、输出保护和最终预测评分路径无阻断。远端首次由launcher执行本轮fresh模型无query检查，随后继续训练，不设额外审批。

本机B512教师计时：FP32约27.04→8.39ms，AMP约29.11→10.16ms。仅是教师前向微基准，不代表整轮训练提速，也不代表准确率提升。

## 完成标准与解释边界

每行E200 own final checkpoint及source-only训练日志完整后，复用既有opaque目标输入包`runs/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4/target_inputs`。predictor不读truth；prediction成功退出后，单独scorer连接既有truth sidecar。clean和三个`leo_*_weak`场景、全receiver指标与artifact全部闭合才标记ARTIFACTS_COMPLETE。

F1−F0评价执行效率与数值稳定性；F2−F1、F3−F1分别记录独立学习变化。报告同row clean/LEO均值、弱接收机、耗时、显存、跳步和机制实际日志；不跨row拼最高值、不用目标分数调参/选择/重跑。单seed只能形成描述性证据，不能晋级默认方法。性能假设允许负结果。

技术停止仅限所属run的协议越界、错split/stage/checkout、输出碰撞、无法执行、prediction/scorer无法合法闭合等确定性故障。低准确率、慢收敛不停止；不改变参数自动重跑。保留所有产物，不影响其他run。

## N607发布与运行核验

代码commit`89758ce1d5af25f8c73abad4dd0a29d51685e829`已push，独立远端OID一致。release为`releases/a1_fast_v2_89758ce1`，归档本地/远端SHA256一致：`cac6250944945e5130fac1314b799b97faa2b4894b7fcb13514d19081598e39b`；远端编译及fresh模型CUDA检查PASS。01:42左右启动，dispatcher4174118；F0/F1/F2/F3分别为4174259/4174264/4174269/4174274，GPU4/5/6/7。

独立核实四个训练PID存活、CWD、完整argv、GPU占用及`init=scratch L/U/V=6300/56700/27000`。后续读回F0完成E10，F1/F2/F3完成E11，最新epoch成功更新率均1.0，loss和gradient跳步均0。首批曾记录AMP非有限梯度跳步，已保存诊断包；不把早期跳步误写为全程无异常。尚未训练完毕或评分。

用户随后增加ECRS相关机制并明确每卡允许两个训练进程。原V2release和配置保留，新矩阵使用第二个名额，另见[八行扩展报告](../a1_ecrs_cross_rx_s392005_20260909_r1/report.md)。本报告开头的单卡单进程是本次初始发布安排，已被新的资源授权扩展，不限制新增合法任务。
