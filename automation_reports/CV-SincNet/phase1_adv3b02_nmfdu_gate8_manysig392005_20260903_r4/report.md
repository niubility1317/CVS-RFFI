# ADV3B02-NMFDU-Gate8 ManySig r4修复与重发报告

## 1.结论与边界

- r3的E2、E3、E4、E5、E6、E8均在epoch80完成后、首次进入Stage2时触发相同的CUDA AMP异常；E1与E7绕过null BCE，未触发该异常。
- 根因是外层autocast开启时调用概率形式`binary_cross_entropy`。修复仅在该局部关闭autocast，并将softmax后的`null_weight`概率和物理质量目标转为FP32计算BCE。
- 未把概率误当logit，未改变null-aware softmax、物理可辨识性、可分性、稳定性、不确定性、损失权重、三阶段调度或梯度链路。
- 数据、8个候选、seed、200epochs和科学选择规则保持不变；不加入ADV3B02基线。r3输出只读保留，r4使用新run ID和输出根。

## 2.需求—实现—验证追溯

|ID|原子要求|状态|验证证据|
|---|---|---|---|
|FIX-01|null概率校准在CUDA AMP下可执行且有限|VERIFIED|本机CUDA FP16 autocast下6个受影响候选均通过|
|FIX-02|保留概率BCE语义及门控梯度|VERIFIED|逐候选与显式概率BCE对值，sample gate梯度存在且有限|
|FIX-03|无证据不改非有限梯度相关算法|NONBLOCKING_MONITOR|跳步发生于共享Stage1，loss有限且现有保护有效|
|FIX-04|新run ID重发，不覆盖r3|STOPPED_BY_USER|r4曾完成启动绑定，随后按用户指令精确停止；无性能结果，后续不得复用r4 run ID|

## 3.修复与本地验证

- 代码提交：`1f56a830df9ebf7bbc58ad6e62f32f4dcae87a87`。
- RED：修复前，新增的6项CUDA FP16 autocast测试全部复现N607的unsafe BCE异常。
- GREEN：修复后6项全部通过；null损失为FP32且有限，与显式概率BCE一致，sample gate梯度有限。
- NMFDU聚焦回归：7个测试文件共57项通过。
- Python语法检查：通过。
- 独立P0/P1审查：无P0/P1；未发现概率语义改变、query越权、输出覆盖、启动或prediction闭合风险。
- Git远端核对：本地`HEAD`与远端分支OID均为`1f56a830df9ebf7bbc58ad6e62f32f4dcae87a87`。

### 非有限梯度判断

r3只持久化了每轮聚合的`train_skipped_nonfinite_grad`，没有首个异常参数名。跳步集中在所有候选共享的Stage1，`train_skipped_nonfinite_loss=0`且训练持续推进，符合GradScaler拦截偶发溢出批次的行为，不是6行第81轮一致退出的原因。现有证据不足以支持修改损失权重、学习率或GradScaler，因此本轮将其保留为非阻断监控项。

## 4.冻结实验协议

- run ID：`phase1_adv3b02_nmfdu_gate8_manysig392005_20260903_r4`。
- 矩阵：E1 equal、E2 i_only、E3 i_d、E4 i_d_s、E5 physical_fixed、E6 physical_full、E7 full_no_null、E8 full。
- 数据：ManySig、equalized=true、split seed=`392005`；source RX=`1,3,4,6,8`、day=`1,2,3`、`L_s/U_s/V=6300/56700/27000`；target RX=`0,2,5,7,9,10,11`、day=`0,1,2,3`。
- 训练：200epochs；target clean与3个`leo_*_weak`场景只用于最终测试，不参与训练或选择。
- 停止规则：只对协议/路径/覆盖、确定性系统异常或无法产生合法prediction等技术失败停止；不因性能低停止。
- 预期artifact：每行最终checkpoint、训练日志、clean和3个LEO弱场景评估、prediction与独立评分。

## 5.N607发布、启动与暂停

- 状态：`STOPPED_BY_USER_BEFORE_EPOCH1 / NO_PERFORMANCE_RESULT / PAUSED_AWAITING_CONFIRMATION`。
- release归档：`E:\type10-7\local_artifacts\releases\adv3b02_nmfdu_gate8_manysig392005_1f56a830.tar.gz`→`/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate8_manysig392005_1f56a830.tar.gz`。
- release SHA256：`ade39216eb638e39f533dc98ebe3d2a4a9ce89fe31dddb682b88ed76d7842042`，本地与远端一致。
- release目录：`/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate8_manysig392005_1f56a830`；远端关键模块编译通过。
- 真实checkpoint无query smoke：`PASS`；严格加载195个state tensor，missing/unexpected均为0；初始化52个NMFDU state key，得到23个有限非零梯度；query truth与Phase2访问均为`false`。
- 首次以文件路径直接调用smoke时，包内`logging.py`遮蔽标准库，异常发生于导入Torch前且未生成输出；改用正确的模块入口后通过。N607环境未安装pytest，未安装新包；CUDA AMP回归已在本机CUDA环境闭合。
- 启动命令：`env ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate8_manysig392005_1f56a830 WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl RUN_ID=phase1_adv3b02_nmfdu_gate8_manysig392005_20260903_r4 RUNS_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_nmfdu_gate8_manysig392005_20260903_r4 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_nmfdu_gate8_manysig392005_20260903_r4 MAX_ACTIVE_PER_GPU=4 bash /home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate8_manysig392005_1f56a830/code/scripts/launch_phase1_adv3b02_nmfdu_gate8_manysig392005_20260902.sh`。
- 启动时间：`2026-09-03 10:22 +0800`；dispatcher PID=`611221`。
- 行PID/GPU：E1=`611287/GPU0`、E2=`611295/GPU1`、E3=`611282/GPU2`、E4=`611299/GPU3`、E5=`611296/GPU4`、E6=`611283/GPU5`、E7=`611290/GPU6`、E8=`611302/GPU7`。
- 启动后GPU0–7计算进程数：`4/2/2/2/2/2/4/2`，未超过用户授权的每GPU最多4个。
- PID/CWD/cmdline读回：dispatcher及8个训练PID均存活；CWD均为`/home/szu2070436088`，cmdline绑定新release、r4输出根、ManySig、split seed=`392005`及对应消融模式。
- 8份独立日志和PID文件已生成；初始扫描没有`Traceback`、`RuntimeError`或OOM。
- r3未续跑、覆盖、停止或删除。

## 6.用户暂停指令与停止闭合

- 用户在r4完成启动后明确要求“先别启动实验，等我确认后再启动实验”。收到该指令后，没有继续等待训练，也没有启动其他实验。
- 停止时间：`2026-09-03 10:30:36 +0800`前完成TERM停止与读回核验。
- 归属核验：以dispatcher PID=`611221`为根解析完整父子进程树，并逐项核对固定release路径、固定launcher或r4 run ID；共锁定81个r4归属进程，包括dispatcher、8个行包装进程、8个训练主进程及其数据加载子进程。
- 停止范围：仅向上述81个精确PID发送信号；未使用`pkill`、模糊进程名或跨实验终止命令。
- 独立读回：上述81个PID均已不存在；GPU上仅剩启动r4前已存在的r3、FCR和ECRS进程，未影响无关任务。
- artifact状态：8份启动日志均以6325bytes原样保留；r4输出根尚未产生checkpoint、metrics、prediction或评分文件，因此本次没有可解释的性能数据。
- 不可复用约束：r4 run ID及其输出根视为已使用并永久保留。待用户明确确认后，只能以新的不可覆盖run ID（预期r5）启动原冻结8行矩阵。
- 修复状态：代码、回归测试、Git提交、release归档及真实checkpoint无query smoke仍保持`VERIFIED`；当前暂停仅来自用户调度指令，不是技术失败或低性能停止。
