# P46：原普通时域 Tukey 窗的同抽样反事实

状态：SOURCE_COMPLETE，独立读回原进程已自然结束；待完整独立复算。唯一矩阵见 [experiment.json](experiment.json)。联合 Ag＋As 目标 ACTIVE；本轮不训练、不读取正式目标。上一轮 P45 已完成并关闭 L0，其正式微增不作为本项依据。

代码依据：实际普通 augmentor 的 bandedge_taper 是时域 IQ 两端的 Tukey 窗，先于原 general_channel（含循环移位）及 normal PA；不是接收频带滤波。未查到这个窗的直接消融。PA旁路、移位旁路、FIR、LA16、增益包络、DC等已失败方向保持关闭。

唯一假设：该时域窗额外损害当前源观测的默认头身份证据。最小检验冻结 P43 B00/B01 固定 FINAL30，在全部 L6300、两命名普通流比较 RAW/FULL/NO_TAPER。NO_TAPER仍消费原gate与alpha抽样、调用原_tukey_window后只返回ones；其余DC/PA/噪声/多径/CFO原生代码与随机流保持，不强制触发。未触发的IQ/logits须逐点相等。实际E13增强由保存base及native make/configure核对；不改变窗口范围或单包概率。

源数据：all5 RX1/3/4/6/8、day1/2/3、6TX、L6300/U56700隐藏/V27000、split392005、256IQ/25MHz、原EQ1/nativeRMS。无训练RX留出、无Ldev。仅用合法L；旧cache其他源角色可反序列化但不消费IQ/标签/统计；U真值不读取，V/target不访问，正式7视图IQ不变。无未知TX硬件反演。

固定F/SIG_TIME_X3、统一Sinc CENTER_25K/CENTER_UNITS、NoExplicitCFO、原内部门控/参考响应、joint160/defaultcos30。两个来源各scratch20+own10/fixedFINAL30，原角色与祖先由pinned P43 own_model/parent_payload严格核对；不按正式排序挑父，不拟合头或更新归一化。历史60modelE/13320更新单列，不是本次新训练。

固定B00 RAW正确集合：全部4个model/draw的NO_TAPER−FULL保持率≥+.2个百分点、整体准确率差≥0；每模型合并两draw至少4/5源RX正增，最差RX≥−.25个百分点。全部通过也只支持冻结输入效应，不证明训练收益、不自动训练；任一失败关闭该固定窗旁路前提，不改窗、draw、RX、强度、阈值或预算救跑。90单元、未触发和触发样本全部保留，不能按子集晋级。

实际预算2×(6300RAW+2draw×2view×6300)+4CP合成smoke=63004NN；6300唯一L、12600普通IQ配对；0新信道/teacher/梯度/训练/headfit/target。两个合法checkpoint各2合成NN的noquery smoke为入口首步，PASS后立即源诊断。不存在新模型，所以不重复已完成的来源七视图正式测试。

本地8项聚焦检查PASS：原生FULL逐位一致、原窗几何、未触发零效应、RNG/函数异常恢复、协议/shape/标签反例、四model/draw及五RX条件正反例；8文件及两内嵌远端脚本AST PASS。本地既有Torch/NumPy ABI警告保留，未用Tensor.numpy，不改环境。唯一限定直接P0/P1审查PASS，未读真实IQ/CP/target、未改代码/远程。

N607普通账号、GPU2/CPU2，所有owner合计≤3进程/GPU且新slot≥12GB空闲；原capacitylock；不停止或修改其他健康任务。唯一owner codex/root/native-taper-source46，独占 20261018-diagnostic-FcrossRX-nativeTaper-L6300-s2026092701-m2-r01 run/release/log根。CWD cvs_FcrossRX_nativeTaper_L6300_20261018_r01 release；python -u experiments/cvs_taper_source46/runner.py。先一次只读preflight、一归档一次SHA比较、remotecompile；发布器本地字符串语法错误已在任何远端操作前修正，两个内嵌脚本AST通过。首次预检在本地PATH找不到whoami时停止、尚未SSH；使用已验证env15/System32路径后预检通过，未变更系统环境或重复远端作业。

技术停止仅限所属角色/来源/非有限/shape/NN预算/输出冲突/系统失败，保留不自动重试。预期原模块__file__/真实E13/config、CP4NNsmoke、默认logits/opaqueID私有NPZ（不含IQ）、完整result/completion、全部RX/TX/day/触发/MSE、独立全量复算、实际PNG/PDF与同row报告/登记。p2最小句柄本Phase1不适用，复用原VALIDATED_ONCE源数据；不建额外哈希、签名、receipt、报告审查或发布许可。

## 实际完成读回

执行代码 ce6eee755301e942acc38e7d8937240ff44ef997 已推送并独立OID一致，单归档比较/remotecompile/远端8项检查PASS。PID1002504 的CWD/argv/startticks与GPU2曾独立核实live；之后stdout从0增至144字节，完整smoke4NN及两draw进度齐全，现自然退出、stderr空、无failure。完成63004NN/0训练/0target。源固定条件2PASS/2FAIL，原假设关闭，不启动去窗训练。

实际E13不是base默认值：p_taper=.0626500835，alpha=.0050120067至.0250600334；时移最大8点。两draw分别原生触发256/512次观察。整体保持率差为B00两draw0/0、B01两draw0/+.015898pp；不是尚未执行。触发不一定意味着输出有变化，待NPZ完整复算核查。原launch.json的CUDA字段是静态空字符串，/proc独立证明实际GPU2，保留原产物；仅修正后续本地发布器元数据，不改已完成release或重跑。
