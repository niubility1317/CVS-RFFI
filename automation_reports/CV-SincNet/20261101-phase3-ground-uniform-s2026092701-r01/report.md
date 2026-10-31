# 地面均匀OE训练准备

LOCAL_VERIFIED，真实性能未测；CUDA不可用，未发布/启动；完整 Phase3 ACTIVE。

# 地面源域均匀 OE 训练候选

Run `20261101-phase3-ground-uniform-s2026092701-r01`，owner `codex-phase3-ground-uniform-20261101`。继续地面训练优先；完整跨RX星地拒识、无标签目标适应、未知发现与未来匿名注册目标不缩减。95%宏RU、95%CK、原同DURA F0闭集−AK≤2pp及预登记主场景95%保持，历史模型/数据/他人任务不改变。

## 假设与训练

上一轮原生对比未带来稳定综合收益，本轮不继承其权重或损失。排序允许通过改变相对间隔降低风险，均匀OE则直接要求源outside-six样本的六类后验接近均匀；是否改善未注册样本的拒识必须检验，不以训练OE均匀或loss下降代替测试。

单学生UNIFORM从完整原F0 Tri-SIG-TIME SSL E50/11100复制reader与默认六类头，新建AdamW微调，不从CPAIR、对比学生或PCD初始化。232993参数/232945可训练/头960；默认头与骨干共同更新，Sinc48和全部buffers固定。无新头、投影、EMA、PL、Us或目标U更新，原F0/DURA/旧方法保持。

固定目标为CE_L＋T2教师KL×4＋mean_d mean_(OE in d)[KL(Uniform6 || softmax(z_OE))]，三项权重1。15个公共RX/day域等权，不读OE真TX或真实K，不构造OE身份正负对。KL(U6||p)=−mean(log_softmax(z))−log6，采用稳定FP32运算且不clamp；每记录共同logit平移不改变此项。L/OE的centered pair-risk保留为同模型诊断，不参与新目标。

这是[OE原论文](https://arxiv.org/html/1812.04606v3)均匀后验思想在当前地面RFFI数据上的匹配实验；源教师、RX/day等权和预算是项目设置，不称论文完整复现。历史`20261031-phase3-logit-energy-s2026092701-r01`的UNIFORM使用源G10星地P1，不能替代本轮G(clean)的独立known TEST。

与固定参考CPAIR（`20261101-phase3-ground-centered-s2026092701-r01` / code `1ae125733a0f25eb9c887351eb63f11f730d8cea`）匹配原F0初值、每步L192/OE64、SourceSampler种子2026103017/18、全部2220步L/OE索引、AdamW lr/wd1e−4/eps1e−8/clip5、FP32/TF32off/eval保留梯度。仅替换OE损失；固定2220更新/568320学生前向后冻结，不用CAL/NV/Q选checkpoint或扫参数。每步记录CE/KD/selected uniform、15域uniform、pair-risk诊断、能量/centered/logitmean、梯度/clip、源索引与曝光。

## 数据与固定测试

直接复用VALIDATED_ONCE D7 `20261101-phase3-ground-open-s2026092701-r03-source-clean-v1` / `sourceV-half-y-rx-day-s2026110101`。训练G(clean) L6300真实六类y、OE6134仅outside-six成员监督/公共RX/day、原F0同G(clean)缓存L教师。Us/Us IDs/真值、目标U/UID、CAL/TEST/NV/query训练读取0；原F0有标签和源Us祖先保留，不称全系统无标签。

六已知14-10/14-7/20-15/20-19/6-15/8-20；源RX1/3/4/6/8=1-19/18-2/19-2/2-19/3-19，2021-03-08/15/23、split392005/eq1/256点。历史V27000按y/RX/day格300固定拆CAL150/TEST150，总各13500、物理互斥；TEST每类2250。旧builder合法读V已知y分层后私有导出TEST真值，本轮不在训练/预测读TEST标签，历史曝光不改称新盲测。

源Q每条件16650=knownTEST13500+NV3150，未知19-15/19-16/7-1各1050、public角色隐藏。跨RX仍开发接收机3-2/index34，未来03-15/23 Q10238（5525/4713），known4508六类1491/1182/150/406/1186/93，unknown5730为5-5/10-7/2-19/7-11各1454/1439/1427/1410。03-23缺20-19/8-20，完整已知指标N/A，不用present-only认证。未注册不证明从未在OE或历史出现，不新读OE真TX审查互斥。

四条件source_clean_DURA/source_star_DURA/cross_clean_DURA/cross_star_DURA全部固定DURA v1.0 R69/266951参数/FP32/TF32off/reuse+graph/nativeoff。源G(clean)/G10P1复用，目标clean只测试，跨RX同序固定物理记录逐条经G；G10MIX01的600km/25MHz/2.45GHz/post_sync/noEQ/refSNR30dB/reflect、一次输入RMS/输出无RMS、scene3001/channel4001/future3133/3134/receiver2027保持，不重抽观测，不用clean参考恢复star。

固定四方法：OLD_CPAIR_energy/OLD_CPAIR_centered/UNIFORM_energy/UNIFORM_centered，主index3。旧两列来自参考CPAIR的col2/3，原F0独立baseline_logits/closed不能用其col0DMEAN替代；旧NN不新增前向。新两读出共用UNIFORM默认头身份，energy=LSE六logits，centered=LSE−mean−log6，高分接收，不依据结果换主。

网络冻结后仅CAL13500按原known24组规则拟合两个新六阈值，再加旧12值，共24个FP64/192B、主6/48B。新所有阈值先于Q冻结；零尾保护失败则报告并继续合法预测，不以未知/测试修复。4×4共16点完整后，独立CPU接sourceknownTEST/sourceNV/crossclean/crossstar四真值各一次，两个旧CPAIR四条件共8个overall与原JSON精确一致。CPU只重建分数/路由/指标，不NN、源训练数组、重新校准或拟合。

正式macroCK95/RU95/原F0损≤2pp及名单完整；源clean全部24组、crossclean总体+clean、两star总体+六景。ground_source/cross_clean/ground_stage/frozen4分别报告；不通过NO_PROMOTION、不扩seed、不按Q调阈值。即使冻结拒识通过，UDA/发现/未来匿名注册仍缺，完整ACTIVE、AN/H_OW=N/A，正式RX24-18未访问。

## 成本与运行边界

新训练568320；新学生预测CAL13500+两源33300+两跨20476=67276，合计635596；旧CPAIR/F0真实新前向0、G20476。源data G99284/F036450、原F0/G/CPAIR训练与cache分别继承。合成smoke：训练student6/F02/G2，预测student2/F02/G2，r03/PCD/legacy0。主部署499944神经参数+六阈值48B，无额外bank/密度/投影；阶段墙钟不是B1时延。

默认N607，每卡至多新增2个本任务worker、启动free≥4096MiB，保护他人任务和所有历史输出。单owner不可覆盖root；源train首真实CP无query smoke PASS即继续2220→冻结→predict首smoke→CAL→四Q→CPUscore→全日志报告/Git。仅原最小流程的一次P0/P1、聚焦负测、一次发布archive SHA/compile/实时PID-CWD-argv-GPU-log核实，不加审批/哈希链。低分不当技术失败重跑。

当前环境说明：本轮准备时N607 GPU7设备句柄仍报Unknown Error；独立无模型、无数据进程以健康GPU1的完整UUID与PCI顺序初始化CUDA也失败。该事实不能证明故障根因或两者因果。未做重置、驱动修改或终止他人作业。环境恢复前只完成本地实现、合成测试、代码提交与登记，不发布新的真实训练；实际checkpoint smoke、训练、预测和评分尚待可用CUDA，不能据准备完成填报新性能。


code `d1e029abe74726f245e54f4bc5d7f91e1ae239f4`；计划Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；计划CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261101-phase3-ground-uniform-s2026092701-r01-release`；待资源恢复后command `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.ground_uniform_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261101-phase3-ground-uniform-s2026092701-r01-release/runtime.json`。


# 地面均匀 OE 候选：本地准备状态

**当前仅完成本地实现与验证，没有本轮实测性能。** N607 的 CUDA 初始化失败，真实训练尚未启动。完整 Phase3 目标保持 ACTIVE，未知拒识宏平均 95%、已知接收宏平均 95%、相对同 DURA 原 F0 闭集的已知正确接收损失不超过 2 个百分点，以及预登记主场景要求均不降低。

Run `20261101-phase3-ground-uniform-s2026092701-r01`，代码 `d1e029abe74726f245e54f4bc5d7f91e1ae239f4`。方法和固定矩阵见[候选计划](../../../docs/PAMF_PHASE3_GROUND_UNIFORM_PLAN_20261101.md)。

## 方法和待执行数据

单个 UNIFORM 学生从原 Tri-SIG-TIME SSL F0 完整权重独立初始化；默认六类头与骨干联合微调，Sinc 参数和 buffers 固定。损失为已知类 CE、原 F0 同输入教师的 T=2 蒸馏，以及按 15 个公共接收机/日期域等权的 KL(Uniform6 || softmax(OE))，三项权重均为 1。中心化配对风险仅记录诊断，不进入目标；不继承上一轮对比学生，不加新头、投影或新的目标端更新。

计划复用源地面 G(clean) 的 L=6300、OE=6134；OE 仅提供六类之外的成员监督和公共接收机/日期，不读取其真实身份或类别数。源侧五个接收机、三个日期、六个已知身份保持，源 Us 本轮新增使用为 0，目标 U 新增使用为 0。网络冻结后的已知 CAL=13500 与已知 TEST=13500 物理互斥，源未知 TEST=3150（三个身份各 1050）。每个源测试条件共 16650 条。

跨接收机仍使用已暴露的开发接收机 3-2/index34、未来两日期 query=10238，其中已知 4508、未知 5730（四个身份）。不称新盲测，正式 RX24-18 未访问。四条件为源接收机 clean、源接收机星地、跨接收机 clean、跨接收机星地，均连接固定 DURA v1.0。具体身份、日期、信道和计数列于计划。

固定比较 OLD_CPAIR_energy、OLD_CPAIR_centered、UNIFORM_energy、UNIFORM_centered，主报最后一项。两个旧方法与阈值精确复用；固定 2220 步、每步 L192/OE64。全部 16 个预测点完成后才独立连接四份真值评分。计划数值为训练学生前向 568320、预测 67276，**并非已经执行的数量**。

## 已完成的验证

五个模块共 15 个独立合成用例、85 个子检查按模块及修复后验证通过，覆盖损失、来源与标签边界、训练冻结、旧两列与原 F0 基线区分、阈值先冻结、完整预测后评分、部分输出保护和阶段衔接。早期缺模块 RED 与 core 浮点断言超差记录保留；后者仅把断言容差改为机器精度，生产数学未改。独立一次 P0/P1 检查 PASS，没有直接 P0/P1 缺陷。

这些证据只证明本地合成契约。真实 checkpoint 的无 query smoke 尚未执行，不能据此声称真实权重兼容、训练成功或指标提升。当前真实 checkpoint/IQ/真值读取、优化步数、训练与预测前向均为 0；未传输发布包、未执行远端编译或启动本轮真实实验。

## 资源故障及后续

2026-11-01T04:57:27+08:00 的独立检查显示 GPU 7 设备句柄报 Unknown Error；GPU 1 有空闲显存，但以其完整 UUID 在子进程导入 Torch 前选择设备，CUDA 初始化仍失败，尚未加载模型或数据。2026-11-01T05:18:27+08:00 只读枚举仍返回设备句柄错误。不能据此确认故障根因，也不能断言 GPU 7 导致其他卡初始化失败。

未重置 GPU、修改驱动、重启机器或停止其他用户作业。可继续真实实验的条件是 N607 CUDA 恢复，或取得另一个已获授权且具有同一数据与权重的可用节点信息；之后先重新核实资源，按既定首 smoke、固定训练、预测、独立评分流程继续。不因本次资源故障降低科学指标或修改实验候选。

最近一次已完成实验仍为[地面对比实测](../../../docs/PAMF_PHASE3_GROUND_CONTRAST_RESULTS_20261101.md)：主 OECON_centered 在源 clean 的 AK/CK/RU/H 为 97.962963%/98.592593%/67.968254%/80.254598%，未通过晋级。历史 CPAIR_centered 同条件 RU=73.968254%，其附报 energy RU=78.380952%；这些都是旧结果，不能填为本轮成绩。
