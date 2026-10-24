# 完整特征接收机外推

LOCAL_VERIFIED_NOT_LAUNCHED：110 项相关检查和一次独立 P0/P1 PASS；真实原 E20 检查尚待释放的匹配卡位，通过即丢弃探针、重载原权重继续。

三 seed 2026092701/02/03，各从原同 seed F21 E20/4440 独立执行原生10/2220＋20/4440，累计 E50/11100＝13L＋37LU。保留已验证 F0 物理数据契约 L6300/U56700/V27000，核对父来源/实际 cfg/payload；U TX 真值隐藏，V 只诊断、不更新模型/归一化/EMA/原型/头。N607 普通账户、CVS-RFFI 环境、FP32/TF32off、每 worker CPU2。命令、CWD、不可覆盖路径、GPU、停止规则和预期产物见 experiment.json。

源域固定网格 alpha=0/.25/.5/1，预先规定每 seed V 真值准确率≥99%且有效≥600，U 原伪类一致率≥99%且有效≥800、hardCE≥原1.25倍；仅 .25 通过。V 有效样本661/665/665，准确率100/99.699248/99.849624%；U898/902/903，原伪类一致率均100%，CE增至2.527858/2.852442/2.483095倍。U 真值正确率仍未知。较高损失仅说明这些固定源特征视图更难，不能证明真实信道保真或目标增益。已知 Joint/Complement 和更早目标结果；Path/ReliableAs/组合目标分数未读，本候选源依据和矩阵已固定。

以原 Joint6G 为配方，从原 E20 而非候选权重起步。前两次普通 U 已有 joint160 特征复用 sharedstyle 的 samepseudo/rawaccepted/differentRX 伙伴；有效位置 h'=h+0.25(h-hpartner)，保留完整签名/参考/主干。普通 U CE0.5 拆为 full0.25＋外推0.25，总 CE 分母1.25、LS0.01；无效伙伴保留原 CE 且差额精确为0，保留全部原接受样本。共享特征仍依参与项正常求梯度，不以 V 或 U 真值选伙伴。

保持 rawgate0.95、U权重6、六学生前向、ordinary/raw RDrop1、AsKL0、浅中层跨RX0.1/joint跨RX0.05、共享style、原L/defaulthead联合、完整L保护分组投影及范数帽1、nativeAdamW/clip5/decay；完整新增U梯度只进入guard一次，不保证AdamW步或目标性能单调。新增每步2次已有头调用，每模型6660步/13320调用/3402000条已有特征，实际有效≤2×accepted；无额外骨干/teacher/物理As/新头/推理参数。累计rawteacher2097900、studentU10602900、rawAs850500，逐seed分别报告父/新增/累计时间和实际有效计数，不称相同FLOP或独占GPU小时。

GPU5/6/7固定对应三seed，每卡本任务最多2条、计外部compute进程<3且free≥10000MiB。等待同seed ReliableAs全部GPU预测结束且PathReliableAs已派发，预留两侧pendingCUDA PID；CPU唯一scorer接组合终态。首可用匹配卡真实E20 smoke验证新CE对签名/参考/主干/头梯度均非零、2头512条、L768/U1536/rawteacher256/physicalU0/rawAs128/LPractical64、CE1.25/有效上限/完整L显式梯度；探针丢弃，正式重载原E20。两个同类系统性技术错误只停止新派发并保留健康任务；不按目标结果调参、重排或选择性重跑。

每row冻结最终E50学生后自动完成raw和固定DURA各clean＋六practical，168000/view、三seed42cell；整row预测固定后独立truth-last。报告全部逐seed/景/SD/MacroF1/TX/RX/day/RX×day/净纠错/成本，F20/F21同读出及预先固定Joint对照。raw clean相对两个控制均≥+1个百分点、六景均严格>+1.5个百分点才满足原目标，DURA单列。恢复器固定DURA v1.0 R69/266951、FP32/TF32off/reuse=True/graph=True/native=False，不用于新F训练，不沿用旧F50时延或替换默认部署。目标保持 ACTIVE。
