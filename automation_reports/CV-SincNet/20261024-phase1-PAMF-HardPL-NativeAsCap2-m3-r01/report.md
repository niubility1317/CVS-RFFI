# 完整原生 As 配方的 U 梯度范数帽对照

LOCAL_VERIFIED_NOT_LAUNCHED：29 项 cap2／梯度检查、119 项受影响流程检查、一次独立 P0/P1 检查全部通过。真实原 E20 检查待首个匹配释放的 NativeAs 卡位；通过后丢弃探针、重载原父权重继续，当前没有本候选 GPU 训练。

原同 seed F21 E20/4440，各自独立执行 native10/2220＋20/4440，累计 E50/11100＝13L＋37LU；三个 seed 为 2026092701/02/03。只加载已核实契约及完整继承来源的原 E20，不加载候选权重或原 E30。L6300/U56700/V27000 与 F0 未见 RX 物理划分不变；U TX 真值隐藏，V 仅诊断，不更新模型、归一化、EMA、原型或头。N607 普通账户、CVS-RFFI、FP32/TF32off、每 worker CPU2；命令、CWD、路径、GPU 和停止规则见 experiment.json，发布后核实执行 commit。

固定源计划 44d38cc9、完整源结果与设计 28431a31：只对 Joint 全部 19980 条既有源梯度记录计算预定 cap1／cap2，不读新 IQ、权重、U 真值或目标数据，不新增模型前向。三个 seed 均通过预登记规则；t2 平均 appliedU/L 算术提高 34.88% 至 51.62%，全参数组提高 27.83% 至 29.36%，计算出的 combinednorm>5 频率增加 0 至 0.1352 个百分点。该结果属于相同当前梯度的算术，不是新训练轨迹、实测 clipping、AdamW 更新或目标收益；Joint 不含完整 nativeAs，向 NativeAs 转移必须独立训练和测试。源计划、完整摘要及预先设计随报告保留。实际 U 正确率 UNKNOWN，不能用源 V 的高精度代替。

相对固定 NativeAs cap1，只将 t2、cls_head、other 三组 U/L 范数帽从 1 改为 2。仍以完整当前 L 目标为 anchor，对负点积 U 方向投影，tiny-L 平方范数阈值 1e-12，完整 U 梯度只保护一次并接回原 native backward。保持 raw argmax／0.95、U 权重 6、六学生、原生 As 半批、LS0.01、ordinary/raw RDrop1、As KL0、浅中层跨 RX0.1、joint160 跨 RX0.05、共享 style、CE 分母1.25、原 L／默认头联合训练和 nativeAdamW／clip5／decay。无额外 autograd、教师、学生或既有头前向，无新头或推理参数。投影后的非负点积不保证优化器更新、L 损失或目标性能单调。

每模型新增 6660 步，U 曝光 1701000、学生 U10206000；累计教师2097900、学生 U10602900。As 生成850500、物理学生1701000，有效接受 As CE 视图为2×rawacceptedhalf，raw 回退0；接受集合及全部接受样本保留。源算术 CPU1／65.676 s 成本单列，不称等实测时长。

GPU5/6/7 固定对应三个 seed，每卡最多两个本任务，计外部 compute 进程小于3、空闲显存至少10000 MiB。等待同 seed NativeAs 全部 GPU 预测完成且 BalancedCE 已派发，两侧 pending CUDA PID 均预留。首个匹配卡位真实原 E20 检查核对 L768／U1536／rawteacher256／As128／物理学生256／Practical192；逐组 U≤2L、负点积投影、tiny-L、完整 L 显式参数梯度、一次 guard 和临时 hook 退出全部符合，且至少一组实际 U>L，证明 cap2 确实生效。唯一 CPU scorer 接 BalancedCE 终态。两个相同系统性技术错误只停止新派发，保留健康任务和产物；不按目标分数停止、调参、重排或选择性重跑。

每 row 最终默认学生冻结后自动预测 raw 及固定 DURA 各 clean＋六 practical，168000/view，三 seed 共42 cell；整 row 预测固定后独立 truth-last。报告逐 seed／景、SD、MacroF1、TX／RX／day／RX×day、净纠错及实际时间。NativeAs cap1 为预定匹配比较，F20/F21 为原主控制；raw clean 对两者均至少高1个百分点、raw 六景平均对两者均严格高于1.5个百分点才达到目标，DURA 单列。固定 DURA v1.0 R69／266951参数／FP32/TF32off／reuse=True/graph=True/native=False，不用于新 F 训练，不套用旧 F50 时延，不替换部署默认。此前完整目标曝光如实保留；当前设计未读取 RXE／NativeAs／BalancedCE 新目标分数，亦不作独立盲测声明。目标保持 ACTIVE。
