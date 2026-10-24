# 高权重伪标签与监督分支同时保护

LOCAL_VERIFIED_NOT_LAUNCHED：40 项分支／投影检查及 133 项受影响流程检查通过。独立训练 P0/P1：PASS。真实原 E20 GPU 检查等待首个匹配释放的 BalancedCE 卡位；检查通过后丢弃探针、重载原父权重直接继续。

源设计 2743f9aa 先于 NativeAs 目标读取：原 E20 三 seed 各一个 L128/U256 源步，在同一导数上同时约束完整 L、非物理 L 和物理 L，可保留至少99.744189%的完整L cap2整体U范数并移除9个实质分支冲突。该旧 Gram 算术结果只支持源域几何假设，不冒充新 QR 训练或目标收益；新数值实现待真实 GPU 检查。实现中近线性相关的监督头边界测试触发旧法失败，现用薄 QR 和小矩阵 SVD 枚举至多8个 active set，实际参数dtype检查通过后选最小改动候选，避免正规方程平方条件数。pinv截断仍1e-10但作用于线性矩阵；约束、实际余弦1e-6、cap2、tiny-L平方范数1e-12及原源规则保持。没有任意归零回退或结果后网格调参，技术差异完整记录在 numerical_implementation_note.json。

相对固定 NativeAsCap2，只改变 U 梯度方向算子。完整 L 不变，复用原L hook四对输出：非物理L为0.5ordinary＋0.25nonphysicalmixed＋0.25raw（含原RDrop），物理L为0.25physicalmixed（AsKL0）。计算完整L/U及两个L分支共4次导数，比原cap2多2次/步、每模型13320次。用原显式参数梯度桥接接回 native backward；保持rawargmax/0.95/U权重6/六学生/原生As/raw回退0/LS0.01/ordinary与raw RDrop1/AsKL0/浅中层跨RX0.1/joint160跨RX0.05/共享style/CE1.25/原L/defaulthead联合/nativeAdamW/CENTER_UNITS/clip5/decay。无额外教师、学生或头前向，无新推理参数。非负梯度点积不保证AdamW步、各项损失或目标性能单调。

只加载原同seed F21 E20/4440，实际角色、物理ID、配置、payload及完整从零继承来源核实后，各自独立native10/2220＋20/4440，累计E50/11100＝13L＋37LU；不用候选权重或原E30。seed2026092701/02/03，原L6300/U56700/V27000及F0未见RX物理集不变。U真值隐藏，实际U正确率UNKNOWN；V仅诊断，无模型/归一化/EMA/原型/分类头更新，无目标或G参与新F训练。每模型累计rawteacher2097900、学生U10602900，新增As850500、物理学生1701000、有效接受As视图2×rawacceptedhalf；全部原接受样本保留，生成/前向成本包含拒绝样本。额外导数与共享GPU经过时间单列，不称相同计算成本。

N607普通账户、CVS-RFFI、FP32/TF32off、每worker CPU2，CWD为PROJECT/releases/20261024-phase1-PAMF-HardPL-BranchGuard-m3-r01。完整命令、路径、资源和停止规则见experiment.json。GPU5/6/7固定三个seed，每卡最多两个本任务训练/预测，计外部compute进程小于3且空闲至少10000MiB才派发。等待同seed BalancedCE所有GPU预测完成且Cap2已派发，预留两侧pendingCUDA PID；唯一CPU scorer接Cap2终态，不停止或热改现有健康任务。首卡真实原E20检查验证L768/U1536/rawteacher256/As128/物理学生256/rawAs0/Practical192、完整L显式梯度、4个Lpair、4次梯度加1次joint检查导数、三个非微小锚点、U≤2L且至少一组U>L、所有临时context退出。两个相同系统性技术错误只停止新派发并保留健康任务和产物；不按目标性能停训、调参、重排或选择性重跑。

每row最终E50默认学生/默认头冻结后，自动预测raw及同固定DURA v1.0各clean＋六practical、168000/view，三seed共42cells；整row的全部14项决策固定后独立truth-last。保留逐seed/view、均值/SD/MacroF1、TX/RX/day/RX×day、净纠错和实际时间。主匹配固定NativeAsCap2检验新增分支约束，次匹配NativeAs cap1同时含范数帽变化；不混作单因素。主控制F20/F21：raw clean对两者均至少+1pp，六景平均对两者均严格>+1.5pp，分别至少433348/504000和2180190/3024000正确才达目标；DURA参考单列。DURA固定R69/266951/FP32TF32off/reuse=True graph=True native=False，G原训练sourceLU/TXhidden/Ffree；不用于新F训练、不套用旧F50时延、不换部署默认。

NativeAs正式目标84.831944/72.264385与DURA84.944444/77.428439已知；本轮按此前固定源路线实现，数值修复来自合成边界失败。BalancedCE/Cap2目标未读。早先目标暴露如实保留，不作新的独立盲测声明，不从目标挑父模型、参数、seed或checkpoint。当前没有本候选训练或目标结果，goal ACTIVE。


发布后独立核实：VERIFIED，执行 commit `61b942a4d922d41ff235707534169074293ec565` 的控制器存活，CWD 和完整命令匹配；三行实际 QUEUED，新增 GPU worker 为 0，真实原 E20 检查 PENDING。等待对应 seed BalancedCE 全部 GPU 预测结束且 Cap2 已派发；独立核实两侧控制器及现有运行行的 PID/CWD/命令/GPU/日志增长。保留全部健康任务；排队不代表训练成功或达到目标。
