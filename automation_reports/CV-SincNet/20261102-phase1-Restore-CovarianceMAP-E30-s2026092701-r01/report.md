# 相关性先验物理恢复：时频Transformer估计信道与噪声、单阶段30轮

- run_id：`20261102-phase1-Restore-CovarianceMAP-E30-s2026092701-r01`
- group_id：`restoration-physical-full-covariance-source-operator-learning`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（真实源2 smoke与训练进程已核实）

## 目的与对照

新学习模块只预测源物理Action/噪声，冻结全512维源IQ相关性先验通过Gaussian后验恢复IQ；无学习IQ解码器或DURA祖先，30轮后自动同一地面TriSIGSSL Main七景。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 新结构和源端依据

本版重新定义恢复结构：时域/复频域patch4构成128tokens，宽128、六层四头Transformer预测64点复FIR、主/镜像调制与实部噪声方差，再用保存的统一源均值/C求解512维实Gaussian后验IQ。没有学习IQ解码器、DURA的coarse/prior/CG或旧G权重。1230341个学习参数全部从零，固定源均值/C为额外1050624字节FP32 buffer；求解使用FP64 Cholesky，输出FP32IQ。部署只用同条received和state内源统计，不读真实Action、噪声、F0、clean或其他样本。

依据是已完整复算的源协方差诊断：完整C整体Fclean96.601563%、低档城区87.743733%，相对DURA同条净+149/10240和+55/1436，预先固定支持条件满足。该诊断使用额外真实Action/噪声且统计拟合与诊断包重叠，不能外推Main、TX准确率或理论恢复上限。新实验独立检验盲信道估计和实际TX识别。

## 单阶段训练和权限

从零单阶段30轮/7410更新，原Ls6300Us56700、原R07二十draw再重复前十、原源顺序与LR。逻辑batch256、micro32累积loss.sum()/实际逻辑batch数，每更新只clip/step一次。直接以.1×完整512实basis Action矩阵相对Frobenius误差+.001×真实噪声logMSE训练估计器。真实Action/噪声仅为源模拟监督，不是TX标签。所有更新调用G.estimate，不执行后验求解、F0、教师、IQ或特征分支；本版不冒充F指导训练。源F0仅在源2真实smoke和冻结后的Main执行，方法假设来自源冻结诊断。

统一mu/C只从原63000LsUs clean拟合（a3a90b55数据统计），.001向同trace各向同性收缩、全512维保持；不是按TX/RX分类或原型。FP64统计转存FP32 buffer的实际舍入量记录，buffer全程固定。没有G checkpoint祖先、TX/V/OE/novel/目标更新、早停或选模，OE只冻结诊断。源2固定draw1前两个非clean场景实例，首步完整后验+原F0有限、算子head真实梯度与E0严格重载PASS后立即30，不以源质量取消Main。

## 固定正式验收与资源

原地面完整Tri-SIG-TIME SSL/3ff8/E50-11100/defaulthead/Main，cross_rx126000加cross_both42000，逐row clean+六景全部预测后独立CUDA空CPU truth-last。raw/DURA只复用同ID的r03既有预测，候选全部新预测，不合成最佳场景。每景距同一raw clean≤3个百分点，两个总体各自保持clean；combined门槛140345/168000=83.538690476%。无目标反馈、阈值拟合、选择性补跑或默认DURA变更。

训练和两组预测依次运行，GPU4/6按实际UUID/free≥12000MiB/existing compute≤2/新增1总≤3准入，本row最多一条NN。其他owner和已存产物不改。5项针对性检查与一次独立P0/P1通过，一次archive SHA传输核实和远端compile；真实启动后检查PID/CWD/cmdline/GPU/log增长。技术失败保存产物，不自动重试、迁移或复位GPU。

静态估计器176636032实MAC，后验额外269221888实MAC，512维Cholesky约44739243FLOPs、三角求解等操作另列，另2个complex256点FFT。不是实测时延，不以更少参数或关闭训练分支声称同质量提速。训练不执行后验会减少迭代成本，但实际轮时、峰值显存、完整预测耗时和正式性能仍须实测。完整配置/命令/路径/预算/停止规则/产物见登记。


## 实际新模型训练已核实

VERIFIED / RUNNING。实际source2完整FP64后验+原F0、算子head梯度与E0严格重载均通过，已进入固定单阶段30训练。CPU控制器和worker的PID/CWD/commit/CUDA/GPU与日志增长独立核实；GPU4准入free22885MiB/原2compute/新增1。固定源mu/C已绑定为额外数据统计buffer，更新只执行算子/噪声估计，不执行后验或F0；0TX/V/OE/novel/目标训练。archive单次SHA一致/远端编译通过。完整30后自动原Main2pop×7 truth-last，当前没有本版正式性能。

E01 CPU独立检查PASS：实际E0全部学习状态等于新factory，mu/C等于原FP32源统计且E01不变，TF骨干及三个Action/noise输出head已更新，optimizer有限；30份sourceorder/draw与原M20完全一致，第一轮247更新/63000记录/微批32/LR和加权loss核实。首轮66.222361秒，不视为独立速度或性能结论。
