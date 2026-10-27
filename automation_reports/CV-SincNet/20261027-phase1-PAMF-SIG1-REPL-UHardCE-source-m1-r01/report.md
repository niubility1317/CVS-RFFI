# REPL：真实L监督与U身份目标分开验证

已登记，尚未发布。固定一个模型seed2026092701，从原同契约单SIG_TIME普通PL scratchE20/4440独立+10+20，累计E50/11100＝13L37LU。原完整PAMF和原6×160默认头联合训练；不读取失败E50初始化、DURA、U真TX或目标。训练只使用合法源clean和直接practical residual/post_sync/noEQ衍生IQ。当前三版HQuality及所有其他健康任务保持原配方。

## 唯一变化

在已完成RB_RXTX语义加权／双直接H／关闭拒绝H互教的配方中，六个raw95接受U的CE都从LS0.01变为普通硬CE。保留真实L的全部原监督与LS0.01、raw95身份和接受集合、Uweight6/cap1、原视图剂量和CE分母、ordinary/raw RDrop、浅中层与joint跨RX约束。没有HQuality门控，没有原型替换头，没有额外前向/教师/推理参数。

计算使用严格恒等式：CE(LS0)−CE(LS0.01)=0.01×（六类logit均值−raw伪身份logit），按原ordinary/mixed/raw系数0.25/0.25/0.125与原1.25分母及同一语义权重加回。这条更正无需额外CE/softmax；原RX_TX重分配仍有其原6CE/3KL计算。实际日志明确区分nativebase LS0.01与effectiveU LS0，不能把配置声明当执行证据。5项聚焦检查含完整hardCE/RDrop scalar及梯度对照。

已有12模型冻结源诊断发现极高置信接受项中LS与硬CE的logit方向常不同，它不证明LS对实际参数或目标有害。当前HQuality首轮虽减轻裁剪，分类头冲突仍在，因此本实验只检验一个独立目标类型假设，不以已有目标分数挑checkpoint或改正在运行的方案。历史普通硬PL与此完整六前向机制不同，已完成控制与源读出复用，无重复监督/PL训练。

## 预算和验收

新增6660步、30轮；源F0 L6300/U56700/V27000及物理ID不变。真实E20联合L/U backward无query检查先行，实际六logit梯度对独立hardCE scalar误差≤2e−6后立即训练。固定finalE50后27000×raw/三H预测，全部固定才做源V评分和RX/day/TX分层。原源门槛保持raw≥26568/27000且H3≥62109/81000；无通过不扩种子或目标，通过也只是允许之后独立登记三种子验证，原clean+1pp/六景>1.5pp及绝对门槛未放宽。

普通账户N607、GPU0–7剩余容量，全owner CUDA+pending<2/free≥10000MiB；固定不可覆盖root、一次归档SHA/远端编译、PID/CWD/cmd/GPU/日志读回。首个继承/角色/finite/预算/执行技术错误停本run新派发，保留所有输出且不自动重跑；科学负结果保留。实际执行commit、环境/CWD、命令、逐行路径、所有seed和完整预期产物见experiment.json/runtime。
