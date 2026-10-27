# REPL 强信道身份监督压力：源域三版验证

当前状态：已登记，尚未发布。只使用合法源clean IQ和直接叠加practical residual/post_sync/noEQ衍生IQ；不使用DURA恢复输出、教师或目标信息。三方法都是同一模型seed2026092701，不冒充三种子。各从原同契约单SIG1普通PL scratchE20/4440独立+10+20，累计E50/11100＝13L37LU，不从任何失败E50续训。完整PAMF、单SIG_TIME、原6×160默认头联合训练保持。

## 固定差异

保持已完成RX_TX配方：真实L先验、接受U伪类/RX频率EMA、0.5至2裁剪且接受均值1的CE/RDrop语义权重。raw95标签与接受集合、weight6/cap1、原L768包、U六前向、原ordinary/raw/RDrop和浅中层/联合跨RX学习保持；拒绝U的H互教关闭。只改变U physical H半批上两次已有mixed前向的CE权重，不改变raw/ordinary或L监督，不归一回原H总量。

| 方法 | 每条已接受physical H的CE权重 |
| --- | --- |
| REPL_HQC_Fixed | 固定0.5，检验单纯减轻压力 |
| REPL_HQC_Soft | clamp(p_H(raw伪身份)/0.5,0.05,1)，权重detach |
| REPL_HQC_Gate | 当前H的argmax等于raw伪身份，且其概率≥0.5时为1，否则0 |

质量计算来自已有训练态H logits，无新增F前向、模型EMA、分类头或推理参数。它只是训练课程信号，不能当作U真实正确率或独立身份校验。两H分别计算权重；原分母与1.25保持。完整记录eligible/nonzero/full/zero、H权重质量及语义加权质量，不能把raw接受率当成所有H实际有效曝光。

## 预算、冻结与验收

3×6660新更新、同原E50预算；已完成监督/普通PL、NoPair和RX_TX模型及源预测全部复用，不重训对照。源原F0 L6300/U56700/V27000，U TX隐藏/RXday可见。每版finalE50冻结后27000×raw/三practical源读出，完整后源V评分及RX/day/TX分层。全部三版完成后仍按raw≥26568/27000且H3≥62109/81000门槛，合格者raw、H3排序，完全同分Fixed/Soft/Gate；无通过不扩种子或目标。不把source通过当成原clean+1pp/六景>1.5pp目标达成。

精确命令、commit、环境/CWD和逐行路径在experiment.json/runtime中。GPU0–7按全部owner CUDA+pending<2/free≥10000MiB选择，最多三条本run学生。真实对应E20联合L/U backward smoke是第一步，三方法的实际六logit梯度与独立scalar核对，PASS后即继续训练。只允许最小流程规定的一次P0/P1审查、普通账户预检、一次归档SHA、远端编译和实际PID/CWD/cmd/GPU/log核实。首个输入/继承/finite/预算/技术失败停新派发，不自动重跑或停止其他owner；科学失败保留。


## 启动已核实

执行commit `459f17852eec0d57f2f3d9973f352e096222af6d`，一次归档传输校验VERIFIED、远端编译PASS。三版真实E20联合反向检查全部PASS：L768/U256/六U前向1536、原singleSIG/defaulthead、raw95身份/门控保持，实际六logit梯度对独立scalar误差≤2e−6。随后三条训练已经在GPU5（Fixed）、7（Soft）、1（Gate）启动；2026-10-27 00:33:52 UTC 独立核实PID/CWD/模块/训练参数/真实GPU，以及步骤日志增长，状态VERIFIED_RUNNING。

早期H权重日志已显示三种压力区别，但它既不是完整收敛分析，也不是U真实精度或测试收益。当前仍为一个模型seed的三方法源验证，未完成源性能门槛、不扩种子或目标。完整启动读回及实际权重计数见[launch_evidence.json](launch_evidence.json)。
