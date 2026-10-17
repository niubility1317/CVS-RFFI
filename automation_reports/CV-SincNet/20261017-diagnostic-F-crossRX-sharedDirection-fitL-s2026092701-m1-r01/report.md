# F＋跨 RX 同 TX：共享微弱加性 IQ 方向源检验

状态：ANALYZED_SOURCE_ONLY／VERIFIED；FAIL_FIXED_SHARED_DIRECTION_CLOSED。代码、数据角色、操作幅度与判据在真实源输入前固定。训练基线为 F＋跨 RX 同 TX 对比学习、scratch E20；本行仅冻结已有同契约 final20/default cosine 头，不训练。

## 唯一问题与矩阵

共享微弱加性方向是否可从其他源 RX/day 迁移到留出 RX/day，并比同能量重排产生更明显的身份判决变化？单行、seed 2026092701，原始／方向／重排三个视图。方向对每折全部 TX 相同，生成器不读标签。

仅原 fit L5040：RX3/4/6/8、day1/2/3、六 TX、每单元70包。12折各留出一个 RX/day 的420包；参考2520包同时排除该 RX 和 day。模型曾训练这些 fit L，故这只是方向迁移敏感性，不是新模型盲测。U/V/RX1/target 不参与；原 opaque 物理角色ID原样匹配，不解码猜测。cache可能反序列化其他源角色，但只有fit L进入计算。

## 固定算子、判据与成本

原nativeRMS/EQ1/IQ256/25MHz，NoExplicitCFO及内部补偿保持。一次共同输入探针 xi=1e−4复RMS，用同一个冻结模型eval的无标签概率KL求一个共同输入梯度；全参考包平均，去I/Q各自DC，缩放到epsilon≤1e−3复RMS、分量峰值≤3e−3，FP32相对容差1e−6。不剪裁原IQ，不再RMS；同能量控制仅分别重排新增向量的I/Q时间位置。0梯度记录不可辨识，不随机回退。

全部要求：raw准确率≥98%；permuted−direction准确率≥2个百分点，且至少3/4 RX各≥1个百分点；direction在raw正确固定集合的保持率整体、每RX、每TX均≥95%；permuted整体≥99%、每RX≥98%；12方向可辨识，所有样本/视图完整、预算准确、无target。失败完成全12折后关闭epsilon/xi/峰值/迭代/模型/RX/day/阈值/seed，不调参救跑。通过也仅为敏感性依据，不自动启动E20。

预计参考60480包前向、30240包输入梯度贡献／240次调用，评估15120包前向，真实checkpoint无query合成smoke2包，合计75602新NN包前向。0模型epoch/参数更新/新teacher/头拟合/As生成。历史父模型E20/4440更新/4686678NN另列，不重复训练或作为新训练初始化。

## 发布、验证与停止

维护配置及逐行矩阵唯一源为 [experiment.json](experiment.json)。实际代码commit由publisher读取HEAD并写入release_commit.txt/resolved_config，报告最终回填。普通N607既有CVS-RFFI环境，CPU2、CUDA隐藏、loader0；CWD为PROJECT/releases/cvs_F_sharedDirection_fitL_20261017_r01，命令为CVS-RFFI/bin/python -u experiments/cvs_shared_iq17/runner.py。独占输出根为PROJECT/runs/20261017-diagnostic-F-crossRX-sharedDirection-fitL-s2026092701-m1-r01；owner为codex/root/F-sharedDirection-source-20261017。精确私有运行路径保存在原登记/发布证据中。

一次普通账户preflight、一次archive传输比较、一次remotecompile；首步匹配真实checkpoint no-query smoke PASS后直接进入源执行。独立读回原PID/CWD/argv/隐藏GPU/日志增长。错误角色/来源/非有限/状态或RNG变化/预算/输出冲突均保存失败产物，技术失败不自动重试，不触碰健康旧进程。科学失败完成全部折后关闭固定问题。预期保存smoke、resolved、角色/折、12份全部logits/默认头判决/新增向量/计数、progress、result/completion或failure、完整stdout，独立全量复算与同源报告。

本地相关反例15＋19＋24＝58项通过；真实F结构且参数冻结的合成检查通过。累计研发验证168合成NN包前向／72输入梯度贡献，0真实IQ/CP/参数更新/target。首次CLI检查暴露operator.py遮蔽标准库，已改名iq_direction.py；唯一P0/P1审查发现publisher旧child入口，修正后原审查者仅核对该修复点PASS，其余范围无直接P0/P1。审查不读取IQ/CP或执行远端。

## 声明边界

这是源敏感性诊断，不是身份保持证明、接收机替换、Ag有效或practical提升。完整Ag/As四因子、跨seed、正式clean＋六practical以及联合目标尚未完成。另一对话的量化残留路线不在本任务修改范围。

## 完整实际结果

执行代码commit：361978e3449d254258b9b61141508a0ed4c945ab。原进程自然结束，独立原PID/CWD/argv/CPU配置与日志376→1835字节增长、smoke、12折completion均已核实。完整15,120个logit/default-head判决、5,040个opaque物理ID、216个视图×TX/RX/day单元、每折2520参考与420留出、同能量控制和实际成本均独立复算一致。

|视图|正确／总数|准确率|Macro-F1|raw正确集合保持率|修复／破坏raw判决|
|---|---:|---:|---:|---:|---:|
|raw|5032/5040|99.841270%|99.841222%|100.000000%|0/0|
|direction|5030/5040|99.801587%|99.801421%|99.940382%|1/3|
|permuted|5032/5040|99.841270%|99.841222%|100.000000%|0/0|

|源RX|raw准确率|direction准确率|permuted准确率|permuted−direction（百分点）|
|---|---:|---:|---:|---:|
|3|99.603175%|99.444444%|99.603175%|0.158730|
|4|99.920635%|100.000000%|99.920635%|-0.079365|
|6|100.000000%|100.000000%|100.000000%|0.000000|
|8|99.841270%|99.761905%|99.841270%|0.079365|

总体方向差异仅0.039683个百分点，未达到2个百分点；达到每RX1个百分点者0/4，未达到3/4。其余原判据通过，包括raw≥98%、12/12方向可辨识、固定raw正确集合的保持率、全部预算及无target。direction整体保持率99.940382%，最差RX99.840637%、最差TX99.760479%；permuted无判决变化。由此关闭本固定方向，不将高fit准确率/梯度存在/微扰保持当作Ag证据。

12个新增向量复RMS范围9.9999999677e−4至1.0000000031e−3（在原FP32相对容差内），最大分量峰值2.3203126621e−3。逐I/Q重排的值集合完全相等，DC、能量和边界完整。原IQ功率及相对扰动比保存在每折私有证据，没有依此调整幅度。

实际参考60,480包前向、30,240包输入梯度贡献／240次调用，评估15,120包前向，真实CP no-query smoke2包，合计75,602新NN包前向；源诊断65.279767秒。0模型epoch/参数更新/新teacher/头拟合/As生成。既有父E20/4440更新/4,686,678 NN与本地研发合成168 NN／72输入梯度贡献分别列示，未隐匿额外计算。

本地独立复算CLI通过；图在原普通N607既有matplotlib环境生成并已查看，未新增NN或修改环境。最初本地绘图环境缺matplotlib，保留错误日志；已将图形选项独立于核心复算，未重跑源推断。源原始日志、所有数值、方向、predictions与PNG/PDF保留在local_artifacts/F_shared_direction_20261017/final和原远端独占输出根；未回收原始IQ或checkpoint，未将私有运行信息、预测或原始日志提交Git。

本轮唯一假设及最小三视图12折实验已被原放弃条件否定。没有晋级E20的新Ag候选；不得改变epsilon/xi/峰值/迭代/模型/RX/day/阈值/seed救跑。下一步需提出与关闭路线不同且有源依据的新问题，尚未固定或发布新的训练。联合目标继续ACTIVE：Ag、稳定As、同契约四因子交互、跨seed及正式clean＋六practical均未完成。
