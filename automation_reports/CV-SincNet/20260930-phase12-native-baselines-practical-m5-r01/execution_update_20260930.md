# 原生对比实验执行记录（2026-09-30）

**已实现并实际启动，尚未完成最终评分。**2026-09-30 18:24:43（香港时间），N607启动首批8条源训练；18:26的独立进程、GPU和日志读回确认任务正在训练。当前45条源训练中8条运行、37条排队；45条评估流水线等待对应源权重。不能将这些任务写成已完成实验，不能提前给出目标域准确率。

## 已固定的矩阵

- 源模型：ProtoNet CDA、Feature Separation、DADDA、MRIOR、TwoStage、CSIL、MoPC-HR、Orthogonal Incremental SEI、RadioNet ADA。9个家族，每个5个模型seed，共45条；从零训练200轮，使用最后权重。
- Phase1：每条模型评估既有168000个目标物理样本的clean与satellite视图，按接收机及三个practical场景分层。
- Phase2：`K={1,5,10,20}`，新增类数`N={0,2,5,10,20}`，7个接收机、3个场景、5次support抽样；每条模型420次旧类适应、2100个既有split。新类注册继承相同旧类support上的B状态，旧类query物理ID保持配对。
- DADDA-SDA、MRIOR-SDA、TwoStage-SDA和Feature Separation FT报告A/B；其原生闭集方法不提供新类注册，C记N/A。CSIL、MoPC-HR、Orthogonal使用各自注册机制。ProtoNet原型追加、RadioNet冻结ADA编码器后的k-NN追加明确标为扩展。预计共79800条Phase2预测记录。
- 模型seed为`392005,2026092701,2026092702,2026092703,2026092704`。392005单列历史种子，主统计使用四个预先固定的新种子，同时给出全部五个种子的敏感性结果。
- 所有目标support/query使用同一`residual_noeq` capsule；Phase1增强也是25MHz、post_sync残差、无额外均衡。

已有CVS D92（E0、去RFRF32）、POSTER Fine-Tuning和RadioNet DF Fine-tuning等已评分批次保留。本轮RadioNet主路线采用论文推荐的ADA后冻结编码器并用cosine、距离加权k-NN（`k=K`），与DF Fine-tuning分开记录。固定1000次ADA更新与作者代码默认10000次不同，必须在论文中披露；当前批次不能声称完全复现作者训练预算。CSIL明确披露旧指纹掩码和源域经验Fisher修正，CSIL/MoPC的IQ编码器也属于数据协议扩展。

## 独立启动证据

|项目|已核实状态|
|---|---|
|代码发布|VERIFIED：`577825849c5c69edfbec9e258948b5621e4c643a`，远端分支OID与本地HEAD一致|
|发布包|VERIFIED：本地与N607 SHA256均为`a225442854f4eb739828ac3ee718a93de29b0b7f169f2ab40391523321b87b02`|
|远端编译|VERIFIED：不可变release中compileall成功，release_commit.txt读回一致|
|调度器|VERIFIED：PID `98355`；唯一owner `codex/root/native-comparisons-20260930`|
|GPU运行|VERIFIED：GPU0至7分别出现下列8个训练PID；每GPU当前一条训练|
|本地验证|89项测试通过，最后的日志样本暴露量改动另经12项source测试验证|

|GPU|首批row|PID|
|---|---|---|
|0|protonet-s392005|98407|
|1|protonet-s2026092701|98408|
|2|protonet-s2026092702|98409|
|3|protonet-s2026092703|98410|
|4|protonet-s2026092704|98411|
|5|feature_separation-s392005|98412|
|6|feature_separation-s2026092701|98413|
|7|feature_separation-s2026092702|98414|

读回的首批训练日志已经出现实际optimizer步骤、损失分量、梯度、学习率与样本暴露量；该证据只证明训练在进行，不代表收敛或目标域性能。源角色匹配、scratch来源及source-only权限在各row的初始化和resolved config中记录。

## 自动后续与恢复

调度器对每张GPU维持一个source lane和一个Phase2 lane，总训练任务不超过两个。源row达到200轮并写出真实checkpoint后，依赖评估才启动；加载前再次核对来源与输入契约，再执行真实checkpoint的support-only smoke。所有预测先固定，独立scorer随后连接truth；评分不反馈调参、种子选择或重跑。健康任务不因低性能停止，技术失败保留产物且阻止其依赖row，不自动重试。

Phase1具有独立完成标志和评分器，Phase2失败不撤销已固定的Phase1预测。最终自动产物包括Phase1结果、完整Phase2结果、配对A/B/C表、混淆矩阵、JSON/CSV、按seed分组的统计，以及训练/推理资源记录。未测量项记N/A。

- [完整设计](../../../docs/CVS_NATIVE_COMPARISON_DESIGN_20260930.md)
- [源训练登记](../20260930-phase1-native-baselines-practical-manysig-m5-r01/report.md)
- [本流水线登记](report.md)
- [独立状态快照](local_readback/20260930T1026Z/state.json)

远端release：`/home/szu2070436088/2510044040/CV-SincNet/releases/native_comparisons_20260930_r01`。

远端调度状态：`/home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/runs/20260930-phase12-native-baselines-practical-m5-r01/dispatcher/state.json`。

恢复时先读该状态、核实PID与完成产物，禁止再次运行launch造成重复实验。最终科学结果仍待真实200轮训练、全部预测与独立评分完成；本记录不预填成绩或完成时间。
