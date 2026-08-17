# D92 E0技术报告替换追踪表

日期：2026-08-17

目标文件：`docs/D92_METHOD_COMPLETE_REPORT_20260727.md`

目标：将原“完整D92”技术报告改写为完整的`D92 E0`方法报告。这里的`D92 E0`严格对应全量消融冻结arm`P2-E0`，不是对原报告进行名称替换。

## 权威定义

`P2-E0`在冻结实现中的方法身份是`d92_d81_d46_without_fisher`，配置差异为`fisher_profile=off`。它保留下列计算链：

1.固定接收IQ到288维联合特征；
2.由Phase1封存聚合知识构造类无关扰动谱并稳健化support中心；
3.旧/新任务自动收缩协方差与固定等权合成；
4.full/block3等先验LDA头；
5.support内按shot秩留一可靠性融合；
6.量化编译、封存和逐query统一仿射判决。

`P2-E0`明确关闭下列计算：Fisher残差方向、Fisher增益、残差候选头、逐类Pareto替换门和原子联合检查。

## 设计—报告映射

|ID|要求|报告位置|状态|验证证据|
|---|---|---|---|---|
|E0-01|标题、摘要和方法定义使用`D92 E0`|标题、摘要、阅读约定|verified|术语检索|
|E0-02|说明E0是完整可运行配置，不是“缺失模块的D92”|阅读约定、模块六|verified|冻结arm定义|
|E0-03|删除Fisher残差作为活动方法步骤|流程图、符号表、伪代码、状态构造|verified|`fisher_profile=off`|
|E0-04|第六个活动模块改为量化编译与状态封存|模块六、资源章节|verified|冻结量化状态格式|
|E0-05|注册拟合次数从含Fisher的88次口径切换为E0基础融合口径|计算量章节|verified|`d92_d81_d46_without_fisher`调用链|
|E0-06|实验结果采用`P2-E0`同排矩阵，不把`P2-FULL`或旧D92结果冒充E0|结果、局限、结论|verified|2026-07-31完整矩阵分析|
|E0-07|对比方法中的D92机制描述移除Fisher安全融合|公平比较和资源表|verified|方法定义一致性检查|
|E0-08|保留历史实现名时明确其仅为复用来源|实现映射、证据来源|verified|代码审计|
|E0-09|每个新增或保留公式紧邻符号说明|全篇受影响公式|verified|Markdown结构检查|
|E0-10|使用`DA0_REG0/DA1_REG0/DA0_REG1/DA1_REG1`解释因果状态|指标与结果边界|verified|当前项目报告规范|

## 证据边界

- 方法定义证据：`code/cvsrffi/stage2_ablation_factory.py`、`code/cvsrffi/stage2_ablation_executors.py`、`code/cvsrffi/full_ablation_spec.py`。
- 设计证据：`paper/ieee_transactions_draft_20260727/experiments/CVS_FULL_ABLATION_DESIGN_PHASE1_PHASE2_20260728.md`。
- 性能证据：`E:/type10-7/automation_reports/CV-SincNet/cvs_full_ablation_completed_matrix_analysis_20260731/report.md`。
- `P2-E0`当前证据是75个identity、225个场景单位的screening结果，不是fresh confirmation，也不是飞行级部署证据。
- 后续`E0D`、`E0OCF`、`CCOC`和连续session路线不是基础`D92 E0`本身，不并入本报告的方法定义或主结果。
