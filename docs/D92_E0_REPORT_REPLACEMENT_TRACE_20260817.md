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

## 纯E0报告裁剪（2026-08-17）

本轮目标：报告直接使用方法名`D92 E0`，只解释其实际执行路径；删除用于说明原完整D92、内部开发沿革或非E0模块的正文。论文复现方法仍作为外部对比对象保留，但不得借用非E0历史数值评价D92 E0。

|ID|要求|目标位置|状态|验证证据|
|---|---|---|---|---|
|E0-PURE-01|标题、摘要和正文只使用`D92 E0`方法名，删除另取方法名|标题、摘要、全篇|verified|`RTB-IDR`零匹配|
|E0-PURE-02|删除Fisher residual、Pareto替换和原子联合检查的对照公式与模块说明|模块六|verified|E0方法正文对应术语零匹配|
|E0-PURE-03|删除`P2-FULL`配对效果，只保留D92 E0自身screen结果|结果章节|verified|结果表仅保留D92 E0行|
|E0-PURE-04|删除历史完整D92、Role-Oracle和后续扩展说明|历史附录、证据边界|verified|相关标题和术语零匹配|
|E0-PURE-05|删除D81、D43、D46、D61、D62等内部方法沿革与映射|实现映射、证据来源|verified|开发编号零匹配|
|E0-PURE-06|删除类增量数值章节中不属于E0的历史D92数值|类增量数值对比|verified|非E0数值章节已删除|
|E0-PURE-07|计算量直接说明E0实际工作量，不再通过与原方法相减来定义|资源章节|verified|44次拟合和1.38–1.67GMAC直接口径|
|E0-PURE-08|保留MRIOR、DADDA、CSIL、MoPC等论文复现方法的机制和权限对比|对比章节|verified|域适应、类增量与公平比较章节存在|
|E0-PURE-09|结论、优势、局限和使用建议只陈述E0自身性质与证据边界|末章|verified|末章术语审计|
|E0-PURE-10|所有保留公式继续成对显示并紧邻符号说明|全篇|verified|Markdown结构审计|
