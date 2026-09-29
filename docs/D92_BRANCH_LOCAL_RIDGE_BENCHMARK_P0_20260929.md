# BranchLocalRidge冻结评测P0/P1审查

日期：2026-09-29。独立审查者：`local_env_reconcile`。仅审查实现、配置和合成测试；未读取真实query数据、预测分数或评测结果，未执行远端、Conda或Git提交。主Agent是唯一测试与启动owner。

状态：独立P0/P1审查完成，没有未解决的P0/P1问题。三基线汇总和元数据审计已纳入同一次审查；冻结核心及算法配置未改变。本结论针对代码、配置与合成测试，不代表真实query基准已经执行或通过性能目标。

## 发现

1. **已修复P1：预登记启动命令继承不可用解释器。**初次审查发现`configs/d92_branch_local_ridge_repeat_rx3_20260929.json:152`和`configs/d92_branch_local_ridge_repeat_rx1_20260929.json:150`的`execution.launch_command`使用`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe`，该路径此前已独立核实为失效链接。主Agent在生成器`:65`和两份生成配置中统一改为已验证的`F:/App/miniconda3/python.exe`，并在`tests/test_d92_branch_local_ridge_benchmark.py:66`加入断言。本审查已定点读回三处修复及测试断言；该问题关闭。没有修改原环境链接或系统配置。

## 已核对的关键路径

- `tools/evaluate_d92_branch_local_ridge.py`将当前split的support五块特征传入唯一冻结候选，逐query调用固定state，不做OOF、超参选择、query统计或跨query更新。完全复用冻结核心公式，物理class ID字典序平局与独立scorer一致。
- `tools/export_d92_branch_features.py:124`的既有加载器核对原BranchRidge缓存配置、特征公式、capsule、checkpoint、model seed、scratch及来源一致性、冻结前target访问为false、单物理观测和不可变encoder声明，以及NPZ成员和精确物理ID。仅读取received文件的ID成员，不读取IQ或truth。
- `tools/run_d92_confirmation.py`为四个固定model seed使用不同新输出，显式绑定每行原始feature缓存和原D92行；未进入encoder/export/GPU路径。全部行预测完成后才调用独立scorer，失败保留产物、不重试、不终止其他健康行。
- `tools/score_d92_confirmation.py`先核对完整split/query ID/class/预测与平局规则，再读取truth。query真值不会传入候选；旧基线预测验证属于独立输出完整性检查，不进入拟合。
- `tools/publish_d92_confirmation.py`的LocalRidge分支包含predictor、原始缓存加载器及其本地依赖、冻结核心和两种算法配置。已有release/run/archive拒绝覆盖；不因复用CPU缓存检查GPU空闲。
- 生成器保持两个cohort、每cohort四个模型、K1/5/10/20、新类0/2/5/10/20、原receiver/scene/support seed和capsule。rx3每模型900 split，rx1每模型300 split；候选覆盖4800个split，包含D92及DG的成对评分记录总数为9648。科学声明明确为已有target上的透明重复基准，不声称新独立确认。

- `tools/summarize_d92_branch_local_ridge_benchmark.py:53`核对LocalRidge、BranchRidge与Interaction的六个run、两组cohort、原source/checkpoint/capsule/seed/矩阵和预登记引用；`:225`在读取或散列任何scores之前要求六run全部完成并绑定startup。`:189`先验证全部六份覆盖，再聚合三组成对比较；D92与DG记录逐字典一致，方法名称保留真实身份。替代基线只用于显式analysis-only副本，不回写原记录。汇总保留全部K×新增类数表、包含旧类单独任务的old guard和联合任务H/new目标，不增加每个新增类数都必须正提升的门槛，也不把重复target或共享checkpoint称为独立新确认。
- `tools/collect_d92_branch_local_ridge_audit.py:35`核对每次完整support拟合的物理ID、冻结公式、tau/gamma/trace、残差、目标分解、实际0/1分解次数和五个数组及有效标量的状态字节。`:147`只读producer与candidate元数据，并对缓存文件取大小；不读取特征数组、IQ、truth、prediction或scores。`:228`核对trace/compact/stage及CSV完整覆盖，`:297`要求两cohort共4800次拟合、1200次K1拟合，保留退化时零分解的合法情况。资源报告区分历史缓存/模型存储、新增源数据传输、实际新增encoder调用、CPU工作时间和wall time；未知部署状态保留null。

## 验证与边界

相关合成测试已阅读，包括support-only拟合、query单条/排列不变性、禁止IQ成员访问、污染/错绑负测、退化精确平局、部分失败留证、缓存元数据仅读、CPU运行路径，以及六run屏障、全覆盖后汇总、完整旧类guard、审计元数据篡改和零分解负测。

主Agent提供的执行证据：入口与共享集成218项通过；三基线汇总初次44项通过（38.29秒）；随后新增已暴露target声明检查1项与元数据审计36项合计37项通过（3.99秒）。合计299项通过，分别为入口/共享218项、汇总45项、审计36项。本审查未重复运行测试；未读取真实query数据、预测或结果，未执行远端操作。此前本轮审查发现的失效启动解释器已修复并读回，没有剩余直接正确性阻断项。
