# Tweak配置可移植性复现实验V4：严格hard-mining与高效学习率选择预登记

- run_id：`tweak_config2_portability_20260909_v4`
- 当前状态：`LOCAL_VERIFIED_RELEASE_PENDING`
- 唯一launch owner：Codex主Agent
- 前序产物：V1数组接口技术失败、V2完整但表征塌缩、V3在确认方法定义不符后由唯一owner停止；三者均保留，V4绝不复用任何旧输出根。

## 已冻结的实验边界

- 只使用已核验的公开Config1—4、设备ID1—10和80个`cf32`文件；Config2训练，图13b的4×4单域校准矩阵和图14的四配置联合校准。不会运行vanilla、消融或额外硬件实验。
- 保持原始`[2,128]`IQ、Fig.8的12维encoder、Euclidean triplet margin=.1、SGD momentum=.9、batch=64、每条传输75/25、`N=11,718`、`M=10`。物理帧角色仍为seed=20260908的全记录仿射双射，此顺序是`PAPER_UNDERSPECIFIED_SPLIT_CHOICE`，不冒称作者实现。

## 本次唯一方法与效率修复

| ID | 论文要求或已授权方案 | 实现目标 | 状态 | 验证门槛 |
|---|---|---|---|---|
| T3 | IV-A：仅选择`d(A,N)<d(A,P)`的hard triplet，之后计算Eq.(1) | 一个均匀同类正/异类负候选；严格距离过滤；没有strict-hard候选则不执行optimizer step | verified locally | 先红后绿单测：margin为正但`dAN>=dAP`被排除；严格候选被保留；真实Config2批次得到finite strict-hard loss和16组梯度。 |
| T7 | 不改变每anchor候选类别分布 | 使用GPU向量化掩码、累计秩选择候选，消除逐anchor`.item()`同步 | verified locally | 正例同类非自身、负例异类单测通过；N607同环境采样基准留待release前复核。 |
| T8 | VI-A：在`1e-2`至`1e-6`搜索以确保loss下降，100epoch并保存最佳epoch | 五个fresh source-only一epoch probe；仅在有strict-hard候选且末窗口loss低于首窗口时入选；最低probe均loss的LR以fresh状态训练100epoch | verified locally | 先红后绿选择单测通过；runner记录全部probe及选择理由；正式probe不读取Config1/3/4或测试帧。 |

## 运行与解释规则

- V4仅在本地聚焦/完整测试、模块编译、真实官方Config2前反传、Git提交/push/远端OID读回、N607源/数据读回全部完成后，以新的release、run、log和output目录启动。
- 训练期间只会因可复现的技术错误停止；低指标将完整保留并做数值解释。正式结果出现前，不声称与论文Fig.13b/Fig.14一致。

## 本地验证读回

- 先红：新增的strict-hard和1+100学习率选择测试在实现前因符号缺失而收集失败；实现后`tests/test_gaskin_tweak_2023.py`、`test_gaskin_tweak_official_lora.py`、`test_gaskin_tweak_pipeline.py`、`test_paper_method_configs.py`共29项通过，三个修改模块的`py_compile`通过。
- 真实官方Config2数据只读前反传：10条记录的首个平衡批为`[64,2,128]`，encoder输出`[64,12]`，strict-hard loss=`0.3538506329`且有限，16组参数取得梯度；本次未创建或覆盖任何实验输出。
