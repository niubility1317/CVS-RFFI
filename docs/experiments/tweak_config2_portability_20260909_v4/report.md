# Tweak配置可移植性复现实验V4：严格hard-mining与高效学习率选择预登记

- run_id：`tweak_config2_portability_20260909_v4`
- 当前状态：`RUNNING_HEALTHY_THROUGH_LATEST_PROBE`
- 唯一launch owner：Codex主Agent
- Git提交：`d5472a9403417c7e88b0f643e0ff7a07d2568b07`；已push且远端分支OID独立一致。
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

## 已预登记的N607落地

- 远端项目根：`/home/szu2070436088/2510044040/CV-SincNet`；数据根：`datasets/tweak_official_lora_configurations_20260908/Diff_Configurations_Setup`（80个所需输入文件已只读复核）。
- 新release：`releases/tweak_config2_portability_20260909_v4/source`；新run：`runs/tweak_config2_portability_20260909_v4/official_config2_full`；新log：`logs/tweak_config2_portability_20260909_v4/train.log`。预flight已确认三者均不存在，故无覆盖风险。
- 资源选择：GPU3（启动前探测约3.7GiB/24GiB）；既有用户许可允许本任务在每卡加挂实验，但不会触碰其他任务。实际PID/CWD/cmdline/GPU/log增长必须启动后独立读回。
- 计划命令：`CUDA_VISIBLE_DEVICES=3 PYTHONPATH=<release>/source <CVS-RFFI-python> -m paper_reproduction.gaskin_tweak_2023.official_lora_experiment --data-root <data> --output-dir <run> --device cuda:0`；不传smoke限制，使用默认100epoch与1+100选择方案。
- 源归档：提交`4958069a89003e171f1d45b88f73628dd05640c5`导出`source.tar`，本地与远端SHA-256均为`ff1a48173e50390e139c94232d762ab3ee221d6cb037e703cb5821ebac0dc5c2`；远端解包及三个改动模块编译通过。
- 远端独立CUDA读回：真实Config2批为`[64,2,128]→[64,12]`，strict-hard loss=`0.32596352696`有限，16组梯度存在；向量化采样200批=`0.1342s`。这证明启动前运行链路，不等于实验结果。
- 启动：2026-09-09 16:13 CST，以预登记完整命令启动PID=`358516`。15秒后独立probe显示PID存活（PPID=1、CPU125%）、CWD与cmdline均为本V4 release/run、CUDA可见进程占446MiB；log尚为0字节符合每epoch才打印的实现，无异常或最终产物。该证据仅证明初始健康。

## 本地验证读回

- 先红：新增的strict-hard和1+100学习率选择测试在实现前因符号缺失而收集失败；实现后`tests/test_gaskin_tweak_2023.py`、`test_gaskin_tweak_official_lora.py`、`test_gaskin_tweak_pipeline.py`、`test_paper_method_configs.py`共29项通过，三个修改模块的`py_compile`通过。
- 真实官方Config2数据只读前反传：10条记录的首个平衡批为`[64,2,128]`，encoder输出`[64,12]`，strict-hard loss=`0.3538506329`且有限，16组参数取得梯度；本次未创建或覆盖任何实验输出。
