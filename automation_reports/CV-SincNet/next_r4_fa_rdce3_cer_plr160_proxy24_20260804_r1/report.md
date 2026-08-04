# NEXT-R4 FA-RDCE3×CER-PLR160 Proxy24实验报告

状态：`IMPLEMENTING / NO_PERFORMANCE_RESULT`

## 1.实验身份

- 实验ID：`next_r4_fa_rdce3_cer_plr160_proxy24_20260804_r1`
- 日期：2026-08-04
- 主agent：`gpt-5.6-sol/high`
- 科学实现/审查：`gpt-5.6-terra/max`
- 唯一N607 runner：冻结提交、命令和路径后由`Luna/max`接管；当前尚未发布
- 目标：验证共享3维Fisher锚定域位移`FA-RDCE3`与轻量残差头`CER-PLR160`能否同时改善域适应和新类注册，而不恢复D92的不必要稠密计算
- 对照：同一row、同一query、同一K下的qKNN基座Q，以及四态内的配对差值

## 2.统一性能状态

|状态码|唯一中文主名称|主指标|
|---|---|---|
|`DA0_REG0`|域适应前/新类注册前|old BA、old-floor、总正确数；seen-new/H=`N/A`|
|`DA1_REG0`|域适应后/新类注册前|old BA、old-floor、总正确数；seen-new/H=`N/A`|
|`DA0_REG1`|域适应前/新类注册后|old BA、seen-new、H、all-floor、总正确数|
|`DA1_REG1`|域适应后/新类注册后|同上；联合主结果|

日志、prediction/score artifact、CSV/JSON、表格和结论必须同时保存`state_id`与中文主名称。REG0的`seen_new_acc`和`H_old_new`写`N/A`而不是0。禁止只写before/after；任何提升必须给出同row或同聚合层的起点、终点和差值。

## 3.冻结假设与矩阵

- 候选：`NEXT-R4-FA-RDCE3-CER-PLR160`
- 协议：`p2_min_v1`
- 语义：`phase1_seen_class_loco_directional_proxy`，不构成正式新类注册声明
- receiver：`1-1`、`18-2`
- held-class：每个receiver 6类，运行时唯一排序
- K：1、5；K1是K5逐类support前缀
- 逻辑行：`2×6×2=24`
- prediction：144个唯一预测；artifact：192个含K1 alias的arm记录
- DA1_REG1必须逐字节复用DA1_REG0的FA状态；K1 H必须逐logit精确alias Q
- 禁止参数、seed、receiver、K、阈值或矩阵搜索

主要比较：`DA1_REG0−DA0_REG0`、`DA1_REG1−DA0_REG1`、固定DA下的`REG1−REG0`、K5的`H−Q`及DA×head交互。

## 4.版本与本地验证

当前基线提交：待predictor-safe query绑定修复后冻结。已落地的相关提交包括：

- `5179c541`：CER-PLR160核心
- `09a07ccf`、`8cb723f9`：矩阵与动态计数一致性
- `45e148c1`：FA-RDCE3核心及四态指标命名
- `a5d80db0`：设计到实现追踪
- `66d14379`：关闭CER的R0/R1表示契约与alias前query闭合P1
- `e356c15a`：独立truth-side scorer及四态明确指标输出
- `54f0723d`：新类按held-class宏平均；注册遗忘改为固定DA下注册前减注册后，并补齐总体与逐receiver聚合
- `6f779325`、`b8b26e90`：prepare→predict→score CLI、动态capsule和行身份闭合

当前聚焦验证：FA、CER、matrix、runtime、artifact、scorer共38项通过；CLI生命周期5项通过。提交`54f0723d`已经独立复审为`P0=0/P1=0`。随后对CLI的全链检查发现新的协议P0：predictor package虽不含名为truth的字段，但其`physical_binding_receipt`仍含`query_ids_by_class`和`query_observation_ids_by_class`，等价于向predict暴露query真实类别分组。当前正在把预测侧绑定改为与类别无关的扁平固定顺序；修复、复审和真实checkpoint smoke完成前不得发布。

## 5.发布前最小信息

以下字段在唯一runner接管前填入，不作为当前研发阶段的额外gate：

|字段|冻结值|
|---|---|
|Git commit/文件SHA|`PENDING`|
|本地验证命令与结果|`PENDING`|
|N607工作目录|`PENDING`|
|Conda/Python环境|优先`ssr-gpu`；若服务器不存在则必须使用已验证且依赖闭合的现有环境，并记录解释器版本|
|prepare/predict/score精确命令|`PENDING`|
|GPU分配|`PENDING_RESOURCE_PREFLIGHT`|
|run root/log/prediction/score路径|`PENDING_IMMUTABLE_PATHS`|
|PID/CWD/cmdline证据|`PENDING_AFTER_LAUNCH`|
|预期artifact|prepare package、truth sidecar、prediction、manifest、resource receipt、score、complete log|

技术停止仅允许协议/安全违规、错误checkout/hash、覆盖风险、缺prediction闭合或至少两个不同row出现同一确定性异常指纹；不得按运行中性能停止。

## 6.结果表

当前没有真实性能结果，不填写估计值或单元测试代理值。

|receiver|held-class|K|状态码|中文状态|arm|old BA|old-floor|seen-new|H|all-floor|总正确数|判定|
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
|—|—|—|—|—|—|—|—|—|—|—|—|`PENDING`|

## 7.完成后检查与裁决

完整矩阵返回后检查same-row四态、per-class old accuracy、forgetting、seen-new、H、all-floor、总正确数和receiver×K聚合。若性能弱或为负，按冻结阈值关闭相应组件/路线并研发下一原理方法；不扩大矩阵或盲调参数。
