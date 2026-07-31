# D105 R4独立release审查收据

> 已作废：N607 R3预启动canonical loader发现冻结manifest与Git archive在24/54个runtime文件上字节不一致。该run未detach、无性能结果；必须以LF规范化后的新锁完成R5独立审查。

状态：`LOCAL_RELEASE_GO（仅同版本本地发布）`

审查日期：2026-07-31

审查人：独立terra-max release reviewer

审查对象：`D105-CBRC+LPO-RC-qKNN`R4冻结实现

协议：`p2_min_v1`

本收据只批准同一文件集合完成本地Git提交后，交由唯一runner进行N607预检和Phase1 source-held资产链落地。它不构成Phase1 formal asset、Target25、性能、稳定性或`PROMOTABLE`证据。

## 1.最终裁决

|等级|数量|裁决|
|---|---:|---|
|P0|0|未发现清单外动态执行、协议越界、query真值/角色/配额访问、预测前模型打开、资产自授权、输出覆盖或运行入口绕过。|
|P1|0|未发现阻止同版本本地release的闭包、模型重建、锁绑定、测试、报告镜像或旧收据追溯缺口。|
|P2|2|存在PyTorch弃用警告和旧PyTorch反序列化兼容分支；均不改变本次本地验证结论。|

结论：`LOCAL_RELEASE_GO / P0=0 / P1=0 / P2=2`。生效前仍须完成新的本地Git提交、不可覆盖run ID和run-specific launcher冻结；N607尚未落地。

## 2.冻结身份与报告闭合

|对象|独立复核值|
|---|---|
|checkpoint|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|candidate runtime manifest|`48ce446cb406aad67902c80547a48ffbd95d496c728725f2d99fc51b3433f9da`；54个成员|
|candidate method lock|`cdae572cad22721351620828cd1ec36ae1d3432d4b04c70a4c60359a64339d2a`|
|R5真实checkpoint无query smoke|`347a0b659d8db3b44e8bacbe0e9c5c613de9827f1d77862917a453e350f2d338`|
|Phase1根报告与Git镜像|同为`f61f48c843580388c08b514bae6a84182a441a17f731cd09f8ff042f3ef4d3e0`|
|Target25根报告与Git镜像|同为`94c37eb0d99415dfc8fd96810dbf1e9f9d47423d485696788c1d474107d27b3f`|

旧R3收据顶部已明确标记为作废：其45文件闭包遗漏了tap-cache legacy exporter、query evaluator通用`checkpoint_loading`路径及模型工厂的`model_modified`探测，旧`LOCAL_RELEASE_GO`不可用于N607发布或性能声明。

## 3.独立技术复核

- 对最终runtime manifest逐项执行canonical loader，54个文件散列、checkpoint和method lock绑定均通过。独立AST递归扫描未发现任何清单外本地静态导入；`checkpoint_loading`、legacy exporter、`SSDG`、`paper_reproduction`和`model_modified`均不属于D105正式闭包。
- 在新解释器中对真实冻结checkpoint执行最小CVSincNet重建，实际策略为`weights_only_with_explicit_safe_globals`。195个state tensor无missing/unexpected key；与旧精确loader相比，全部state tensor以及确定性IQ上的`z_id`、`z_dom`、`hidden`、`pre_relu`均逐字节一致。guard未观察到`SSDG`、`cvsrffi.checkpoint_loading`、`baselines`、`model_modified`或`paper_reproduction`导入；该检查未读取target或query。
- query evaluator在模型构造前完成四包authority、split、support/query物理ID、同IQ before/after、Phase1 formal asset、candidate runtime/method lock、qKNN、checkpoint SHA和device预检。公开输入面不含truth、role、quota或真实batch类计数；每个query仅对全部已注册类独立决策，`query_rows_used_for_fit=0`且`query_state_updates=0`。
- 10个D105测试文件在`ssr-gpu`中完成211项回归；54文件`py_compile`、5个正式CLI与4个关键子命令`--help`均退出0；`git diff --check`退出0。唯一观察到的运行时警告是`model.py`中的`torch.cuda.amp.autocast`弃用提示。
- R5 smoke完成400个source-held meta step，K1下`M_HEAD=M0`、`M_JOINT=M_DA`保持精确恒等，目标访问=false、query truth读取=false、性能计算=false。该证据仅证明本地技术链路，不是source-held gate、formal asset或Target性能结果。

## 4.P2与后续约束

1. `model.py`仍使用即将弃用的`torch.cuda.amp.autocast`形式。当前实测未改变数值或闭包，后续独立维护可迁移到新API。
2. 当旧PyTorch不存在`safe_globals`时，代码保留严格checkpoint SHA绑定的历史反序列化兼容分支。本次审查环境实际使用`weights_only`。唯一runner必须在N607预检和首个健康检查中记录实际加载策略；该记录不应改变方法、锁或数据验证。

Phase1 source-held预测、truth-open、独立score、gate、外部authority seal和formal asset仍未执行。只有这些同一资产链完整通过，才可启动完整25job/300 scenario-arm pair/600 state prediction surface的开发screen；任何性能陈述必须来自完整同row、独立评分的artifact。
