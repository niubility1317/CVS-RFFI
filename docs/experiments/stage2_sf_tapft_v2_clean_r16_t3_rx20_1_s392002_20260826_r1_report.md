# SF-TAPFT V2 R0交付报告

## 结论

R0代码、严格bundle接口和query只读预测接口已完成本地验证，当前最高状态为`LOCAL_VERIFIED / NO_N607_SMOKE / NO_PERFORMANCE_RESULT`。N607只读preflight通过，但ADV3B02 CORE90 checkpoint没有同谱系的正式Phase1 deployment bundle，因此未创建不可执行配置、未发布release归档、未启动远端适配、未打开query或truth，也未进入R1。

## 预登记运行合同

|字段|值|
|---|---|
|run ID|`stage2_sf_tapft_v2_clean_r16_t3_rx20_1_s392002_20260826_r1`|
|候选|SF-TAPFT V2 R0 clean reference|
|模型|`ADV3B02_CORE90_SOFT_E200`|
|checkpoint|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|
|checkpoint SHA-256|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|target support|receiver`20-1`、seed`713101`、旧类6类、K=10、共60条|
|capsule|`d18-enrollment-before-rx20-1-seed713101-k10-smoke-reuse`|
|split|`stage2b-rx20-1-seed713101-before-support-prefix`|
|协议|`p2_min_v1 / VALIDATED_ONCE`|
|方法预算|Adapter rank16；A/B/C逐阶段；OOF选择后从fresh checkpoint做全60条support final-step refit|
|预期artifact|`cvs.sf_tapft.clean_single.v2` bundle、`selection.json`、无query smoke证据|
|停止规则|checkpoint/bundle谱系不一致、错误capsule/split/K/类序、输出覆盖、协议越权或无法产生合法bundle|

正式命令、GPU、远端output/log路径和release映射暂不登记，因为缺少可加载的同谱系正式Phase1 bundle；填写虚假路径会形成不可执行合同。

## 已完成实现

- Phase1模型侧绑定：checkpoint lineage、runtime、ordered class registry和不可变聚合组件身份。
- target数据侧绑定：`protocol_schema/phase2_data_status/capsule_id/split_id/support_count/per_class_counts`必须由外部可信事实传入严格loader并逐项核对。
- 精确delta平均：只平均许可参数与target head；其他参数和buffer恢复适配前anchor。
- 统一OOF指标与schedule：BA、macro-F1、class floor、NLL、per-class recall/margin、正负flip和参数移动量。
- 最终模型从fresh checkpoint使用完整support按已选schedule训练，并固定取最后optimizer step；`fold0_as_final=false`。
- `clean_single.v2`严格bundle与V1只读兼容。
- 只读预测接口：三参数签名，无truth/role/quota/global assignment输入；逐行覆盖全部注册类，预测前后state不变。

## 本地验证

在`ssr-gpu`环境运行4个聚焦测试文件，共55项全部通过；4个核心模块`py_compile`通过。唯一提示为既存`torch.cuda.amp.GradScaler`弃用warning，不影响测试结论。Task4、Task5、Task6均完成独立P0/P1审查；Task5发现的“可信target binding可选”和“V1 allowlist拓宽”两项P1已定点修复并复审通过。

当前实现提交为`1ad598b5d4f409f84b07d373fdb33fd3c9227d40`；提交时本地HEAD与远端分支OID一致。

## N607只读证据

- 2026-08-26执行直接`N607` preflight成功：项目根可见，8张RTX 3090均可见且检查时空闲。
- CORE90 checkpoint现场SHA-256为`2699eedc…d59c98`。
- 旧V1无query smoke记录确认该capsule读取60条support，且`source_opened=false`、`query_opened=false`、`query_truth_opened=false`；该记录不能替代本次V2 smoke。
- 服务器现有唯一正式`deployment_binding.json`绑定另一checkpoint，SHA-256为`1eb6d07b…307d7`，不可替代CORE90。
- CORE90旧class binding与6类顺序、checkpoint SHA一致；旧int8 manifest也绑定CORE90，但明确写有`formal_phase2_eligible=false`和`provenance_status=UNVERIFIED_UNDER_CURRENT_PROTOCOL`，因此不能作为正式Phase1 bundle输入。

## 用户“使用Phase1 bundle中的样本”的落实方式

Phase1正式deployment bundle按协议不包含raw/source样本或样本级特征。为避免以后对不齐，正确实现是双绑定：模型侧从同一Phase1 bundle固定checkpoint谱系、runtime、6类有序registry和不可变聚合知识；数据侧从同一已验证Phase2 capsule固定60条target support、物理ID、split和逐类计数。任何一侧不一致，严格loader均拒绝。本轮没有把旧source样本复制进Phase2，也没有把其他checkpoint的bundle冒充CORE90。

## 阻塞与下一步

下一步只能先获得一个与CORE90 SHA`2699eedc…d59c98`一致、能够由现有正式loader通过的Phase1 deployment bundle映射；之后补齐不可覆盖配置、一次release归档SHA核对、一次远端编译和真实checkpoint无query smoke。smoke必须回读`support_count=60`、`fold0_as_final=false`、`nonpermitted_changed_count=0`以及source/query/truth均未打开。此前不得启动R1或宣称性能结果。
